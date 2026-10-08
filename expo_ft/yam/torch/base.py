"""Fold-70 contract: saved LeRobot processors, mixed dtypes, identity grippers."""

import hashlib
import importlib
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from expo_ft.yam.rounds import read, sha

LEROBOT_REVISION = "5f8be1af5380d89ba4f645bc69a9c10c632a8025"
ROLES = ("top", "left", "right")


def check_model_dependencies():
    """Fail before decoding episodes if optional LeRobot model extras disappeared."""
    try:
        for name in ("transformers", "peft", "scipy"):
            importlib.import_module(name)
        module = importlib.import_module(
            "lerobot.policies.molmoact2.modeling_molmoact2"
        )
        importlib.import_module("lerobot.policies.molmoact2.processor_molmoact2")
        if module.MolmoAct2ForConditionalGeneration is None:
            raise ImportError(
                "LeRobot's MolmoAct2 transformer implementation is unavailable"
            )
    except (ImportError, RuntimeError) as exc:
        command = shlex.join(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "transformers==5.5.4",
                "peft>=0.18,<0.20",
                "scipy>=1.14,<2",
            ]
        )
        raise RuntimeError(
            f"MolmoAct2 environment check failed: {exc}\nRepair this Python environment with:\n{command}"
        ) from exc


def checkpoint_identity(path):
    path = Path(path)
    names = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]
    for name in names[2:4]:
        names.extend(
            s["state_file"] for s in read(path / name)["steps"] if "state_file" in s
        )
    files = {n: sha(path / n) for n in sorted(set(names))}
    return {
        "files": files,
        "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


def observation(images, state, task):
    state = np.asarray(state, np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Expected 14 finite state values")
    if np.any((state[[6, 13]] < 0) | (state[[6, 13]] > 1)):
        raise ValueError("Grippers must be [0,1], 1=open; no polarity conversion")
    out = {"observation.state": torch.from_numpy(state.copy()), "task": task}
    for role in ROLES:
        im = np.asarray(images[role])
        if im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3:
            raise ValueError(f"{role}: expected HWC uint8 RGB")
        out["observation.images." + role] = torch.from_numpy(im.copy()).permute(2, 0, 1)
    return out


class Processor:
    """Replay normalization mirrors the saved masked quantiles and clamp steps.

    This deliberately does not use the JAX quantiles helper, which normalizes
    all 14 dimensions. The fold-70 processors leave grippers unnormalized.
    """

    def __init__(self, checkpoint):
        self.path = Path(checkpoint)
        pre = read(self.path / "policy_preprocessor.json")["steps"]
        post = read(self.path / "policy_postprocessor.json")["steps"]
        for step in pre + post:
            if step["registry_name"] in (
                "molmoact2_state_frame_transform",
                "molmoact2_action_frame_transform",
            ) and any(
                step["config"].get(k) is not None
                for k in ("joint_signs", "joint_offsets")
            ):
                raise ValueError(
                    "Only the documented identity joint frame is supported"
                )
        normalizer = next(
            s for s in pre if s["registry_name"] == "molmoact2_masked_normalizer"
        )
        unnormalizer = next(
            s for s in post if s["registry_name"] == "molmoact2_masked_unnormalizer"
        )
        self.stats = load_file(str(self.path / normalizer["state_file"]))
        output = load_file(str(self.path / unnormalizer["state_file"]))
        self.eps = normalizer["config"]["eps"]
        for key in ("action", "observation.state"):
            mask = self.stats[key + ".mask"].bool()
            expected = torch.ones(14, dtype=torch.bool)
            expected[[6, 13]] = False
            if not torch.equal(mask, expected):
                raise ValueError("Unexpected normalization mask")
        for k in ("q01", "q99", "mask"):
            if not torch.equal(self.stats["action." + k], output["action." + k]):
                raise ValueError("Action pre/post statistics disagree")

    def normalize(self, value, key):
        x = torch.as_tensor(np.array(value, dtype=np.float32, copy=True))
        if x.shape[-1] != 14 or not torch.isfinite(x).all():
            raise ValueError("Invalid replay vector")
        if torch.any((x[..., [6, 13]] < 0) | (x[..., [6, 13]] > 1)):
            raise ValueError("Replay grippers outside [0,1]")
        lo, hi = self.stats[key + ".q01"], self.stats[key + ".q99"]
        denom = hi - lo
        denom = torch.where(denom == 0, self.eps, denom)
        y = (2 * (x - lo) / denom - 1).clamp(-1, 1)
        return torch.where(self.stats[key + ".mask"].bool(), y, x).numpy()

    def normalize_actions(self, value):
        return self.normalize(value, "action")

    def prepare_replay(self, views, states, prompt):
        # Preserve original RGB for the VLA. Never reconstruct VLA inputs from
        # resized critic images or from clipped/normalized states.
        return {
            "rgb": np.stack([np.stack(views[k]) for k in ROLES], axis=1),
            "raw_state": np.asarray(states, np.float32),
            "state": self.normalize(states, "observation.state"),
        }


class BasePolicy:
    def __init__(self, checkpoint, device="cuda", base_model=None):
        check_model_dependencies()
        from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
        from lerobot.policies.molmoact2.processor_molmoact2 import (
            make_molmoact2_pre_post_processors_from_pretrained,
        )

        self.device = torch.device(device)
        self.processor = Processor(checkpoint)
        cfg = MolmoAct2Config.from_pretrained(str(checkpoint))
        if cfg.chunk_size != 30 or cfg.n_action_steps != 30 or cfg.model_params_fp32:
            raise ValueError("Requires fold-70 native mixed BF16 export, H30")
        cfg.device = device
        overrides = {"device_processor": {"device": device}}
        if base_model:
            cfg.checkpoint_path, cfg.checkpoint_revision = str(base_model), None
            overrides["molmoact2_pack_inputs"] = {
                "checkpoint_path": str(base_model),
                "checkpoint_revision": None,
            }
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.policy = MolmoAct2Policy.from_pretrained(
            str(checkpoint), config=cfg, strict=True
        ).eval()
        self.policy.requires_grad_(False)
        self.pre, self.post = make_molmoact2_pre_post_processors_from_pretrained(
            cfg,
            str(checkpoint),
            preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    @torch.inference_mode()
    def candidates(
        self,
        images,
        state,
        prompt,
        count,
        seed,
        *,
        cache_prefix=True,
        candidate_batch=1,
    ):
        if (
            count < 1
            or candidate_batch < 1
            or (candidate_batch > 1 and not cache_prefix)
        ):
            raise ValueError("Invalid candidate batch or disabled prefix cache")
        batch = self.pre(observation(images, state, prompt))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        values = []
        # Cache only within this observation. The pinned HF implementation
        # exposes encoder_kv_states and generates its own mask on the uncached
        # path; reproduce that exact mask, including its BOTH-mode behavior.
        inputs = self.policy._model_inputs(batch)
        backbone = self.policy._backbone()
        kv, mask = None, None
        if cache_prefix and count > 1:
            with self.policy._autocast_context():
                output = backbone(
                    **{k: v for k, v in inputs.items() if k != "states"}, use_cache=True
                )
                kv = backbone._extract_kv_states(output.past_key_values)
                mask = backbone._get_encoder_attention_mask(
                    inputs["input_ids"], inputs.get("attention_mask")
                )
                del output
        # Sequential action-expert samples bound VRAM without repeating VLM.
        for start in range(0, count, candidate_batch):
            n = min(candidate_batch, count - start)
            if kv is None:
                a = self.policy.predict_action_chunk(
                    batch,
                    inference_action_mode="continuous",
                    num_steps=10,
                    generator=generator,
                )
            else:
                # Only text/mask and KV batch axes are used when prefix KV is
                # provided. RGB crops are intentionally not replicated.
                tiled = {
                    k: (v.repeat_interleave(n, 0) if n > 1 else v)
                    for k, v in inputs.items()
                    if k in ("input_ids", "attention_mask", "states")
                }
                batch_kv = (
                    tuple(tuple(v.repeat_interleave(n, 0) for v in pair) for pair in kv)
                    if n > 1
                    else kv
                )
                batch_mask = (
                    mask.repeat_interleave(n, 0) if n > 1 and mask is not None else mask
                )
                padding = batch.get("action_dim_is_pad")
                if padding is not None and n > 1:
                    padding = padding.repeat_interleave(n, 0)
                with self.policy._autocast_context():
                    a = backbone.generate_actions_from_inputs(
                        **tiled,
                        action_dim_is_pad=padding,
                        action_horizon=self.policy._generation_action_horizon(),
                        num_steps=10,
                        generator=generator,
                        encoder_kv_states=batch_kv,
                        encoder_attention_mask=batch_mask,
                    )[:, :30, :14].float()
            # Canonicalize through the saved postprocessor: critic must rank the
            # same bounded commands that will actually go onto the wire.
            physical = self.post(a).float().cpu().numpy()
            physical[..., [6, 13]] = physical[..., [6, 13]].clip(0, 1)
            values.extend(self.processor.normalize_actions(physical))
        result = np.stack(values)
        if result.shape != (count, 30, 14) or not np.isfinite(result).all():
            raise ValueError("Invalid base candidates")
        return result

    @torch.inference_mode()
    def physical_actions(self, normalized):
        a = torch.as_tensor(
            normalized, device=self.device, dtype=torch.float32
        ).reshape(1, 30, 14)
        result = self.post(a)[0].float().cpu().numpy()
        result[:, [6, 13]] = result[:, [6, 13]].clip(0, 1)
        if result.shape != (30, 14) or not np.isfinite(result).all():
            raise ValueError("Invalid physical actions")
        return result
