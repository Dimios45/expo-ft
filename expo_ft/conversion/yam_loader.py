"""Offline YAM preprocessing and OpenPI loading, preserving LeRobot semantics.

Inputs are in the SOURCE CHECKPOINT frame. No wire/dataset gripper inversion is
inferred here: hardware adaptation is a separate, explicitly unverified task.
CPU torch is used only for the exact original bilinear image resize operation.
"""

import json
from pathlib import Path

import numpy as np

from .yam_pi05 import CAMERAS, JAX_CAMERAS, load_model, validate_config


class YamProcessor:
    def __init__(self, checkpoint, tokenizer):
        from safetensors.numpy import load_file
        from transformers import AutoTokenizer

        self.root = Path(checkpoint)
        self.config = json.loads((self.root / "config.json").read_text())
        validate_config(self.config)
        pre = json.loads((self.root / "policy_preprocessor.json").read_text())
        post = json.loads((self.root / "policy_postprocessor.json").read_text())
        normalizer = next(
            s for s in pre["steps"] if s["registry_name"] == "normalizer_processor"
        )
        unnormalizer = next(
            s for s in post["steps"] if s["registry_name"] == "unnormalizer_processor"
        )
        if normalizer["config"]["norm_map"] != {
            "VISUAL": "IDENTITY",
            "STATE": "QUANTILES",
            "ACTION": "QUANTILES",
        }:
            raise ValueError("Unsupported normalization")
        for s in pre["steps"]:
            if (
                s["registry_name"] == "rename_observations_processor"
                and s["config"]["rename_map"]
            ):
                raise ValueError("Nonempty saved rename map needs explicit conversion")
        self.stats = load_file(str(self.root / normalizer["state_file"]))
        self.output_stats = load_file(str(self.root / unnormalizer["state_file"]))
        self.eps = np.float32(normalizer["config"]["eps"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer), local_files_only=True
        )
        self.tokenizer_config = next(
            s["config"]
            for s in pre["steps"]
            if s["registry_name"] == "tokenizer_processor"
        )

    def normalize(self, value, key="observation.state"):
        value = np.asarray(value, dtype=np.float32)
        lo, hi = self.stats[key + ".q01"], self.stats[key + ".q99"]
        denom = np.where(hi == lo, self.eps, hi - lo)
        return np.float32(2) * (value - lo) / denom - np.float32(1)

    def unnormalize_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)[..., :14]
        lo, hi = self.output_stats["action.q01"], self.output_stats["action.q99"]
        denom = np.where(hi == lo, self.eps, hi - lo)
        return (actions + np.float32(1)) * denom / np.float32(2) + lo

    def prepare_numpy(self, images, state, prompt="fold the towel"):
        import torch
        import torch.nn.functional as F

        state = np.asarray(state, dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected finite source-frame state with shape (14,)")
        normalized = self.normalize(state)
        bins = np.digitize(normalized, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        text = f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "
        tc = self.tokenizer_config
        tokens = self.tokenizer(
            [text],
            max_length=tc["max_length"],
            truncation=tc["truncation"],
            padding=tc["padding"],
            padding_side=tc["padding_side"],
            return_tensors="np",
        )
        result = {
            "state": np.pad(normalized, (0, 18))[None],
            "tokenized_prompt": tokens["input_ids"].astype(np.int32),
            "tokenized_prompt_mask": tokens["attention_mask"].astype(bool),
        }
        for role, key in zip(CAMERAS, JAX_CAMERAS):
            image = np.asarray(images[role])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{role} must be HWC RGB uint8")
            x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 255
            h, w = image.shape[:2]
            if (h, w) != (224, 224):
                ratio = max(w / 224, h / 224)
                nh, nw = int(h / ratio), int(w / ratio)
                x = F.interpolate(
                    x, (nh, nw), mode="bilinear", align_corners=False
                ).clamp(0, 1)
                top, left = (224 - nh) // 2, (224 - nw) // 2
                x = F.pad(x, (left, 224 - nw - left, top, 224 - nh - top))
            result[key] = (x * 2 - 1).permute(0, 2, 3, 1).numpy()
        return result


def as_observation(data):
    import jax.numpy as jnp
    from openpi.models.model import Observation

    return Observation(
        images={k: jnp.asarray(data[k]) for k in JAX_CAMERAS},
        image_masks={k: jnp.ones((1,), dtype=bool) for k in JAX_CAMERAS},
        state=jnp.asarray(data["state"]),
        tokenized_prompt=jnp.asarray(data["tokenized_prompt"]),
        tokenized_prompt_mask=jnp.asarray(data["tokenized_prompt_mask"]),
    )


def flow(model, obs, noisy_actions, times):
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask

    prefix, pm, pa = model.embed_prefix(obs)
    suffix, sm, sa, cond = model.embed_suffix(obs, noisy_actions, times)
    mask = jnp.concatenate([pm, sm], axis=1)
    (_, result), _ = model.PaliGemma.llm(
        [prefix, suffix],
        mask=make_attn_mask(mask, jnp.concatenate([pa, sa])),
        positions=jnp.cumsum(mask, axis=1) - 1,
        adarms_cond=[None, cond],
    )
    return model.action_out_proj(result[:, -model.action_horizon :])


def inference_functions(model):
    """Pure JIT calls: inference never returns a second copy of NNX state."""
    import jax
    from flax import nnx

    graph, state = nnx.split(model)

    def compile_highest(fn):
        def wrapped(*args):
            with jax.default_matmul_precision("highest"):
                return fn(*args)

        return jax.jit(wrapped)

    prefix = compile_highest(lambda s, o: nnx.merge(graph, s).embed_prefix(o)[0])
    velocity = compile_highest(lambda s, o, x, t: flow(nnx.merge(graph, s), o, x, t))
    sample = compile_highest(
        lambda s, o, n: nnx.merge(graph, s).sample_actions(
            jax.random.key(0), o, noise=n[:, None], num_steps=10
        )
    )
    return state, prefix, velocity, sample


class YamJaxPolicy:
    def __init__(self, checkpoint, tokenizer, dtype="float32"):
        self.dtype = dtype
        if (Path(checkpoint) / "INCOMPLETE").exists():
            raise ValueError("Incomplete conversion")
        self.processor = YamProcessor(checkpoint, tokenizer)
        self.model = load_model(checkpoint, dtype)
        self.model.eval()
        self._state, _, _, self._sample = inference_functions(self.model)

    def predict(self, images, state, prompt="fold the towel", noise=None):
        import jax.numpy as jnp

        if noise is None:
            noise = (
                np.random.default_rng().standard_normal((1, 30, 32)).astype(np.float32)
            )
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (1, 30, 32) or not np.isfinite(noise).all():
            raise ValueError("Noise must have shape (1,30,32) and be finite")
        obs = as_observation(self.processor.prepare_numpy(images, state, prompt))
        normalized = np.asarray(self._sample(self._state, obs, jnp.asarray(noise)))
        result = self.processor.unnormalize_actions(normalized)[0]
        if result.shape != (30, 14) or not np.isfinite(result).all():
            raise ValueError("Invalid generated actions")
        return result
