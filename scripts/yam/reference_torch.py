#!/usr/bin/env python3
"""Run source LeRobot policy offline. No server or robot code is imported."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lerobot-src", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    args = p.parse_args()
    sys.path.insert(0, str(args.lerobot_src))
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors.torch import load_file

    from expo_ft.conversion.yam_loader import YamProcessor

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = PI05Config.from_pretrained(str(args.source))
    cfg.device = "cpu"
    cfg.dtype = args.precision
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    policy = PI05Policy(cfg)
    weights = load_file(str(args.source / "model.safetensors"))
    policy.load_state_dict(weights, strict=True)
    del weights
    # PI05 calls the decoder/vision modules directly; neither vocabulary head
    # participates in flow prediction. Exclude those unused modules from GPU
    # allocation AFTER strict loading, without altering the source checkpoint.
    policy.model.paligemma_with_expert.paligemma.lm_head = torch.nn.Identity()
    policy.model.paligemma_with_expert.gemma_expert.lm_head = torch.nn.Identity()
    policy.to("cuda").eval()
    print(
        "Strict source load complete; unused vocabulary heads excluded from GPU",
        flush=True,
    )
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(args.source),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(args.tokenizer)},
        },
    )
    adapter = YamProcessor(args.source, args.tokenizer)
    args.output.mkdir(parents=True, exist_ok=False)
    captured = {}

    def capture(_module, _inputs, output):
        captured["flow"] = output.detach()

    handle = policy.model.action_out_proj.register_forward_hook(capture)
    records = []
    with torch.inference_mode():
        for path in sorted(args.fixtures.glob("*.npz")):
            raw = np.load(path)
            batch = {
                "observation.state": torch.from_numpy(raw["state_raw"].copy()),
                "task": "fold the towel",
            }
            for role in ["top", "left", "right"]:
                batch["observation.images." + role] = (
                    torch.from_numpy(raw[role].copy()).permute(2, 0, 1).float() / 255
                )
            batch = pre(batch)
            images, masks = policy._preprocess_images(batch)
            tokens = batch["observation.language.tokens"]
            token_mask = batch["observation.language.attention_mask"]
            prepared = adapter.prepare_numpy(
                {k: raw[k] for k in ["top", "left", "right"]}, raw["state_raw"]
            )
            for key, value in [
                ("tokenized_prompt", tokens.cpu().numpy()),
                ("tokenized_prompt_mask", token_mask.cpu().numpy()),
            ]:
                np.testing.assert_array_equal(prepared[key], value)
            np.testing.assert_allclose(
                prepared["state"][0, :14],
                batch["observation.state"][0].cpu().numpy(),
                atol=0,
                rtol=0,
            )
            for key, im in zip(
                ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"], images
            ):
                np.testing.assert_allclose(
                    prepared[key], im.permute(0, 2, 3, 1).cpu().numpy(), atol=0, rtol=0
                )
            noise = torch.from_numpy(raw["noise"].copy()).cuda()
            result = dict(prepared)
            prefix = policy.model.embed_prefix(images, masks, tokens, token_mask)[0]
            result["prefix"] = prefix.float().cpu().numpy()
            for u in [0.1, 0.5, 0.9]:
                # zero target actions makes x_u = u * noise, with explicit common time.
                policy.model(
                    images,
                    masks,
                    tokens,
                    token_mask,
                    torch.zeros_like(noise),
                    noise,
                    torch.tensor([u], device="cuda"),
                )
                result[f"flow_{u}"] = captured["flow"].float().cpu().numpy()
            start = time.perf_counter()
            actions = policy.model.sample_actions(
                images, masks, tokens, token_mask, noise=noise, num_steps=10
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            result["actions_normalized"] = actions.float().cpu().numpy()
            physical = post(actions[:, :, :14]).float().cpu().numpy()
            result["actions_physical"] = physical
            np.testing.assert_allclose(
                adapter.unnormalize_actions(result["actions_normalized"]),
                physical,
                atol=1e-6,
                rtol=1e-6,
            )
            np.savez(args.output / path.name, **result)
            records.append({"fixture": path.name, "sampling_seconds": seconds})
            print(path.name, seconds, flush=True)
    handle.remove()
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "precision": args.precision,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "fixtures": records,
                "preprocessing": "exact_match",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
