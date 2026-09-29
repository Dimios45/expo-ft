#!/usr/bin/env python3
"""Offline real-weight EXPO memory/update test. Never publishes a policy."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--actor-mode", choices=["frozen", "expert"], default="frozen")
    args = p.parse_args()
    import jax
    import numpy as np
    import torch

    from expo_ft.conversion.yam_pi05 import JAX_CAMERAS
    from expo_ft.yam.base import BasePolicy
    from expo_ft.yam.learner import Settings, YamEXPO

    torch.set_num_threads(4)
    start = time.time()
    cfg = Settings(actor_mode=args.actor_mode)
    base = BasePolicy(
        args.checkpoint,
        args.tokenizer,
        cfg.actor_mode,
        cfg.actor_lr,
        args.resume / "actor.msgpack" if args.resume else None,
    )
    if args.resume:
        base.restore_optimizer(args.resume)
    print("BASE LOADED", flush=True)
    frame = np.zeros((224, 224, 3), np.uint8)
    data = base.prepare(
        {k: frame for k in ("top", "left", "right")},
        np.zeros(14, np.float32),
        "fold the towel",
    )
    rng = np.random.default_rng(123)
    candidates = base.candidates(data, cfg.candidates, rng)
    print("SAMPLED", candidates.shape, flush=True)
    agent = YamEXPO(cfg)
    if args.resume:
        agent.restore(args.resume / "expo.msgpack")
    image = np.concatenate([data[k] for k in JAX_CAMERAS], -1)
    b = {
        "images": image,
        "next_images": image,
        "states": data["state"][:, :14],
        "next_states": data["state"][:, :14],
        "actions": candidates[0].reshape(1, 420),
        "next_candidates": candidates.reshape(1, 8, 420),
        "rewards": np.ones(1, np.float32),
        "masks": np.zeros(1, np.float32),
        "steps": np.full(1, 30, np.float32),
    }
    report = {
        "settings": cfg.dictionary(),
        "device": str(jax.devices()[0]),
        "updates": [],
        "restored_rl_updates": int(agent.state["updates"]),
        "actor_optimizer_restored": base.opt is not None,
    }
    if args.resume and (args.resume / "reload_fixture.npz").exists():
        saved = np.load(args.resume / "reload_fixture.npz")
        np.testing.assert_allclose(
            base.candidates(data, 1, np.random.default_rng(321)),
            saved["base_actions"],
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            agent.select(image, b["states"], saved["candidates"], 123)[0],
            saved["selected"],
        )
        report["resume_verified"] = True
        print("RESUME VERIFIED", flush=True)
    for i in range(2):
        t = time.time()
        metrics = agent.update(b)
        metrics["seconds"] = time.time() - t
        report["updates"].append(metrics)
        print("RL UPDATE", i, metrics, flush=True)
    chosen, _info = agent.select(image, b["states"], b["next_candidates"], 123)
    assert chosen.shape == (1, 420) and np.isfinite(chosen).all()
    if args.actor_mode == "expert":
        # Synthetic target tests mechanics/memory only, never used in a real round.
        before = base.candidates(data, 1, np.random.default_rng(321))
        report["actor"] = base.train_success(data, candidates[0], 777)
        after = base.candidates(data, 1, np.random.default_rng(321))
        report["actor"]["action_change_max"] = float(np.max(abs(after - before)))
        print("ACTOR UPDATE", report["actor"], flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    agent.save(args.output / "expo.msgpack")
    agent.restore(args.output / "expo.msgpack")
    np.testing.assert_array_equal(
        agent.select(image, b["states"], b["next_candidates"], 123)[0], chosen
    )
    base.save(args.output)
    np.savez(
        args.output / "reload_fixture.npz",
        base_actions=base.candidates(data, 1, np.random.default_rng(321)),
        candidates=b["next_candidates"],
        selected=chosen,
    )
    report.update(passed=True, elapsed_seconds=time.time() - start, deployable=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
