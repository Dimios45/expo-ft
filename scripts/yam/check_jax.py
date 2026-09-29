#!/usr/bin/env python3
"""Offline JAX inference and numerical comparison against saved PyTorch outputs."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def metrics(a, b, atol, rtol):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {"passed": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    error = np.abs(a - b)
    return {
        "passed": finite and bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "rms": float(np.sqrt(np.mean((a - b) ** 2))),
        "atol": atol,
        "rtol": rtol,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    import jax
    import jax.numpy as jnp

    from expo_ft.conversion.yam_loader import (
        YamProcessor,
        as_observation,
        inference_functions,
    )
    from expo_ft.conversion.yam_pi05 import load_model

    jax.config.update("jax_default_matmul_precision", "highest")
    processor = YamProcessor(args.checkpoint, args.tokenizer)
    model = load_model(args.checkpoint, args.precision)
    model.eval()
    state, prefix_fn, fn, sample_fn = inference_functions(model)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.fixtures.glob("*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    reports = []
    for path in paths:
        raw = np.load(path)
        ref = np.load(args.reference / path.name)
        data = processor.prepare_numpy(
            {k: raw[k] for k in ["top", "left", "right"]}, raw["state_raw"]
        )
        obs = as_observation(data)
        checks = {k: metrics(v, ref[k], 0, 0) for k, v in data.items()}
        prefix = np.asarray(prefix_fn(state, obs))
        checks["prefix"] = metrics(prefix, ref["prefix"], 1e-4, 1e-3)
        out = {"prefix": prefix}
        for u in [0.1, 0.5, 0.9]:
            value = np.asarray(
                fn(
                    state,
                    obs,
                    jnp.asarray(raw["noise"]) * u,
                    jnp.array([u], dtype=jnp.float32),
                )
            )
            out[f"flow_{u}"] = value
            checks[f"flow_{u}"] = metrics(value, ref[f"flow_{u}"], 1e-4, 1e-3)
        print(
            path.name,
            "prefix_max",
            checks["prefix"]["max_abs"],
            "flow_max",
            checks["flow_0.5"]["max_abs"],
            flush=True,
        )
        noise = jnp.asarray(raw["noise"])
        start = time.perf_counter()
        normalized = np.asarray(sample_fn(state, obs, noise))
        elapsed = time.perf_counter() - start
        physical = processor.unnormalize_actions(normalized)
        out.update(actions_normalized=normalized, actions_physical=physical)
        checks["actions_normalized"] = metrics(
            normalized, ref["actions_normalized"], 1e-4, 1e-3
        )
        checks["actions_physical"] = metrics(
            physical, ref["actions_physical"], 1e-4, 1e-3
        )
        # A second call verifies deterministic repeated inference after compilation.
        start = time.perf_counter()
        again = np.asarray(sample_fn(state, obs, noise))
        warmed = time.perf_counter() - start
        checks["repeat"] = metrics(again, normalized, 0, 0)
        np.savez(args.output / path.name, **out)
        reports.append(
            {
                "fixture": path.name,
                "checks": checks,
                "first_sampling_seconds": elapsed,
                "warm_sampling_seconds": warmed,
            }
        )
        print(
            path.name,
            "flow_max",
            checks["flow_0.5"]["max_abs"],
            "action_max",
            checks["actions_normalized"]["max_abs"],
            flush=True,
        )
        (args.output / "report.json").write_text(
            json.dumps(
                {"precision": args.precision, "complete": False, "fixtures": reports},
                indent=2,
            )
        )
    report = {
        "precision": args.precision,
        "complete": True,
        "device": str(jax.devices()[0]),
        "memory_stats": jax.devices()[0].memory_stats(),
        "fixtures": reports,
        "passed": all(c["passed"] for r in reports for c in r["checks"].values()),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if args.precision == "float32" and not report["passed"]:
        raise SystemExit(
            "Parity did not meet fixed FP32 thresholds; inspect report.json"
        )


if __name__ == "__main__":
    main()
