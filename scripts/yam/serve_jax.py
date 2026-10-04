#!/usr/bin/env python3
"""Serve the converted YAM policy through Karma's json_numpy HTTP protocol."""

import argparse
import io
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json_numpy
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

LOG = logging.getLogger("yam-jax")
NORM_TAG = "yam_dual_molmoact2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8204)
    parser.add_argument(
        "--dtype",
        choices=("fp32", "float32", "bf16", "bfloat16"),
        default="float32",
        help="Weight and model compute precision (default: float32).",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("device", "unified"),
        help="device: GPU allocations; unified: allow spill into system RAM. "
        "Defaults to unified if TF_FORCE_UNIFIED_MEMORY=true, otherwise device.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        help="Unified allocator budget relative to VRAM (default: 1.5). "
        "Only applicable to unified mode; does not reserve physical RAM.",
    )
    args = parser.parse_args(argv)
    args.dtype = {"fp32": "float32", "bf16": "bfloat16"}.get(args.dtype, args.dtype)
    return args


def configure_memory(mode=None, fraction=None):
    """Configure the allocator before importing JAX or initializing CUDA.

    Explicit CLI mode/fraction take precedence over inherited environment values.
    Preserve the previously documented environment-only unified-memory command.
    """
    if mode is None:
        unified = os.environ.get("TF_FORCE_UNIFIED_MEMORY", "").lower() in ("true", "1")
        mode = "unified" if unified else "device"
    if mode not in ("device", "unified"):
        raise ValueError("memory mode must be device or unified")
    if mode == "device" and fraction is not None:
        raise ValueError("--memory-fraction requires --memory-mode unified")
    if mode == "unified":
        if fraction is None:
            fraction = float(
                os.environ.get("XLA_CLIENT_MEM_FRACTION")
                or os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
                or "1.5"
            )
        if not math.isfinite(fraction) or fraction < 1:
            raise ValueError("unified --memory-fraction must be finite and >= 1")
    # Remove both spellings before setting one: JAX rejects conflicting aliases.
    os.environ.pop("XLA_CLIENT_MEM_FRACTION", None)
    os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "true" if mode == "unified" else "false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = (
        "default" if mode == "unified" else "platform"
    )
    if mode == "unified":
        os.environ["XLA_CLIENT_MEM_FRACTION"] = str(fraction)
    return {
        "memory_mode": mode,
        "memory_fraction": fraction,
        "allocator": os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"],
    }


def decode_image(value):
    image = np.asarray(value)
    if image.dtype != np.uint8:
        raise ValueError("Camera must be uint8 RGB or encoded JPEG/PNG bytes")
    if image.ndim == 1:
        with Image.open(io.BytesIO(image.tobytes())) as decoded:
            image = np.asarray(decoded.convert("RGB"))
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Camera must have shape (H,W,3)")
    return image


def build_app(policy, checkpoint, dtype=None, memory_config=None):
    app = FastAPI(title="YAM pi05 JAX server")
    lock = threading.Lock()

    @app.post("/online/reload")
    def reload_online():
        # Pod-configured coordinator workspace; no request-controlled paths.
        if not hasattr(policy, "reload_candidate"):
            return JSONResponse({"error": "Reload unsupported"}, status_code=400)
        try:
            with lock:
                training_root = os.environ.get('YAM_ONLINE_TRAINING_ROOT', str(ROOT / 'artifacts/yam-overlap-test'))
                return policy.reload_candidate(training_root)
        except Exception:
            LOG.exception("Candidate reload failed; previous policy retained where possible")
            return JSONResponse({"error": "Candidate reload failed; inspect server log"}, status_code=500)

    @app.get("/healthz")
    @app.get("/act")
    def health():
        return {
            "status": "ok",
            "policy_type": "pi05",
            "backend": "jax",
            "checkpoint": str(checkpoint),
            "norm_tag": NORM_TAG,
            "state_dim": 14,
            "num_cameras": 3,
            "action_horizon": 30,
            "num_steps": 10,
            "normalization": "saved checkpoint quantiles",
            "action_representation": "absolute joint targets in radians; grippers passed through",
            "gripper_adapter": "none; matches PyTorch pass-through",
            "hardware_mapping_verified": False,
            **getattr(policy, "metadata", {}),
            "dtype": dtype or getattr(policy, "dtype", "float32"),
            **(memory_config or {}),
        }

    @app.post("/act")
    async def act(request: Request):
        try:
            payload = json_numpy.loads((await request.body()).decode())
            if not isinstance(payload, dict):
                raise TypeError("Request must be an object")
            if payload.get("num_steps", 10) != 10:
                raise ValueError("This verified sampler requires num_steps=10")
            if payload.get("normalization_tag", NORM_TAG) != NORM_TAG:
                raise ValueError("Expected YAM normalization_tag=" + NORM_TAG)
            prompt = payload.get("instruction", "fold the towel")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("instruction must be a nonempty string")
            state = np.asarray(payload["state"], dtype=np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError("state must contain 14 finite joint/gripper values")
            images = {
                role: decode_image(payload[role + "_cam"])
                for role in ("top", "left", "right")
            }
            start = time.perf_counter()
            with lock:
                # Match the working PyTorch HTTP wrapper: no coordinate inversion.
                actions = policy.predict(images, state, prompt=prompt)
            return Response(
                content=json_numpy.dumps(
                    {
                        "actions": actions,
                        "dt_ms": (time.perf_counter() - start) * 1000,
                        "dtype": dtype or getattr(policy, "dtype", "float32"),
                    }
                ),
                media_type="application/json",
            )
        except (KeyError, ValueError, TypeError, OSError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            LOG.exception("Inference failed")
            return JSONResponse(
                {"error": "Inference failed; see server log"}, status_code=500
            )

    return app


def main():
    args = parse_args()
    try:
        memory_config = configure_memory(args.memory_mode, args.memory_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    logging.basicConfig(level=logging.INFO)
    LOG.info("Starting YAM: dtype=%s memory=%s", args.dtype, memory_config)
    import torch
    import uvicorn

    from expo_ft.conversion.yam_loader import YamJaxPolicy

    torch.set_num_threads(4)
    policy = YamJaxPolicy(args.checkpoint, args.tokenizer, dtype=args.dtype)
    LOG.info("Warming up JAX; initial compilation can take several minutes")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    actions = policy.predict(
        {k: frame for k in ("top", "left", "right")}, np.zeros(14, dtype=np.float32)
    )
    LOG.info("Warmup passed: actions=%s; ready to serve", actions.shape)
    uvicorn.run(
        build_app(policy, args.checkpoint, args.dtype, memory_config),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
