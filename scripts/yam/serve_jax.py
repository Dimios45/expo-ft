#!/usr/bin/env python3
"""Serve the converted YAM policy through Karma's json_numpy HTTP protocol."""

import argparse
import io
import logging
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json_numpy
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

LOG = logging.getLogger("yam-jax")
NORM_TAG = "yam_dual_molmoact2"


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


def build_app(policy, checkpoint):
    app = FastAPI(title="YAM pi05 JAX server")
    lock = threading.Lock()

    @app.get("/healthz")
    @app.get("/act")
    def health():
        return {
            "status": "ok",
            "policy_type": "pi05",
            "backend": "jax",
            "checkpoint": str(checkpoint),
            "dtype": "float32",
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
                    {"actions": actions, "dt_ms": (time.perf_counter() - start) * 1000}
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8204)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    import torch
    import uvicorn

    from expo_ft.conversion.yam_loader import YamJaxPolicy

    torch.set_num_threads(4)
    policy = YamJaxPolicy(args.checkpoint, args.tokenizer, dtype="float32")
    LOG.info("Warming up JAX; initial compilation can take several minutes")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    actions = policy.predict(
        {k: frame for k in ("top", "left", "right")}, np.zeros(14, dtype=np.float32)
    )
    LOG.info("Warmup passed: actions=%s; ready to serve", actions.shape)
    uvicorn.run(
        build_app(policy, args.checkpoint), host=args.host, port=args.port, workers=1
    )


if __name__ == "__main__":
    main()
