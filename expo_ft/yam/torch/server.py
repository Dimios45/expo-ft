"""KARMA json_numpy HTTP contract, serialized inference and boundary reloads."""

import hmac
import io
import logging
import threading
import time

import json_numpy
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image
from starlette.concurrency import run_in_threadpool

LOG = logging.getLogger(__name__)


def build_app(policy, token):
    app = FastAPI(title="YAM fold-70 PyTorch EXPO")
    mutex = threading.Lock()

    def health():
        return dict(
            status="ok",
            policy_type="molmoact2",
            backend="pytorch",
            norm_tag="yam_dual_molmoact2",
            state_dim=14,
            num_cameras=3,
            action_horizon=30,
            num_steps=10,
            dtype="native mixed bf16/fp32",
            hardware_mapping_verified=False,
            gripper_adapter="identity; 1=open",
            **{k: v for k, v in policy.metadata.items() if k != "backend"},
        )

    @app.get("/healthz")
    @app.get("/act")
    def status():
        with mutex:
            return health()

    @app.post("/online/reload")
    def reload(request: Request):
        if not hmac.compare_digest(
            request.headers.get("Authorization", ""), "Bearer " + token
        ):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            with mutex:
                policy.reload_candidate()
                return health()
        except Exception:
            LOG.exception("Candidate reload failed; live policy retained")
            return JSONResponse(
                {"error": "Reload failed; inspect server log"}, status_code=500
            )

    def predict(payload):
        if (
            payload.get("num_steps", 10) != 10
            or payload.get("normalization_tag", "yam_dual_molmoact2")
            != "yam_dual_molmoact2"
        ):
            raise ValueError("Requires YAM normalization tag and 10 flow steps")
        prompt = payload.get("instruction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Missing task instruction")
        images = {}
        for role in ("top", "left", "right"):
            im = np.asarray(payload[role + "_cam"])
            if im.dtype != np.uint8:
                raise ValueError("Camera must be uint8 RGB or compressed bytes")
            if im.ndim == 1:
                with Image.open(io.BytesIO(im.tobytes())) as frame:
                    im = np.asarray(frame.convert("RGB"))
            images[role] = im
        started = time.perf_counter()
        with mutex:
            actions = policy.predict(
                images, np.asarray(payload["state"], np.float32), prompt
            )
            version = policy.metadata["policy_version"]
        return {
            "actions": actions,
            "dt_ms": (time.perf_counter() - started) * 1000,
            "policy_version": version,
        }

    @app.post("/act")
    async def act(request: Request):
        try:
            data = await request.body()
            if len(data) > 20 * 1024**2:
                raise ValueError("Observation exceeds 20 MiB")
            payload = json_numpy.loads(data.decode())
            if not isinstance(payload, dict):
                raise TypeError("Expected observation object")
            result = await run_in_threadpool(predict, payload)
            return Response(json_numpy.dumps(result), media_type="application/json")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            LOG.exception("Inference failed")
            return JSONResponse(
                {"error": "Inference failed; inspect server log"}, status_code=500
            )

    return app
