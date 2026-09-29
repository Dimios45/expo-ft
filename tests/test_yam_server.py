"""No hardware/GPU: verify state/action pass-through matching the PyTorch wrapper."""

import asyncio
import importlib.util
from pathlib import Path

import json_numpy
import numpy as np
from starlette.requests import Request

spec = importlib.util.spec_from_file_location(
    "yam_server", Path(__file__).resolve().parents[1] / "scripts/yam/serve_jax.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_state_and_actions_pass_through():
    wire = np.arange(14, dtype=np.float32) / 14
    wire[[6, 13]] = [1, 0]
    predicted = np.arange(30 * 14, dtype=np.float32).reshape(30, 14) / 420
    predicted[:, 6] = np.linspace(0, 1, 30)
    predicted[:, 13] = np.linspace(1, 0, 30)
    original_prediction = predicted.copy()

    class Policy:
        def predict(self, images, state, prompt):
            np.testing.assert_array_equal(state, wire)
            assert prompt == "fold the towel"
            assert set(images) == {"top", "left", "right"}
            return predicted

    app = server.build_app(Policy(), "fixture")
    health = next(r.endpoint for r in app.routes if r.path == "/healthz")()
    assert health["gripper_adapter"] == "none; matches PyTorch pass-through"
    assert health["dtype"] == "float32"
    assert health["num_steps"] == 10
    assert health["action_horizon"] == 30
    endpoint = next(
        r.endpoint for r in app.routes if r.path == "/act" and "POST" in r.methods
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    payload = {k + "_cam": frame for k in ("top", "left", "right")}
    payload.update(state=wire, instruction="fold the towel", num_steps=10)

    async def receive():
        return {"type": "http.request", "body": json_numpy.dumps(payload).encode()}

    request = Request({"type": "http", "method": "POST", "path": "/act"}, receive)
    response = asyncio.run(endpoint(request))
    assert response.status_code == 200
    actions = json_numpy.loads(response.body)["actions"]
    np.testing.assert_array_equal(actions, original_prediction)
    np.testing.assert_array_equal(predicted, original_prediction)
    np.testing.assert_array_equal(wire[[6, 13]], [1, 0])
