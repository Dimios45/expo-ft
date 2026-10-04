"""No hardware/GPU: verify state/action pass-through matching the PyTorch wrapper."""

import asyncio
import importlib.util
from pathlib import Path

import json_numpy
import numpy as np
import pytest
from starlette.requests import Request

spec = importlib.util.spec_from_file_location(
    "yam_server", Path(__file__).resolve().parents[1] / "scripts/yam/serve_jax.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_state_and_actions_pass_through(dtype):
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

    app = server.build_app(Policy(), "fixture", dtype=dtype)
    health = next(r.endpoint for r in app.routes if r.path == "/healthz")()
    assert health["gripper_adapter"] == "none; matches PyTorch pass-through"
    assert health["dtype"] == dtype
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
    assert json_numpy.loads(response.body)["dtype"] == dtype
    np.testing.assert_array_equal(actions, original_prediction)
    np.testing.assert_array_equal(predicted, original_prediction)
    np.testing.assert_array_equal(wire[[6, 13]], [1, 0])


@pytest.fixture
def memory_env(monkeypatch):
    for name in ("TF_FORCE_UNIFIED_MEMORY", "XLA_CLIENT_MEM_FRACTION",
                 "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR",
                 "XLA_PYTHON_CLIENT_PREALLOCATE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize("value,expected", [
    ("fp32", "float32"), ("float32", "float32"),
    ("bf16", "bfloat16"), ("bfloat16", "bfloat16"),
])
def test_precision_cli_aliases(value, expected):
    args = server.parse_args(["--checkpoint", "fixture", "--tokenizer", "fixture",
                              "--dtype", value])
    assert args.dtype == expected


def test_unified_overrides_platform_allocator(memory_env):
    memory_env.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    memory_env.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    memory_env.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    config = server.configure_memory("unified", 1.5)
    assert config == {"memory_mode": "unified", "memory_fraction": 1.5, "allocator": "default"}
    assert server.os.environ["TF_FORCE_UNIFIED_MEMORY"] == "true"
    assert server.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert server.os.environ["XLA_CLIENT_MEM_FRACTION"] == "1.5"
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in server.os.environ


def test_legacy_environment_command(memory_env):
    memory_env.setenv("TF_FORCE_UNIFIED_MEMORY", "true")
    memory_env.setenv("XLA_CLIENT_MEM_FRACTION", "1.5")
    assert server.configure_memory()["memory_mode"] == "unified"


def test_explicit_device_clears_inherited_unified_settings(memory_env):
    memory_env.setenv("TF_FORCE_UNIFIED_MEMORY", "true")
    memory_env.setenv("XLA_CLIENT_MEM_FRACTION", "1.5")
    config = server.configure_memory("device")
    assert config["allocator"] == "platform"
    assert server.os.environ["TF_FORCE_UNIFIED_MEMORY"] == "false"
    assert "XLA_CLIENT_MEM_FRACTION" not in server.os.environ


@pytest.mark.parametrize("fraction", [0.75, -1, float("nan"), float("inf")])
def test_invalid_unified_budget(memory_env, fraction):
    with pytest.raises(ValueError, match="finite and >= 1"):
        server.configure_memory("unified", fraction)


def test_device_rejects_unused_fraction(memory_env):
    with pytest.raises(ValueError, match="requires"):
        server.configure_memory("device", 1.5)


def test_health_reports_policy_precision_and_memory_mode():
    class Policy:
        dtype = "bfloat16"

    app = server.build_app(Policy(), "fixture", memory_config={"memory_mode": "device"})
    health = next(r.endpoint for r in app.routes if r.path == "/healthz")()
    assert health["dtype"] == "bfloat16"
    assert health["memory_mode"] == "device"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_checkpoint_restored_in_selected_precision(tmp_path, monkeypatch, dtype):
    # Run this suite with JAX_PLATFORMS=cpu: tiny real Orbax restore, no model/GPU.
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from expo_ft.conversion import yam_pi05

    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(tmp_path / "params", {"params": {"w": np.ones((2, 3), np.float32)}})

    class Config:
        def load(self, params, *, remove_extra_params):
            assert not remove_extra_params
            return params

    monkeypatch.setattr(yam_pi05, "model_config", lambda selected: Config())
    params = yam_pi05.load_model(tmp_path, dtype=dtype)
    assert params["w"].dtype == jnp.dtype(dtype)
    np.testing.assert_array_equal(np.asarray(params["w"]).astype(np.float32), np.ones((2, 3)))
