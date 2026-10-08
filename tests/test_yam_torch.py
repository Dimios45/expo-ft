"""CPU algorithm/transport tests; never construct a hardware controller."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from expo_ft.yam.rounds import atomic_json, sha
from expo_ft.yam.torch.base import Processor, observation
from expo_ft.yam.torch.learner import Agent, Settings
from expo_ft.yam.torch.runtime import CandidateStore
from expo_ft.yam.torch.server import build_app


def small():
    return Settings(
        filters=4,
        stages=(1, 1, 1, 1),
        hidden=(16,),
        image_latent=16,
        state_latent=8,
        image_size=32,
        batch_size=2,
        updates=2,
        editor_interval=1,
    )


def batch():
    return {
        "rgb": np.zeros((1, 3, 32, 32, 3), np.uint8),
        "next_rgb": np.ones((1, 3, 32, 32, 3), np.uint8),
        "states": np.zeros((1, 14), np.float32),
        "next_states": np.zeros((1, 14), np.float32),
        "actions": np.zeros((1, 420), np.float32),
        "next_candidates": np.zeros((1, 8, 420), np.float32),
        "rewards": np.ones(1, np.float32),
        "masks": np.zeros(1, np.float32),
        "steps": np.full(1, 30, np.float32),
    }


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_torch_update_checkpoint_rng_and_gripper_mask(tmp_path):
    agent = Agent(small())
    before = {k: v.clone() for k, v in agent.critic.state_dict().items()}
    b = batch()
    result = agent.critic_update([b])
    assert result["target_mean"] == 1
    assert any(
        not torch.equal(v, before[k]) for k, v in agent.critic.state_dict().items()
    )
    enc = {k: v.clone() for k, v in agent.encoder.state_dict().items()}
    q = {k: v.clone() for k, v in agent.critic.state_dict().items()}
    result = agent.editor_update([b])
    assert np.isfinite(list(result.values())).all()
    assert all(torch.equal(v, enc[k]) for k, v in agent.encoder.state_dict().items())
    assert all(torch.equal(v, q[k]) for k, v in agent.critic.state_dict().items())
    agent.save(tmp_path)
    expected, _ = agent.select(b["rgb"], b["states"], b["next_candidates"])
    restored = Agent(small())
    restored.restore(tmp_path, training=True)
    actual, _ = restored.select(b["rgb"], b["states"], b["next_candidates"])
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert actual.abs().max() <= agent.cfg.edit_scale
    assert torch.count_nonzero(actual.reshape(30, 14)[:, [6, 13]]) == 0
    assert restored.updates == 1
    # Resumed optimizer states produce exactly the same next update.
    agent.critic_update([b])
    restored.critic_update([b])
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_chunk_target_terminal_and_truncation():
    a = Agent(small())
    b = batch()
    for h in a.target:
        for p in h.parameters():
            p.data.zero_()
        h.net[-1].bias.data.fill_(2)
    b["masks"][:] = 1
    _, m = a.critic_loss(b)
    assert float(m["target_mean"]) == pytest.approx(1 + 0.99**30 * 2)
    with pytest.raises(ValueError, match="bootstrap"):
        a.critic_update([b], [b])


def test_nonfinite_update_does_not_publish_parameters():
    a = Agent(small())
    b = batch()
    b["rewards"][:] = np.nan
    before = {k: v.clone() for k, v in a.state_dict().items()}
    with pytest.raises(FloatingPointError):
        a.critic_update([b])
    assert all(torch.equal(v, before[k]) for k, v in a.state_dict().items())


def test_accumulation_matches_effective_batch():
    a = Agent(small())
    b = Agent(small())
    sample = batch()
    # Constant images make crop choices immaterial; terminal targets remove
    # stochastic next selection. Each microbatch has the same sample.
    a.critic_update([sample, sample])
    b.critic_update([{k: np.concatenate([v, v]) for k, v in sample.items()}])
    for p, q in zip(a.critic.parameters(), b.critic.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)


def make_processor(path):
    mask = torch.ones(14)
    mask[[6, 13]] = 0
    stats = {
        f"{k}.{n}": v
        for k in ("action", "observation.state")
        for n, v in {
            "mask": mask,
            "q01": torch.ones(14) * -2,
            "q99": torch.ones(14) * 2,
        }.items()
    }
    save_file({k: v.clone() for k, v in stats.items()}, str(path / "stats.safetensors"))
    for file, reg in [
        ("policy_preprocessor.json", "molmoact2_masked_normalizer"),
        ("policy_postprocessor.json", "molmoact2_masked_unnormalizer"),
    ]:
        atomic_json(
            path / file,
            {
                "steps": [
                    {
                        "registry_name": reg,
                        "state_file": "stats.safetensors",
                        "config": {"eps": 1e-8},
                    }
                ]
            },
        )
    return Processor(path)


def test_processor_and_wire_gripper_identity(tmp_path):
    p = make_processor(tmp_path)
    x = np.ones((2, 14), np.float32)
    x[:, 13] = 0.2
    y = p.normalize_actions(x)
    np.testing.assert_array_equal(y[:, [6, 13]], x[:, [6, 13]])
    np.testing.assert_allclose(y[:, :6], 0.5)
    images = {k: np.zeros((12, 16, 3), np.uint8) for k in ("top", "left", "right")}
    obs = observation(images, x[0], "fold the towel")
    assert list(obs["observation.state"][[6, 13]]) == list(
        torch.from_numpy(x[0, [6, 13]])
    )
    x[0, 6] = -1
    with pytest.raises(ValueError):
        p.normalize_actions(x)
    with pytest.raises(ValueError):
        observation(images, x[0], "task")


def test_karma_http_prompt_reload_and_finite_actions():
    from fastapi.testclient import TestClient

    class Policy:
        def __init__(self):
            self.metadata = {
                "policy_version": 0,
                "prompt": "fold the towel",
                "backend": "pytorch",
            }

        def predict(self, images, state, prompt):
            observation(images, state, prompt)
            if prompt != self.metadata["prompt"]:
                raise ValueError("Prompt must exactly match experiment prompt")
            return np.zeros((30, 14), np.float32)

        def reload_candidate(self):
            self.metadata["policy_version"] += 1

    import json_numpy

    client = TestClient(build_app(Policy(), "t" * 32))
    payload = {
        "instruction": "fold the towel",
        "state": np.zeros(14, np.float32),
        **{
            k + "_cam": np.zeros((16, 16, 3), np.uint8)
            for k in ("top", "left", "right")
        },
    }
    r = client.post("/act", content=json_numpy.dumps(payload))
    assert r.status_code == 200
    assert json_numpy.loads(r.text)["actions"].shape == (30, 14)
    payload["instruction"] = "wrong"
    assert client.post("/act", content=json_numpy.dumps(payload)).status_code == 400
    assert client.post("/online/reload").status_code == 401
    assert (
        client.post(
            "/online/reload", headers={"Authorization": "Bearer " + "t" * 32}
        ).json()["policy_version"]
        == 1
    )
    assert client.get("/healthz").json()["backend"] == "pytorch"


def test_atomic_candidate_corruption_and_rollback(tmp_path, monkeypatch):
    from io import BytesIO

    import expo_ft.yam.torch.runtime as rt

    (tmp_path / "versions").mkdir()
    weights = tmp_path / "source"
    weights.write_bytes(b"verified payload")
    exp = {"experiment_id": "test", "source_sha": "base", "settings": {}}
    manifest = {
        "version": 1,
        "settings": {},
        "backend": "pytorch",
        "experiment_id": "test",
        "base_sha256": "base",
        "files": {"policy.safetensors": sha(weights)},
    }
    candidate = {
        "version": 1,
        "manifest": manifest,
        "experiment_id": "test",
        "base_sha256": "base",
    }
    monkeypatch.setattr(rt, "request_json", lambda *a: candidate)
    monkeypatch.setattr(
        rt.urllib.request, "urlopen", lambda *a, **k: BytesIO(b"partial")
    )
    store = CandidateStore(tmp_path, "http://unused", "token")
    with pytest.raises(ValueError, match="corrupt"):
        store.install(exp, 0)
    assert not (tmp_path / "versions/0001").exists()
    monkeypatch.setattr(
        rt.urllib.request, "urlopen", lambda *a, **k: BytesIO(weights.read_bytes())
    )
    folder, version = store.install(exp, 0)
    assert version == 1
    assert (folder / "policy.safetensors").read_bytes() == weights.read_bytes()
    assert store.install(exp, 1) is None
    with pytest.raises(ValueError, match="rollback"):
        store.install(exp, 2)


def test_torch_imports_do_not_load_jax():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; import expo_ft.yam.torch.rounds; import expo_ft.yam.torch.runtime; assert "jax" not in sys.modules',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def write_episode(folder, exp, session, frames=61):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir()
    (folder / "meta/episodes").mkdir(parents=True)
    (folder / "data").mkdir()
    (folder / "videos").mkdir()
    atomic_json(
        folder / "expo_session.json",
        {
            "experiment_id": exp["experiment_id"],
            "policy_version": session["policy_version"],
            "server_session": session["session_id"],
            "prompt": exp["prompt"],
            "reward": 1,
            "terminal": "success",
            "prefetch": False,
            "dataset_frame": "karma-recorded",
            "server_metadata": session,
        },
    )
    atomic_json(
        folder / "openpi_control_rollouts.json",
        {
            "speed": 1,
            "chunk_size": 30,
            "episodes": [{"prompt": exp["prompt"], "saved": True, "episode_index": 0}],
        },
    )
    atomic_json(
        folder / "meta/info.json",
        {
            "fps": 30,
            "total_episodes": 1,
            "data_path": "data/frames.parquet",
            "video_path": "videos/{video_key}.mp4",
        },
    )
    rows = []
    for i in range(frames):
        state = [0.0] * 14
        state[6] = 0.25
        state[13] = 0.75
        rows.append(
            dict(
                episode_index=0,
                frame_index=i,
                timestamp=i / 30,
                **{"observation.state": state, "action": state},
            )
        )
    pq.write_table(pa.Table.from_pylist(rows), folder / "data/frames.parquet")
    episode = dict(
        episode_index=0, length=frames, **{"data/chunk_index": 0, "data/file_index": 0}
    )
    for key in (
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ):
        episode.update(
            {
                f"videos/{key}/chunk_index": 0,
                f"videos/{key}/file_index": 0,
                f"videos/{key}/from_timestamp": 0.0,
            }
        )
        with av.open(str(folder / f"videos/{key}.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = 32
            stream.height = 32
            stream.pix_fmt = "yuv420p"
            for i in range(frames):
                frame = av.VideoFrame.from_ndarray(
                    np.full((32, 32, 3), i, np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    pq.write_table(
        pa.Table.from_pylist([episode]), folder / "meta/episodes/episode.parquet"
    )


def test_full_offline_round_replay_resume_and_two_versions(tmp_path, monkeypatch):
    from expo_ft.yam.rounds import read, validate_version
    from expo_ft.yam.torch import rounds

    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    make_processor(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"fixture not a deployed model")
    atomic_json(checkpoint / "config.json", {})
    root = tmp_path / "experiment"
    rounds.initialize(root, checkpoint, "fold the towel", small(), device="cpu")
    exp = read(root / "experiment.json")

    class FixtureBase:
        def __init__(self, *args):
            pass

        def candidates(self, images, state, prompt, count, seed, **kwargs):
            result = np.zeros((count, 30, 14), np.float32)
            result[:, :, [6, 13]] = state[[6, 13]]
            return result

    monkeypatch.setattr(rounds, "BasePolicy", FixtureBase)
    for version in (0, 1):
        session = {
            "experiment_id": exp["experiment_id"],
            "prompt": exp["prompt"],
            "policy_version": version,
            "session_id": f"session{version}",
            "base_sha256": exp["source_sha"],
            "policy_weights_sha256": sha(root / "versions/0001/policy.safetensors")
            if version
            else None,
        }
        (root / "sessions" / session["session_id"]).mkdir()
        atomic_json(root / "sessions" / session["session_id"] / "session.json", session)
        dataset = tmp_path / f"episode{version}"
        write_episode(dataset, exp, session, 61 + version)
        rounds.prepare(root, dataset)
        pending = read(root / "pending.json")
        assert len(pending["replay_inventory"]) == version + 1
        replay_path = Path(pending["replay_inventory"][-1]["path"])
        # KARMA recording inversion is undone once, and masked normalization
        # preserves the resulting live-wire gripper values.
        np.testing.assert_array_equal(
            np.load(replay_path / "state.npy")[0, [6, 13]], [0.75, 0.25]
        )
        before = sha(replay_path / "candidates.npy")
        rounds.prepare(root, dataset)  # Prepared retry reuses the verified cache.
        assert sha(replay_path / "candidates.npy") == before
        rounds.train(root, microbatch=1)
        current = read(root / "current.json")
        assert current["version"] == version + 1
        manifest = validate_version(Path(current["checkpoint"]))
        assert manifest["backend"] == "pytorch"
        assert len(current["episodes"]) == version + 1
        assert not (root / "pending.json").exists()
    # Immutable cached data cannot be silently altered and consumed.
    with (replay_path / "candidates.npy").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        rounds.Replay(manifest["replay_inventory"])


def test_hot_reload_failure_keeps_live_policy_and_session(tmp_path, monkeypatch):
    from expo_ft.yam.rounds import read
    from expo_ft.yam.torch import rounds, runtime

    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    make_processor(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    atomic_json(checkpoint / "config.json", {})
    root = tmp_path / "serve"
    rounds.initialize(root, checkpoint, "fold the towel", small(), device="cpu")
    exp = read(root / "experiment.json")

    class Base:
        def __init__(self, *args):
            self.device = torch.device("cpu")
            self.policy = type("Policy", (), {"reset": lambda self: None})()

    monkeypatch.setattr(runtime, "BasePolicy", Base)
    candidate = root / "versions/0001"
    candidate.mkdir()
    Agent(small()).save(candidate)
    atomic_json(
        candidate / "manifest.json",
        {
            "backend": "pytorch",
            "version": 1,
            "settings": small().dictionary(),
            "experiment_id": exp["experiment_id"],
            "base_sha256": exp["source_sha"],
            "files": {"policy.safetensors": sha(candidate / "policy.safetensors")},
        },
    )

    class Store:
        def install(self, *args):
            return candidate, 1

    policy = runtime.RoundPolicy(root, checkpoint, device="cpu", store=Store())
    assert policy.metadata["policy_version"] == 0
    policy.reload_candidate()
    live = policy.agent
    session = policy.metadata.copy()
    registry = read(root / "current.json")
    with (candidate / "policy.safetensors").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        policy.reload_candidate()
    assert policy.agent is live and policy.metadata == session
    assert read(root / "current.json") == registry


def test_real_coordinator_process_http_auth_and_websocket(tmp_path):
    import json
    import socket
    import subprocess
    import time
    import urllib.error
    import urllib.request

    import msgpack
    from websockets.sync.client import connect

    from expo_ft.yam.torch import rounds

    def port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    make_processor(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    atomic_json(checkpoint / "config.json", {})
    root = tmp_path / "learner"
    rounds.initialize(root, checkpoint, "fold the towel", small(), device="cpu")
    token = "a" * 64
    (root / "control.token").write_text(token)
    ws_port, http_port = port(), port()
    repo = Path(__file__).resolve().parents[1]
    with (tmp_path / "service.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(repo / "scripts/yam/torch/online.py"),
                "learner",
                "--root",
                str(root),
                "--token-file",
                str(root / "control.token"),
                "--host",
                "127.0.0.1",
                "--port",
                str(ws_port),
                "--store-port",
                str(http_port),
                "--policy-url",
                "http://127.0.0.1:1",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{http_port}/status",
                headers={"Authorization": "Bearer " + token},
            )
            for _ in range(100):
                try:
                    with urllib.request.urlopen(req, timeout=1) as response:
                        status = json.load(response)
                    break
                except OSError:
                    assert proc.poll() is None, (tmp_path / "service.log").read_text()
                    time.sleep(0.1)
            else:
                pytest.fail("Coordinator did not start")
            assert status["current"]["version"] == 0 and status["episodes"] == []
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/candidate", timeout=1
                )
            assert error.value.code == 401
            with connect(f"ws://127.0.0.1:{ws_port}", open_timeout=2) as ws:
                ws.send(msgpack.packb({"op": "status", "token": token}))
                reply = msgpack.unpackb(ws.recv(), raw=False)
                assert reply["ok"] and reply["result"]["lease"] is None
                ws.send(msgpack.packb({"op": "status", "token": "bad"}))
                reply = msgpack.unpackb(ws.recv(), raw=False)
                assert not reply["ok"]
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def test_offline_actions_q_script_uses_recorded_rewards(tmp_path, monkeypatch):
    import importlib.util
    from dataclasses import replace

    from expo_ft.yam.rounds import read

    path = Path(__file__).resolve().parents[1] / "scripts/yam/torch/test_actions_q.py"
    spec = importlib.util.spec_from_file_location("test_actions_q_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    make_processor(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    atomic_json(checkpoint / "config.json", {})
    dataset = tmp_path / "recording"
    source = {"experiment_id": "historical-policy", "prompt": "fold the towel"}
    session = {"policy_version": 7, "session_id": "historical-session"}
    write_episode(dataset, source, session)
    label_before = sha(dataset / "expo_session.json")

    class FixtureBase:
        def __init__(self, *args):
            self.post = lambda tensor: tensor

        def candidates(self, images, state, prompt, count, seed, **kwargs):
            actions = np.zeros((count, 30, 14), np.float32)
            actions[:, :, [6, 13]] = state[[6, 13]]
            return actions

        def physical_actions(self, actions):
            return actions.copy()

    monkeypatch.setattr(runner, "BasePolicy", FixtureBase)
    monkeypatch.setattr(runner, "check_model_dependencies", lambda: None)
    monkeypatch.setattr(runner, "Settings", lambda **kwargs: replace(small(), **kwargs))
    output = tmp_path / "diagnostic"
    runner.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--transitions",
            "2",
            "--updates",
            "2",
            "--batch-size",
            "2",
            "--candidates",
            "2",
        ]
    )
    report = read(output / "report.json")
    assert report["sources"][0]["behavior_version"] == 7
    assert report["sources"][0]["reward"] == 1
    assert not report["hardware_control"]
    assert report["after_terminal_mse"] is not None
    assert sha(dataset / "expo_session.json") == label_before
    assert np.load(output / "base_actions.npz")["physical"].shape == (2, 30, 14)
    assert np.load(output / "selected_actions.npz")["base_q_ensemble"].shape == (10, 2)
    assert not (output / "current.json").exists()


def test_model_dependency_preflight_gives_repair_command(monkeypatch):
    from types import SimpleNamespace

    from expo_ft.yam.torch import base

    def missing(name):
        raise ModuleNotFoundError("No module named 'transformers'")

    monkeypatch.setattr(base, "importlib", SimpleNamespace(import_module=missing))
    with pytest.raises(RuntimeError, match="uv pip install --python") as error:
        base.check_model_dependencies()
    assert "transformers==5.5.4" in str(error.value)
