"""Offline EXPO mathematics, action dimensions, resumability and round isolation."""

from pathlib import Path
from typing import ClassVar

import jax
import numpy as np
import pytest

from expo_ft.yam.learner import Settings, YamEXPO, chunk_target
from expo_ft.yam.replay import dataset_to_wire, windows
from expo_ft.yam.rounds import atomic_json, commit_round, lock, read


def tiny():
    return Settings(
        candidates=2,
        edits=2,
        stages=(1,),
        filters=4,
        image_latent=8,
        state_latent=4,
        hidden=(16,),
    )


def batch():
    return {
        "images": np.zeros((1, 8, 8, 9), np.float32),
        "next_images": np.zeros((1, 8, 8, 9), np.float32),
        "states": np.zeros((1, 14), np.float32),
        "next_states": np.zeros((1, 14), np.float32),
        "actions": np.zeros((1, 420), np.float32),
        "next_candidates": np.zeros((1, 2, 420), np.float32),
        "rewards": np.ones(1, np.float32),
        "masks": np.zeros(1, np.float32),
        "steps": np.full(1, 30, np.float32),
    }


def test_windows_and_gripper_recording_boundary():
    for size in (31, 32, 60, 61, 9000):
        starts = windows(size)
        assert np.all(starts + 30 < size) and starts[-1] + 30 == size - 1
    with pytest.raises(ValueError):
        windows(30)
    raw = np.arange(14, dtype=np.float32)
    raw[[6, 13]] = [0, 1]
    native = dataset_to_wire(raw, "karma-recorded")
    np.testing.assert_array_equal(native[[6, 13]], [1, 0])
    np.testing.assert_array_equal(dataset_to_wire(native, "karma-recorded"), raw)
    np.testing.assert_array_equal(dataset_to_wire(raw, "wire"), raw)


def test_terminal_bootstrap():
    value = chunk_target(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([30, 30]),
        np.array([100.0, 2.0]),
        0.99,
    )
    np.testing.assert_allclose(value, [1.0, 2 * 0.99**30])


def test_update_target_editor_and_roundtrip(tmp_path):
    cfg = tiny()
    agent = YamEXPO(cfg, image_size=8)
    old = jax.device_get(agent.state["target_q"])
    metrics = agent.update(batch())
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["critic_grad_norm"] > 0 and metrics["editor_grad_norm"] > 0
    for before, current, target in zip(
        jax.tree.leaves(old),
        jax.tree.leaves(agent.state["critic"].params),
        jax.tree.leaves(agent.state["target_q"]),
    ):
        np.testing.assert_allclose(
            target,
            (1 - cfg.tau) * before + cfg.tau * np.asarray(current),
            rtol=1e-5,
            atol=1e-7,
        )
    b = batch()
    args = (b["images"], b["states"], b["next_candidates"], 7)
    a, info = agent.select(*args)
    assert a.shape == (1, 420) and info["q_values"].shape == (10, 1, 4)
    assert len(set(info["pair"].tolist())) == 2
    f = tmp_path / "expo.msgpack"
    agent.save(f)
    restored = YamEXPO(cfg, image_size=8)
    restored.restore(f)
    np.testing.assert_array_equal(restored.select(*args)[0], a)
    m1 = agent.update(b)
    m2 = restored.update(b)
    for key in m1:
        np.testing.assert_allclose(m1[key], m2[key], rtol=1e-6)


def test_round_lock_and_publication(tmp_path):
    (tmp_path / "versions").mkdir()
    parent = {"version": 0, "episodes": [], "checkpoint": None}
    atomic_json(tmp_path / "current.json", parent)
    with lock(tmp_path):
        with pytest.raises(RuntimeError), lock(tmp_path):
            pass
        work = tmp_path / "versions" / ".training"
        work.mkdir()
        (work / "ok").write_text("yes")
        nxt = commit_round(tmp_path, work, parent, "episode-a")
    assert read(tmp_path / "current.json") == nxt and nxt["episodes"] == ["episode-a"]
    assert (Path(nxt["checkpoint"]) / "ok").exists()


@pytest.mark.parametrize("aborted", [False, True])
@pytest.mark.parametrize("video_prefix", ["", "videos/"])
def test_import_lerobot_episode_and_stale_version(tmp_path, aborted, video_prefix):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    from expo_ft.yam.replay import CAMERAS, Episode, import_episode

    dataset = tmp_path / "dataset"
    (dataset / "meta/episodes").mkdir(parents=True)
    (dataset / "data").mkdir()
    info = {
        "fps": 30,
        "total_episodes": 1,
        "data_path": "data/data.parquet",
        "video_path": "videos/{video_key}.mp4",
    }
    atomic_json(dataset / "meta/info.json", info)
    exp = {
        "experiment_id": "test",
        "prompt": "fold the towel",
        "settings": {"discount": 0.99},
    }
    cur = {"version": 0, "episodes": []}
    session = {
        "experiment_id": "test",
        "policy_version": 0,
        "prompt": exp["prompt"],
        "reward": 1,
        "terminal": "success",
        "dataset_frame": "karma-recorded",
    }
    atomic_json(dataset / "expo_session.json", session)
    atomic_json(
        dataset / "openpi_control_rollouts.json",
        {
            "speed": 1,
            "chunk_size": 30,
            "episodes": [{"prompt": exp["prompt"], "aborted": aborted}],
        },
    )
    meta = {
        "episode_index": 0,
        "length": 31,
        "data/chunk_index": 0,
        "data/file_index": 0,
    }
    (dataset / "videos").mkdir()
    for key in CAMERAS.values():
        meta.update(
            {
                video_prefix + key + "/chunk_index": 0,
                video_prefix + key + "/file_index": 0,
                video_prefix + key + "/from_timestamp": 0.0,
            }
        )
        with av.open(str(dataset / "videos" / f"{key}.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = 16
            stream.height = 16
            stream.pix_fmt = "yuv420p"
            for i in range(31):
                frame = av.VideoFrame.from_ndarray(
                    np.full((16, 16, 3), i, np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    pq.write_table(pa.Table.from_pylist([meta]), dataset / "meta/episodes/file.parquet")
    rows = [
        {
            "episode_index": 0,
            "frame_index": i,
            "timestamp": i / 30,
            "observation.state": [0.0] * 14,
            "action": [0.0] * 14,
        }
        for i in range(31)
    ]
    pq.write_table(pa.Table.from_pylist(rows), dataset / "data/data.parquet")

    class Processor:
        output_stats: ClassVar[dict] = {
            "action.q01": np.zeros(14),
            "action.q99": np.ones(14),
        }
        eps = 1e-8

        def prepare_numpy(self, images, state, prompt):
            from expo_ft.conversion.yam_pi05 import JAX_CAMERAS

            return dict(
                **{
                    k: images[r][None].astype(np.float32) / 255
                    for k, r in zip(JAX_CAMERAS, ("top", "left", "right"))
                },
                state=np.pad(state, (0, 18))[None],
                tokenized_prompt=np.zeros((1, 200), np.int32),
                tokenized_prompt_mask=np.ones((1, 200), bool),
            )

    dest = tmp_path / "replay"
    import_episode(dataset, dest, Processor(), exp, cur)
    ep = Episode(dest)
    assert len(ep) == 1
    b = ep.batch(0, np.zeros((2, 30, 14), np.float32))
    np.testing.assert_allclose(b["rewards"], [0.99**29])
    assert b["masks"][0] == 0
    if aborted:
        atomic_json(
            dataset / "expo_session.json",
            {**session, "reward": 0, "terminal": "failure"},
        )
        with pytest.raises(ValueError, match="must be marked truncated"):
            import_episode(dataset, tmp_path / "invalid", Processor(), exp, cur)
        atomic_json(
            dataset / "expo_session.json",
            {**session, "reward": 0, "terminal": "truncated"},
        )
        truncated = tmp_path / "truncated"
        import_episode(dataset, truncated, Processor(), exp, cur)
        assert Episode(truncated).arrays["masks"][-1] == 1
        assert Episode(truncated).arrays["rewards"][-1] == 0
        atomic_json(dataset / "expo_session.json", session)
    np.testing.assert_array_equal(b["states"][0, [6, 13]], [1, 1])
    # Wire-frame grippers are normalized once; joints remain unflipped.
    np.testing.assert_array_equal(
        b["actions"].reshape(30, 14)[0], [-1] * 6 + [1] + [-1] * 6 + [1]
    )
    with pytest.raises(ValueError, match="stale"):
        import_episode(
            dataset,
            tmp_path / "stale",
            Processor(),
            exp,
            {"version": 1, "episodes": []},
        )


def test_expert_filter_and_optimizer_serialization():
    import optax
    from flax import nnx

    from expo_ft.yam.base import dump_tree, expert_filter, restore_tree

    class Small(nnx.Module):
        def __init__(self):
            self.action_in_proj = nnx.Linear(2, 2, rngs=nnx.Rngs(1))
            self.vision = nnx.Linear(2, 2, rngs=nnx.Rngs(2))

    _, train, frozen = nnx.split(Small(), expert_filter, ...)
    assert list(train.keys()) == ["action_in_proj"] and list(frozen.keys()) == [
        "vision"
    ]
    tx = optax.adam(1e-5)
    opt = tx.init(train)
    for value in (train, opt):
        restored = restore_tree(value, dump_tree(value))
        assert jax.tree.structure(restored) == jax.tree.structure(value)
        for a, b in zip(jax.tree.leaves(value), jax.tree.leaves(restored)):
            np.testing.assert_array_equal(a, b)


def test_editor_bounds_and_change_of_variables():
    import jax.numpy as jnp

    from expo_ft.yam.learner import gaussian_edit

    mean = jnp.zeros((2, 420))
    std = jnp.zeros_like(mean)
    key = jax.random.PRNGKey(99)
    edit, logp = gaussian_edit(mean, std, key, 0.2)
    assert np.max(np.abs(edit)) <= 0.2
    u = np.asarray(jax.random.normal(key, mean.shape))
    expected = (
        -0.5 * (u * u + np.log(2 * np.pi)) - np.log(1 - np.tanh(u) ** 2) - np.log(0.2)
    ).sum(-1)
    np.testing.assert_allclose(logp, expected, rtol=1e-5, atol=1e-3)


def test_version_integrity_and_two_commits(tmp_path):
    from expo_ft.yam.rounds import sha, validate_version

    (tmp_path / "versions").mkdir()
    old = {"version": 0, "episodes": [], "checkpoint": None}
    atomic_json(tmp_path / "current.json", old)
    for n in range(2):
        work = tmp_path / "versions" / f".stage{n}"
        work.mkdir()
        (work / "expo.msgpack").write_bytes(b"weights")
        atomic_json(
            work / "manifest.json",
            {"version": n + 1, "files": {"expo.msgpack": sha(work / "expo.msgpack")}},
        )
        with lock(tmp_path):
            old = commit_round(tmp_path, work, old, f"ep{n}")
        validate_version(old["checkpoint"])
    assert old["version"] == 2 and old["episodes"] == ["ep0", "ep1"]
    (Path(old["checkpoint"]) / "expo.msgpack").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        validate_version(old["checkpoint"])
