"""Import one explicitly versioned LeRobot v3 attempt, without robot access."""

import json
from pathlib import Path

import numpy as np

from .rounds import atomic_json, read, sha

CAMERAS = {
    "top": "observation.images.top",
    "left": "observation.images.left_wrist",
    "right": "observation.images.right_wrist",
}


def windows(length, horizon=30):
    # Each transition has H actual commands and a real next measured state.
    if length < horizon + 1:
        raise ValueError("Need at least 31 recorded frames")
    return np.unique(
        np.r_[np.arange(0, length - horizon, horizon), length - horizon - 1]
    ).astype(np.int64)


def dataset_to_wire(x, frame):
    x = np.array(x, dtype=np.float32, copy=True)
    if frame == "karma-recorded":
        x[..., [6, 13]] = 1 - x[..., [6, 13]]
    elif frame != "wire":
        raise ValueError("Unknown dataset frame")
    return x


def normalize_action(processor, a):
    lo = processor.output_stats["action.q01"]
    hi = processor.output_stats["action.q99"]
    denom = np.where(hi == lo, processor.eps, hi - lo)
    return 2 * (a - lo) / denom - 1


def decode_selected(path, indices):
    import av

    selected = set(map(int, indices))
    out = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in selected:
                out[i] = frame.to_ndarray(format="rgb24")
            if len(out) == len(selected):
                break
    if len(out) != len(selected):
        raise ValueError(f"Video missing requested frames: {path}")
    return out


def hitl_prior_metadata(dataset):
    """Read explicit offline-prior labels without fabricating a server session."""
    dataset = Path(dataset)
    label = read(dataset / "hitl_reward.json")
    rollout = read(dataset / "openpi_control_hitl.json")
    saved = [
        e for e in rollout["episodes"] if e.get("saved") and not e.get("discarded")
    ]
    if len(saved) != 1:
        raise ValueError("HITL prior requires exactly one saved attempt")
    entry = saved[0]
    if (
        label["episode_index"] != entry["episode_index"]
        or label["prompt"] != entry["prompt"]
    ):
        raise ValueError("HITL reward does not identify the saved episode")
    session = dict(label, dataset_frame="karma-recorded", role="explicit offline prior")
    return session, dict(rollout, episodes=saved)


def import_episode(dataset, dest, processor, experiment, current, *, hitl_prior=False):
    import pyarrow.parquet as pq

    dataset = Path(dataset)
    dest = Path(dest)
    if hitl_prior:
        session, rollout = hitl_prior_metadata(dataset)
        label_path = dataset / "hitl_reward.json"
    else:
        session = read(dataset / "expo_session.json")
        rollout = read(dataset / "openpi_control_rollouts.json")
        label_path = dataset / "expo_session.json"
    info = read(dataset / "meta/info.json")
    if not hitl_prior and (
        session["experiment_id"] != experiment["experiment_id"]
        or session["policy_version"] != current["version"]
    ):
        raise ValueError(
            "Episode belongs to another experiment or stale policy version"
        )
    if session["prompt"] != experiment["prompt"]:
        raise ValueError("Prompt changed")
    if session["reward"] not in (0, 1) or session["terminal"] not in (
        "success",
        "failure",
        "truncated",
    ):
        raise ValueError("Invalid human reward/termination")
    if (session["reward"] == 1) != (session["terminal"] == "success"):
        raise ValueError("Success reward/terminal disagree")
    if info["fps"] != 30 or info["total_episodes"] != 1:
        raise ValueError("Require exactly one episode at 30 Hz")
    if rollout["speed"] != 1 or rollout["chunk_size"] != 30:
        raise ValueError("First EXPO experiment requires speed=1 and chunk_size=30")
    if (
        len(rollout["episodes"]) != 1
        or rollout["episodes"][0]["prompt"] != experiment["prompt"]
    ):
        raise ValueError("Unexpected attempts/task")
    # Karma marks Ctrl+C as aborted even when the operator stops after success.
    # The wrapper's explicit human success label defines an absorbing terminal.
    if rollout["episodes"][0].get("aborted") and session["terminal"] not in (
        "success",
        "truncated",
    ):
        raise ValueError("Interrupted unsuccessful attempt must be marked truncated")
    eps = pq.read_table(dataset / "meta/episodes").to_pylist()
    if len(eps) != 1:
        raise ValueError("Expected one episode metadata row")
    e = eps[0]
    data_path = dataset / info["data_path"].format(
        chunk_index=e["data/chunk_index"], file_index=e["data/file_index"]
    )
    rows = [
        r
        for r in pq.read_table(data_path).to_pylist()
        if r["episode_index"] == e["episode_index"]
    ]
    if len(rows) != e["length"]:
        raise ValueError("Episode frame count mismatch")
    intervention = None
    if hitl_prior:
        intervention = np.asarray(
            [r["intervention"] for r in rows], dtype=np.float32
        ).reshape(-1)
        if (
            intervention.shape != (len(rows),)
            or not np.isin(intervention, [0, 1]).all()
        ):
            raise ValueError("Invalid HITL intervention labels")
    t = np.array([r["timestamp"] for r in rows])
    fi = np.array([r["frame_index"] for r in rows])
    if not np.array_equal(fi, np.arange(len(rows))) or not np.allclose(
        np.diff(t), 1 / 30, atol=1e-4
    ):
        raise ValueError("Noncontiguous recording")
    raw_s = dataset_to_wire(
        [r["observation.state"] for r in rows], session["dataset_frame"]
    )
    raw_a = dataset_to_wire([r["action"] for r in rows], session["dataset_frame"])
    if (
        raw_s.shape != (len(rows), 14)
        or raw_a.shape != raw_s.shape
        or not np.isfinite(raw_s).all()
        or not np.isfinite(raw_a).all()
    ):
        raise ValueError("Invalid state/action arrays")
    starts = windows(len(rows))
    ends = starts + 30
    anchors = np.unique(np.r_[starts, ends])
    index = {v: i for i, v in enumerate(anchors)}
    views = {}
    video_paths = []
    for role, key in CAMERAS.items():
        video_meta = "videos/" + key if "videos/" + key + "/chunk_index" in e else key
        path = dataset / info["video_path"].format(
            video_key=key,
            chunk_index=e[video_meta + "/chunk_index"],
            file_index=e[video_meta + "/file_index"],
        )
        offset = e[video_meta + "/from_timestamp"]
        absolute = np.rint((offset + t[anchors]) * 30).astype(int)
        decoded = decode_selected(path, absolute)
        views[role] = [decoded[int(i)] for i in absolute]
        video_paths.append(path)
    prepared = [
        processor.prepare_numpy(
            {k: v[i] for k, v in views.items()}, raw_s[f], experiment["prompt"]
        )
        for i, f in enumerate(anchors)
    ]
    from expo_ft.conversion.yam_pi05 import JAX_CAMERAS

    arrays = {
        "images": np.stack(
            [np.concatenate([d[k][0] for k in JAX_CAMERAS], axis=-1) for d in prepared]
        ),
        "state": np.concatenate([d["state"] for d in prepared]),
        "tokens": np.concatenate([d["tokenized_prompt"] for d in prepared]),
        "token_mask": np.concatenate([d["tokenized_prompt_mask"] for d in prepared]),
        "actions": np.stack(
            [normalize_action(processor, raw_a[i : i + 30]) for i in starts]
        ),
        "current": np.array([index[v] for v in starts]),
        "next": np.array([index[v] for v in ends]),
        "rewards": np.where(
            ends == len(rows) - 1,
            session["reward"] * experiment["settings"]["discount"] ** 29,
            0,
        ).astype(np.float32),
        "masks": np.where(
            (ends == len(rows) - 1) & (session["terminal"] != "truncated"), 0, 1
        ).astype(np.float32),
    }
    if intervention is not None:
        arrays["intervention"] = np.stack([intervention[i : i + 30] for i in starts])
    provenance = {
        str(p.relative_to(dataset)): sha(p)
        for p in [data_path, label_path, *video_paths]
    }
    import hashlib

    eid = hashlib.sha256(
        json.dumps(
            {k: v for k, v in provenance.items() if k != label_path.name},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if eid in current["episodes"]:
        raise ValueError("Episode already consumed")
    dest.mkdir(parents=True, exist_ok=False)
    for name, a in arrays.items():
        np.save(dest / (name + ".npy"), a)
    atomic_json(
        dest / "metadata.json",
        {
            "episode_id": eid,
            "session": session,
            "source_hashes": provenance,
            "collection_manifest_sha256": sha(
                dataset
                / (
                    "openpi_control_hitl.json"
                    if hitl_prior
                    else "openpi_control_rollouts.json"
                )
            ),
            "human_frames": int(intervention.sum())
            if intervention is not None
            else None,
            "transitions": len(starts),
            "frames": len(rows),
            "representation": "wire-frame normalized by saved checkpoint; full 30-command windows",
            "limitations": "LeRobot timestamps are frame indices, not measured wall-clock execution times.",
        },
    )
    return eid


class Episode:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = read(self.path / "metadata.json")
        self.arrays = {
            p.stem: np.load(p, mmap_mode="r") for p in self.path.glob("*.npy")
        }

    def __len__(self):
        return len(self.arrays["current"])

    def observation(self, i, next_state=False):
        from expo_ft.conversion.yam_pi05 import JAX_CAMERAS

        a = self.arrays
        j = int(a["next" if next_state else "current"][i])
        im = a["images"][j]
        return dict(
            **{k: im[None, ..., n * 3 : n * 3 + 3] for n, k in enumerate(JAX_CAMERAS)},
            state=a["state"][j : j + 1],
            tokenized_prompt=a["tokens"][j : j + 1],
            tokenized_prompt_mask=a["token_mask"][j : j + 1],
        )

    def batch(self, i, candidates):
        a = self.arrays
        j = int(a["current"][i])
        k = int(a["next"][i])
        return {
            "images": np.array(a["images"][j : j + 1]),
            "next_images": np.array(a["images"][k : k + 1]),
            "states": np.array(a["state"][j : j + 1, :14]),
            "next_states": np.array(a["state"][k : k + 1, :14]),
            "actions": np.array(a["actions"][i : i + 1]).reshape(1, 420),
            "next_candidates": candidates.reshape(1, -1, 420),
            "rewards": np.array(a["rewards"][i : i + 1]),
            "masks": np.array(a["masks"][i : i + 1]),
            "steps": np.array([30], np.float32),
        }
