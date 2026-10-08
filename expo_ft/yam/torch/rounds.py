"""Transactional episode preparation, cached base candidates, and Torch updates."""

import time
import uuid
from pathlib import Path

import numpy as np
import torch

from expo_ft.yam.replay import import_episode
from expo_ft.yam.rounds import atomic_json, commit_round, read, sha, validate_version

from .base import LEROBOT_REVISION, ROLES, BasePolicy, Processor, checkpoint_identity
from .learner import Agent, Settings


def initialize(root, checkpoint, prompt, settings=None, base_model=None, device="cuda"):
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    if not prompt.strip():
        raise ValueError("Empty prompt")
    identity = checkpoint_identity(checkpoint)
    Processor(checkpoint)  # Validate masks and frame before creating the run.
    root.mkdir(parents=True, exist_ok=False)
    for name in ("replay", "versions", "sessions"):
        (root / name).mkdir()
    atomic_json(
        root / "experiment.json",
        {
            "schema": 1,
            "backend": "pytorch",
            "profile": "torch-stable-v1",
            "checkpoint": str(checkpoint),
            "base_model": str(base_model) if base_model else None,
            "device": device,
            "prompt": prompt,
            "experiment_id": uuid.uuid4().hex,
            "settings": (settings or Settings()).dictionary(),
            "checkpoint_provenance": identity,
            "source_sha": identity["sha256"],
            "lerobot_revision": LEROBOT_REVISION,
        },
    )
    atomic_json(
        root / "current.json", {"version": 0, "episodes": [], "checkpoint": None}
    )


def experiment(root):
    exp, cur = read(root / "experiment.json"), read(root / "current.json")
    if exp.get("profile") != "torch-stable-v1":
        raise ValueError("Not a native Torch experiment")
    return exp, cur


def prepare(root, dataset, allow_older_policy=False):
    exp, cur = experiment(root)
    dataset = Path(dataset).resolve()
    label = read(dataset / "expo_session.json")
    server = read(root / "sessions" / label["server_session"] / "session.json")
    version = label["policy_version"]
    if type(version) != int or not 0 <= version <= cur["version"]:
        raise ValueError("Invalid behavior version")
    if version != cur["version"] and not allow_older_policy:
        raise ValueError("Older policy requires explicit off-policy admission")
    if label.get("prefetch") is not False:
        raise ValueError("Requires no-prefetch rollout")
    for k, v in (
        ("experiment_id", exp["experiment_id"]),
        ("policy_version", version),
        ("prompt", exp["prompt"]),
    ):
        if label[k] != v or server[k] != v:
            raise ValueError("Collection identity mismatch: " + k)
    if server.get("base_sha256") != exp["source_sha"]:
        raise ValueError("Rollout base checkpoint differs")
    if (
        label["server_metadata"]["policy_weights_sha256"]
        != server["policy_weights_sha256"]
    ):
        raise ValueError("Behavior weights mismatch")
    if version:
        source = root / "versions" / f"{version:04d}"
        validate_version(source)
        if sha(source / "policy.safetensors") != server["policy_weights_sha256"]:
            raise ValueError("Changed behavior checkpoint")
    elif server["policy_weights_sha256"] is not None:
        raise ValueError("Unexpected base-version editor")
    pending_path = root / "pending.json"
    if pending_path.exists():
        pending = read(pending_path)
        if pending["episode_id"] in cur["episodes"]:
            pending_path.unlink()
            return
        if (
            pending["parent"] != cur["version"]
            or pending["dataset"] != str(dataset)
            or pending["label_sha"] != sha(dataset / "expo_session.json")
        ):
            raise ValueError("Conflicting pending round")
    else:
        prior = (
            validate_version(Path(cur["checkpoint"]))["replay_inventory"]
            if cur["checkpoint"]
            else []
        )
        path = root / "replay" / uuid.uuid4().hex
        eid = import_episode(
            dataset,
            path,
            Processor(exp["checkpoint"]),
            exp,
            {**cur, "version": version},
        )
        pending = {
            "parent": cur["version"],
            "episode_id": eid,
            "dataset": str(dataset),
            "label_sha": sha(dataset / "expo_session.json"),
            "replay_inventory": [*prior, {"episode_id": eid, "path": str(path)}],
        }
        atomic_json(pending_path, pending)
    if checkpoint_identity(exp["checkpoint"]) != exp["checkpoint_provenance"]:
        raise ValueError("Frozen base assets changed")
    base = None
    started = time.monotonic()
    for item in pending["replay_inventory"]:
        path = Path(item["path"])
        if item.get("cache_sha"):
            verify_replay(item)
            continue
        rgb = np.load(path / "rgb.npy", mmap_mode="r")
        state = np.load(path / "raw_state.npy", mmap_mode="r")
        if base is None:
            base = BasePolicy(exp["checkpoint"], exp["device"], exp["base_model"])
        cache = path / "candidates.npy"
        progress = path / "cache-progress.json"
        count = Settings(**exp["settings"]).candidates
        shape = (len(rgb), count, 30, 14)
        done = read(progress)["completed"] if progress.exists() else 0
        pool = np.lib.format.open_memmap(
            cache, mode="r+" if done else "w+", dtype=np.float32, shape=shape
        )
        if pool.shape != shape:
            raise ValueError("Cache geometry changed")
        for i in range(done, len(rgb)):
            images = dict(zip(ROLES, rgb[i], strict=True))
            seed = (int(item["episode_id"][:8], 16) + i) % 2**31
            pool[i] = base.candidates(
                images, state[i], exp["prompt"], count, seed, candidate_batch=count
            )
            pool.flush()
            atomic_json(
                progress, {"completed": i + 1, "base_sha256": exp["source_sha"]}
            )
            print(
                f"Candidate cache {i + 1}/{len(rgb)} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
        item["cache_sha"] = sha(cache)
        item["files"] = {p.name: sha(p) for p in path.glob("*.npy")}
        item["files"]["metadata.json"] = sha(path / "metadata.json")
        item["base_sha256"] = exp["source_sha"]
        atomic_json(pending_path, pending)
    pending["prepared"] = True
    atomic_json(pending_path, pending)


def verify_replay(item):
    path = Path(item["path"])
    for name, digest in item["files"].items():
        if Path(name).name != name or sha(path / name) != digest:
            raise ValueError("Replay/cache integrity failure: " + name)
    if sha(path / "candidates.npy") != item["cache_sha"]:
        raise ValueError("Candidate cache changed")


class Replay:
    def __init__(self, inventory):
        self.episodes = []
        self.transitions = []
        self.terminals = []
        for e, item in enumerate(inventory):
            verify_replay(item)
            a = {
                p.stem: np.load(p, mmap_mode="r")
                for p in Path(item["path"]).glob("*.npy")
            }
            self.episodes.append(a)
            for i in range(len(a["current"])):
                self.transitions.append((e, i))
                if a["masks"][i] == 0:
                    self.terminals.append((e, i))

    def batch(self, ids):
        out = []
        for e, i in ids:
            a = self.episodes[e]
            j, k = int(a["current"][i]), int(a["next"][i])
            out.append(
                {
                    "rgb": a["rgb"][j],
                    "next_rgb": a["rgb"][k],
                    "states": a["state"][j],
                    "next_states": a["state"][k],
                    "actions": a["actions"][i].reshape(420),
                    "next_candidates": a["candidates"][k].reshape(-1, 420),
                    "rewards": a["rewards"][i],
                    "masks": a["masks"][i],
                    "steps": np.float32(30),
                }
            )
        return {k: np.stack([b[k] for b in out]) for k in out[0]}

    def sample(self, rng, size):
        return [
            self.transitions[int(i)]
            for i in rng.integers(len(self.transitions), size=size)
        ]


def train(root, microbatch=1):
    exp, cur = experiment(root)
    pending = read(root / "pending.json")
    if pending["episode_id"] in cur["episodes"]:
        (root / "pending.json").unlink()
        return
    if not pending.get("prepared") or pending["parent"] != cur["version"]:
        raise ValueError("Prepare round first")
    if sha(Path(pending["dataset"]) / "expo_session.json") != pending["label_sha"]:
        raise ValueError("Reward label changed")
    c = Settings(**exp["settings"])
    if c.batch_size % microbatch:
        raise ValueError("Microbatch must divide batch size")
    replay = Replay(pending["replay_inventory"])
    agent = Agent(c, exp["device"])
    if cur["checkpoint"]:
        validate_version(Path(cur["checkpoint"]))
        agent.restore(Path(cur["checkpoint"]), training=True)
    rng = np.random.default_rng(c.seed + cur["version"])
    work = root / "versions" / (".training-" + uuid.uuid4().hex)
    work.mkdir()
    start = time.monotonic()

    def batches(ids):
        return [
            replay.batch(ids[i : i + microbatch])
            for i in range(0, len(ids), microbatch)
        ]

    with (work / "metrics.jsonl").open("w") as log:
        for step in range(c.updates):
            ids = replay.sample(rng, c.batch_size)
            terminal = []
            if replay.terminals:
                tids = [
                    replay.terminals[int(i)]
                    for i in rng.integers(len(replay.terminals), size=c.batch_size)
                ]
                terminal = batches(tids)
            metrics = agent.critic_update(batches(ids), terminal)
            metrics.update(
                step=agent.updates,
                editor_updated=False,
                policy_version=cur["version"] + 1,
            )
            if agent.updates % c.editor_interval == 0:
                metrics.update(
                    agent.editor_update(batches(replay.sample(rng, c.batch_size))),
                    editor_updated=True,
                )
            metrics["seconds"] = time.monotonic() - start
            line = __import__("json").dumps(metrics)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
    # Validate save/load and finite selection before atomic publication.
    agent.save(work)
    check = Agent(c, exp["device"], training=False)
    check.restore(work)
    sample = replay.batch([replay.transitions[0]])
    actions, _ = check.select(
        sample["rgb"], sample["states"], sample["next_candidates"]
    )
    if not torch.isfinite(actions).all():
        raise ValueError("Candidate validation failed")
    manifest = {
        "schema": 1,
        "backend": "pytorch",
        "experiment_id": exp["experiment_id"],
        "version": cur["version"] + 1,
        "settings": exp["settings"],
        "base_sha256": exp["source_sha"],
        "replay_inventory": pending["replay_inventory"],
        "elapsed_seconds": time.monotonic() - start,
        "serving_files": ["policy.safetensors"],
        "files": {p.name: sha(p) for p in work.iterdir() if p.is_file()},
        "limitations": [
            "Frozen base; no base-model RL updates",
            "Nominal recording timestamps",
            "No closed-loop validation",
        ],
    }
    atomic_json(work / "manifest.json", manifest)
    commit_round(root, work, cur, pending["episode_id"])
    (root / "pending.json").unlink()
