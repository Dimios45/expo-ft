#!/usr/bin/env python3
"""Continue conservative EXPO from a served checkpoint; no hardware access."""

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from expo_ft.yam.rounds import (
    atomic_json,
    commit_round,
    lock,
    read,
    sha,
    validate_version,
)


def parent_state(root):
    exp, cur = read(root / "experiment.json"), read(root / "current.json")
    parent = Path(cur["checkpoint"])
    manifest = validate_version(parent)
    if exp.get("profile") != "stable-v2" or exp["settings"]["actor_mode"] != "frozen":
        raise ValueError("Requires stable-v2 frozen-base profile")
    if manifest["settings"] != exp["settings"] or manifest["version"] != cur["version"]:
        raise ValueError("Parent configuration/version mismatch")
    return exp, cur, parent, manifest


def prepare(root, dataset):
    from stability_experiment import observation

    from expo_ft.conversion.yam_loader import YamProcessor
    from expo_ft.yam.base import BasePolicy
    from expo_ft.yam.replay import Episode, import_episode

    exp, cur, parent, manifest = parent_state(root)
    session = read(dataset / "expo_session.json")
    server = read(root / "sessions" / session["server_session"] / "session.json")
    for key, expected in [
        ("experiment_id", exp["experiment_id"]),
        ("policy_version", cur["version"]),
        ("prompt", exp["prompt"]),
    ]:
        if session[key] != expected or server[key] != expected:
            raise ValueError("Collecting policy identity mismatch: " + key)
    if session.get("prefetch") is not False:
        raise ValueError("Require no-prefetch collection")
    if session["server_metadata"]["policy_weights_sha256"] != sha(
        parent / "expo.msgpack"
    ):
        raise ValueError("Rollout weight hash differs from parent")
    pending_path = root / "pending.json"
    if pending_path.exists():
        pending = read(pending_path)
        if pending["parent"] != cur["version"] or pending["dataset"] != str(
            dataset.resolve()
        ):
            raise ValueError("Another/stale episode pending")
        if pending["session_sha"] != sha(dataset / "expo_session.json"):
            raise ValueError("Pending label changed")
    else:
        dest = root / "replay" / uuid.uuid4().hex
        eid = import_episode(
            dataset, dest, YamProcessor(exp["checkpoint"], exp["tokenizer"]), exp, cur
        )
        if "replay_inventory" in manifest:
            prior = manifest["replay_inventory"]
        else:
            p = manifest["prior_data"]
            prior = [{"episode_id": p["episode_id"], "path": p["replay"]}]
        if eid in [p["episode_id"] for p in prior]:
            raise ValueError("Identical episode payload already in replay")
        inventory = [{"episode_id": p["episode_id"], "path": p["path"]} for p in prior]
        inventory.append({"episode_id": eid, "path": str(dest)})
        pending = {
            "parent": cur["version"],
            "episode_id": eid,
            "path": str(dest),
            "dataset": str(dataset.resolve()),
            "session_sha": sha(dataset / "expo_session.json"),
            "replay_inventory": inventory,
            "parent_sha": sha(parent / "expo.msgpack"),
        }
        atomic_json(pending_path, pending)
    base = BasePolicy(exp["checkpoint"], exp["tokenizer"], "frozen")
    start = time.time()
    for eidx, item in enumerate(pending["replay_inventory"]):
        ep = Episode(item["path"])
        if ep.meta["episode_id"] != item["episode_id"]:
            raise ValueError("Replay identity mismatch")
        folder = (
            root
            / "candidate_caches"
            / f"parent-{cur['version']:04d}"
            / item["episode_id"]
        )
        folder.mkdir(parents=True, exist_ok=True)
        progress = folder / "progress.json"
        cached = folder / "candidates.npy"
        count = len(ep.arrays["images"])
        done = read(progress)["completed"] if progress.exists() else 0
        shape = (count, 16, 30, 14)
        if done:
            pool = np.lib.format.open_memmap(cached, mode="r+")
            if pool.shape != shape:
                raise ValueError("Cache shape mismatch")
        else:
            pool = np.lib.format.open_memmap(
                cached, mode="w+", dtype=np.float32, shape=shape
            )
        seedbase = 100000 + cur["version"] * 10000 + eidx * 1000
        for anchor in range(done, count):
            pool[anchor] = base.candidates(
                observation(ep, anchor), 16, np.random.default_rng(seedbase + anchor)
            )
            pool.flush()
            atomic_json(
                progress,
                {"completed": anchor + 1, "seedbase": seedbase, "count": count},
            )
            if anchor % 10 == 0 or anchor == count - 1:
                print(
                    json.dumps(
                        {
                            "episode": eidx,
                            "anchor": anchor + 1,
                            "total": count,
                            "seconds": time.time() - start,
                        }
                    ),
                    flush=True,
                )
        item["cache"] = str(cached)
        item["cache_sha"] = sha(cached)
        item["replay_files"] = {
            p.name: sha(p) for p in Path(item["path"]).iterdir() if p.is_file()
        }
    pending["prepared"] = True
    atomic_json(pending_path, pending)
    print("REPLAY AND FRESH CANDIDATE POOLS READY", flush=True)


def train(root):
    from expo_ft.yam.learner import Settings
    from expo_ft.yam.replay import Episode
    from expo_ft.yam.stable import StableEXPO, crop_batch

    exp, cur, parent, _manifest = parent_state(root)
    pending = read(root / "pending.json")
    if pending["episode_id"] in cur["episodes"]:
        (root / "pending.json").unlink()
        print("Round already committed; cleared stale pending marker", flush=True)
        return
    if not pending.get("prepared") or pending["parent"] != cur["version"]:
        raise ValueError("Prepare this round first")
    if pending["parent_sha"] != sha(parent / "expo.msgpack"):
        raise ValueError("Parent checkpoint changed")
    inventory = pending["replay_inventory"]
    episodes = []
    pools = []
    for item in inventory:
        if sha(item["cache"]) != item["cache_sha"]:
            raise ValueError("Cache integrity failure")
        for name, digest in item["replay_files"].items():
            if sha(Path(item["path"]) / name) != digest:
                raise ValueError("Replay integrity failure")
        episodes.append(Episode(item["path"]))
        pools.append(np.load(item["cache"], mmap_mode="r"))
    cfg = Settings.from_dict(exp["settings"])
    agent = StableEXPO(cfg)
    agent.restore(parent / "expo.msgpack")
    before = {
        "critic": int(agent.state["updates"]),
        "editor": int(agent.state["editor"].step),
        "temperature": int(agent.state["temperature"].step),
    }
    rng = np.random.default_rng(cfg.seed + cur["version"] + 1000)
    weights = np.array([len(e) for e in episodes], float)
    weights /= weights.sum()
    steps = min(40, 20 * math.ceil(len(episodes[-1]) / 40))
    work = root / "versions" / (".training-" + uuid.uuid4().hex)
    work.mkdir()
    terminals = [
        (eidx, int(i))
        for eidx, ep in enumerate(episodes)
        for i in np.flatnonzero(ep.arrays["masks"] == 0)
    ]
    counts = np.zeros(len(episodes), int)
    aux_counts = np.zeros(len(episodes), int)
    start = time.time()

    def terminal_values():
        results = []
        for e, i in terminals:
            ep = episodes[e]
            b = ep.batch(i, pools[e][ep.arrays["next"][i], :8])
            z = agent.encoder.apply(
                {"params": agent.state["encoder"].params}, b["images"]
            )
            q = agent.critic.apply(
                {"params": agent.state["critic"].params}, z, b["actions"], p=b["states"]
            )
            results.append(
                {
                    "episode": e,
                    "reward_label": ep.meta["session"]["reward"],
                    "target": float(b["rewards"][0]),
                    "q_mean": float(q.mean()),
                }
            )
        return results

    terminal_before = terminal_values()

    def make(eidx, i, augment=True):
        ep = episodes[eidx]
        anchor = ep.arrays["next"][i]
        b = ep.batch(
            i, pools[eidx][anchor, rng.choice(16, cfg.candidates, replace=False)]
        )
        return crop_batch(b, rng) if augment else b

    def sample():
        ids = [(int(rng.choice(len(episodes), p=weights)), 0) for _ in range(8)]
        return [(e, int(rng.integers(len(episodes[e])))) for e, _ in ids]

    with (work / "metrics.jsonl").open("w") as log:
        for step in range(1, steps + 1):
            ids = sample()
            for e, _ in ids:
                counts[e] += 1
            # Balanced terminal auxiliary loss: both the success and failure
            # target appear each step, independent of episode duration.
            aux = [make(e, i) for e, i in terminals]
            for e, _ in terminals:
                aux_counts[e] += 1
            metrics = agent.critic_update(
                [make(e, i) for e, i in ids], aux, terminal_weight=0.25
            )
            metrics.update(
                step=step,
                transition_ids=ids,
                terminal_aux_ids=terminals,
                editor_updated=False,
            )
            if int(agent.state["updates"]) % 20 == 0:
                edit_ids = sample()
                metrics.update(agent.editor_update([make(e, i) for e, i in edit_ids]))
                metrics.update(editor_updated=True, editor_transition_ids=edit_ids)
            metrics["seconds"] = time.time() - start
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)
    terminal_checks = terminal_values()
    b = make(len(episodes) - 1, 0, False)
    selected, details = agent.select(
        b["next_images"], b["next_states"], b["next_candidates"], 77
    )
    agent.save(work / "expo.msgpack")
    agent.restore(work / "expo.msgpack")
    np.testing.assert_array_equal(
        selected,
        agent.select(b["next_images"], b["next_states"], b["next_candidates"], 77)[0],
    )
    residual = details["candidates"][0, 8:] - b["next_candidates"][0]
    assert np.abs(residual).max() <= 0.050001
    np.testing.assert_array_equal(residual.reshape(8, 30, 14)[..., [6, 13]], 0)
    total = {
        "critic": int(agent.state["updates"]),
        "editor": int(agent.state["editor"].step),
        "temperature": int(agent.state["temperature"].step),
    }
    result = {
        "version": cur["version"] + 1,
        "parent_version": cur["version"],
        "parent_checkpoint": str(parent),
        "parent_sha256": pending["parent_sha"],
        "new_episode": pending["episode_id"],
        "replay_inventory": inventory,
        "settings": cfg.dictionary(),
        "profile": "stable-v2",
        "base_checkpoint": exp["checkpoint"],
        "base_unchanged": True,
        "actor_updates": 0,
        "restored_steps": before,
        "total_steps": total,
        "round_steps": {k: total[k] - before[k] for k in total},
        "replay_sample_counts": counts.tolist(),
        "terminal_aux_counts": aux_counts.tolist(),
        "terminal_checks": terminal_checks,
        "terminal_before": terminal_before,
        "effective_batch_size": 8,
        "terminal_auxiliary_weight": 0.25,
        "elapsed_seconds": time.time() - start,
        "deployable": False,
        "limitations": [
            "No held-out evaluation",
            "Recorded nominal timestamps and fixed 30-step windows",
            "Finite frozen-base candidate pools, refreshed this round",
        ],
        "files": {p.name: sha(p) for p in work.iterdir() if p.is_file()},
    }
    atomic_json(work / "manifest.json", result)
    new = commit_round(root, work, cur, pending["episode_id"])
    (root / "pending.json").unlink()
    print("ROUND COMPLETE", json.dumps(new), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "train"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=ROOT / "round-0001")
    args = p.parse_args()
    import torch

    torch.set_num_threads(4)
    with lock(args.root):
        if args.stage == "prepare":
            prepare(args.root, args.dataset)
        else:
            train(args.root)


if __name__ == "__main__":
    main()
