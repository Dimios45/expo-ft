#!/usr/bin/env python3
"""Offline-only stability experiment. Does not serve or access robot hardware."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from expo_ft.yam.rounds import (
    atomic_json,
    initialize,
    lock,
    read,
    sha,
    validate_version,
)


def profile():
    from expo_ft.yam.learner import Settings

    return Settings(
        actor_mode="frozen",
        edit_scale=0.05,
        lr=1e-4,
        editor_lr=3e-5,
        temperature_lr=1e-5,
        init_temperature=0.01,
        gradient_clip=1.0,
        mask_gripper_edits=True,
        initial_editor_logstd=-3.0,
        scale_entropy_target=True,
    )


def setup(args):
    """Explicit prior import: preserve original session identity and hashes."""
    from expo_ft.conversion.yam_loader import YamProcessor
    from expo_ft.yam.replay import import_episode

    src = read(args.source / "experiment.json")
    session = read(args.dataset / "expo_session.json")
    if session["policy_version"] != 0:
        raise ValueError("This comparison requires the original base rollout")
    server = read(args.source / "sessions" / session["server_session"] / "session.json")
    if server["experiment_id"] != src["experiment_id"] or server["policy_version"] != 0:
        raise ValueError("Original session identity mismatch")
    validate_version(args.source / "versions/0001")
    initialize(
        args.root,
        src["checkpoint"],
        src["tokenizer"],
        profile().dictionary(),
        src["prompt"],
    )
    exp = read(args.root / "experiment.json")
    exp.update(
        diagnostic_only=True,
        profile="stable-v2",
        source_experiment=str(args.source),
        note="Offline candidate only; no deployment approval inferred from training",
    )
    atomic_json(args.root / "experiment.json", exp)
    dest = args.root / "replay/prior-round-0000"
    # Validate against the actual collecting experiment, not a fabricated session.
    eid = import_episode(
        args.dataset,
        dest,
        YamProcessor(src["checkpoint"], src["tokenizer"]),
        src,
        {"version": 0, "episodes": []},
    )
    atomic_json(
        args.root / "prior_data.json",
        {
            "episode_id": eid,
            "replay": str(dest),
            "dataset": str(args.dataset.resolve()),
            "source_experiment": str(args.source.resolve()),
            "session": session,
            "source_manifest_sha256": sha(args.source / "versions/0001/manifest.json"),
            "role": "authorized prior data, not a rollout collected by this experiment",
        },
    )
    print("SETUP", args.root, flush=True)


def observation(ep, anchor):
    from expo_ft.conversion.yam_pi05 import JAX_CAMERAS

    a = ep.arrays
    return {
        **{
            k: a["images"][anchor : anchor + 1, ..., n * 3 : n * 3 + 3]
            for n, k in enumerate(JAX_CAMERAS)
        },
        "state": a["state"][anchor : anchor + 1],
        "tokenized_prompt": a["tokens"][anchor : anchor + 1],
        "tokenized_prompt_mask": a["token_mask"][anchor : anchor + 1],
    }


def audit(args):
    """Compare recorded wire-frame commands to saved server outputs, read-only."""
    import pyarrow.parquet as pq

    from expo_ft.yam.replay import dataset_to_wire

    session = read(args.dataset / "expo_session.json")
    rows = pq.read_table(args.dataset / "data").to_pylist()
    actions = dataset_to_wire([r["action"] for r in rows], session["dataset_frame"])
    traces = sorted(
        (args.source / "sessions" / session["server_session"]).glob("*.npz")
    )
    report = []
    for f in traces:
        with np.load(f) as saved:
            distances = np.linalg.norm(
                actions[:, None] - saved["actions"][None], axis=-1
            )
            closest = distances.argmin(1)
            frames = np.flatnonzero(distances.min(1) < 1e-4)
            report.append(
                {
                    "request": f.name,
                    "exact_frame_count": len(frames),
                    "recorded_frames": frames.tolist(),
                    "server_action_indices": closest[frames].tolist(),
                }
            )
    atomic_json(
        args.root / "execution_audit.json",
        {
            "dataset": str(args.dataset),
            "session": session["server_session"],
            "requests": report,
            "requests_with_exactly_29_matching_frames": sum(
                r["exact_frame_count"] == 29 for r in report
            ),
            "warning": "Exact matches establish command identity, not authoritative timestamps. "
            "Fixed 30-command replay windows can cross the observed request boundaries. "
            "Do not deploy this diagnostic candidate until execution alignment is resolved.",
        },
    )
    print("EXECUTION AUDIT SAVED", flush=True)


def summarize_delta(output, reference, processor):
    delta = output - reference
    physical = processor.unnormalize_actions(
        output.reshape(-1, 14)
    ) - processor.unnormalize_actions(reference.reshape(-1, 14))
    joints = [i for i in range(14) if i not in (6, 13)]
    return {
        "normalized_abs_p50_p95_max": np.quantile(
            np.abs(delta), [0.5, 0.95, 1]
        ).tolist(),
        "joint_radians_abs_p50_p95_max": np.quantile(
            np.abs(physical[:, joints]), [0.5, 0.95, 1]
        ).tolist(),
        "gripper_abs_p50_p95_max": np.quantile(
            np.abs(physical[:, [6, 13]]), [0.5, 0.95, 1]
        ).tolist(),
        "within_chunk_adjacent_delta_p95": float(
            np.quantile(np.abs(np.diff(output, axis=1)), 0.95)
        ),
    }


def prepare(args, exp, ep):
    from expo_ft.yam.base import BasePolicy, restore_tree
    from expo_ft.yam.learner import Settings, YamEXPO

    cache = args.root / "base_candidates.npz"
    if cache.exists():
        raise FileExistsError(
            "Candidate cache already exists; do not overwrite comparison seeds"
        )
    base = BasePolicy(exp["checkpoint"], exp["tokenizer"], "frozen")
    # A finite pool is valid only while the base is frozen. Draw fresh pools next
    # round; sample subsets for training. Persist seeds and scope this approximation.
    pools = []
    start = time.time()
    count = len(ep.arrays["images"])
    for anchor in range(count):
        pools.append(
            base.candidates(
                observation(ep, anchor), 16, np.random.default_rng(7000 + anchor)
            )
        )
        if anchor % 5 == 0 or anchor == count - 1:
            print(
                json.dumps(
                    {
                        "cache_anchor": anchor + 1,
                        "total": count,
                        "seconds": time.time() - start,
                    }
                ),
                flush=True,
            )
    pools = np.stack(pools)
    np.savez(cache, candidates=pools, seeds=np.arange(count) + 7000)
    atomic_json(
        args.root / "cache_manifest.json",
        {
            "sha256": sha(cache),
            "source_sha": exp["source_sha"],
            "actor_mode": "frozen",
            "pool_per_anchor": 16,
            "next_batch_subset": 8,
            "limitation": "Finite frozen-base noise pool; refresh for each new training round",
        },
    )
    anchors = np.unique(np.linspace(0, count - 1, min(12, count), dtype=int))
    old = args.source / "versions/0001"
    settings = Settings.from_dict(read(old / "manifest.json")["settings"])
    agent = YamEXPO(settings)
    agent.restore(old / "expo.msgpack")
    original, ranking, edited, expert, full = [], [], [], [], []
    selected = []

    def rank(candidates, anchor, seed):
        images = ep.arrays["images"][anchor : anchor + 1]
        states = ep.arrays["state"][anchor : anchor + 1, :14]
        out, info = agent.select(
            images, states, candidates[None].reshape(1, 8, 420), seed
        )
        # Same Q pair and original candidates as edited selection: isolate edits.
        index = int(np.argmax(info["scores"][0, :8]))
        return candidates[index], out.reshape(30, 14), int(info["selected"][0])

    for anchor in anchors:
        candidates = pools[anchor, :8]
        r, e, i = rank(candidates, anchor, int(anchor) + 9000)
        original.append(candidates[0])
        ranking.append(r)
        edited.append(e)
        selected.append(i)
    # Replace expert leaves in the same loaded model, keeping the frozen VLM.
    base.trainable = restore_tree(base.trainable, (old / "actor.msgpack").read_bytes())
    base.sync()
    for anchor in anchors:
        candidates = base.candidates(
            observation(ep, anchor), 8, np.random.default_rng(7000 + int(anchor))
        )
        expert.append(candidates[0])
        full.append(rank(candidates, anchor, int(anchor) + 9000)[1])
    reference = np.stack(original)
    outputs = {
        "base": reference,
        "old_expert_only": np.stack(expert),
        "old_ranking_only": np.stack(ranking),
        "old_edits_on_original": np.stack(edited),
        "old_full": np.stack(full),
    }
    np.savez(args.root / "ablations.npz", anchors=anchors, **outputs)
    report = {
        name: summarize_delta(value, reference, base.processor)
        for name, value in outputs.items()
    }
    report.update(
        observations=len(anchors),
        selected_indices=selected,
        note="Recorded-observation output comparison, not counterfactual task success",
        cache_seconds=time.time() - start,
    )
    atomic_json(args.root / "ablation_report.json", report)
    print("ABLATIONS", json.dumps(report), flush=True)


def train(args, exp, ep):
    from expo_ft.yam.learner import Settings
    from expo_ft.yam.stable import StableEXPO, crop_batch

    cfg = Settings.from_dict(exp["settings"])
    cache = args.root / "base_candidates.npz"
    if sha(cache) != read(args.root / "cache_manifest.json")["sha256"]:
        raise ValueError("Cache integrity failure")
    pool = np.load(cache)["candidates"]
    rng = np.random.default_rng(20260929)
    agent = StableEXPO(cfg)
    terminal = np.flatnonzero(ep.arrays["masks"] == 0)
    if len(terminal) != 1 or ep.arrays["rewards"][terminal[0]] <= 0:
        raise ValueError("Expected successful terminal in round-0000")
    output = args.root / "candidate"
    output.mkdir(exist_ok=False)
    steps = min(40, 20 * math.ceil(len(ep) / 40))
    start = time.time()
    sampled, auxiliary = [], []

    def make(i, augment=True):
        anchor = ep.arrays["next"][i]
        indices = rng.choice(16, cfg.candidates, replace=False)
        b = ep.batch(i, pool[anchor, indices])
        return crop_batch(b, rng) if augment else b

    with (output / "metrics.jsonl").open("w") as log:
        for step in range(1, steps + 1):
            ids = rng.integers(len(ep), size=8).tolist()
            sampled.extend(ids)
            batches = [make(i) for i in ids]
            ti = int(terminal[(step - 1) % len(terminal)])
            auxiliary.append(ti)
            metrics = agent.critic_update(batches, [make(ti)], terminal_weight=0.25)
            metrics.update(
                step=step,
                transition_ids=ids,
                terminal_aux_ids=[ti],
                editor_updated=False,
            )
            if step % 20 == 0:
                edit_ids = rng.integers(len(ep), size=8).tolist()
                metrics.update(agent.editor_update([make(i) for i in edit_ids]))
                metrics.update(editor_updated=True, editor_transition_ids=edit_ids)
            metrics["seconds"] = time.time() - start
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)
    agent.save(output / "expo.msgpack")
    # Predictions on all recorded actions. These are training-set diagnostics,
    # not held-out validation: only one episode is available.

    checks = []
    deltas, grips = [], []
    for i in range(len(ep)):
        b = make(i, False)
        z = agent.encoder.apply({"params": agent.state["encoder"].params}, b["images"])
        q = agent.critic.apply(
            {"params": agent.state["critic"].params}, z, b["actions"], p=b["states"]
        )
        anchor = ep.arrays["current"][i]
        chosen, info = agent.select(
            b["images"], b["states"], pool[anchor, :8].reshape(1, 8, 420), 10000 + i
        )
        residual = info["candidates"][0, 8:].reshape(8, 30, 14) - pool[anchor, :8]
        deltas.append(np.abs(residual).ravel())
        grips.append(np.abs(residual[..., [6, 13]]).max())
        checks.append(
            {
                "transition": i,
                "q_mean": float(q.mean()),
                "recorded_return": float(
                    cfg.discount ** (int(ep.meta["frames"]) - 2 - i * 30)
                )
                if i < len(ep) - 1
                else float(ep.arrays["rewards"][i]),
                "selected": int(info["selected"][0]),
            }
        )
    agent.restore(output / "expo.msgpack")
    restored, _ = agent.select(
        b["images"],
        b["states"],
        pool[anchor, :8].reshape(1, 8, 420),
        10000 + len(ep) - 1,
    )
    np.testing.assert_array_equal(restored, chosen)
    manifest = {
        "profile": "stable-v2",
        "settings": cfg.dictionary(),
        "critic_updates": int(agent.state["updates"]),
        "editor_updates": int(agent.state["editor"].step),
        "temperature_updates": int(agent.state["temperature"].step),
        "actor_updates": 0,
        "effective_batch_size": 8,
        "terminal_auxiliary_weight": 0.25,
        "terminal_auxiliary_samples": len(auxiliary),
        "uniform_terminal_samples": sampled.count(int(terminal[0])),
        "base_unchanged": True,
        "deployable": False,
        "prior_data": read(args.root / "prior_data.json"),
        "cache": read(args.root / "cache_manifest.json"),
        "elapsed_seconds": time.time() - start,
        "edit_abs_p50_p95_max": np.quantile(
            np.concatenate(deltas), [0.5, 0.95, 1]
        ).tolist(),
        "gripper_residual_max": float(max(grips)),
        "training_set_q": checks,
        "limitations": [
            "One episode; no held-out success evaluation",
            "Finite frozen-base candidate pool",
            "Nominal timestamps; request alignment audit still required",
        ],
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    atomic_json(output / "manifest.json", manifest)
    print("OFFLINE CANDIDATE SAVED; NOT PUBLISHED", output, flush=True)


def evaluate(args, exp, ep):
    from expo_ft.conversion.yam_loader import YamProcessor
    from expo_ft.yam.learner import Settings
    from expo_ft.yam.stable import StableEXPO

    validate_version(args.root / "candidate")
    agent = StableEXPO(Settings.from_dict(exp["settings"]))
    agent.restore(args.root / "candidate/expo.msgpack")
    processor = YamProcessor(exp["checkpoint"], exp["tokenizer"])
    cache = args.root / "base_candidates.npz"
    if sha(cache) != read(args.root / "cache_manifest.json")["sha256"]:
        raise ValueError("Cache integrity failure")
    pool = np.load(cache)["candidates"]
    prior = np.load(args.root / "ablations.npz")
    ranking, edited, residuals = [], [], []
    for anchor in prior["anchors"]:
        candidates = pool[anchor, :8]
        out, info = agent.select(
            ep.arrays["images"][anchor : anchor + 1],
            ep.arrays["state"][anchor : anchor + 1, :14],
            candidates.reshape(1, 8, 420),
            int(anchor) + 9000,
        )
        ranking.append(candidates[int(np.argmax(info["scores"][0, :8]))])
        edited.append(out.reshape(30, 14))
        residuals.append(info["candidates"][0, 8:].reshape(8, 30, 14) - candidates)
    outputs = {
        "stable_ranking_only": np.stack(ranking),
        "stable_full": np.stack(edited),
    }
    report = {
        k: summarize_delta(v, prior["base"], processor) for k, v in outputs.items()
    }
    r = np.stack(residuals)
    report["residual_abs_p50_p95_max"] = np.quantile(np.abs(r), [0.5, 0.95, 1]).tolist()
    report["gripper_residual_max"] = float(np.abs(r[..., [6, 13]]).max())
    report["deployable"] = False
    report["interpretation"] = (
        "Output differences only; same episode used for training. No success-rate claim."
    )
    np.savez(args.root / "stable_ablations.npz", anchors=prior["anchors"], **outputs)
    atomic_json(args.root / "stable_ablation_report.json", report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["setup", "audit", "prepare", "train", "evaluate"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--source", type=Path, default=Path("/usr/local/models/sra-expo-ft/towel-expo")
    )
    p.add_argument("--dataset", type=Path, default=ROOT / "round-0000")
    args = p.parse_args()
    import torch

    torch.set_num_threads(4)
    if args.stage == "setup":
        setup(args)
        return
    if args.stage == "audit":
        audit(args)
        return
    from expo_ft.yam.replay import Episode

    with lock(args.root):
        exp = read(args.root / "experiment.json")
        ep = Episode(read(args.root / "prior_data.json")["replay"])
        {"prepare": prepare, "train": train, "evaluate": evaluate}[args.stage](
            args, exp, ep
        )


if __name__ == "__main__":
    main()
