#!/usr/bin/env python3
"""Offline fold-70 action generation and EXPO Q-learning on labeled KARMA data.

No robot/network control and no deployment. Old behavior-policy labels retain
 their identity; this is an explicitly off-policy diagnostic, not a new rollout.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch

from expo_ft.yam.replay import import_episode
from expo_ft.yam.rounds import atomic_json, read, sha
from expo_ft.yam.torch.base import (
    ROLES,
    BasePolicy,
    Processor,
    check_model_dependencies,
    checkpoint_identity,
)
from expo_ft.yam.torch.learner import Agent, Settings
from expo_ft.yam.torch.rounds import Replay


@torch.no_grad()
def evaluate(agent, replay, ids):
    """Fixed observations and random seed; in-sample diagnostics, no augmentation."""
    saved_rng = agent.generator.get_state()
    agent.generator.manual_seed(12345)
    rows = []
    try:
        for episode, transition in ids:
            b = replay.batch([(episode, transition)])
            s = agent._tensor(b, "states")
            z = agent.encoder(agent.images(b["rgb"]))
            q = agent.qs(agent.critic, z, s, agent._tensor(b, "actions"))
            ns = agent._tensor(b, "next_states")
            nz = agent.encoder(agent.images(b["next_rgb"]))
            chosen, _ = agent.select_encoded(
                nz, ns, agent._tensor(b, "next_candidates")
            )
            nq = agent.qs(agent.target, nz, ns, chosen)[:2].amin(0)
            target = (
                agent._tensor(b, "rewards")
                + agent.cfg.discount**30 * agent._tensor(b, "masks") * nq
            )
            rows.append(
                {
                    "episode": episode,
                    "transition": transition,
                    "q_mean": float(q.mean()),
                    "q_min": float(q.min()),
                    "q_max": float(q.max()),
                    "target": float(target.item()),
                    "bootstrap_mask": float(b["masks"][0]),
                    "mse": float((q - target[None]).square().mean()),
                }
            )
    finally:
        agent.generator.set_state(saved_rng)
    terminal = [r["mse"] for r in rows if r["bootstrap_mask"] == 0]
    return {
        "transitions": rows,
        "td_mse": float(np.mean([r["mse"] for r in rows])),
        "terminal_mse": float(np.mean(terminal)) if terminal else None,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="Repeat for multiple explicitly labeled single-episode KARMA directories",
    )
    p.add_argument("--output", type=Path, required=True, help="Must be a NEW directory")
    p.add_argument(
        "--transitions",
        type=int,
        default=16,
        help="Evenly spaced per episode; always includes the last window",
    )
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--critic-only", action="store_true", help="Disable editor/temperature updates"
    )
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    if min(a.transitions, a.updates, a.candidates, a.batch_size) < 1:
        p.error("All counts must be positive")
    check_model_dependencies()
    labels = [read(d / "expo_session.json") for d in a.dataset]
    if len({label["prompt"] for label in labels}) != 1:
        p.error("All episodes must have the same task string")
    torch.set_num_threads(4)
    c = Settings(
        candidates=a.candidates,
        edits=a.candidates,
        updates=a.updates,
        batch_size=a.batch_size,
    )
    processor = Processor(a.checkpoint)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    identity = checkpoint_identity(a.checkpoint)
    inventory = []
    ids = []
    sources = []
    for e, (dataset, label) in enumerate(zip(a.dataset, labels, strict=True)):
        print(
            f"Importing recorded episode {e}: reward={label['reward']}, terminal={label['terminal']}",
            flush=True,
        )
        path = a.output / f"replay-{e:03d}"
        # Validate using the recorded behavior identity; do not rewrite labels
        # to claim that fold-70 generated these historical robot commands.
        exp = {
            "experiment_id": label["experiment_id"],
            "prompt": label["prompt"],
            "settings": c.dictionary(),
        }
        eid = import_episode(
            dataset,
            path,
            processor,
            exp,
            {"version": label["policy_version"], "episodes": []},
        )
        actions = np.load(path / "actions.npy", mmap_mode="r")
        chosen = np.unique(
            np.linspace(
                0, len(actions) - 1, min(a.transitions, len(actions)), dtype=int
            )
        )
        ids.extend((e, int(i)) for i in chosen)
        inventory.append(
            {
                "path": str(path),
                "episode_id": eid,
                "selected_transitions": chosen.tolist(),
            }
        )
        sources.append(
            {
                "dataset": str(dataset.resolve()),
                "label_sha256": sha(dataset / "expo_session.json"),
                "behavior_experiment": label["experiment_id"],
                "behavior_version": label["policy_version"],
                "reward": label["reward"],
                "terminal": label["terminal"],
            }
        )
    print(
        "Loading real fold-70 model (previous measurement: about 100 seconds)...",
        flush=True,
    )
    base = BasePolicy(a.checkpoint, a.device)
    postprocessor = base.post
    prompt = labels[0]["prompt"]
    for e, item in enumerate(inventory):
        path = Path(item["path"])
        rgb = np.load(path / "rgb.npy", mmap_mode="r")
        state = np.load(path / "raw_state.npy", mmap_mode="r")
        current = np.load(path / "current.npy")
        nxt = np.load(path / "next.npy")
        chosen = item["selected_transitions"]
        anchors = np.unique(np.r_[current[chosen], nxt[chosen]])
        pool = np.lib.format.open_memmap(
            path / "candidates.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rgb), a.candidates, 30, 14),
        )
        # Uncomputed anchors cannot be used accidentally as valid candidate pools.
        pool[:] = np.nan
        for j, anchor in enumerate(anchors):
            images = dict(zip(ROLES, rgb[anchor], strict=True))
            t = time.monotonic()
            pool[anchor] = base.candidates(
                images,
                state[anchor],
                prompt,
                a.candidates,
                70 + e * 100000 + int(anchor),
                candidate_batch=a.candidates,
            )
            pool.flush()
            print(
                f"Episode {e}: generated {j + 1}/{len(anchors)} anchors ({time.monotonic() - t:.3f}s)",
                flush=True,
            )
        if e == 0:
            observation_anchor = int(current[chosen[0]])
            proposals = np.array(pool[observation_anchor])
            physical = np.stack([base.physical_actions(x) for x in proposals])
            np.savez(
                a.output / "base_actions.npz",
                normalized=proposals,
                physical=physical,
                state_raw=np.array(state[observation_anchor]),
            )
            print(
                "Saved base_actions.npz: physical actions have shape",
                physical.shape,
                flush=True,
            )
        del pool
        item["cache_sha"] = sha(path / "candidates.npy")
        item["files"] = {p.name: sha(p) for p in path.glob("*.npy")}
        item["files"]["metadata.json"] = sha(path / "metadata.json")
    del base
    gc.collect()
    if a.device.startswith("cuda"):
        torch.cuda.empty_cache()
    replay = Replay(inventory)
    # Only the explicitly sampled transitions have generated candidate pools.
    replay.transitions = ids
    replay.terminals = [(e, i) for e, i in ids if replay.episodes[e]["masks"][i] == 0]
    agent = Agent(c, a.device)
    before = evaluate(agent, replay, ids)
    atomic_json(a.output / "q_before.json", before)
    rng = np.random.default_rng(70)
    print(
        "Training EXPO on actual recorded actions and human reward labels...",
        flush=True,
    )
    with (a.output / "metrics.jsonl").open("w") as log:
        for step in range(a.updates):
            sampled = replay.sample(rng, a.batch_size)
            batches = [replay.batch([i]) for i in sampled]
            terminal = []
            if replay.terminals:
                terminal = [
                    replay.batch([replay.terminals[int(i)]])
                    for i in rng.integers(len(replay.terminals), size=a.batch_size)
                ]
            metrics = agent.critic_update(batches, terminal)
            if not a.critic_only and (step + 1) % c.editor_interval == 0:
                metrics.update(agent.editor_update(batches))
            metrics.update(step=step + 1, seconds=time.monotonic() - started)
            line = json.dumps(metrics)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    after = evaluate(agent, replay, ids)
    atomic_json(a.output / "q_after.json", after)
    b = replay.batch([ids[0]])
    proposals = np.load(a.output / "base_actions.npz")["normalized"]
    with torch.no_grad():
        z = agent.encoder(agent.images(b["rgb"]))
        s = agent._tensor(b, "states")
        tensor = torch.as_tensor(proposals.reshape(a.candidates, 420), device=a.device)
        q = agent.qs(
            agent.critic,
            z.repeat_interleave(a.candidates, 0),
            s.repeat_interleave(a.candidates, 0),
            tensor,
        )
        selected, details = agent.select(
            b["rgb"], b["states"], proposals.reshape(1, a.candidates, 420)
        )
        physical = postprocessor(selected.reshape(1, 30, 14).cpu())[0].float().numpy()
        physical[:, [6, 13]] = physical[:, [6, 13]].clip(0, 1)
    np.savez(
        a.output / "selected_actions.npz",
        physical=physical,
        normalized=selected.cpu().numpy().reshape(30, 14),
        base_q_ensemble=q.cpu().numpy(),
        selection_scores=details["scores"].cpu().numpy(),
        selected=details["selected"].cpu().numpy(),
    )
    (a.output / "checkpoint").mkdir()
    agent.save(a.output / "checkpoint")
    report = {
        "kind": "offline off-policy diagnostic; NOT deployable",
        "hardware_control": False,
        "base_identity": identity,
        "prompt": prompt,
        "sources": sources,
        "settings": c.dictionary(),
        "transitions": len(ids),
        "updates": a.updates,
        "critic_only": a.critic_only,
        "before_td_mse": before["td_mse"],
        "after_td_mse": after["td_mse"],
        "before_terminal_mse": before["terminal_mse"],
        "after_terminal_mse": after["terminal_mse"],
        "seconds": time.monotonic() - started,
        "limitations": [
            "In-sample fit; no held-out policy evaluation",
            "Historical behavior is not fold-70",
            "Q scores are learned return estimates, not calibrated success probabilities",
            "No checkpoint publication or robot execution",
        ],
    }
    atomic_json(a.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("Finished. Results:", a.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
