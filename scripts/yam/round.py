#!/usr/bin/env python3
"""One episode -> EXPO update -> immutable deployable version. No robot access."""

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from expo_ft.yam.rounds import atomic_json, commit_round, initialize, lock, read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--checkpoint", required=True)
    init.add_argument("--tokenizer", required=True)
    init.add_argument("--prompt", default="fold the towel")
    init.add_argument("--actor-mode", choices=["frozen", "expert"], default="expert")
    init.add_argument("--candidates", type=int, default=8)
    init.add_argument("--seed", type=int, default=42)
    ing = sub.add_parser("ingest")
    ing.add_argument("--dataset", type=Path, required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--updates", type=int, default=100)
    tr.add_argument("--actor-every", type=int, default=10)
    se = sub.add_parser("serve")
    se.add_argument("--host", default="0.0.0.0")
    se.add_argument("--port", type=int, default=8204)
    se.add_argument(
        "--allow-diagnostic-network",
        action="store_true",
        help="Explicitly expose a diagnostic candidate without promoting it",
    )
    sub.add_parser("status")
    args = p.parse_args()
    root = args.root.resolve()
    if args.command == "init":
        from expo_ft.yam.learner import Settings

        cfg = Settings.from_dict(
            asdict(
                Settings(
                    actor_mode=args.actor_mode,
                    candidates=args.candidates,
                    edits=args.candidates,
                    seed=args.seed,
                )
            )
        )
        initialize(root, args.checkpoint, args.tokenizer, cfg.dictionary(), args.prompt)
        print("Initialized round 0 (unchanged base policy):", root)
        return
    if args.command == "status":
        print(
            json.dumps(
                {
                    "experiment": read(root / "experiment.json"),
                    "current": read(root / "current.json"),
                    "pending": read(root / "pending.json")
                    if (root / "pending.json").exists()
                    else None,
                },
                indent=2,
            )
        )
        return
    with lock(root):
        exp = read(root / "experiment.json")
        cur = read(root / "current.json")
        if exp.get("profile") == "stable-v2" and args.command in ("ingest", "train"):
            raise ValueError(
                "Use scripts/yam/continue_stable.py prepare/train for this profile; "
                "the legacy trainer has a different update schedule"
            )
        if args.command == "ingest":
            if (root / "pending.json").exists():
                raise RuntimeError("One episode already pending; train it first")
            from expo_ft.conversion.yam_loader import YamProcessor
            from expo_ft.yam.replay import import_episode

            session = read(args.dataset / "expo_session.json")
            server = read(
                root / "sessions" / session["server_session"] / "session.json"
            )
            for key, expected in [
                ("experiment_id", exp["experiment_id"]),
                ("policy_version", cur["version"]),
                ("prompt", exp["prompt"]),
            ]:
                if server[key] != expected:
                    raise ValueError("Collection server identity/version mismatch")
            if session.get("prefetch") is not False:
                raise ValueError("This round runner requires no-prefetch collection")
            dest = root / "replay" / uuid.uuid4().hex
            eid = import_episode(
                args.dataset,
                dest,
                YamProcessor(exp["checkpoint"], exp["tokenizer"]),
                exp,
                cur,
            )
            atomic_json(
                root / "pending.json",
                {"episode_id": eid, "path": str(dest), "parent": cur["version"]},
            )
            print("Accepted one episode:", eid)
            return
        if args.command == "serve":
            if (
                exp.get("diagnostic_only")
                and not args.allow_diagnostic_network
                and args.host
                not in (
                    "127.0.0.1",
                    "localhost",
                )
            ):
                raise ValueError(
                    "Diagnostic checkpoints default to localhost; use "
                    "--allow-diagnostic-network for an explicit network test"
                )
            import numpy as np
            import torch
            import uvicorn
            from serve_jax import build_app

            from expo_ft.yam.runtime import RoundPolicy

            torch.set_num_threads(4)
            policy = RoundPolicy(root)
            frame = np.zeros((224, 224, 3), np.uint8)
            policy.predict(
                {k: frame for k in ("top", "left", "right")},
                np.zeros(14, np.float32),
                exp["prompt"],
            )
            policy.record = True
            print("Warmup passed. Serving version", cur["version"], flush=True)
            uvicorn.run(
                build_app(policy, exp["checkpoint"]),
                host=args.host,
                port=args.port,
                workers=1,
            )
            return
        if args.updates < 1 or args.actor_every < 1:
            raise ValueError("Update counts must be positive")
        pending = read(root / "pending.json")
        if pending["episode_id"] in cur["episodes"]:
            (root / "pending.json").unlink()
            print(
                "Previous round was already committed; cleared stale pending marker. No retraining."
            )
            return
        if (
            pending["parent"] != cur["version"]
            or pending["episode_id"] in cur["episodes"]
        ):
            raise ValueError("Stale/duplicate pending episode")
        import numpy as np
        import torch

        from expo_ft.yam.base import BasePolicy
        from expo_ft.yam.learner import Settings, YamEXPO
        from expo_ft.yam.replay import Episode

        torch.set_num_threads(4)
        cfg = Settings.from_dict(exp["settings"])
        parent = Path(cur["checkpoint"]) if cur["checkpoint"] else None
        if parent:
            from expo_ft.yam.rounds import validate_version

            manifest = validate_version(parent)
            if (
                manifest["settings"] != exp["settings"]
                or manifest["version"] != cur["version"]
            ):
                raise ValueError("Parent configuration/version mismatch")
        base = BasePolicy(
            exp["checkpoint"],
            exp["tokenizer"],
            cfg.actor_mode,
            cfg.actor_lr,
            parent / "actor.msgpack" if parent else None,
        )
        if parent:
            base.restore_optimizer(parent)
        agent = YamEXPO(cfg)
        if parent:
            agent.restore(parent / "expo.msgpack")
        allowed = set(cur["episodes"] + [pending["episode_id"]])
        episodes = []
        for path in (root / "replay").iterdir():
            if (path / "metadata.json").exists():
                ep = Episode(path)
                if ep.meta["episode_id"] in allowed:
                    episodes.append(ep)
        if {ep.meta["episode_id"] for ep in episodes} != allowed:
            raise ValueError("Replay inventory incomplete")
        successes = [ep for ep in episodes if ep.meta["session"]["reward"] == 1]
        rng = np.random.default_rng(cfg.seed + cur["version"])
        work = root / "versions" / (".training-" + uuid.uuid4().hex)
        work.mkdir()
        start = time.time()
        actor_updates = 0
        with (work / "metrics.jsonl").open("w") as log:
            for step in range(args.updates):
                weights = np.array([len(ep) for ep in episodes], float)
                weights /= weights.sum()
                ep = episodes[int(rng.choice(len(episodes), p=weights))]
                i = int(rng.integers(len(ep)))
                # Fresh candidates from the current base, never cached across actor updates.
                candidates = base.candidates(
                    ep.observation(i, True), cfg.candidates, rng
                )
                metrics = agent.update(ep.batch(i, candidates))
                if (
                    cfg.actor_mode == "expert"
                    and successes
                    and (step + 1) % args.actor_every == 0
                ):
                    good = successes[int(rng.integers(len(successes)))]
                    j = int(rng.integers(len(good)))
                    metrics.update(
                        base.train_success(
                            good.observation(j),
                            good.arrays["actions"][j],
                            int(rng.integers(2**31 - 1)),
                        )
                    )
                    actor_updates += 1
                metrics.update(step=step + 1, elapsed_seconds=time.time() - start)
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                print(json.dumps(metrics), flush=True)
        agent.save(work / "expo.msgpack")
        base.save(work)
        manifest = {
            "parent_version": cur["version"],
            "version": cur["version"] + 1,
            "new_episode": pending["episode_id"],
            "replay_episodes": sorted(allowed),
            "updates": args.updates,
            "actor_updates": actor_updates,
            "actor_mode": cfg.actor_mode,
            "settings": cfg.dictionary(),
            "base_checkpoint": exp["checkpoint"],
            "elapsed_seconds": time.time() - start,
            "files": {p.name: sha(p) for p in work.iterdir() if p.is_file()},
        }
        atomic_json(work / "manifest.json", manifest)
        # Reload RL state before publication to catch serialization/schema failures.
        agent.restore(work / "expo.msgpack")
        next_state = commit_round(root, work, cur, pending["episode_id"])
        (root / "pending.json").unlink()
        print("READY TO DEPLOY", json.dumps(next_state), flush=True)


if __name__ == "__main__":
    main()
