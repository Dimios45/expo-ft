#!/usr/bin/env python3
"""Bootstrap stable EXPO from an explicitly imported Karma HITL offline prior.

No hardware access. Collection-time policy identity was not logged by the basic
server, so this prior is never represented as a verified versioned rollout.
"""

import argparse
import uuid
from pathlib import Path

from continue_stable import parent_state, prepare_pools, train
from stability_experiment import profile

from expo_ft.yam.rounds import atomic_json, initialize, lock, read, sha


def prepare_hitl(root, dataset, expected_parent):
    """Explicit off-policy admission; collecting version remains unverified."""
    from expo_ft.conversion.yam_loader import YamProcessor
    from expo_ft.yam.replay import import_episode

    exp, cur, parent, manifest = parent_state(root)
    if cur["version"] != expected_parent or expected_parent < 1:
        raise ValueError("Current checkpoint differs from --expected-parent")
    source_hashes = {
        str(p.relative_to(dataset)): sha(p)
        for p in sorted(dataset.rglob("*"))
        if p.is_file()
    }
    pending_path = root / "pending.json"
    if pending_path.exists():
        pending = read(pending_path)
        if (
            pending["parent"] != expected_parent
            or pending["dataset"] != str(dataset)
            or pending.get("hitl_source_hashes") != source_hashes
        ):
            raise ValueError(
                "Different dataset, changed source, or stale pending round"
            )
    else:
        dest = root / "replay" / uuid.uuid4().hex
        eid = import_episode(
            dataset,
            dest,
            YamProcessor(exp["checkpoint"], exp["tokenizer"]),
            exp,
            cur,
            hitl_prior=True,
        )
        inventory = [
            {"episode_id": p["episode_id"], "path": p["path"]}
            for p in manifest["replay_inventory"]
        ]
        if eid in [p["episode_id"] for p in inventory]:
            raise ValueError("Episode already in replay")
        inventory.append({"episode_id": eid, "path": str(dest)})
        pending = {
            "parent": expected_parent,
            "episode_id": eid,
            "path": str(dest),
            "dataset": str(dataset),
            "parent_sha": sha(parent / "expo.msgpack"),
            "hitl_source_hashes": source_hashes,
            "replay_inventory": inventory,
            "prior_provenance": {
                "label": read(dataset / "hitl_reward.json"),
                "collection_policy_identity_verified": False,
                "admission": "explicit unversioned HITL off-policy prior",
                "training_parent": expected_parent,
                "manifest_sha": sha(dataset / "openpi_control_hitl.json"),
            },
        }
        atomic_json(pending_path, pending)
    prepare_pools(root, exp, cur, pending)


def setup(args):
    from expo_ft.conversion.yam_loader import YamProcessor
    from expo_ft.yam.replay import hitl_prior_metadata, import_episode
    from expo_ft.yam.stable import StableEXPO

    label, _ = hitl_prior_metadata(args.dataset)
    if label["prompt"] != args.prompt:
        raise ValueError("Prompt must match the recorded task exactly")
    cfg = profile()
    initialize(
        args.root, args.checkpoint, args.tokenizer, cfg.dictionary(), args.prompt
    )
    with lock(args.root):
        exp = read(args.root / "experiment.json")
        exp.update(
            profile="stable-v2",
            diagnostic_only=True,
            prior_type="explicit HITL offline data",
            collection_policy_identity_verified=False,
        )
        atomic_json(args.root / "experiment.json", exp)
        replay = args.root / "replay" / "initial-hitl-prior"
        eid = import_episode(
            args.dataset,
            replay,
            YamProcessor(args.checkpoint, args.tokenizer),
            exp,
            {"version": 0, "episodes": []},
            hitl_prior=True,
        )
        parent = args.root / "versions" / "0000"
        parent.mkdir()
        agent = StableEXPO(cfg)
        agent.save(parent / "expo.msgpack")
        atomic_json(
            parent / "manifest.json",
            {
                "version": 0,
                "settings": cfg.dictionary(),
                "profile": "stable-v2",
                "replay_inventory": [],
                "base_unchanged": True,
                "actor_updates": 0,
                "bootstrap_only": True,
                "deployable": False,
                "files": {"expo.msgpack": sha(parent / "expo.msgpack")},
            },
        )
        atomic_json(
            args.root / "current.json",
            {
                "version": 0,
                "episodes": [],
                "checkpoint": str(parent),
            },
        )
        atomic_json(
            args.root / "pending.json",
            {
                "parent": 0,
                "episode_id": eid,
                "dataset": str(args.dataset),
                "path": str(replay),
                "parent_sha": sha(parent / "expo.msgpack"),
                "prior_provenance": {
                    "label": label,
                    "collection_policy_identity_verified": False,
                    "manifest_sha": sha(args.dataset / "openpi_control_hitl.json"),
                },
                "replay_inventory": [{"episode_id": eid, "path": str(replay)}],
            },
        )
        print(
            "INITIALIZED. Next run prepare, then train. Do not serve bootstrap version 0."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["init", "prepare", "prepare-hitl", "train"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--expected-parent", type=int)
    parser.add_argument("--allow-unversioned-prior", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/usr/local/models/sra-expo-ft/yam_pi05_jax"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/home/sra/molmoact2/outputs/models/paligemma-tokenizer"),
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    import torch

    torch.set_num_threads(4)
    if args.stage == "init":
        if args.dataset is None or not args.prompt:
            parser.error("init requires --dataset and --prompt")
        args.dataset = args.dataset.resolve()
        setup(args)
        return
    with lock(args.root):
        if args.stage == "prepare-hitl":
            if (
                args.dataset is None
                or args.expected_parent is None
                or not args.allow_unversioned_prior
            ):
                parser.error(
                    "prepare-hitl requires --dataset, --expected-parent, and --allow-unversioned-prior"
                )
            prepare_hitl(args.root, args.dataset.resolve(), args.expected_parent)
        elif args.stage == "prepare":
            exp, cur, _, _ = parent_state(args.root)
            pending = read(args.root / "pending.json")
            if cur["version"] != 0 or pending["parent"] != 0:
                raise ValueError("Bootstrap prepare only supports the initial prior")
            prepare_pools(args.root, exp, cur, pending)
        else:
            train(args.root)


if __name__ == "__main__":
    main()
