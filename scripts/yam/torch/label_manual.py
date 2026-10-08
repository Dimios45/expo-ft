#!/usr/bin/env python3
"""Attach an explicitly operator-attested label to a manual versioned rollout.

This does not claim the episode had a collector lease or boundary verification.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from expo_ft.yam.rounds import atomic_json, read, sha, validate_version


def label(dataset, root, reward, terminal, policy_version=None):
    exp, current = read(root / "experiment.json"), read(root / "current.json")
    version = current["version"] if policy_version is None else policy_version
    if type(version) is not int or not 0 <= version <= current["version"]:
        raise ValueError("Invalid behavior policy version")
    sessions = [read(p) for p in (root / "sessions").glob("*/session.json")]
    sessions = [
        s
        for s in sessions
        if s.get("policy_version") == version
        and s.get("experiment_id") == exp["experiment_id"]
    ]
    if len(sessions) != 1:
        raise ValueError(
            "Requires one unambiguous serving session for the specified version; inspect history manually"
        )
    server = sessions[0]
    if (
        server.get("prompt") != exp["prompt"]
        or server.get("base_sha256") != exp["source_sha"]
    ):
        raise ValueError("Serving identity mismatch")
    expected = None
    if version:
        folder = root / "versions" / f"{version:04d}"
        manifest = validate_version(folder)
        if (
            manifest["version"] != version
            or manifest["experiment_id"] != exp["experiment_id"]
        ):
            raise ValueError("Behavior checkpoint identity mismatch")
        expected = sha(folder / "policy.safetensors")
    if server.get("policy_weights_sha256") != expected:
        raise ValueError("Behavior weights differ from the serving session")
    info = read(dataset / "meta/info.json")
    rollout = read(dataset / "openpi_control_rollouts.json")
    entries = rollout["episodes"]
    if info["total_episodes"] != 1 or info["total_frames"] < 31 or info["fps"] != 30:
        raise ValueError("Requires one saved episode with at least 31 frames at 30 Hz")
    if (
        len(entries) != 1
        or not entries[0].get("saved")
        or entries[0]["prompt"] != exp["prompt"]
    ):
        raise ValueError("Saved attempt/task does not match this server")
    if rollout["speed"] != 1 or rollout["chunk_size"] != 30:
        raise ValueError("Execution settings differ from EXPO contract")
    if (reward == 1) != (terminal == "success"):
        raise ValueError("Reward and terminal label disagree")
    if type(entries[0].get("success")) is bool and entries[0]["success"] != bool(
        reward
    ):
        raise ValueError("Reward disagrees with KARMA outcome label")
    value = {
        "experiment_id": exp["experiment_id"],
        "policy_version": version,
        "server_session": server["session_id"],
        "prompt": exp["prompt"],
        "reward": reward,
        "terminal": terminal,
        "dataset_frame": "karma-recorded",
        "prefetch": False,
        "server_metadata": server,
        "provenance": "Retrospective operator attestation of the manual no-prefetch rollout; no collection lease",
        "boundary_verified": False,
    }
    path = dataset / "expo_session.json"
    if path.exists() and read(path) != value:
        raise ValueError("Existing label differs; refusing overwrite")
    atomic_json(path, value)
    print(f"Labeled {dataset}: reward={reward}, terminal={terminal}, version={version}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reward", type=int, choices=[0, 1], required=True)
    p.add_argument(
        "--policy-version",
        type=int,
        help="Explicit behavior version; otherwise current version",
    )
    p.add_argument(
        "--terminal", choices=["success", "failure", "truncated"], required=True
    )
    a = p.parse_args()
    label(a.dataset, a.root, a.reward, a.terminal, a.policy_version)


if __name__ == "__main__":
    main()
