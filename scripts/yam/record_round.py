#!/usr/bin/env python3
"""Run ON THE NUC, in its configured Karma checkout. This command moves robots.

Only Python stdlib is required here. The assistant does not execute this script.
"""

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def health(server):
    with urllib.request.urlopen(
        server.rstrip("/").removesuffix("/act") + "/healthz", timeout=10
    ) as r:
        result = json.load(r)
    if result.get("status") != "ok" or "experiment_id" not in result:
        raise ValueError("Not an EXPO round server")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("karma_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    extra = args.karma_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    reserved = {
        "--root",
        "--server",
        "--episodes",
        "--episode-seconds",
        "--fps",
        "--speed",
        "--chunk-size",
        "--num-steps",
        "--prefetch",
        "--no-prefetch",
        "--repo-id",
        "--policy-frame",
    }
    if any(v.split("=")[0] in reserved for v in extra):
        raise ValueError("Do not override versioned collection/execution settings")
    if args.out.exists():
        raise FileExistsError("Use a new dataset directory for each episode")
    before = health(args.server)
    start = time.time()
    print(
        "Collecting policy version",
        before["policy_version"],
        "task:",
        before["prompt"],
        flush=True,
    )
    print(
        "Use that exact task when Karma prompts. Its y/n label is followed by the authoritative 0/1 label here.",
        flush=True,
    )
    command = [
        "uv",
        "run",
        "karma",
        "rollout",
        "--rig",
        "yam_bimanual",
        "--server",
        args.server,
        "--root",
        str(args.out.resolve()),
        "--repo-id",
        "local/yam-expo-round",
        "--episodes",
        "1",
        "--episode-seconds",
        str(args.seconds),
        "--fps",
        "30",
        "--speed",
        "1",
        "--chunk-size",
        "30",
        "--num-steps",
        "10",
        "--no-prefetch",
        *extra,
    ]
    process = subprocess.Popen(command)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        print(
            "Waiting for Karma to finish saving the interrupted episode...", flush=True
        )
        returncode = process.wait()
    after = health(args.server)
    for key in ("experiment_id", "policy_version", "session_id"):
        if before[key] != after[key]:
            raise RuntimeError("Server changed during episode; recording not admitted")
    if not (args.out / "meta/info.json").exists():
        raise RuntimeError("Karma did not save a dataset")
    label = input("Task success reward [0/1]: ").strip()
    if label not in ("0", "1"):
        raise ValueError("Reward must be 0 or 1")
    terminal = (
        "success"
        if label == "1"
        else input(
            "Failure or interrupted/time-limit truncation [failure/truncated]: "
        ).strip()
    )
    if terminal not in ("success", "failure", "truncated"):
        raise ValueError("Invalid terminal reason")
    session = {
        "experiment_id": before["experiment_id"],
        "policy_version": before["policy_version"],
        "server_session": before["session_id"],
        "prompt": before["prompt"],
        "reward": int(label),
        "terminal": terminal,
        "dataset_frame": "karma-recorded",
        "prefetch": False,
        "collection_started": start,
        "collection_finished": time.time(),
        "client_returncode": returncode,
        "server_metadata": before,
    }
    (args.out / "expo_session.json").write_text(json.dumps(session, indent=2) + "\n")
    print(
        "Saved versioned episode:",
        args.out,
        "Transfer this entire folder to the GPU host before ingest/train.",
    )


if __name__ == "__main__":
    main()
