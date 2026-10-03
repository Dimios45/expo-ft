#!/usr/bin/env python3
"""Label one saved Karma HITL episode without connecting to hardware."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def choose(prompt, options):
    while True:
        value = input(prompt).strip().lower()
        if value in options:
            return value
        print("Choose " + "/".join(options))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    info = json.loads((args.dataset / "meta/info.json").read_text())
    manifest = json.loads((args.dataset / "openpi_control_hitl.json").read_text())
    entries = [e for e in manifest["episodes"] if e.get("saved") and not e.get("discarded")]
    if info.get("total_episodes") != 1 or len(entries) != 1:
        raise ValueError("Expected exactly one saved episode; refusing ambiguous assignment")
    entry = entries[0]
    output = args.dataset / "hitl_reward.json"
    if output.exists():
        raise FileExistsError(f"Reward already exists: {output}; inspect it before relabelling")
    print(f"Task: {entry['prompt']}; Karma success label: {entry.get('success')}")
    reward = int(choose("Task success reward [0/1]: ", ("0", "1")))
    prior = entry.get("success")
    if prior is not None and bool(reward) != prior:
        if choose("This differs from Karma's label. Keep the new reward? [y/n]: ", ("y", "n")) != "y":
            print("No reward written. Run this command again to label.")
            return
    terminal = "success" if reward else choose(
        "Failure or incomplete/time-limit truncation [failure/truncated]: ",
        ("failure", "truncated"),
    )
    payload = {
        "schema": 1,
        "collection_mode": "karma-hitl",
        "episode_index": entry["episode_index"],
        "prompt": entry["prompt"],
        "reward": reward,
        "terminal": terminal,
        "karma_success": prior,
        "labelled_at": datetime.now(timezone.utc).isoformat(),
        "note": "Human-labelled outcome of a mixed policy/human episode; not autonomous success. Not an EXPO admission manifest.",
    }
    with output.open("x") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {output}. Original Karma manifest and intervention labels preserved.")


if __name__ == "__main__":
    main()
