#!/usr/bin/env python3
"""Read-only snapshots of versions, rewards, jobs and learning curves."""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from expo_ft.yam.rounds import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-url", default="http://192.168.0.167:18309")
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--interval", type=float, default=0, help="0: once; 300: every five minutes"
    )
    a = p.parse_args()
    if a.interval < 0:
        p.error("interval cannot be negative")
    token = a.token_file.read_text().strip()
    a.output.mkdir(parents=True, exist_ok=True)
    while True:
        req = urllib.request.Request(
            a.store_url.rstrip("/") + "/status",
            headers={"Authorization": "Bearer " + token},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.load(response)
        except (OSError, ValueError) as exc:
            result = {"error": str(exc)}
        result["observed_at"] = time.time()
        atomic_json(a.output / "latest.json", result)
        with (a.output / "progress.jsonl").open("a") as log:
            log.write(json.dumps(result) + "\n")
        print(
            json.dumps({k: v for k, v in result.items() if k != "learning_curve"}),
            flush=True,
        )
        if not a.interval:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
