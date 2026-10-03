#!/usr/bin/env python3
"""Run a command and record sampled whole-GPU usage, including process exit status."""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
args = p.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    p.error("a command is required")
args.output.parent.mkdir(parents=True, exist_ok=True)
start = time.time()
code = -1
with args.output.with_suffix(".csv").open("w") as log:
    monitor = subprocess.Popen([
        "nvidia-smi", "--id=0",
        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits", "--loop-ms=200"], stdout=log)
    try:
        code = subprocess.call(command)
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
rows = list(csv.reader(args.output.with_suffix(".csv").open()))
used = [float(r[1]) for r in rows if len(r) == 4]
util = [float(r[3]) for r in rows if len(r) == 4]
report = {"command": command, "exit_code": code, "elapsed_seconds": time.time() - start,
          "sample_interval_ms": 200, "samples": len(used),
          "scope": "whole GPU 0; sampled peak may miss shorter spikes",
          "peak_used_mib": max(used) if used else None,
          "peak_used_gib": max(used) / 1024 if used else None,
          "mean_utilization_percent": sum(util) / len(util) if util else None}
args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
raise SystemExit(code)
