#!/usr/bin/env python3
"""Load the public YAM adapter in a fresh process and check an offline fixture."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "expo_ft/agents/vla/openpi/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Previous JAX output for exact reload comparison",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from expo_ft.conversion.yam_loader import YamJaxPolicy

    policy = YamJaxPolicy(args.checkpoint, args.tokenizer)
    data = np.load(args.fixture)
    images = {k: data[k] for k in ["top", "left", "right"]}
    first = policy.predict(images, data["state_raw"], noise=data["noise"])
    start = time.perf_counter()
    second = policy.predict(images, data["state_raw"], noise=data["noise"])
    elapsed = time.perf_counter() - start
    np.testing.assert_array_equal(first, second)
    reference = np.load(args.reference)["actions_physical"][0]
    np.testing.assert_array_equal(first, reference)
    args.output.write_text(
        json.dumps(
            {
                "passed": True,
                "shape": list(first.shape),
                "finite": bool(np.isfinite(first).all()),
                "reload_bitwise_equal": True,
                "repeat_bitwise_equal": True,
                "warm_seconds": elapsed,
            },
            indent=2,
        )
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
