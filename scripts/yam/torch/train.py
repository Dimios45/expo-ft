#!/usr/bin/env python3
"""Initialize or update a native PyTorch fold-70 experiment; no robot access."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from expo_ft.yam.rounds import lock
from expo_ft.yam.torch.learner import Settings
from expo_ft.yam.torch.rounds import initialize, prepare, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["init", "prepare", "train"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--base-model", type=Path)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--prompt", default="fold the towel")
    p.add_argument("--device", default="cuda")
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--allow-older-policy", action="store_true")
    p.add_argument(
        "--reuse-frozen-caches",
        action="store_true",
        help="Frozen caches are always verified and reused",
    )
    a = p.parse_args()
    a.root = a.root.resolve()
    import torch

    torch.set_num_threads(4)
    if a.stage == "init":
        if not a.checkpoint:
            p.error("init requires --checkpoint")
        initialize(
            a.root,
            a.checkpoint,
            a.prompt,
            Settings(candidates=a.candidates, edits=a.candidates, updates=a.updates),
            a.base_model,
            a.device,
        )
    else:
        with lock(a.root):
            if a.stage == "prepare":
                if not a.dataset:
                    p.error("prepare requires --dataset")
                prepare(a.root, a.dataset, a.allow_older_policy)
            else:
                train(a.root, a.microbatch)


if __name__ == "__main__":
    main()
