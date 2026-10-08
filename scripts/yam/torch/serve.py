#!/usr/bin/env python3
"""Standalone fold-70 HTTP inference; no learner or robot process is started."""

import argparse
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prompt", default="fold the towel")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18304)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    import numpy as np
    import torch
    import uvicorn

    from expo_ft.yam.rounds import lock, read
    from expo_ft.yam.torch.base import check_model_dependencies
    from expo_ft.yam.torch.rounds import initialize
    from expo_ft.yam.torch.runtime import RoundPolicy
    from expo_ft.yam.torch.server import build_app

    check_model_dependencies()
    torch.set_num_threads(4)
    a.root = a.root.resolve()
    if not a.root.exists():
        initialize(a.root, a.checkpoint, a.prompt, device=a.device)
    exp = read(a.root / "experiment.json")
    if exp.get("backend") != "pytorch" or exp["prompt"] != a.prompt:
        raise ValueError("Existing run differs; use a new --root")
    with lock(a.root):
        token_path = a.root / "control.token"
        if not token_path.exists():
            fd = os.open(token_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as file:
                file.write(secrets.token_hex(32))
        token = token_path.read_text().strip()
        print(
            "Loading fold-70; initial loading can take about 100 seconds...", flush=True
        )
        policy = RoundPolicy(a.root, a.checkpoint, device=a.device)
        frame = np.zeros((360, 640, 3), np.uint8)
        state = np.zeros(14, np.float32)
        state[[6, 13]] = 1
        policy.predict(
            {role: frame for role in ("top", "left", "right")}, state, a.prompt
        )
        print(
            f"Ready: http://{a.host}:{a.port}/act; policy version {policy.cur['version']}; task {a.prompt!r}",
            flush=True,
        )
        uvicorn.run(build_app(policy, token), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
