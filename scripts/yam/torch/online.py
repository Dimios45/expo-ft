#!/usr/bin/env python3
"""Native PyTorch services; the NUC remains the only robot-control process."""

import argparse
import asyncio
import hmac
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from expo_ft.yam.rounds import atomic_json, lock, read


def token_at(path):
    token = path.read_text().strip()
    if len(token) < 32:
        raise ValueError("Control token must contain at least 32 characters")
    return token


async def learner(a):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse
    from websockets.asyncio.server import serve

    from expo_ft.yam.online import CHUNK, Coordinator, handle_connection
    from expo_ft.yam.torch.runtime import request_json

    exp = read(a.root / "experiment.json")
    token = token_at(a.token_file)
    if exp.get("backend") != "pytorch":
        raise ValueError("Wrong experiment backend")
    if (a.root / "online/STOPPED.json").exists():
        raise ValueError("Experiment stopped; inspect before restarting")

    class TorchCoordinator(Coordinator):
        async def dispatch(self, message):
            if message.get("op") == "begin_episode" and self.lease_path.exists():
                recorded = read(self.lease_path)["health"]
                live = await asyncio.to_thread(self.http, "/healthz")
                if any(
                    recorded.get(k) != live.get(k)
                    for k in ("session_id", "policy_version", "experiment_id")
                ):
                    raise ValueError(
                        "Server restarted during an episode lease; inspect the interrupted attempt before releasing it"
                    )
            return await super().dispatch(message)

        def http(self, path, post=False):
            return request_json(self.policy_url + path, token, b"{}" if post else None)

        async def prepare_lease(self, health):
            if (
                health.get("experiment_id") != exp["experiment_id"]
                or health.get("base_sha256") != exp["source_sha"]
                or health.get("prompt") != exp["prompt"]
                or health.get("backend") != "pytorch"
            ):
                raise ValueError("Serving identity mismatch")
            from expo_ft.yam.online import identifier

            session = identifier(health["session_id"])
            folder = self.root / "sessions" / session
            folder.mkdir(exist_ok=True)
            path = folder / "session.json"
            if path.exists() and read(path) != health:
                raise ValueError("Serving session changed")
            atomic_json(path, health)

    c = TorchCoordinator(
        a.root,
        REPO,
        policy_url=a.policy_url,
        microbatch=a.microbatch,
        trainer_script="scripts/yam/torch/train.py",
    )
    app = FastAPI(title="Torch EXPO checkpoint store")

    @app.middleware("http")
    async def auth(request, call_next):
        if not hmac.compare_digest(
            request.headers.get("Authorization", ""), "Bearer " + token
        ):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/experiment")
    def get_experiment():
        return exp

    @app.get("/candidate")
    def candidate():
        cur = read(a.root / "current.json")
        # Even an authenticated direct reload cannot advance an active lease.
        version = (
            read(c.lease_path)["health"]["policy_version"]
            if c.lease_path.exists()
            else cur["version"]
        )
        manifest = (
            read(a.root / "versions" / f"{version:04d}" / "manifest.json")
            if version
            else None
        )
        return {
            "experiment_id": exp["experiment_id"],
            "base_sha256": exp["source_sha"],
            "version": version,
            "manifest": manifest,
        }

    @app.get("/weights/{version}")
    def weights(version: int):
        cur = read(a.root / "current.json")
        if not 1 <= version <= cur["version"]:
            return JSONResponse({"error": "Unknown version"}, status_code=404)
        folder = a.root / "versions" / f"{version:04d}"
        # Atomic published directories are immutable. Full checksum validation is
        # done at publication and installation, not once per network chunk.
        return FileResponse(
            folder / "policy.safetensors", media_type="application/octet-stream"
        )

    @app.get("/status")
    async def status():
        result = await c.dispatch({"op": "status"})
        labels = [read(p) for p in (c.store / "episodes").glob("*/expo_session.json")]
        result["episodes"] = [
            {k: label[k] for k in ("policy_version", "reward", "terminal", "prompt")}
            for label in labels
        ]
        result["success_rate"] = (
            sum(x["reward"] for x in labels) / len(labels) if labels else None
        )
        result["current"] = read(a.root / "current.json")
        metrics = []
        for folder in sorted((a.root / "versions").glob("[0-9]*")):
            path = folder / "metrics.jsonl"
            if path.exists():
                metrics.extend(
                    __import__("json").loads(line)
                    for line in path.read_text().splitlines()
                )
        result["learning_curve"] = metrics[-1000:]
        return result

    server = uvicorn.Server(
        uvicorn.Config(app, host=a.host, port=a.store_port, log_level="info")
    )

    async def handler(ws):
        await handle_connection(ws, c.dispatch, token)

    async with serve(handler, a.host, a.port, max_size=CHUNK + 4096, compression=None):
        worker = asyncio.create_task(c.worker())
        print(
            f"Torch learner ws://{a.host}:{a.port}; store http://{a.host}:{a.store_port}",
            flush=True,
        )
        try:
            await server.serve()
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass


def inference(a):
    import numpy as np
    import torch
    import uvicorn

    from expo_ft.yam.torch.runtime import CandidateStore, RoundPolicy, request_json
    from expo_ft.yam.torch.server import build_app

    torch.set_num_threads(4)
    token = token_at(a.token_file)
    exp = request_json(a.store_url.rstrip("/") + "/experiment", token)
    if a.device.startswith("cuda"):
        _free, total = torch.cuda.mem_get_info(a.device)
        # Measured fold-70 base alone peaks above 12 GiB. Do not expose a
        # health endpoint on a known undersized GPU and fail on first act.
        if total < 14 * 1024**3:
            raise RuntimeError(
                "fold-70 measured base peak is 12.06 GiB before EXPO; this GPU is too small. "
                "Use a >=16 GiB inference GPU or validate a separate offload/quantized backend first."
            )
    a.root.mkdir(parents=True, exist_ok=True)
    for name in ("versions", "sessions"):
        (a.root / name).mkdir(exist_ok=True)
    path = a.root / "experiment.json"
    if path.exists() and read(path) != exp:
        raise ValueError("Existing inference run differs; use a fresh root")
    atomic_json(path, exp)
    if not (a.root / "current.json").exists():
        atomic_json(
            a.root / "current.json", {"version": 0, "episodes": [], "checkpoint": None}
        )
    with lock(a.root):
        policy = RoundPolicy(
            a.root,
            a.checkpoint,
            a.device,
            a.base_model,
            CandidateStore(a.root, a.store_url, token),
        )
        # Do not implicitly reload a newer version after restart: a durable
        # coordinator lease may still pin the last served version.
        frame = np.zeros((360, 640, 3), np.uint8)
        state = np.zeros(14, np.float32)
        state[[6, 13]] = 1
        policy.predict(
            {k: frame for k in ("top", "left", "right")}, state, exp["prompt"]
        )
        print(
            f"Warm Torch policy version {policy.cur['version']}; http://{a.host}:{a.port}",
            flush=True,
        )
        uvicorn.run(build_app(policy, token), host=a.host, port=a.port)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=["learner", "inference"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int)
    p.add_argument("--store-port", type=int, default=18309)
    p.add_argument("--store-url", default="http://192.168.0.167:18309")
    p.add_argument("--policy-url", default="http://192.168.0.119:18304")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--base-model", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--microbatch", type=int, default=1)
    a = p.parse_args()
    a.root = a.root.resolve()
    a.port = a.port or (18308 if a.role == "learner" else 18304)
    if a.role == "learner":
        # Separate from the round lock held by the training subprocess.
        import fcntl

        with (a.root / ".coordinator.lock").open("a+") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(learner(a))
    else:
        if not a.checkpoint:
            p.error("inference requires --checkpoint")
        inference(a)


if __name__ == "__main__":
    main()
