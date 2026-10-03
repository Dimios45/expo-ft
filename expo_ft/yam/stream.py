"""Durable event journal and opt-in relay to the existing local action policy.

This transport does not run a learner or execute robot commands. Archive-based
online replay admission is handled separately by expo_ft.yam.online.

Sequence numbers are per episode, start at zero, and survive reconnection.
An acknowledgement means the exact bytes were committed to SQLite, not that
the content was accepted as an aligned training transition.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import urllib.request

import msgpack

MAX_MESSAGE = 8 * 1024 * 1024
KINDS = {"begin", "observation", "execution", "end"}


def unpack(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or len(raw) > MAX_MESSAGE:
        raise ValueError("expected a binary MessagePack frame <= 8 MiB")
    message = msgpack.unpackb(raw, raw=False, strict_map_key=True)
    if not isinstance(message, dict) or message.get("schema") != 1:
        raise ValueError("expected schema 1 object")
    return message


class Journal:
    """One writer per database; transactions give crash-safe retry semantics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS records ("
                       "episode TEXT, seq INTEGER, kind TEXT, digest TEXT, payload BLOB, "
                       "received_ns INTEGER, PRIMARY KEY (episode, seq))")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def append(self, raw: bytes) -> dict:
        message = unpack(raw)
        episode, seq, kind = (message.get(k) for k in ("episode", "seq", "kind"))
        if not isinstance(episode, str) or not 1 <= len(episode) <= 128:
            raise ValueError("episode must be a nonempty ID of at most 128 characters")
        if type(seq) is not int or not 0 <= seq < 2**63:
            raise ValueError("seq must be a nonnegative signed 64-bit integer")
        if kind not in KINDS:
            raise ValueError("unknown record kind")
        if kind == "end":
            if message.get("terminal") not in {"success", "failure", "truncated"}:
                raise ValueError("end requires an explicit terminal label")
            reward = message.get("reward")
            if type(reward) not in (int, float) or not math.isfinite(reward):
                raise ValueError("end requires a finite human reward")
        digest = hashlib.sha256(raw).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT digest FROM records WHERE episode=? AND seq=?",
                                  (episode, seq)).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError("sequence reused with different bytes")
            else:
                last = db.execute("SELECT seq, kind FROM records WHERE episode=? "
                                  "ORDER BY seq DESC LIMIT 1", (episode,)).fetchone()
                if (last is None and (seq != 0 or kind != "begin")) or (
                    last is not None and (seq != last[0] + 1 or last[1] == "end" or kind == "begin")
                ):
                    raise ValueError("expected contiguous sequence in an open episode")
                db.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
                           (episode, seq, kind, digest, raw, time.time_ns()))
        return {"schema": 1, "kind": "ack", "episode": episode, "seq": seq,
                "sha256": digest, "duplicate": bool(existing), "training_admitted": False}


def infer_recorded(body):
    """Bridge to existing local policy; timing includes HTTP and CPU processing."""
    import json_numpy
    import numpy as np

    if not isinstance(body, bytes):
        raise ValueError("observation_json must be bytes")
    request = urllib.request.Request("http://127.0.0.1:8204/act", data=body,
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=180) as reply:
        result = json_numpy.loads(reply.read())
    elapsed = (time.monotonic_ns() - start) / 1e6
    actions = np.asarray(result["actions"])
    if actions.shape != (30, 14) or not np.isfinite(actions).all():
        raise ValueError("policy returned invalid actions")
    return {"actions": actions.tolist(), "policy_reported_ms": result.get("dt_ms"),
            "backend_roundtrip_ms": elapsed, "hardware_accessed": False,
            "mode": "recorded_inference_via_existing_http_policy"}


async def handle(websocket, journal: Journal, inference_lock=None, live_event_path=None):
    async for raw in websocket:
        started = time.monotonic_ns()
        try:
            message = unpack(raw)
            if message.get("kind") == "probe":
                response = {"schema": 1, "kind": "probe_ack", "seq": message.get("seq"),
                            "payload_bytes": len(raw), "mode": "shadow_transport"}
            elif message.get("kind") == "health":
                def health():
                    with urllib.request.urlopen("http://127.0.0.1:8204/healthz", timeout=5) as reply:
                        return json.load(reply)
                response = {"schema": 1, "kind": "health_result", "seq": message.get("seq"),
                            "health": await asyncio.to_thread(health)}
            elif message.get("kind") in {"infer_recorded", "infer_live"}:
                if inference_lock is None:
                    raise ValueError("recorded inference disabled")
                if inference_lock.locked():
                    raise ValueError("inference busy; retry later")
                async with inference_lock:
                    response = await asyncio.to_thread(infer_recorded, message.get("observation_json"))
                response.update(schema=1, kind="inference_result", seq=message.get("seq"))
                if message["kind"] == "infer_live":
                    response.pop("hardware_accessed", None)
                    response["mode"] = "sequential_hardware_transport_test"
                    if live_event_path:
                        from expo_ft.yam.rounds import atomic_json
                        atomic_json(live_event_path, {"completed_at": time.time(), "seq": message.get("seq")})
            else:
                response = await asyncio.to_thread(journal.append, raw)
            response["server_elapsed_ns"] = time.monotonic_ns() - started
        except (ValueError, TypeError, OverflowError, OSError, msgpack.UnpackException) as exc:
            response = {"schema": 1, "kind": "error", "message": str(exc)}
        await websocket.send(msgpack.packb(response, use_bin_type=True))


async def serve(path: str, port: int, recorded_inference: bool = False, live_event_path=None):
    from websockets.asyncio.server import serve as ws_serve

    journal = Journal(path)
    lock = asyncio.Lock() if recorded_inference else None
    async with ws_serve(lambda ws: handle(ws, journal, lock, live_event_path), "127.0.0.1", port,
                        max_size=MAX_MESSAGE, max_queue=4, compression=None):
        print(f"Shadow transport ws://127.0.0.1:{port}; recorded inference={recorded_inference}", flush=True)
        await asyncio.Future()
