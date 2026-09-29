"""Transactional one-episode-per-round state; serving and training are exclusive."""

import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


@contextmanager
def lock(root):
    with (Path(root) / ".round.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Experiment is serving or training. Stop its process first."
            ) from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def initialize(root, checkpoint, tokenizer, settings, prompt):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    checkpoint = Path(checkpoint).resolve()
    tokenizer = Path(tokenizer).resolve()
    if (
        not (checkpoint / "params").exists()
        or not (tokenizer / "tokenizer_config.json").exists()
    ):
        raise ValueError("Missing checkpoint/tokenizer")
    root.mkdir(parents=True)
    (root / "replay").mkdir()
    (root / "versions").mkdir()
    (root / "sessions").mkdir()
    atomic_json(
        root / "experiment.json",
        {
            "schema": 1,
            "checkpoint": str(checkpoint),
            "tokenizer": str(tokenizer),
            "settings": settings,
            "prompt": prompt,
            "experiment_id": uuid.uuid4().hex,
            "source_sha": read(checkpoint / "conversion_manifest.json")[
                "source_sha256"
            ],
        },
    )
    atomic_json(
        root / "current.json", {"version": 0, "episodes": [], "checkpoint": None}
    )
    return root


def commit_round(root, work, old, episode_id):
    root = Path(root)
    if read(root / "current.json") != old:
        raise RuntimeError("Parent version changed during training")
    final = root / "versions" / f"{old['version'] + 1:04d}"
    if final.exists():
        raise RuntimeError("Version destination exists")
    os.replace(work, final)
    next_state = {
        "version": old["version"] + 1,
        "episodes": old["episodes"] + [episode_id],
        "checkpoint": str(final),
    }
    atomic_json(root / "current.json", next_state)
    return next_state


def validate_version(folder):
    folder = Path(folder)
    manifest = read(folder / "manifest.json")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    return manifest
