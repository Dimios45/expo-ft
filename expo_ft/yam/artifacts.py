"""Immutable inference bundles; learner checkpoints never cross this interface."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import threading
import uuid
from dataclasses import dataclass

import numpy as np
from safetensors.numpy import load_file, save_file

from .lan_protocol import digest, fingerprint, identifier, integer
from .rounds import atomic_json, read, sha

COMPONENTS = {'encoder', 'editor', 'target_q'}


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def flatten(tree, prefix=''):
    result = {}
    for k, v in tree.items():
        if not isinstance(k, str) or '/' in k:
            raise ValueError('invalid parameter name')
        name = f'{prefix}/{k}' if prefix else k
        if hasattr(v, 'items'):
            result.update(flatten(v, name))
        else:
            a = np.asarray(v)
            if a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('inference tensors must be finite FP32')
            result[name] = np.ascontiguousarray(a)
    return result


def unflatten(tensors):
    result = {}
    for name, value in tensors.items():
        parts = name.split('/')
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def tensor_spec(tensors):
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in sorted(tensors.items())}


def publish(root, *, run, version, parent, base_sha256, preprocessing_sha256,
            config, params, replay_cutoff, counters):
    root = Path(root)
    identifier(run)
    integer(version, 'version', 1)
    integer(parent, 'parent')
    if version != parent + 1 or set(params) != COMPONENTS:
        raise ValueError('invalid version or inference components')
    digest(base_sha256)
    digest(preprocessing_sha256)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'current.json').exists() and read(root / 'current.json')['version'] != parent:
        raise ValueError('publication parent mismatch')
    tensors = flatten(params)
    stage = root / ('.stage-' + uuid.uuid4().hex)
    stage.mkdir()
    try:
        save_file(tensors, str(stage / 'weights.safetensors'))
        with (stage / 'weights.safetensors').open('rb') as f:
            os.fsync(f.fileno())
        manifest = dict(schema=1, run=run, version=version, parent=parent,
                        base_sha256=base_sha256, preprocessing_sha256=preprocessing_sha256,
                        config=config, config_sha256=fingerprint(config),
                        tensors=tensor_spec(tensors), replay_cutoff=replay_cutoff,
                        counters=counters, bytes=(stage / 'weights.safetensors').stat().st_size,
                        sha256=sha(stage / 'weights.safetensors'))
        atomic_json(stage / 'manifest.json', manifest)
        fsync_dir(stage)
        final = root / str(version)
        if final.exists():
            raise FileExistsError(final)
        os.rename(stage, final)
        fsync_dir(root)
        atomic_json(root / 'current.json', manifest)
        fsync_dir(root)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_bundle(path, *, run, base_sha256, preprocessing_sha256, config, expected_spec=None):
    path = Path(path)
    m = read(path / 'manifest.json')
    for k, expected in dict(schema=1, run=run, base_sha256=base_sha256,
                            preprocessing_sha256=preprocessing_sha256,
                            config_sha256=fingerprint(config)).items():
        if m.get(k) != expected:
            raise ValueError(f'incompatible bundle: {k}')
    integer(m['version'], 'version', 1)
    if fingerprint(m['config']) != m['config_sha256']:
        raise ValueError('config hash mismatch')
    f = path / 'weights.safetensors'
    if f.stat().st_size != m['bytes'] or sha(f) != m['sha256']:
        raise ValueError('artifact checksum mismatch')
    tensors = load_file(str(f))
    if tensor_spec(tensors) != m['tensors'] or (expected_spec is not None and tensor_spec(tensors) != expected_spec):
        raise ValueError('tensor schema mismatch')
    params = unflatten(tensors)
    if set(params) != COMPONENTS:
        raise ValueError('unexpected inference components')
    flatten(params)  # finite/dtype validation
    return m, params


@dataclass(frozen=True)
class Snapshot:
    version: int
    params: object
    manifest: dict


class Snapshots:
    """Episode-pinned immutable references; expensive staging happens outside lock."""
    def __init__(self, initial=None):
        self.active = initial or Snapshot(0, None, {})
        self.ready = None
        self.previous = None
        self.episode = None
        self.completed = set()
        self._lock = threading.Lock()

    def stage(self, manifest, params, validate):
        staged = validate(params)  # device_put + warmup must complete here
        candidate = Snapshot(manifest['version'], staged, manifest)
        with self._lock:
            if candidate.version > max(self.active.version, self.ready.version if self.ready else -1):
                self.ready = candidate
        return candidate

    def begin(self, episode):
        identifier(episode)
        with self._lock:
            if self.episode is not None and self.episode != episode:
                raise ValueError('another episode holds the lease')
            if episode in self.completed:
                raise ValueError('episode already completed')
            if self.episode is None:
                if self.ready is not None:
                    self.previous, self.active, self.ready = self.active, self.ready, None
                self.episode = episode
            return self.active

    def get(self, episode, version):
        with self._lock:
            if self.episode != episode or self.active.version != version:
                raise ValueError('episode/version lease mismatch')
            return self.active

    def end(self, episode):
        with self._lock:
            if self.episode != episode:
                raise ValueError('episode lease mismatch')
            self.completed.add(episode)
            self.episode = None
