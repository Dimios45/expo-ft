"""NUC-side durable segment writer/uploader; never imports JAX or controls arms."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import queue
import shutil
import threading
import time

from .artifacts import fsync_dir
from .lan_protocol import MAX_MESSAGE, identifier, pack, unpack
from .rounds import atomic_json, read


class Spool:
    def __init__(self, root, identity, *, queue_size=120, reserve_bytes=2 * 1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        self.reserve = reserve_bytes
        self.queue = queue.Queue(maxsize=queue_size)
        self.error = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, episode, record):
        """Nonblocking; caller must stop locally on failure, never silently drop data.

        Record and nested buffers must be immutable after handoff.
        """
        if self.error:
            raise RuntimeError('spool worker failed') from self.error
        identifier(episode)
        try:
            self.queue.put_nowait(('tick', episode, record))
        except queue.Full as exc:
            raise RuntimeError('durable spool cannot keep up') from exc

    def finish(self, episode, outcome):
        """Call outside the control tick. The commit follows all queued records."""
        if self.error:
            raise RuntimeError('spool worker failed') from self.error
        event = threading.Event()
        self.queue.put(('finish', episode, (outcome, event)), timeout=5)
        if not event.wait(30):
            raise RuntimeError('episode spool finalization timed out')
        if self.error:
            raise RuntimeError('spool worker failed') from self.error
        return self.root / episode / 'commit.msgpack'

    def close(self):
        if self.thread.is_alive():
            self.queue.put(('stop', None, None), timeout=5)
            self.thread.join(10)
        if self.thread.is_alive():
            raise RuntimeError('spool worker did not stop')

    def _write(self, path, data):
        if shutil.disk_usage(self.root).free - len(data) < self.reserve:
            raise RuntimeError('spool disk reserve reached')
        temp = path.with_suffix(path.suffix + '.tmp')
        with temp.open('wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
        fsync_dir(path.parent)

    def _worker(self):
        current, records, hashes, ticks = None, [], [], 0
        def seal():
            nonlocal records
            if not records:
                return
            blob = pack(dict(schema=1, run=self.identity['run'], episode=current,
                             index=len(hashes), records=records))
            key = hashlib.sha256(blob).hexdigest()
            self._write(self.root / current / f'{len(hashes):06d}-{key}.msgpack', blob)
            hashes.append(key)
            records = []
        try:
            while True:
                kind, episode, data = self.queue.get()
                if kind == 'stop':
                    # Preserve an uncommitted partial attempt for diagnosis/recovery.
                    seal()
                    return
                if current is None:
                    current = episode
                    (self.root / current).mkdir(exist_ok=False)
                if episode != current:
                    raise ValueError('finish the current episode before starting another')
                if kind == 'tick':
                    # At most 30 ticks per segment; seal earlier for large frames.
                    if records and (len(records) >= 30 or len(pack(dict(schema=1, records=records))) + len(pack(dict(schema=1, record=data))) > MAX_MESSAGE - 4096):
                        seal()
                    records.append(data)
                    ticks += 1
                elif kind == 'finish':
                    outcome, event = data
                    seal()
                    manifest = dict(self.identity, **outcome, schema=1, episode=current,
                                    ticks=ticks, segments=hashes)
                    self._write(self.root / current / 'commit.msgpack', pack(manifest))
                    current, records, hashes, ticks = None, [], [], 0
                    event.set()
                else:
                    raise ValueError('unknown spool operation')
        except BaseException as exc:
            self.error = exc
            if 'event' in locals():
                event.set()


def upload_pending(root, client):
    """Idempotent single pass. Sealed segments stream before episode completion."""
    count = 0
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir() or (folder / 'ack.json').exists():
            continue
        for segment in sorted(folder.glob('[0-9]*-*.msgpack')):
            marker = segment.with_suffix('.ack')
            if not marker.exists():
                key = client.upload(segment)
                atomic_json(marker, {'sha256': key})
        commit = folder / 'commit.msgpack'
        if commit.exists():
            result = client.json('/episodes', 'POST', commit.read_bytes())
            atomic_json(folder / 'ack.json', result)
            count += 1
    return count


def upload_forever(root, client, stop, on_error=lambda exc: None):
    while not stop.is_set():
        try:
            upload_pending(root, client)
        except Exception as exc:
            on_error(exc)
        stop.wait(1)
