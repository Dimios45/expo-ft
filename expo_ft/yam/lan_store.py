"""Resumable content-addressed HTTP store and transactional episode admission."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import urllib.error
import urllib.request

from .artifacts import fsync_dir
from .lan_protocol import (MAX_MESSAGE, digest, fingerprint, identifier, integer,
                           observation, unpack, vector)
from .rounds import atomic_json, read, sha

BLOCK = 256 * 1024


class Store:
    def __init__(self, root, run, base_sha256, preprocessing_sha256):
        self.root = Path(root)
        self.run = identifier(run)
        self.base_sha256 = digest(base_sha256)
        self.preprocessing_sha256 = digest(preprocessing_sha256)
        for p in ('objects', 'partial', 'bundles'):
            (self.root / p).mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        identity = dict(run=run, base_sha256=base_sha256, preprocessing_sha256=preprocessing_sha256)
        identity_path = self.root / 'identity.json'
        if identity_path.exists() and read(identity_path) != identity:
            raise ValueError('store belongs to another experiment')
        atomic_json(identity_path, identity)
        with self.db() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS episodes ('
                       'seq INTEGER PRIMARY KEY AUTOINCREMENT, episode TEXT UNIQUE, '
                       'digest TEXT, manifest TEXT, committed_ns INTEGER)')

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.root / 'episodes.sqlite', timeout=30)
        try:
            conn.execute('PRAGMA synchronous=FULL')
            with conn:
                yield conn
        finally:
            conn.close()

    def offset(self, key):
        digest(key)
        with self.mutex:
            complete = self.root / 'objects' / key
            partial = self.root / 'partial' / key
            return dict(offset=(complete if complete.exists() else partial).stat().st_size
                        if complete.exists() or partial.exists() else 0, complete=complete.exists())

    def put(self, key, offset, total, data):
        digest(key)
        integer(offset, 'offset')
        integer(total, 'total', 1)
        if total > MAX_MESSAGE or len(data) > BLOCK or not data or offset + len(data) > total:
            raise ValueError('invalid upload bounds')
        with self.mutex:
            final = self.root / 'objects' / key
            path = final if final.exists() else self.root / 'partial' / key
            size = path.stat().st_size if path.exists() else 0
            if offset < size:
                with path.open('rb') as f:
                    f.seek(offset)
                    if f.read(len(data)) != data:
                        raise ValueError('conflicting upload retry')
            elif offset == size and not final.exists():
                with path.open('ab') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            else:
                raise ValueError('unexpected upload offset')
            if path.stat().st_size == total and not final.exists():
                if sha(path) != key:
                    path.unlink()  # this incomplete, uncommitted upload is corrupt
                    raise ValueError('upload checksum mismatch')
                os.rename(path, final)
                fsync_dir(final.parent)
            return self.offset(key)

    def commit(self, m):
        identifier(m['episode'])
        if m.get('schema') != 1 or m.get('run') != self.run:
            raise ValueError('episode run/schema mismatch')
        if (m.get('base_sha256') != self.base_sha256 or
                m.get('preprocessing_sha256') != self.preprocessing_sha256):
            raise ValueError('episode base/preprocessing mismatch')
        integer(m['version'], 'policy version')
        integer(m['ticks'], 'ticks', 1)
        if m.get('terminal') not in ('success', 'failure', 'truncated') or m.get('reward') not in (0, 1):
            raise ValueError('explicit outcome required')
        if (m['terminal'] == 'success') != (m['reward'] == 1) or m.get('saved') is not True:
            raise ValueError('invalid outcome or unsaved episode')
        if not isinstance(m['segments'], list) or not 1 <= len(m['segments']) <= 100000:
            raise ValueError('invalid segment inventory')
        content_hash = episode_fingerprint(m)
        with self.mutex, self.db() as db:
            old = db.execute('SELECT seq,digest FROM episodes WHERE episode=?', (m['episode'],)).fetchone()
            if old:
                if old[1] != content_hash:
                    raise ValueError('episode ID reused with different content')
                return dict(seq=old[0], duplicate=True)
            count, last_dispatch, last_capture = 0, -1, -1
            for index, key in enumerate(m['segments']):
                digest(key)
                path = self.root / 'objects' / key
                if sha(path) != key:
                    raise ValueError('segment checksum mismatch')
                segment = unpack(path.read_bytes())
                if (segment['run'], segment['episode'], segment['index']) != (self.run, m['episode'], index):
                    raise ValueError('segment identity/order mismatch')
                if not segment['records']:
                    raise ValueError('empty segment')
                for r in segment['records']:
                    observation(r['observation'])
                    if r['observation']['tick'] != count or r['version'] != m['version']:
                        raise ValueError('tick gap or mixed policy versions')
                    dispatch = integer(r['dispatch_ns'], 'dispatch timestamp')
                    capture = r['observation']['capture_ns']
                    if not (dispatch > last_dispatch and capture > last_capture and dispatch >= capture):
                        raise ValueError('nonmonotonic capture/dispatch timestamps')
                    vector(r['action'], (14,), 'executed command')
                    integer(r['plan_start'], 'plan start')
                    if not r['plan_start'] <= count:
                        raise ValueError('invalid plan alignment')
                    last_dispatch, last_capture = dispatch, capture
                    count += 1
            observation(m['final_observation'])
            if (count != m['ticks'] or m['final_observation']['tick'] != count or
                    m['final_observation']['capture_ns'] <= last_dispatch):
                raise ValueError('missing true successor observation')
            # Final observation contains arrays/JPEG bytes; manifests travel as MessagePack.
            # SQLite stores the canonical MessagePack object in a content-addressed file.
            from .lan_protocol import pack
            blob = pack(m)
            final_key = hashlib.sha256(blob).hexdigest()
            final = self.root / 'objects' / final_key
            if not final.exists():
                tmp = self.root / 'partial' / final_key
                with tmp.open('wb') as f:
                    f.write(blob); f.flush(); os.fsync(f.fileno())
                os.replace(tmp, final)
                fsync_dir(final.parent)
            row = db.execute('INSERT INTO episodes(episode,digest,manifest,committed_ns) VALUES (?,?,?,?)',
                             (m['episode'], content_hash, final_key, time.time_ns()))
            return dict(seq=row.lastrowid, duplicate=False)

    def episodes(self, after=0):
        with self.db() as db:
            return [dict(seq=r[0], episode=r[1], manifest=r[2]) for r in db.execute(
                'SELECT seq,episode,manifest FROM episodes WHERE seq>? ORDER BY seq', (after,))]


def episode_fingerprint(m):
    from .lan_protocol import pack
    return hashlib.sha256(pack(m)).hexdigest()


class ArtifactServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store, token):
        if address[0] in ('', '0.0.0.0', '::'):
            raise ValueError('bind an explicit wired or loopback address')
        if not token:
            raise ValueError('LAN_TOKEN is required')
        self.store, self.token = store, token
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # never log bearer tokens or observation payloads

    def do_HEAD(self):
        self._dispatch('HEAD')

    def do_GET(self):
        self._dispatch('GET')

    def do_PUT(self):
        self._dispatch('PUT')

    def do_POST(self):
        self._dispatch('POST')

    def _json(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def _body(self, limit):
        size = int(self.headers.get('Content-Length', '-1'))
        if not 0 <= size <= limit:
            raise ValueError('invalid content length')
        self.connection.settimeout(10)
        body = self.rfile.read(size)
        if len(body) != size:
            raise ValueError('partial HTTP body')
        return body

    def _dispatch(self, method):
        if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + self.server.token):
            return self._json({'error': 'unauthorized'}, 401)
        try:
            parts = self.path.strip('/').split('/')
            store = self.server.store
            if len(parts) == 2 and parts[0] == 'objects':
                key = digest(parts[1])
                if method == 'HEAD':
                    state = store.offset(key)
                    self.send_response(200)
                    self.send_header('X-Offset', str(state['offset']))
                    self.send_header('X-Complete', str(int(state['complete'])))
                    self.end_headers()
                    return
                if method == 'PUT':
                    return self._json(store.put(key, int(self.headers['X-Offset']),
                                               int(self.headers['X-Total-Size']), self._body(BLOCK)))
                if method == 'GET':
                    return self._file(store.root / 'objects' / key)
            if parts == ['episodes'] and method == 'POST':
                return self._json(store.commit(unpack(self._body(MAX_MESSAGE))))
            if parts == ['episodes'] and method == 'GET':
                return self._json(store.episodes())
            if parts == ['current'] and method == 'GET':
                return self._json(read(store.root / 'bundles/current.json'))
            if len(parts) == 3 and parts[0] == 'bundles' and method == 'GET':
                version = integer(int(parts[1]), 'version', 1)
                if parts[2] not in ('manifest.json', 'weights.safetensors'):
                    raise ValueError('unknown bundle file')
                return self._file(store.root / 'bundles' / str(version) / parts[2])
            self._json({'error': 'not found'}, 404)
        except FileNotFoundError:
            self._json({'error': 'not found'}, 404)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self._json({'error': str(exc)}, 400)

    def _file(self, path):
        size = path.stat().st_size
        start = 0
        requested = self.headers.get('Range')
        if requested:
            import re
            match = re.fullmatch(r'bytes=(\d+)-', requested)
            if not match:
                raise ValueError('unsupported Range')
            start = int(match[1])
            if start >= size:
                self.send_response(416); self.end_headers(); return
        self.send_response(206 if requested else 200)
        self.send_header('Content-Length', str(size - start))
        if requested:
            self.send_header('Content-Range', f'bytes {start}-{size-1}/{size}')
        self.end_headers()
        with path.open('rb') as f:
            f.seek(start)
            while data := f.read(BLOCK):
                self.wfile.write(data)


class Client:
    def __init__(self, url, token, bytes_per_second=20_000_000):
        self.url, self.token = url.rstrip('/'), token
        if bytes_per_second <= 0:
            raise ValueError('positive transfer rate required')
        self.rate = bytes_per_second

    def request(self, path, method='GET', data=None, headers=None):
        h = {'Authorization': 'Bearer ' + self.token, **(headers or {})}
        return urllib.request.urlopen(urllib.request.Request(self.url + path, data=data, headers=h, method=method), timeout=15)

    def json(self, path, method='GET', data=None):
        with self.request(path, method, data) as reply:
            return json.load(reply)

    def upload(self, path):
        path = Path(path)
        key, total = sha(path), path.stat().st_size
        if not 0 < total <= MAX_MESSAGE:
            raise ValueError('invalid segment size')
        with self.request('/objects/' + key, 'HEAD') as reply:
            offset = int(reply.headers['X-Offset'])
            if reply.headers['X-Complete'] == '1':
                if offset != total:
                    raise ValueError('remote object size mismatch')
                return key
        with path.open('rb') as f:
            f.seek(offset)
            while data := f.read(BLOCK):
                started = time.monotonic()
                with self.request('/objects/' + key, 'PUT', data,
                                  {'X-Offset': str(offset), 'X-Total-Size': str(total)}) as reply:
                    result = json.load(reply)
                offset += len(data)
                if result['offset'] != offset:
                    raise ValueError('remote offset mismatch')
                time.sleep(max(0, len(data) / self.rate - (time.monotonic() - started)))
        return key

    def download(self, remote, destination, expected_sha, expected_size):
        destination = Path(destination)
        digest(expected_sha)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) == expected_sha:
            return destination
        partial = destination.with_suffix(destination.suffix + '.partial')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= expected_size:
            if offset == expected_size and sha(partial) == expected_sha:
                os.replace(partial, destination)
                return destination
            partial.unlink(); offset = 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        with self.request(remote, headers=headers) as reply:
            if offset and (reply.status != 206 or not reply.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                raise ValueError('server did not honor resume offset')
            with partial.open('ab' if offset else 'wb') as f:
                while data := reply.read(BLOCK):
                    started = time.monotonic()
                    if f.tell() + len(data) > expected_size:
                        raise ValueError('oversized artifact')
                    f.write(data)
                    time.sleep(max(0, len(data) / self.rate - (time.monotonic() - started)))
                f.flush(); os.fsync(f.fileno())
        if partial.stat().st_size != expected_size or sha(partial) != expected_sha:
            partial.unlink()
            raise ValueError('artifact checksum mismatch')
        os.replace(partial, destination)
        fsync_dir(destination.parent)
        return destination
