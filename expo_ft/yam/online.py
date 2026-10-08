"""Episode uploads, one collector lease, and a serial background learner.

Control/upload WebSocket is separate from latency-sensitive action transport.
Only loopback is exposed; connect through SSH. No robot commands originate here.
"""
import asyncio
import hashlib
import json
import os
import signal
import logging
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid

import msgpack
from expo_ft.yam.rounds import atomic_json, read, sha

MAX_ARCHIVE = 2 * 1024**3
CHUNK = 256 * 1024


def identifier(value):
    if not isinstance(value, str) or not value or len(value) > 128 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in value):
        raise ValueError('Invalid identifier')
    return value


def extract_archive(archive, destination):
    """Reject traversal, links, special files, duplicate paths and tar bombs."""
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        seen, size = set(), 0
        if len(members) > 100000:
            raise ValueError('Too many archive entries')
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts or str(path) in seen:
                raise ValueError('Invalid archive path')
            if not (member.isfile() or member.isdir()):
                raise ValueError('Only regular files/directories allowed')
            seen.add(str(path))
            size += member.size
            if size > MAX_ARCHIVE:
                raise ValueError('Unpacked episode exceeds limit')
        destination.mkdir()
        try:
            for member in members:
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush()
                        os.fsync(dst.fileno())
            for name in ('expo_session.json', 'meta/info.json', 'openpi_control_rollouts.json'):
                if not (destination/name).is_file():
                    raise ValueError('Missing episode metadata: ' + name)
        except BaseException:
            shutil.rmtree(destination)
            raise


class Coordinator:
    def __init__(self, root, repo, *, policy_url='http://127.0.0.1:8204', microbatch=1, candidate_batch=4,
                 trainer_script='scripts/yam/continue_stable.py'):
        self.trainer_script = trainer_script
        self.policy_url = policy_url.rstrip('/')
        self.microbatch, self.candidate_batch = microbatch, candidate_batch
        self.root, self.repo = Path(root).resolve(), Path(repo).resolve()
        self.store = self.root/'online'
        self.store.mkdir(exist_ok=True)
        for name in ('uploads', 'episodes', 'queue'):
            (self.store/name).mkdir(exist_ok=True)
        self.mutex = asyncio.Lock()
        self.enabled = False
        self.lease_path = self.store/'lease.json'
        self.status_path = self.store/'learner.json'
        # A crash during prepare/train is fail-closed; recovery is explicit.
        for path in (self.store/'queue').glob('*.json'):
            job = read(path)
            if job['state'] == 'training':
                job.update(state='failed', error='Worker interrupted; inspect pending checkpoint before retry')
                atomic_json(path, job)
                atomic_json(self.status_path, job)

    def http(self, path, post=False):
        request = urllib.request.Request(self.policy_url+path,
                                         data=b'{}' if post else None,
                                         headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    def enqueue(self, key, dataset, prepared=False):
        path = self.store/'queue'/(key+'.json')
        if not path.exists():
            atomic_json(path, dict(dataset=str(dataset), state='queued', created_at=time.time(), prepared=prepared))

    async def prepare_lease(self, health):
        """Complete deployment-specific validation before persisting a lease."""

    async def dispatch(self, m):
        async with self.mutex:
            op = m.get('op')
            if op == 'status':
                return dict(learner=read(self.status_path) if self.status_path.exists() else {},
                            lease=read(self.lease_path) if self.lease_path.exists() else None,
                            jobs=[read(p) for p in sorted((self.store/'queue').glob('*.json'))])
            if op == 'begin_episode':
                episode = identifier(m['episode'])
                if self.lease_path.exists():
                    lease = read(self.lease_path)
                    if lease['episode'] != episode:
                        raise ValueError('Another episode owns the policy lease: '+lease['episode'])
                    return lease
                # A short boundary pause is allowed; never change weights mid-episode.
                health = await asyncio.to_thread(self.http, '/online/reload', True)
                health = await asyncio.to_thread(self.http, '/healthz')
                await self.prepare_lease(health)
                lease = dict(episode=episode, health=health, acquired_at=time.time())
                atomic_json(self.lease_path, lease)
                self.enabled = True
                return lease
            if op == 'end_episode':
                episode = identifier(m['episode'])
                if self.lease_path.exists():
                    if read(self.lease_path)['episode'] != episode:
                        raise ValueError('Lease owner mismatch')
                    self.lease_path.unlink()
                return dict(released=True)
            digest = m.get('sha256', '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid SHA256')
            part = self.store/'uploads'/(digest+'.tar')
            meta = self.store/'uploads'/(digest+'.json')
            dest = self.store/'episodes'/digest
            if op == 'upload_begin':
                size = m['size']
                if type(size) is not int or not 0 < size <= MAX_ARCHIVE:
                    raise ValueError('Archive size exceeds limit')
                if meta.exists() and read(meta)['size'] != size:
                    raise ValueError('Upload identity mismatch')
                if not meta.exists():
                    if shutil.disk_usage(self.store).free < 3*size + 2*1024**3:
                        raise ValueError('Insufficient disk space for episode')
                    atomic_json(meta, dict(size=size))
                return dict(offset=part.stat().st_size if part.exists() else 0, complete=dest.exists())
            if op == 'upload_chunk':
                expected = read(meta)['size']
                data, offset = m['data'], m['offset']
                if not isinstance(data, bytes) or not 0 < len(data) <= CHUNK or type(offset) is not int or offset < 0:
                    raise ValueError('Invalid chunk')
                current = part.stat().st_size if part.exists() else 0
                if offset < current:
                    with part.open('rb') as f:
                        f.seek(offset)
                        if f.read(len(data)) == data:
                            return dict(offset=offset+len(data))
                    raise ValueError('Conflicting retry')
                if offset != current or offset+len(data) > expected:
                    raise ValueError('Noncontiguous upload')
                with part.open('ab') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                return dict(offset=offset+len(data))
            if op == 'upload_finish':
                if not dest.exists():
                    if part.stat().st_size != read(meta)['size'] or await asyncio.to_thread(sha, part) != digest:
                        raise ValueError('Archive integrity failure')
                    temporary = dest.with_name('.extract-'+uuid.uuid4().hex)
                    await asyncio.to_thread(extract_archive, part, temporary)
                    session = read(temporary/'expo_session.json')
                    if session['experiment_id'] != read(self.root/'experiment.json')['experiment_id']:
                        shutil.rmtree(temporary)
                        raise ValueError('Wrong experiment')
                    os.replace(temporary, dest)
                self.enqueue(digest, dest)
                return dict(stored=True, queued=True, sha256=digest)
            raise ValueError('Unknown operation')

    async def worker(self):
        while True:
            jobs = sorted(((p,read(p)) for p in (self.store/'queue').glob('*.json')), key=lambda x:x[1]['created_at'])
            if not self.enabled or any(j['state']=='failed' for _,j in jobs):
                await asyncio.sleep(1)
                continue
            next_job = next(((p,j) for p,j in jobs if j['state']=='queued'), None)
            if next_job is None:
                await asyncio.sleep(1)
                continue
            path, job = next_job
            job.update(state='training', started_at=time.time())
            atomic_json(path,job)
            atomic_json(self.status_path,job)
            env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       XLA_PYTHON_CLIENT_ALLOCATOR='platform', OMP_NUM_THREADS='4', YAM_CANDIDATE_BATCH=str(self.candidate_batch))
            env.pop('LD_LIBRARY_PATH',None)
            code = -1
            process = None
            try:
                for phase in (['train'] if job['prepared'] else ['prepare','train']):
                    job['phase'] = phase
                    atomic_json(path, job); atomic_json(self.status_path, job)
                    args = [sys.executable,self.trainer_script,phase,'--root',str(self.root),
                            '--dataset',job['dataset'],'--reuse-frozen-caches','--allow-older-policy','--microbatch',str(self.microbatch)]
                    prefix = f'logs/online-{path.stem}-{phase}'
                    command = [sys.executable,'scripts/yam/measure_gpu.py','--output',prefix,'--',*args]
                    (self.repo/'logs').mkdir(exist_ok=True)
                    with open(self.repo/(prefix+'.log'),'ab') as log:
                        process = await asyncio.create_subprocess_exec(*command,cwd=self.repo,env=env,
                            stdout=log,stderr=log,start_new_session=True)
                        job['worker_pid'] = process.pid
                        atomic_json(path,job); atomic_json(self.status_path,job)
                        code = await process.wait()
                    if code: break
                job.update(state='ready' if code==0 else 'failed',exit_code=code,finished_at=time.time())
                if code==0: job['candidate_version'] = read(self.root/'current.json')['version']
            except asyncio.CancelledError:
                job.update(state='failed',error='Coordinator stopped during update; inspect pending state before retry',finished_at=time.time())
                if process is not None:
                    # This process group was created exclusively for this learner job.
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: await asyncio.wait_for(process.wait(),timeout=10)
                    except asyncio.TimeoutError:
                        try: os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                        await process.wait()
                raise
            except Exception as exc:
                job.update(state='failed',error=str(exc),finished_at=time.time())
            finally:
                atomic_json(path,job)
                atomic_json(self.status_path,job)



async def handle_connection(ws, dispatch, token=None):
    import hmac
    from websockets.exceptions import ConnectionClosed
    try:
        async for raw in ws:
            try:
                message = msgpack.unpackb(raw,raw=False)
                if not isinstance(message,dict): raise ValueError('Expected message object')
                if token is not None and not hmac.compare_digest(str(message.pop('token','')),token):
                    raise ValueError('Unauthorized')
                result = await dispatch(message)
                response = dict(ok=True,result=result)
            except Exception as exc:
                response = dict(ok=False,error=str(exc))
            await ws.send(msgpack.packb(response,use_bin_type=True))
    except ConnectionClosed:
        logging.getLogger(__name__).info('Collector disconnected; durable uploads and episode lease retained')


async def serve(root, repo, port):
    from websockets.asyncio.server import serve as ws_serve
    if (Path(root)/'online/STOPPED.json').exists():
        raise RuntimeError('Experiment concluded by operator. Training restart requires explicit removal of online/STOPPED.json.')
    coordinator = Coordinator(root,repo)
    pending = coordinator.root/'pending.json'
    if pending.exists() and read(pending).get('prepared'):
        coordinator.enqueue('seed-'+read(pending)['episode_id'],Path(read(pending)['dataset']),True)
    async def handle(ws):
        await handle_connection(ws, coordinator.dispatch)
    async with ws_serve(handle,'127.0.0.1',port,max_size=CHUNK+4096,compression=None):
        print(f'Online collection coordinator listening on {port}',flush=True)
        await coordinator.worker()
