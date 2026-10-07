"""Persistent binary WebSocket RTC service with explicit episode leases.

GPU work is serialized, not assumed preemptible. Staging is admitted only when
no proposal/selection is in flight; loaded deadline tests are still required.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hmac
from pathlib import Path
import time
import urllib.error

import numpy as np

from .artifacts import Snapshots, load_bundle
from .lan_protocol import identifier, integer, observation, pack, unpack, vector, MAX_MESSAGE
from .rounds import atomic_json


class Inference:
    def __init__(self, policy, config):
        self.policy, self.config = policy, config
        self.snapshots = Snapshots()
        self.proposals = {}
        self.gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix='yam-inference')
        self.lock = asyncio.Lock()
        self.proposal_lock = asyncio.Lock()
        self.busy = 0
        self.timings = []

    async def compute(self, fn, *args):
        self.busy += 1
        try:
            async with self.lock:
                return await asyncio.get_running_loop().run_in_executor(self.gpu, fn, *args)
        finally:
            self.busy -= 1

    async def dispatch(self, m):
        if m.get('run') != self.config['run']:
            raise ValueError('run mismatch')
        op = m['op']
        if op == 'health':
            return dict(version=self.snapshots.active.version, ready=self.snapshots.ready.version if self.snapshots.ready else None,
                        episode=self.snapshots.episode, hardware_ready=all(self.config['gates'].values()))
        episode = identifier(m['episode'])
        if op == 'begin':
            s = self.snapshots.begin(episode)
            return dict(version=s.version, config_sha256=m.get('config_sha256'),
                        base_sha256=self.config['base_sha256'],
                        preprocessing_sha256=self.config['preprocessing_sha256'])
        if op == 'end':
            async with self.proposal_lock:
                if self.busy:
                    raise ValueError('inference still in flight')
                self.snapshots.end(episode)
                self.proposals.clear()
            return dict(ended=episode)
        s = self.snapshots.get(episode, m['version'])
        obs = observation(m['observation'])
        request_id = identifier(m['request_id'])
        seed = integer(m['seed'], 'seed')
        if not 0 < m['budget_ms'] <= 10000:
            raise ValueError('invalid inference budget')
        started = time.monotonic()
        if op == 'propose':
            async with self.proposal_lock:
                if self.proposals:
                    raise ValueError('one outstanding proposal allowed')
                prefix = np.asarray(m['prefix'], dtype=np.float32)
                vector(prefix, (len(prefix), 14), 'prefix')
                if len(prefix) not in (0, self.config['delay']):
                    raise ValueError('invalid prefix delay')
                if m['start_tick'] != obs['tick'] + len(prefix):
                    raise ValueError('proposal observation/prefix alignment mismatch')
                candidate = await self.compute(self.policy.propose, obs, prefix, seed)
                if (time.monotonic() - started) * 1000 > m['budget_ms']:
                    raise ValueError('proposal deadline missed')
                self.snapshots.get(episode, s.version)
                self.proposals[request_id] = (s, m['start_tick'], candidate)
            return dict(proposal_id=request_id, version=s.version)
        if op == 'select':
            proposal_id = identifier(m['proposal_id'])
            async with self.proposal_lock:
                if proposal_id not in self.proposals:
                    raise ValueError('unknown or already consumed proposal')
                snapshot, tick, candidates = self.proposals.pop(proposal_id)
                if snapshot is not s or not tick - 1 <= obs['tick'] <= tick:
                    raise ValueError('selection is not aligned to fresh observation')
                actions = await self.compute(self.policy.select, s.params, obs, candidates, seed)
            elapsed = (time.monotonic() - started) * 1000
            if elapsed > m['budget_ms']:
                raise ValueError('selection deadline missed')
            vector(actions, (self.config['replan_steps'], 14), 'actions')
            self.timings.append(elapsed)
            self.timings = self.timings[-1000:]
            return dict(episode=episode, version=s.version, proposal_id=proposal_id,
                        start_tick=tick, expires_ns=integer(m['expires_ns'], 'expiry'), actions=actions,
                        server_ms=elapsed)
        if op == 'cancel':
            async with self.proposal_lock:
                self.proposals.clear()
            return dict(cancelled=True)
        raise ValueError('unknown operation')

    async def stage(self, manifest, params):
        # Busy is checked and lock acquired without yielding between them.
        if self.busy or self.lock.locked() or self.proposals:
            return False
        async with self.lock:
            staged = await asyncio.get_running_loop().run_in_executor(self.gpu, self.policy.selector.stage, params)
            self.snapshots.stage(manifest, staged, lambda p: p)
        return True

    def close(self):
        self.gpu.shutdown(wait=True)


async def pull_forever(service, client, cache, stop, on_error=lambda exc: None):
    cache = Path(cache)
    staged_version = 0
    pending = None
    while not stop.is_set():
        try:
            if pending is None:
                m = await asyncio.to_thread(client.json, '/current')
                if m['version'] > max(staged_version, service.snapshots.active.version):
                    folder = cache / str(m['version'])
                    await asyncio.to_thread(client.download, f"/bundles/{m['version']}/weights.safetensors",
                                            folder / 'weights.safetensors', m['sha256'], m['bytes'])
                    atomic_json(folder / 'manifest.json', m)
                    c = service.config
                    pending = await asyncio.to_thread(load_bundle, folder, run=c['run'],
                        base_sha256=c['base_sha256'], preprocessing_sha256=c['preprocessing_sha256'],
                        config=c['policy_config'], expected_spec=service.policy.selector.spec)
            if pending is not None and await service.stage(*pending):
                staged_version = pending[0]['version']
                pending = None
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                on_error(exc)
        except Exception as exc:
            pending = None
            on_error(exc)
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
        except asyncio.TimeoutError:
            pass


async def serve(service, host, port, token, stop):
    from websockets.asyncio.server import serve as ws_serve
    if not token or host in ('', '0.0.0.0', '::'):
        raise ValueError('explicit wired address and LAN_TOKEN required')

    async def handler(ws):
        if not hmac.compare_digest(ws.request.headers.get('Authorization', ''), 'Bearer ' + token):
            await ws.close(code=1008, reason='unauthorized')
            return
        async for raw in ws:
            try:
                m = unpack(raw)
                result = await service.dispatch(m)
                reply = dict(schema=1, ok=True, request_id=m.get('request_id'), **result)
            except Exception as exc:
                reply = dict(schema=1, ok=False, error=str(exc))
            await ws.send(pack(reply))

    async with ws_serve(handler, host, port, compression=None, max_size=MAX_MESSAGE, max_queue=2):
        await stop.wait()
