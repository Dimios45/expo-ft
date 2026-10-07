"""CPU/offline correctness tests; no camera, robot, network reconfiguration or GPU."""
import asyncio
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from expo_ft.yam.artifacts import Snapshots, Snapshot, load_bundle, publish
from expo_ft.yam.lan_protocol import pack, unpack, validate_plan
from expo_ft.yam.lan_store import ArtifactServer, Client, Store
from expo_ft.yam.lan_spool import Spool, upload_pending

BASE = 'a' * 64
PRE = 'b' * 64


def obs(tick):
    return dict(tick=tick, capture_ns=1_000_000 + tick * 33_333_333,
                state=np.zeros(14, np.float32),
                images={k: np.zeros((8, 8, 3), np.uint8) for k in ('top', 'left', 'right')},
                frame_ids={k: tick for k in ('top', 'left', 'right')},
                image_ns={k: 1_000_000 + tick * 33_333_333 for k in ('top', 'left', 'right')})


def params(value=0):
    return {k: {'x': np.full((2, 3), value, np.float32)} for k in ('encoder', 'editor', 'target_q')}


def record(tick, version=0, c=8, d=5):
    start = tick // c * c
    proposal_tick = max(0, start-d)
    return dict(observation=obs(tick), dispatch_ns=obs(tick)['capture_ns'] + 1000,
                version=version, action=np.zeros(14, np.float32), plan_start=start,
                proposal_tick=proposal_tick, prefix=np.zeros((start-proposal_tick,14),np.float32),
                intervention=False)


def identity():
    return dict(run='test', base_sha256=BASE, preprocessing_sha256=PRE)


@pytest.fixture
def http(tmp_path):
    store = Store(tmp_path/'store', **dict(run='test', base_sha256=BASE, preprocessing_sha256=PRE))
    server = ArtifactServer(('127.0.0.1', 0), store, 'test-token')
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    client = Client(f'http://127.0.0.1:{server.server_port}', 'test-token', bytes_per_second=1e10)
    yield store, client
    server.shutdown(); server.server_close(); thread.join()


def test_protocol_preserves_binary_arrays_and_rejects_stale_actions():
    m = unpack(pack(dict(schema=1, observation=obs(0))))
    np.testing.assert_array_equal(m['observation']['state'], obs(0)['state'])
    p = dict(episode='one', version=2, start_tick=8, expires_ns=20, actions=np.zeros((8,14)))
    validate_plan(p, episode='one', version=2, tick=8, now_ns=19, c=8)
    for change in (dict(version=3), dict(episode='two'), dict(tick=9), dict(now_ns=21)):
        args=dict(episode='one',version=2,tick=8,now_ns=19,c=8);args.update(change)
        with pytest.raises(ValueError):validate_plan(p,**args)


def test_bundle_roundtrip_and_tamper(tmp_path):
    m = publish(tmp_path, run='test',version=1,parent=0,base_sha256=BASE,
                preprocessing_sha256=PRE, config={'C':8},params=params(),replay_cutoff=0,counters={})
    kwargs=dict(run='test',base_sha256=BASE,preprocessing_sha256=PRE,config={'C':8})
    got, p=load_bundle(tmp_path/'1',**kwargs)
    assert got==m
    assert set(p)==set(params())
    with pytest.raises(ValueError,match='incompatible'):
        load_bundle(tmp_path/'1',**dict(kwargs,base_sha256='c'*64))
    f=tmp_path/'1/weights.safetensors'; f.write_bytes(f.read_bytes()+b'x')
    with pytest.raises(ValueError,match='checksum'):load_bundle(tmp_path/'1',**kwargs)


def test_episode_lease_atomic_swap_and_failed_validation():
    s=Snapshots();old=s.begin('a')
    s.stage({'version':1},params(1),lambda p:p)
    assert s.begin('a') is old and s.get('a',0) is old
    with pytest.raises(ValueError):s.begin('b')
    s.end('a');new=s.begin('b')
    assert new.version==1 and old.version==0
    def fail(_):raise ValueError('bad warmup')
    with pytest.raises(ValueError):s.stage({'version':2},params(),fail)
    assert s.get('b',1) is new
    s.end('b')
    with pytest.raises(ValueError):s.begin('a')


def test_resumable_upload_and_download(http,tmp_path):
    store,client=http
    data=b'x'*700000; path=tmp_path/'segment';path.write_bytes(data)
    key=hashlib.sha256(data).hexdigest()
    store.put(key,0,len(data),data[:10000]);store.put(key,0,len(data),data[:10000])
    with pytest.raises(ValueError):store.put(key,0,len(data),b'bad')
    assert client.upload(path)==key and store.offset(key)['complete']
    dest=tmp_path/'received'
    dest.with_suffix('.partial').write_bytes(data[:10000])
    client.download('/objects/'+key,dest,key,len(data))
    assert dest.read_bytes()==data


def test_spool_stream_commit_retry_and_gap_rejection(http,tmp_path):
    store,client=http
    spool=Spool(tmp_path/'spool', identity(),reserve_bytes=0)
    for tick in range(32):spool.submit('episode',record(tick))
    spool.finish('episode',dict(version=0,saved=True,reward=0,terminal='truncated',final_observation=obs(32)))
    spool.close()
    assert upload_pending(tmp_path/'spool',client)==1
    assert upload_pending(tmp_path/'spool',client)==0
    assert len(store.episodes())==1
    manifest=unpack((tmp_path/'spool/episode/commit.msgpack').read_bytes())
    assert store.commit(manifest)['duplicate']
    manifest['reward']=1;manifest['terminal']='success'
    with pytest.raises(ValueError,match='reused'):store.commit(manifest)
    seg=dict(schema=1,run='test',episode='gap',index=0,records=[record(1)])
    blob=pack(seg);key=hashlib.sha256(blob).hexdigest()
    (store.root/'objects'/key).write_bytes(blob)
    bad=dict(identity(),schema=1,episode='gap',version=0,ticks=1,segments=[key],saved=True,
             reward=0,terminal='truncated',final_observation=obs(2))
    with pytest.raises(ValueError,match='tick gap'):store.commit(bad)


def test_inference_two_phase_versioning():
    from expo_ft.yam.lan_inference import Inference
    class Fake:
        def propose(self,obs,prefix,seed):return np.zeros((2,8,14),np.float32)
        def select(self,params,obs,candidates,seed):return candidates[0]
    config=dict(identity(),gates={'measured':False},replan_steps=8,delay=5)
    service=Inference(Fake(),config)
    async def run():
        await service.dispatch(dict(op='begin',run='test',episode='ep'))
        common=dict(run='test',episode='ep',version=0,observation=obs(0),seed=0,budget_ms=1000)
        await service.dispatch(dict(common,op='propose',request_id='p0',prefix=np.empty((0,14)),start_tick=0))
        service.snapshots.stage({'version':1},params(),lambda p:p)
        result=await service.dispatch(dict(common,op='select',request_id='s0',proposal_id='p0',expires_ns=10000000))
        assert result['version']==0
        await service.dispatch(dict(op='end',run='test',episode='ep'))
        assert (await service.dispatch(dict(op='begin',run='test',episode='ep2')))['version']==1
    try:asyncio.run(run())
    finally:service.close()
