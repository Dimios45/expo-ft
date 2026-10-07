import asyncio
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest
from expo_ft.yam.online import Coordinator, extract_archive
from expo_ft.yam.rounds import atomic_json


def archive(path, files):
    with tarfile.open(path,'w:') as t:
        for name, data in files.items():
            b=json.dumps(data).encode()
            m=tarfile.TarInfo(name);m.size=len(b);t.addfile(m,io.BytesIO(b))


def setup(tmp_path):
    root=tmp_path/'learner';root.mkdir()
    atomic_json(root/'experiment.json',{'experiment_id':'exp'})
    return Coordinator(root,tmp_path)


def test_upload_resume_dedup_and_queue(tmp_path):
    c=setup(tmp_path)
    path=tmp_path/'ep.tar'
    archive(path,{'expo_session.json':{'experiment_id':'exp'},'meta/info.json':{},'openpi_control_rollouts.json':{}})
    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    async def run():
        assert (await c.dispatch(dict(op='upload_begin',sha256=digest,size=len(data))))['offset']==0
        message=dict(op='upload_chunk',sha256=digest,offset=0,data=data[:1024])
        await c.dispatch(message);await c.dispatch(message)
        assert (await c.dispatch(dict(op='upload_begin',sha256=digest,size=len(data))))['offset']==1024
        with pytest.raises(ValueError,match='Conflicting'):
            await c.dispatch(dict(message,data=b'bad'))
        await c.dispatch(dict(op='upload_chunk',sha256=digest,offset=1024,data=data[1024:]))
        for _ in range(2):assert (await c.dispatch(dict(op='upload_finish',sha256=digest)))['queued']
        assert len(list((c.store/'queue').glob('*.json')))==1
    asyncio.run(run())


@pytest.mark.parametrize('name',['../outside','/absolute'])
def test_archive_rejects_escape(tmp_path,name):
    path=tmp_path/'bad.tar';archive(path,{name:{}})
    with pytest.raises(ValueError):extract_archive(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_episode_lease_pins_policy_until_boundary(tmp_path):
    c=setup(tmp_path);calls=[]
    def http(path,post=False):
        calls.append(path);return {'policy_version':4,'status':'ok'}
    c.http=http
    async def run():
        first=await c.dispatch(dict(op='begin_episode',episode='one'))
        again=await c.dispatch(dict(op='begin_episode',episode='one'))
        assert first==again and calls.count('/online/reload')==1
        with pytest.raises(ValueError,match='Another episode'):
            await c.dispatch(dict(op='begin_episode',episode='two'))
        with pytest.raises(ValueError,match='owner mismatch'):
            await c.dispatch(dict(op='end_episode',episode='two'))
        await c.dispatch(dict(op='end_episode',episode='one'))
        await c.dispatch(dict(op='begin_episode',episode='two'))
        assert calls.count('/online/reload')==2
    asyncio.run(run())


def test_ten_episode_runner_uploads_without_scp(tmp_path,monkeypatch):
    import importlib.util
    import sys
    import threading
    spec=importlib.util.spec_from_file_location('collect_online_test',Path(__file__).parents[1]/'scripts/yam/collect_online.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    recorder=tmp_path/'recorder.py'
    recorder.write_text("""import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a,_=p.parse_known_args()
a.out.mkdir();(a.out/'expo_session.json').write_text(json.dumps(dict(reward=0,terminal='truncated',policy_version=3)))
""")
    shared={'lease':None,'uploads':{},'jobs':[],'begins':0}
    mutex=threading.Lock()
    class FakeRPC:
        def __init__(self,url):pass
        def close(self):pass
        def call(self,**m):
            with mutex:
                op=m['op']
                if op=='begin_episode':
                    assert shared['lease'] is None
                    shared['lease']=m['episode'];shared['begins']+=1
                    return {'health':{'policy_version':3+len(shared['jobs'])}}
                if op=='end_episode':shared['lease']=None;return {}
                if op=='status':return dict(lease=None,jobs=list(shared['jobs']),learner={'state':'ready'})
                d=m['sha256']
                if op=='upload_begin':shared['uploads'].setdefault(d,bytearray());return dict(offset=len(shared['uploads'][d]),complete=False)
                if op=='upload_chunk':
                    assert len(shared['uploads'][d])==m['offset']
                    shared['uploads'][d].extend(m['data']);return {'offset':len(shared['uploads'][d])}
                if op=='upload_finish':
                    assert hashlib.sha256(shared['uploads'][d]).hexdigest()==d
                    shared['jobs'].append(dict(state='ready',dataset='/episodes/'+d))
                    return dict(sha256=d,stored=True,queued=True)
                raise AssertionError(op)
    monkeypatch.setattr(module,'RPC',FakeRPC)
    monkeypatch.setattr('builtins.input',lambda _: '')
    monkeypatch.setattr(sys,'argv',['collect_online','--root',str(tmp_path/'run'),'--episodes','10','--recorder',str(recorder)])
    module.main()
    report=json.loads((tmp_path/'run/summary.json').read_text())
    assert report['episodes']==10 and report['truncated']==10
    assert len(shared['jobs'])==10 and shared['begins']==11
    assert shared['lease'] is None


def test_repeated_interrupt_waits_for_child_cleanup():
    import importlib.util
    spec=importlib.util.spec_from_file_location('collect_online_interrupt',Path(__file__).parents[1]/'scripts/yam/collect_online.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Child:
        calls=0
        def wait(self):
            self.calls+=1
            if self.calls<3:raise KeyboardInterrupt
            return 0
    child=Child()
    assert module.wait_for_recorder(child)==(0,True)
    assert child.calls==3


def test_archive_discarded_preserves_files_and_rejects_saved(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('collector_recovery',Path(__file__).parents[1]/'scripts/yam/collect_online.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=tmp_path/'episode-005';(folder/'meta').mkdir(parents=True)
    (folder/'meta/info.json').write_text(json.dumps({'total_episodes':0}))
    (folder/'evidence.txt').write_text('preserve')
    module.archive_discarded(folder)
    assert not folder.exists()
    assert list((tmp_path/'discarded-attempts').glob('*/evidence.txt'))[0].read_text()=='preserve'
    (folder/'meta').mkdir(parents=True)
    (folder/'meta/info.json').write_text(json.dumps({'total_episodes':1}))
    with pytest.raises(RuntimeError,match='saved episodes'):
        module.archive_discarded(folder)
    assert folder.exists()


def test_disconnected_client_does_not_crash_handler():
    from expo_ft.yam.online import handle_connection
    from websockets.exceptions import ConnectionClosedError
    class Socket:
        def __aiter__(self): return self
        async def __anext__(self): raise ConnectionClosedError(None,None)
    async def dispatch(message): raise AssertionError('must not dispatch')
    asyncio.run(handle_connection(Socket(),dispatch))


def test_session_sync_failure_does_not_create_lease(tmp_path):
    c=setup(tmp_path)
    c.http=lambda *a: {'policy_version':0}
    async def fail(health): raise OSError('session transfer failed')
    c.prepare_lease=fail
    with pytest.raises(OSError): asyncio.run(c.dispatch({'op':'begin_episode','episode':'one'}))
    assert not c.lease_path.exists()


def test_restart_marks_job_and_status_interrupted(tmp_path):
    c=setup(tmp_path)
    job=dict(state='training',dataset='test',created_at=1)
    atomic_json(c.store/'queue/test.json',job);atomic_json(c.status_path,job)
    restored=Coordinator(c.root,tmp_path)
    assert json.loads(restored.status_path.read_text())['state']=='failed'
    assert json.loads((c.store/'queue/test.json').read_text())['state']=='failed'


def test_cancelled_worker_stops_its_process_group(tmp_path,monkeypatch):
    import signal
    import expo_ft.yam.online as online
    c=setup(tmp_path);c.enabled=True;c.enqueue('one',tmp_path/'data')
    ready=asyncio.Event();calls=[]
    class Process:
        pid=123456
        returncode=None
        async def wait(self):
            ready.set()
            if self.returncode is None: await asyncio.Future()
            return self.returncode
    proc=Process()
    async def spawn(*a,**kw):
        assert kw['start_new_session'];return proc
    def kill(pid,sig):
        calls.append((pid,sig));proc.returncode=-sig
    monkeypatch.setattr(online.asyncio,'create_subprocess_exec',spawn)
    monkeypatch.setattr(online.os,'killpg',kill)
    async def run():
        task=asyncio.create_task(c.worker());await ready.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(run())
    assert calls==[(123456,signal.SIGTERM)]
    assert json.loads(c.status_path.read_text())['state']=='failed'
