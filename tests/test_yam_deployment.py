import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from expo_ft.yam.deployment import candidate_paths, publish_candidate
from expo_ft.yam.rounds import atomic_json


def fixture(root):
    learner=root/'learner';(learner/'versions').mkdir(parents=True)
    atomic_json(learner/'experiment.json',{'settings':{'test':True}})
    base={'version':0,'checkpoint':None,'episodes':[]};atomic_json(learner/'current.json',base)
    current={'version':1,'checkpoint':str(learner/'versions/0001'),'episodes':['one']}
    payload=b'checkpoint'
    manifest={'version':1,'settings':{'test':True},'files':{'expo.msgpack':hashlib.sha256(payload).hexdigest()}}
    text=json.dumps(manifest).encode();digest=hashlib.sha256(text).hexdigest()
    stage,final=candidate_paths(root,current,digest);stage.mkdir()
    (stage/'manifest.json').write_bytes(text);(stage/'expo.msgpack').write_bytes(payload)
    return base,current,digest,stage,final


def test_partial_or_corrupt_candidate_cannot_publish(tmp_path):
    base,current,digest,stage,final=fixture(tmp_path)
    (stage/'expo.msgpack').write_bytes(b'corrupt')
    with pytest.raises(ValueError):publish_candidate(tmp_path,current,digest)
    assert json.loads((tmp_path/'learner/current.json').read_text())==base
    assert not final.exists()


def test_verified_candidate_is_atomic_and_idempotent(tmp_path):
    base,current,digest,stage,final=fixture(tmp_path)
    publish_candidate(tmp_path,current,digest)
    assert final.is_dir() and not stage.exists()
    publish_candidate(tmp_path,current,digest)
    with pytest.raises(ValueError,match='downgrade'):publish_candidate(tmp_path,base,None)
    assert json.loads((tmp_path/'learner/current.json').read_text())==current


def test_fixed_task_requires_blank_confirmation_and_preserves_interrupt():
    spec=importlib.util.spec_from_file_location('karma_episode',Path(__file__).parents[1]/'scripts/yam/karma_episode.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    answers=iter(['fold the wel',''])
    assert m.task_confirmation('fold the towel',lambda _:next(answers))=='fold the towel'
    def stop(_): raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):m.task_confirmation('fold the towel',stop)


def test_recovery_requires_verified_pending_label_and_no_lease(tmp_path):
    spec=importlib.util.spec_from_file_location('recover_online',Path(__file__).parents[1]/'scripts/yam/recover_online.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    learner=tmp_path/'learner';(learner/'online/queue').mkdir(parents=True)
    dataset=tmp_path/'dataset';dataset.mkdir();(dataset/'expo_session.json').write_text('{}')
    atomic_json(learner/'current.json',{'version':0,'checkpoint':None})
    atomic_json(learner/'pending.json',{'prepared':True,'parent':0,'dataset':str(dataset),'session_sha':hashlib.sha256(b'{}').hexdigest()})
    atomic_json(learner/'online/queue/job.json',{'state':'training','dataset':str(dataset),'created_at':1})
    atomic_json(learner/'online/lease.json',{})
    with pytest.raises(ValueError,match='lease'):m.retry_prepared(tmp_path)
    (learner/'online/lease.json').unlink();(dataset/'expo_session.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='label changed'):m.retry_prepared(tmp_path)
    (dataset/'expo_session.json').write_text('{}')
    result=m.retry_prepared(tmp_path)
    assert result['state']=='queued' and result['prepared']
    assert result['recovery_history'][0]['state']=='training'
    with pytest.raises(ValueError,match='exactly one'):m.retry_prepared(tmp_path)
