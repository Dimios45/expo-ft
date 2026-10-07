import importlib.util
import json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('record_round',Path(__file__).parents[1]/'scripts/yam/record_round.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_discarded_dataset_is_not_labeled(tmp_path):
    (tmp_path/'meta').mkdir()
    (tmp_path/'meta/info.json').write_text(json.dumps(dict(total_episodes=0,total_frames=0)))
    (tmp_path/'openpi_control_rollouts.json').write_text(json.dumps(dict(episodes=[])))
    with pytest.raises(RuntimeError,match='no complete episode'):
        module.require_saved_episode(tmp_path,0)


def test_error_exit_is_not_labeled(tmp_path):
    with pytest.raises(RuntimeError,match='exited with an error'):
        module.require_saved_episode(tmp_path,1)


def test_saved_episode_accepted(tmp_path):
    (tmp_path/'meta').mkdir()
    (tmp_path/'meta/info.json').write_text(json.dumps(dict(total_episodes=1,total_frames=50)))
    (tmp_path/'openpi_control_rollouts.json').write_text(json.dumps(dict(episodes=[dict(saved=True)])))
    module.require_saved_episode(tmp_path,0)


def test_invalid_label_reprompts(monkeypatch):
    replies=iter(['yes','x','1'])
    monkeypatch.setattr('builtins.input',lambda _:next(replies))
    assert module.ask_choice('reward',{'0','1'})=='1'


def test_native_logs_preserved_for_failed_attempt(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'logs/runtime').mkdir(parents=True)
    (tmp_path/'logs/runtime/rollout.log').write_text('native fault')
    episode=tmp_path/'episode';episode.mkdir()
    module.preserve_runtime_logs(episode)
    assert (episode/'runtime-logs/rollout.log').read_text()=='native fault'
