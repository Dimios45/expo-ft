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
