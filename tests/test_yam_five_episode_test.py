import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('five_test',Path(__file__).parents[1]/'scripts/yam/five_episode_test.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_configuration_is_five_minutes_five_episodes_and_pinned(tmp_path):
    config={'run':'test','prompt':'a new task','episodes':5,'seconds':300,'initial_policy':'base-only'}
    m.validate(config);path=tmp_path/'config.json';m.save_new(path,config);m.save_new(path,config)
    with pytest.raises(ValueError):m.save_new(path,dict(config,prompt='changed'))
    with pytest.raises(ValueError):m.validate(dict(config,seconds=180))
    with pytest.raises(ValueError):m.validate(dict(config,episodes=10))
    with pytest.raises(ValueError):m.validate(dict(config,prompt=' '))
