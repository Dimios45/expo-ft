"""Published checkpoints may omit the historical conversion-source manifest."""
import pytest
import importlib.util
from pathlib import Path

from expo_ft.yam.rounds import atomic_json, initialize, read


def assets(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "weights").write_bytes(b"fixture weights")
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        atomic_json(checkpoint / name, {})
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    atomic_json(tokenizer / "tokenizer_config.json", {})
    return checkpoint, tokenizer


def test_published_checkpoint_fingerprint_tracks_weights(tmp_path):
    checkpoint, tokenizer = assets(tmp_path)
    first = initialize(tmp_path / "first", checkpoint, tokenizer, {}, "fold the towel")
    a = read(first / "experiment.json")
    assert a["checkpoint_provenance"]["kind"] == "published-checkpoint-assets"
    assert "params/weights" in a["checkpoint_provenance"]["files"]
    (checkpoint / "params" / "weights").write_bytes(b"changed weights")
    second = initialize(tmp_path / "second", checkpoint, tokenizer, {}, "fold the towel")
    assert read(second / "experiment.json")["source_sha"] != a["source_sha"]


def test_legacy_source_identity_preserved(tmp_path):
    checkpoint, tokenizer = assets(tmp_path)
    atomic_json(checkpoint / "conversion_manifest.json", {"source_sha256": "legacy"})
    root = initialize(tmp_path / "run", checkpoint, tokenizer, {}, "fold the towel")
    assert read(root / "experiment.json")["source_sha"] == "legacy"


def test_missing_assets_do_not_leave_partial_experiment(tmp_path):
    checkpoint, tokenizer = assets(tmp_path)
    (checkpoint / "config.json").unlink()
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="Missing checkpoint asset"):
        initialize(root, checkpoint, tokenizer, {}, "fold the towel")
    assert not root.exists()


def test_stable_first_round_parent_and_missing_trained_parent(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/yam/continue_stable.py"
    spec = importlib.util.spec_from_file_location("continue_stable_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    atomic_json(tmp_path / "experiment.json", {
        "profile": "stable-v2", "settings": {"actor_mode": "frozen"}})
    atomic_json(tmp_path / "current.json", {
        "version": 0, "checkpoint": None, "episodes": []})
    _, _, parent, manifest = module.parent_state(tmp_path)
    assert parent is None and manifest["replay_inventory"] == []
    atomic_json(tmp_path / "current.json", {
        "version": 1, "checkpoint": None, "episodes": ["episode"]})
    with pytest.raises(ValueError, match="Missing trained parent"):
        module.parent_state(tmp_path)
