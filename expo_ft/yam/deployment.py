"""Immutable checkpoint installation; no model imports or GPU allocation."""
from pathlib import Path

from .rounds import atomic_json, read, sha, validate_version


def candidate_paths(root, current, manifest_sha):
    root = Path(root).resolve()
    version = current['version']
    if type(version) is not int or version < 1:
        raise ValueError('Expected positive candidate version')
    if len(manifest_sha) != 64 or any(c not in '0123456789abcdef' for c in manifest_sha):
        raise ValueError('Invalid manifest hash')
    final = root/'learner/versions'/f'{version:04d}'
    if Path(current['checkpoint']) != final:
        raise ValueError('Checkpoint path does not match registry/version')
    stage = final.with_name(f'.incoming-{version:04d}-{manifest_sha}')
    return stage, final


def publish_candidate(root, current, manifest_sha):
    root = Path(root)
    path = root/'learner/current.json'
    old = read(path)
    if current['version'] < old['version']:
        raise ValueError('Refusing registry downgrade')
    if current['version'] == 0:
        if current != old: raise ValueError('Conflicting base registry')
        return
    stage, final = candidate_paths(root, current, manifest_sha)
    source = final if final.exists() else stage
    if sha(source/'manifest.json') != manifest_sha:
        raise ValueError('Manifest checksum mismatch')
    manifest = validate_version(source)
    exp = read(root/'learner/experiment.json')
    if manifest['version'] != current['version'] or manifest['settings'] != exp['settings']:
        raise ValueError('Candidate configuration mismatch')
    if old['version'] == current['version'] and old != current:
        raise ValueError('Conflicting immutable version')
    if not final.exists(): stage.rename(final)
    atomic_json(path, current)
