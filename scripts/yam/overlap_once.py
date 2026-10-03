"""Historical one-shot overlap pilot; superseded by online_server.py.

Wait for next successful live inference, then train an isolated candidate once.
Do not launch alongside the online coordinator for the same learner registry.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from expo_ft.yam.rounds import atomic_json

root = ROOT / 'artifacts/yam-overlap-test'
pending = json.loads((root / 'pending.json').read_text())
if not pending.get('prepared'):
    raise RuntimeError('Prepare replay before arming')
armed = time.time()
status = root / 'overlap-status.json'
atomic_json(status, dict(state='waiting_for_live_inference', armed_at=armed,
                         serving_version=3, training_episode=pending['episode_id']))
print('ARMED: training starts after the next successful infer_live request', flush=True)
event = root / 'live-event.json'
while True:
    if event.exists() and json.loads(event.read_text())['completed_at'] > armed:
        break
    if time.time() - armed > 3600:
        atomic_json(status, dict(state='expired', armed_at=armed))
        raise SystemExit('No live inference within one hour; not trained')
    time.sleep(1)
atomic_json(status, dict(state='training', started_at=time.time(), armed_at=armed))
env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
           XLA_PYTHON_CLIENT_ALLOCATOR='platform', OMP_NUM_THREADS='4')
env.pop('LD_LIBRARY_PATH', None)
command = [sys.executable, 'scripts/yam/measure_gpu.py', '--output', 'logs/yam-overlap-train',
           '--', sys.executable, 'scripts/yam/continue_stable.py', 'train', '--root', str(root),
           '--microbatch', '1']
code = subprocess.call(command, env=env)
atomic_json(status, dict(state='candidate_ready' if code == 0 else 'failed',
                         exit_code=code, finished_at=time.time(), promoted=False))
raise SystemExit(code)
