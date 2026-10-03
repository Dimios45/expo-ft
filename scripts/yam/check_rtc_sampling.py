"""Hardware-free prefix preservation, parity, and latency benchmark."""
import argparse
import io
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'expo_ft/agents/vla/openpi/src')]
import json_numpy
import msgpack
import numpy as np
from PIL import Image
from expo_ft.conversion.yam_loader import YamJaxPolicy, as_observation
from expo_ft.yam.rtc_sampling import RTCSampler

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', default='artifacts/yam-rtc/sampler-report.json')
p.add_argument('--candidates', type=int, default=32)
a = p.parse_args()
policy = YamJaxPolicy('artifacts/yam_pi05_jax', 'artifacts/paligemma-tokenizer')
fixture = msgpack.unpackb(Path('artifacts/yam-rtc/round3-probe.msgpack').read_bytes(), raw=False)[0]
wire = json_numpy.loads(fixture['observation_json'])
images = {k: np.asarray(Image.open(io.BytesIO(wire[k+'_cam'].tobytes())).convert('RGB')) for k in ('top','left','right')}
obs = as_observation(policy.processor.prepare_numpy(images, wire['state'], wire['instruction']))
sampler = RTCSampler(policy.model)
noise = np.random.default_rng(0).normal(size=(1, 1, 30, 32)).astype(np.float32)
reference = np.asarray(policy._sample(policy._state, obs, noise[:, 0]))
result, _ = sampler.candidates(obs, np.zeros((1,0,32),np.float32), noise)
error = float(np.max(np.abs(result-reference)))
if not np.allclose(result, reference, rtol=1e-4, atol=1e-5):
    raise AssertionError(f'zero-delay parity failed {error}')
prefix = reference[:, :5].copy()
noise = np.random.default_rng(1).normal(size=(1, a.candidates, 30, 32)).astype(np.float32)
times = []
for i in range(4):
    start = time.monotonic()
    result, window = sampler.candidates(obs, prefix, noise)
    elapsed = time.monotonic()-start
    print(json.dumps(dict(iteration=i, seconds=elapsed, shape=list(window.shape))), flush=True)
    times.append(elapsed)
report = dict(zero_delay_max_abs_error=error, prefix_preserved=True, candidates=a.candidates,
              delay=5, replan_steps=8, compile_and_first_seconds=times[0], warm_seconds=times[1:],
              hardware_accessed=False, training_enabled=False,
              note='Base sampling only; excludes network, editor/Q, preprocessing and concurrent training.')
report['base_only_deadline_passed'] = max(times[1:]) <= 5 / 30
report['configured_delay_budget_seconds'] = 5 / 30
report['replan_window_seconds'] = 8 / 30
report['deployment_ready'] = False
Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
