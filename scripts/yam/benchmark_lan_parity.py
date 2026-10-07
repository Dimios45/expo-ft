#!/usr/bin/env python3
"""Real-weight bf16 zero-delay parity and committed-prefix checks, no robot."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'expo_ft/agents/vla/openpi/src')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
os.environ.setdefault('OMP_NUM_THREADS', '4')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--dtype', choices=['bfloat16','float32'], default='bfloat16')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():p.error('output exists')
    import jax
    import numpy as np
    import torch
    from expo_ft.conversion.yam_loader import YamJaxPolicy, as_observation
    from expo_ft.yam.rtc_sampling import RTCSampler
    torch.set_num_threads(4)
    policy = YamJaxPolicy(a.checkpoint, a.tokenizer, dtype=a.dtype)
    rng = np.random.default_rng(17)
    images = {k:rng.integers(0,256,(360,640,3),dtype=np.uint8) for k in ('top','left','right')}
    obs = as_observation(policy.processor.prepare_numpy(images,np.zeros(14,np.float32),'fold the t-shirt'))
    noise = rng.standard_normal((1,1,30,32)).astype(np.float32)
    reference = np.asarray(policy._sample(policy._state,obs,noise[:,0]))
    sampler = RTCSampler(policy.model)
    t = time.perf_counter()
    result,_ = sampler.candidates(obs,np.empty((1,0,32),np.float32),noise)
    cold = time.perf_counter()-t
    times=[]
    for _ in range(10):
        t=time.perf_counter();sampler.candidates(obs,np.empty((1,0,32),np.float32),noise)
        times.append(time.perf_counter()-t)
    error=float(np.max(np.abs(result-reference)))
    physical_error=float(np.max(np.abs(policy.processor.unnormalize_actions(result)-policy.processor.unnormalize_actions(reference))))
    prefix=reference[:,:5].copy()
    full,_=sampler.candidates(obs,prefix,np.repeat(noise,8,axis=1))
    report=dict(dtype=a.dtype,device=str(jax.devices()[0]),zero_delay_max_abs_error=error,
                zero_delay_wire_max_abs_error=physical_error,zero_delay_allclose=bool(np.allclose(result,reference,rtol=1e-4,atol=1e-5)),
                prefix_exact=bool(np.array_equal(full[:,:5],np.broadcast_to(prefix,(8,5,32)))),
                bootstrap_compile_seconds=cold,bootstrap_warm_seconds=times,
                robot_accessed=False,behavior_quality_validated=False,checkpoint=a.checkpoint)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['zero_delay_allclose'] or not report['prefix_exact']:raise SystemExit(1)


if __name__=='__main__':main()
