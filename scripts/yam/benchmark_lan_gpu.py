#!/usr/bin/env python3
"""Isolated real-weight LAN benchmarks. Never commands robots or publishes weights.

Use measure_gpu.py around each process for sampled whole-GPU peak VRAM.
Synthetic observations measure mechanics, not policy quality or real robot timing.
"""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'expo_ft/agents/vla/openpi/src')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['serve', 'train'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, help='Optional recorded LeRobot v3 frame triplet, encoded as JPEG q95')
    p.add_argument('--candidates', type=int, default=8)
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--delay', type=int, default=5)
    p.add_argument('--scratch', type=Path, default=Path('/tmp'))
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--critic-steps', type=int, default=40)
    p.add_argument('--prepare-transitions', type=int, default=0,
                   help='Time this many uncached next-candidate preparations before updating')
    a = p.parse_args()
    if a.repeats < 1 or a.critic_steps < 1 or a.prepare_transitions < 0 or a.microbatch not in (1, 2, 4, 8):
        p.error('positive repeats/steps, nonnegative preparation count, and microbatch 1/2/4/8 required')
    if a.output.exists():
        p.error('output exists; choose a new report')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(a.scratch).free < 2 * 1024**3:
        p.error('scratch filesystem needs at least 2 GiB free')
    scratch_context = tempfile.TemporaryDirectory(prefix='yam-lan-gpu-', dir=a.scratch)
    scratch = Path(scratch_context.name)
    import jax
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from expo_ft.yam.lan_policy import RTCPolicy, settings
    from expo_ft.conversion.yam_pi05 import JAX_CAMERAS
    config = dict(checkpoint=a.checkpoint, tokenizer=a.tokenizer, dtype='bfloat16',
                  prompt='fold the t-shirt', replan_steps=a.window, delay=a.delay, candidates=a.candidates)
    report = dict(schema=1, mode=a.mode, config=config, device=str(jax.devices()[0]),
                  synthetic_observations=True, robot_accessed=False, published=False,
                  completed=False, measurements={})
    if jax.devices()[0].platform != 'gpu':
        raise RuntimeError('GPU benchmark requires a GPU backend')
    def save():
        a.output.write_text(json.dumps(report, indent=2) + '\n')
    def measure(label, fn, repeats):
        times = []
        result = None
        for i in range(repeats):
            t = time.perf_counter()
            result = fn(i)
            jax.block_until_ready(result)
            times.append(time.perf_counter() - t)
            print(json.dumps(dict(label=label, iteration=i, seconds=times[-1])), flush=True)
        report['measurements'][label] = dict(seconds=times, min=min(times), median=float(np.median(times)),
                                           p95=float(np.percentile(times, 95)), max=max(times))
        save()
        return result
    report['hostname'] = platform.node()
    report['jax_version'] = jax.__version__
    report['gpu_inventory'] = subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    report['scratch_filesystem'] = str(a.scratch)
    save()
    t = time.perf_counter()
    policy = RTCPolicy(config)
    report['load_seconds'] = time.perf_counter() - t
    report['base_parameter_bytes'] = sum(x.nbytes for x in jax.tree.leaves(policy.sampler.state))
    report['base_dtypes'] = sorted(set(str(x.dtype) for x in jax.tree.leaves(policy.sampler.state)))
    rng = np.random.default_rng(123)
    obs = dict(state=np.zeros(14, np.float32), images={k: rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
                                                     for k in ('top', 'left', 'right')})
    if a.dataset is not None:
        import av
        import io
        import pyarrow.parquet as pq
        from PIL import Image
        from expo_ft.yam.replay import dataset_to_wire
        images = {}
        for role, key in [('top', 'top'), ('left', 'left_wrist'), ('right', 'right_wrist')]:
            video = sorted((a.dataset / 'videos' / ('observation.images.' + key)).rglob('*.mp4'))[0]
            with av.open(str(video)) as stream:
                frame = next(stream.decode(video=0)).to_ndarray(format='rgb24')
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format='JPEG', quality=95)
            images[role] = buf.getvalue()
        data_file = sorted((a.dataset / 'data').rglob('*.parquet'))[0]
        state = pq.read_table(data_file, columns=['observation.state'])['observation.state'][0].as_py()
        obs = dict(state=dataset_to_wire(state, 'karma-recorded'), images=images)
        report.update(synthetic_observations=False, recorded_input=str(a.dataset),
                      synthetic_prefix_actions=True, synthetic_rewards=True, jpeg_quality=95,
                      repeated_single_observation=True)
    prepared = measure('preprocess', lambda _: policy.prepare(obs), a.repeats)
    prefix = np.zeros((a.delay, 14), np.float32)
    candidate = measure('proposal_compile', lambda _: policy.propose(obs, prefix, 4), 1)
    measure('proposal_warm', lambda i: policy.propose(obs, prefix, i), a.repeats)
    report['proposal_budget_seconds'] = a.delay / 30
    report['proposal_meets_budget_on_this_host'] = report['measurements']['proposal_warm']['max'] < a.delay / 30
    save()
    from expo_ft.yam.stable import StableEXPO
    t = time.perf_counter()
    agent = StableEXPO(settings(config), microbatch=a.microbatch)
    jax.block_until_ready(agent.state)
    report['learner_initialization_seconds'] = time.perf_counter() - t
    params = dict(encoder=agent.state['encoder'].params, editor=agent.state['editor'].params,
                  target_q=agent.state['target_q'])
    report['inference_component_bytes'] = {k: sum(x.nbytes for x in jax.tree.leaves(v)) for k,v in params.items()}
    report['inference_tensor_bytes'] = sum(report['inference_component_bytes'].values())
    report['training_state_bytes'] = sum(x.nbytes if hasattr(x, "nbytes") else np.asarray(x).nbytes for x in jax.tree.leaves(agent.state))
    save()
    if a.mode == 'serve':
        # Free optimizer/current-Q tensors before timing inference-only staging.
        host_params = jax.device_get(params)
        del params, agent
        gc.collect()
        staged = measure('stage_compile', lambda _: policy.selector.stage(host_params), 1)
        measure('select_warm', lambda i: policy.select(staged, obs, candidate, i), a.repeats)
        def full(i):
            candidates = policy.propose(obs, prefix, i)
            return policy.select(staged, obs, candidates, i)
        measure('propose_and_select_warm', full, a.repeats)
        # Retain both old and new buffers as required for atomic episode swaps.
        second = measure('stage_second_buffer', lambda _: policy.selector.stage(host_params), 1)
        measure('select_after_swap', lambda i: policy.select(second, obs, candidate, i), a.repeats)
        report['jax_memory_stats'] = jax.devices()[0].memory_stats()
    else:
        image = np.concatenate([prepared[k] for k in JAX_CAMERAS], -1)
        batch = dict(images=image, next_images=image, states=prepared['state'][:, :14],
                     next_states=prepared['state'][:, :14], actions=candidate[0].reshape(1, -1),
                     next_candidates=candidate.reshape(1, a.candidates, -1), rewards=np.ones(1, np.float32),
                     masks=np.ones(1, np.float32), steps=np.full(1, a.window, np.float32))
        batches = [batch for _ in range(8)]
        measure('critic_compile', lambda _: agent.critic_update(batches, []), 1)
        measure('editor_compile', lambda _: agent.editor_update(batches), 1)
        # TrainState integer steps become device scalars after the first update.
        # Warm both paths after that transition, including asynchronous apply work.
        def settle(_):
            agent.critic_update(batches, [])
            agent.editor_update(batches)
            jax.block_until_ready(agent.state)
        measure('state_signature_warmup', settle, 2)
        if a.prepare_transitions:
            preparation_start = time.perf_counter()
            for index in range(a.prepare_transitions):
                policy.propose(obs, prefix, index + 1000)
                if (index + 1) % 100 == 0:
                    print(json.dumps(dict(prepared_transitions=index+1,
                        seconds=time.perf_counter()-preparation_start)), flush=True)
            report['uncached_preparation_seconds'] = time.perf_counter() - preparation_start
            report['prepared_transitions'] = a.prepare_transitions
            save()
        cycle = time.perf_counter()
        updates = []
        for step in range(a.critic_steps):
            start = time.perf_counter()
            metrics = agent.critic_update(batches, [])
            if (step + 1) % 20 == 0:
                metrics.update(agent.editor_update(batches))
            jax.block_until_ready(agent.state)
            updates.append(dict(step=step+1, seconds=time.perf_counter()-start, **metrics))
            report['updates'] = updates
            save()
            print(json.dumps(updates[-1]), flush=True)
        report['warm_update_cycle_seconds'] = time.perf_counter() - cycle
        report['effective_batch'] = 8
        report['microbatch'] = a.microbatch
        # Serialize to an isolated benchmark directory, never the serving registry.
        from expo_ft.yam.artifacts import flatten
        from safetensors.numpy import save_file, load_file
        params = dict(encoder=agent.state['encoder'].params, editor=agent.state['editor'].params,
                      target_q=agent.state['target_q'])
        tensors = measure('device_to_host', lambda _: flatten(jax.device_get(params)), 1)
        target = scratch / 'weights.safetensors'
        def write(_):
            save_file(tensors, str(target))
            with target.open('rb') as f:
                os.fsync(f.fileno())
        measure('serialize_fsync', write, 1)
        measure('deserialize', lambda _: load_file(str(target)), 1)
        report['artifact_bytes'] = target.stat().st_size
        # Keep only benchmark reports; these synthetic updates are not deployable.
        target.unlink()
        checkpoint = scratch / 'state.msgpack'
        def checkpoint_write(_):
            agent.save(checkpoint)
            with checkpoint.open('rb') as f:
                os.fsync(f.fileno())
        measure('checkpoint_fsync', checkpoint_write, 1)
        report['checkpoint_bytes'] = checkpoint.stat().st_size
        checkpoint.unlink()
        report['jax_memory_stats'] = jax.devices()[0].memory_stats()
    report['completed'] = True
    report['deployment_certified'] = False
    report['limitations'] = ['4090 timings do not establish 3060 deadlines or memory fit',
                            'no robot, network, camera timing, replay preparation or reset included',
                            'synthetic rewards do not measure learning quality']
    save()
    scratch_context.cleanup()
    print('REPORT ' + str(a.output), flush=True)


if __name__ == '__main__':
    main()
