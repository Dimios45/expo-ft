#!/usr/bin/env python3
"""Measure literal all-parameter YAM base fine-tuning on successful replay.

This is an isolated flow-matching benchmark, not the paper's LoRA parameter mask
or RTC prefix-training recipe. It never deploys a policy or updates an EXPO registry.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'expo_ft/agents/vla/openpi/src')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR','platform')


def group(path):
    text='/'.join(map(str,path))
    if text.startswith('PaliGemma/img'):return 'vision'
    if '_1/' in text or text.endswith('_1') or not text.startswith('PaliGemma/'):
        return 'action_expert_and_projections'
    return 'language'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-root',type=Path,default=ROOT/'artifacts/yam-online-base/learner')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--save-weights',action='store_true',help='Export new inference weights, not optimizer state')
    a=p.parse_args()
    if a.steps<1 or a.batch_size<1:p.error('Positive steps and batch size required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx
    from expo_ft.conversion.yam_loader import as_observation
    from expo_ft.conversion.yam_pi05 import load_model
    from expo_ft.yam.replay import Episode
    from expo_ft.yam.rounds import read,validate_version,atomic_json

    exp=read(a.replay_root/'experiment.json');cur=read(a.replay_root/'current.json')
    manifest=validate_version(cur['checkpoint'])
    episodes=[]
    for item in manifest['replay_inventory']:
        ep=Episode(item['path'])
        if ep.meta['session']['reward']==1 and ep.meta['session']['terminal']=='success':
            episodes.append(ep)
    if not episodes:raise ValueError('No successful finalized episodes')
    report=dict(mode='literal_all_weights_flow_matching',paper_lora_recipe=False,
                base_checkpoint=exp['checkpoint'],source_experiment=exp['experiment_id'],
                replay=[dict(path=str(e.path),episode_id=e.meta['episode_id'],transitions=len(e)) for e in episodes],
                batch_size=a.batch_size,requested_steps=a.steps,steps=[],learning_rate=2.5e-5,
                optimizer='AdamW b1=.9 b2=.95 eps=1e-8 wd=1e-10 clip_global_norm=1',
                dtype='float32',rtc_prefix_training=False,hardware_accessed=False,deployed=False)
    atomic_json(a.output/'report.json',report)
    model=load_model(exp['checkpoint'],'float32')
    model.train()
    graph,params,other=nnx.split(model,nnx.Param,...)
    del model
    counts={}
    for path,value in params.flat_state().items():
        label=group(path);counts[label]=counts.get(label,0)+value.value.size
    report.update(trainable_parameters=sum(counts.values()),parameter_groups=counts,
                  parameter_bytes=sum(x.nbytes for x in jax.tree.leaves(params)),
                  parameter_dtypes=sorted(set(str(x.dtype) for x in jax.tree.leaves(params))))
    print(json.dumps({'parameter_groups':counts,'trainable_parameters':sum(counts.values())}),flush=True)
    atomic_json(a.output/'report.json',report)
    if a.save_weights and shutil.disk_usage(a.output).free < report['parameter_bytes']*1.2:
        raise RuntimeError('Insufficient disk for separate inference checkpoint')
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
    opt=tx.init(params)

    def step(parameters,opt_state,obs,actions,key):
        def loss_fn(p):
            net=nnx.merge(graph,p,other)
            with jax.default_matmul_precision('highest'):
                return net.compute_loss(key,obs,actions,train=True).mean()
        loss,grads=jax.value_and_grad(loss_fn)(parameters)
        metrics={'loss':loss,'gradient_norm':optax.global_norm(grads)}
        for label in counts:
            leaves=[v.value for path,v in grads.flat_state().items() if group(path)==label]
            metrics[label+'_gradient_norm']=optax.global_norm(leaves)
        updates,opt_state=tx.update(grads,opt_state,parameters)
        metrics['update_norm']=optax.global_norm(updates)
        return optax.apply_updates(parameters,updates),opt_state,metrics

    compiled=jax.jit(step,donate_argnums=(0,1))
    rng=np.random.default_rng(42)
    weights=np.array([len(e) for e in episodes],float);weights/=weights.sum()
    for i in range(a.steps):
        samples=[];obs_list=[];actions=[]
        for _ in range(a.batch_size):
            e=int(rng.choice(len(episodes),p=weights));index=int(rng.integers(len(episodes[e])))
            samples.append([e,index]);obs_list.append(episodes[e].observation(index))
            actions.append(np.pad(episodes[e].arrays['actions'][index],((0,0),(0,18))))
        data={k:np.concatenate([o[k] for o in obs_list],axis=0) for k in obs_list[0]}
        obs=as_observation(data)
        # The existing helper builds masks for a single item; expand for this batch.
        obs=obs.replace(image_masks={k:jnp.ones((a.batch_size,),dtype=bool) for k in obs.images}) if hasattr(obs,'replace') else obs
        started=time.monotonic()
        params,opt,metrics=compiled(params,opt,obs,jnp.asarray(np.stack(actions)),jax.random.key(i))
        metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError(str(metrics))
        if not all(metrics[k+'_gradient_norm']>0 for k in counts):raise AssertionError('A base component received no gradient')
        record=dict(step=i+1,seconds=time.monotonic()-started,samples=samples,**metrics)
        report['steps'].append(record);atomic_json(a.output/'report.json',report)
        print(json.dumps(record),flush=True)
    report['completed']=True
    if a.save_weights:
        import orbax.checkpoint as ocp
        target=a.output/'checkpoint';target.mkdir()
        for path in Path(exp['checkpoint']).iterdir():
            if path.is_file() and (path.name in ('config.json','policy_preprocessor.json','policy_postprocessor.json') or path.suffix=='.safetensors'):
                shutil.copy2(path,target/path.name)
        with ocp.PyTreeCheckpointer() as saver:
            saver.save(target/'params',{'params':nnx.State.merge(params,other).to_pure_dict()})
        report['saved_weights']=str(target)
        report['optimizer_saved']=False
    atomic_json(a.output/'report.json',report)
    print('COMPLETED',str(a.output/'report.json'),flush=True)


if __name__=='__main__':main()
