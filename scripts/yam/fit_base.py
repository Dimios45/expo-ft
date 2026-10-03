#!/usr/bin/env python3
"""Offline all-weight YAM fitting with episode holdout and best-checkpoint selection.

Not RTC/LoRA training. Never promotes a model into an existing online experiment.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'expo_ft/agents/vla/openpi/src'),str(Path(__file__).parent)]
from benchmark_base_update import group


def mean_gradients(loss_fn, params, batches, keys):
    """Average independent microbatch gradients before one optimizer update."""
    import jax
    import jax.numpy as jnp
    initial=(jnp.array(0.,jnp.float32),jax.tree.map(jnp.zeros_like,params))
    def accumulate(carry,item):
        batch,key=item
        loss,grad=jax.value_and_grad(loss_fn)(params,batch,key)
        return (carry[0]+loss,jax.tree.map(lambda a,b:a+b,carry[1],grad)),None
    (loss,grads),_=jax.lax.scan(accumulate,initial,(batches,keys))
    n=keys.shape[0]
    return loss/n,jax.tree.map(lambda x:x/n,grads)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-root',type=Path,default=ROOT/'artifacts/yam-online-base/learner')
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--accumulate',type=int,default=4)
    p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--learning-rate',type=float,default=5e-6)
    p.add_argument('--seed',type=int,default=123)
    a=p.parse_args()
    if min(a.steps,a.accumulate,a.eval_every)<1 or a.learning_rate<=0:p.error('Positive training settings required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    from flax import nnx
    from expo_ft.conversion.yam_loader import as_observation
    from expo_ft.conversion.yam_pi05 import load_model
    from expo_ft.yam.replay import Episode
    from expo_ft.yam.rounds import read,atomic_json,validate_version,sha
    exp=read(a.replay_root/'experiment.json');cur=read(a.replay_root/'current.json')
    manifest=validate_version(cur['checkpoint'])
    episodes=[]
    for item in manifest['replay_inventory']:
        e=Episode(item['path'])
        if e.meta['session']['reward']==1 and e.meta['session']['terminal']=='success':
            for name,digest in item['replay_files'].items():
                if sha(Path(item['path'])/name)!=digest:raise ValueError('Replay file changed: '+name)
            episodes.append(e)
    if len(episodes)<3:raise ValueError('Need at least three successful episodes for this split')
    # Latest successful episode is entirely held out; start from original base,
    # not any earlier benchmark which already used all successful episodes.
    train,heldout=episodes[:-1],episodes[-1]
    validation=np.unique(np.linspace(0,len(heldout)-1,8,dtype=int)).tolist()
    report=dict(mode='all_weights_flow_matching',source_checkpoint=exp['checkpoint'],source_experiment=exp['experiment_id'],
                steps=[],evaluations=[],train_episode_ids=[e.meta['episode_id'] for e in train],
                validation_episode_id=heldout.meta['episode_id'],validation_indices=validation,
                train_windows=sum(len(e) for e in train),validation_windows=len(heldout),
                microbatch=1,accumulate=a.accumulate,effective_batch=a.accumulate,learning_rate=a.learning_rate,
                seed=a.seed,requested_steps=a.steps,optimizer_saved=False,hardware_accessed=False,deployed=False,
                limitations=['Only one held-out successful episode; noisy small-data model selection',
                             'Fixed LeRobot windows lack authoritative RTC timing','No physical behavior validation'])
    atomic_json(a.output/'report.json',report)
    model=load_model(exp['checkpoint'],'float32');model.train()
    graph,params,other=nnx.split(model,nnx.Param,...);del model
    count=sum(x.size for x in jax.tree.leaves(params));nbytes=sum(x.nbytes for x in jax.tree.leaves(params))
    if shutil.disk_usage(a.output).free < nbytes*1.15:raise RuntimeError('Insufficient disk for best checkpoint export')
    report.update(trainable_parameters=count,parameter_bytes=nbytes)
    atomic_json(a.output/'report.json',report)
    warmup=min(5,max(1,a.steps//4))
    lr=optax.warmup_cosine_decay_schedule(a.learning_rate/5,a.learning_rate,warmup,max(a.steps,warmup+1),end_value=a.learning_rate/10)
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(lr,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
    opt=tx.init(params)
    def loss_fn(p,batch,key,training):
        obs,actions=batch
        net=nnx.merge(graph,p,other)
        if not training:net.eval()
        with jax.default_matmul_precision('highest'):
            return net.compute_loss(key,obs,actions,train=training).mean()
    def step(p,opt_state,batches,keys):
        loss,grads=mean_gradients(lambda p,b,k:loss_fn(p,b,k,True),p,batches,keys)
        updates,opt_state=tx.update(grads,opt_state,p)
        metrics=dict(loss=loss,gradient_norm=optax.global_norm(grads),update_norm=optax.global_norm(updates))
        for label in ('vision','language','action_expert_and_projections'):
            metrics[label+'_gradient_norm']=optax.global_norm([v.value for path,v in grads.flat_state().items() if group(path)==label])
        return optax.apply_updates(p,updates),opt_state,metrics
    step=jax.jit(step,donate_argnums=(0,1))
    evaluate=jax.jit(lambda p,b,k:loss_fn(p,b,k,False))
    def batch(ep,i):
        return as_observation(ep.observation(i)),jnp.asarray(np.pad(ep.arrays['actions'][i],((0,0),(0,18)))[None])
    validation_batches=[batch(heldout,i) for i in validation]
    def score(p,stepnum):
        losses=[float(evaluate(p,b,jax.random.key(9000+i))) for i,b in enumerate(validation_batches)]
        if not np.isfinite(losses).all():raise FloatingPointError('Nonfinite validation')
        entry=dict(step=stepnum,loss=float(np.mean(losses)),per_sample=losses)
        report['evaluations'].append(entry);print('VALIDATION',json.dumps(entry),flush=True)
        return entry['loss']
    best=baseline=score(params,0);best_step=0;best_host=None
    rng=np.random.default_rng(a.seed)
    weights=np.array([len(e) for e in train],float);weights/=weights.sum()
    for index in range(a.steps):
        draws=[(int(rng.choice(len(train),p=weights)),0) for _ in range(a.accumulate)]
        draws=[(e,int(rng.integers(len(train[e])))) for e,_ in draws]
        batches=jax.tree.map(lambda *xs:jnp.stack(xs),*[batch(train[e],i) for e,i in draws])
        keys=jax.random.split(jax.random.key(a.seed+index),a.accumulate)
        start=time.monotonic();params,opt,metrics=step(params,opt,batches,keys)
        metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
        if not np.isfinite(list(metrics.values())).all():raise FloatingPointError(str(metrics))
        entry=dict(step=index+1,seconds=time.monotonic()-start,samples=draws,learning_rate=float(lr(index)),**metrics)
        report['steps'].append(entry);print('TRAIN',json.dumps(entry),flush=True)
        if (index+1)%a.eval_every==0 or index+1==a.steps:
            value=score(params,index+1)
            if value<best:
                best=value;best_step=index+1
                # Host snapshot avoids retaining a second 13 GB model on GPU.
                best_host=jax.device_get(params)
        report.update(best_step=best_step,best_validation_loss=best,baseline_validation_loss=baseline)
        atomic_json(a.output/'report.json',report)
    report['completed']=True
    if best_host is None:
        report['candidate_exported']=False
        atomic_json(a.output/'report.json',report)
        print('No candidate improved held-out loss; original retained',flush=True)
        return
    target=a.output/'checkpoint';target.mkdir()
    for path in Path(exp['checkpoint']).iterdir():
        if path.is_file() and (path.name in ('config.json','policy_preprocessor.json','policy_postprocessor.json') or path.suffix=='.safetensors'):
            shutil.copy2(path,target/path.name)
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(target/'params',{'params':nnx.State.merge(best_host,jax.device_get(other)).to_pure_dict()})
    report.update(candidate_exported=True,checkpoint=str(target),validation_relative_change=best/baseline-1)
    atomic_json(a.output/'report.json',report)
    print('COMPLETE',json.dumps({k:report[k] for k in ['best_step','baseline_validation_loss','best_validation_loss','checkpoint']}),flush=True)


if __name__=='__main__':main()
