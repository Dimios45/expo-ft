#!/usr/bin/env python3
"""Isolated recorded-input two-phase RPC benchmark. Never connects to robot APIs."""
import argparse
import asyncio
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'expo_ft/agents/vla/openpi/src')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR','platform')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
from expo_ft.yam.lan_protocol import pack,unpack

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode',choices=['server','client'])
p.add_argument('--host',required=True)
p.add_argument('--port',type=int,default=18765)
p.add_argument('--token-file',type=Path,required=True)
p.add_argument('--checkpoint')
p.add_argument('--tokenizer')
p.add_argument('--fixture',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=100)
a=p.parse_args()
if a.repeats < 1: p.error('positive repeat count required')
episode='rpc-'+uuid.uuid4().hex
token=a.token_file.read_text().strip()
config=dict(run='benchmark',checkpoint=a.checkpoint,tokenizer=a.tokenizer,dtype='bfloat16',
            prompt='fold the t-shirt',replan_steps=15,delay=14,candidates=2,
            base_sha256='a'*64,preprocessing_sha256='b'*64,gates={'robot_authorized':False})

async def server():
    import jax
    import torch
    from expo_ft.yam.lan_policy import RTCPolicy,settings
    from expo_ft.yam.stable import StableEXPO
    from expo_ft.yam.lan_inference import Inference,serve
    torch.set_num_threads(4)
    policy=RTCPolicy(config)
    agent=StableEXPO(settings(config))
    params=dict(encoder=agent.state['encoder'].params,editor=agent.state['editor'].params,target_q=agent.state['target_q'])
    service=Inference(policy,config)
    await service.stage({'version':1},jax.device_get(params))
    del agent,params
    frames=dict(np.load(a.fixture,allow_pickle=False))
    from PIL import Image
    images={}
    for k,v in frames.items():
        buf=io.BytesIO();Image.fromarray(v).save(buf,format='JPEG',quality=95);images[k]=buf.getvalue()
    obs=dict(state=np.zeros(14,np.float32),images=images)
    candidates=policy.propose(obs,np.zeros((14,14),np.float32),0)
    policy.select(service.snapshots.ready.params,obs,candidates,0)
    stop=asyncio.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):asyncio.get_running_loop().add_signal_handler(sig,stop.set)
    a.output.write_text(json.dumps(dict(ready=True,pid=os.getpid(),robot_accessed=False)))
    try:await serve(service,a.host,a.port,token,stop)
    finally:service.close()

async def client():
    from PIL import Image
    from websockets.asyncio.client import connect
    frames=dict(np.load(a.fixture,allow_pickle=False))
    def observation(tick):
        images={}
        t=time.monotonic_ns()
        for k,v in frames.items():
            buf=io.BytesIO();Image.fromarray(v).save(buf,format='JPEG',quality=95);images[k]=buf.getvalue()
        return dict(tick=tick,state=np.zeros(14,np.float32),images=images,capture_ns=t,
                    image_ns={k:t for k in frames},frame_ids={k:tick for k in frames})
    records=[]
    async with connect(f'ws://{a.host}:{a.port}',additional_headers={'Authorization':'Bearer '+token},compression=None,max_size=16*1024**2) as ws:
        async def rpc(m):
            await ws.send(pack(dict(schema=1,run='benchmark',episode=episode,**m)))
            reply=unpack(await ws.recv())
            if not reply['ok']:raise RuntimeError(reply)
            return reply
        version=(await rpc(dict(op='begin')))['version']
        for i in range(a.repeats):
            start=time.monotonic()
            tick=i*15
            obs=observation(tick)
            await rpc(dict(op='propose',version=version,observation=obs,request_id=f'p{i}',
                           seed=i,budget_ms=1000,prefix=np.zeros((14,14),np.float32),start_tick=tick+14))
            proposal=time.monotonic()-start
            select_at=start+13/30
            await asyncio.sleep(max(0,select_at-time.monotonic()))
            fresh=time.monotonic()
            fresh_obs=observation(tick+13)
            encoded=time.monotonic()
            response=await rpc(dict(op='select',version=version,observation=fresh_obs,
                               request_id=f's{i}',proposal_id=f'p{i}',seed=i,budget_ms=1000,
                               expires_ns=int((start+14/30)*1e9)))
            end=time.monotonic()
            records.append(dict(proposal_seconds=proposal,selection_seconds=end-fresh,
                                selection_encode_seconds=encoded-fresh,server_selection_ms=response["server_ms"],
                                response_at_seconds=end-start,deadline_missed=end>start+14/30))
            await asyncio.sleep(max(0,start+15/30-time.monotonic()))
        await rpc(dict(op='end'))
    report=dict(robot_accessed=False,recorded_images=True,synthetic_states_prefixes=True,
                checkpoint_version=version,config={'C':15,'delay':14,'candidates':2,'control_hz':30},
                samples=records,deadline_misses=sum(r['deadline_missed'] for r in records))
    for k in ('proposal_seconds','selection_seconds','response_at_seconds'):
        values=[r[k] for r in records]
        report[k]=dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),max=max(values))
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)

asyncio.run(server() if a.mode=='server' else client())
