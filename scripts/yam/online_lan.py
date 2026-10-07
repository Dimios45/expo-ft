#!/usr/bin/env python3
"""Three-host adaptation of the supervised A100 episode-wise online run.

Inference and upload/control use separate wired endpoints. Only the NUC's
collector starts hardware. Boundary publication can pause, as in the A100 run.
"""
import argparse
import asyncio
import hashlib
import fcntl
import signal
import hmac
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'expo_ft/agents/vla/openpi/src')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR','platform')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('YAM_CANDIDATE_BATCH','1')
from expo_ft.yam.rounds import atomic_json, read, sha

REMOTE = 'sra@192.168.0.119'
SSH = ['ssh','-i',str(Path.home()/'.ssh/yam_lan_bench'),'-o','BatchMode=yes','-o','ConnectTimeout=10']


def remote(code, value=None):
    return subprocess.check_output(SSH+[REMOTE,'python3 -c '+shlex.quote(code)],
        input=json.dumps(value).encode() if value is not None else None,timeout=120)


def fingerprint(policy, checkpoint, tokenizer):
    """Hash actual bf16 parameters, processor assets and tokenizer assets."""
    import jax
    import numpy as np
    digest=hashlib.sha256()
    leaves, tree=jax.tree.flatten(policy._state)
    digest.update(str(tree).encode())
    for leaf in leaves:
        a=np.asarray(jax.device_get(leaf))
        digest.update(str((a.shape,str(a.dtype))).encode()); digest.update(a.tobytes())
    for root,names in [(Path(checkpoint),['config.json','policy_preprocessor.json','policy_postprocessor.json']),
                       (Path(tokenizer),sorted(p.name for p in Path(tokenizer).iterdir() if p.is_file() and p.suffix in ('.json','.model')) )]:
        for name in names:
            digest.update(name.encode()); digest.update((root/name).read_bytes())
    return digest.hexdigest()


def bootstrap(root, checkpoint, tokenizer, prompt):
    learner=root/'learner'
    if not (learner/'experiment.json').exists():
        subprocess.run([sys.executable,str(REPO/'scripts/yam/continue_stable.py'),'init',
            '--root',str(learner),'--checkpoint',str(checkpoint),'--tokenizer',str(tokenizer),
            '--prompt',prompt],check=True)
        exp=read(learner/'experiment.json'); exp['base_dtype']='bfloat16'
        atomic_json(learner/'experiment.json',exp)
    exp=read(learner/'experiment.json')
    if exp.get('base_dtype')!='bfloat16' or exp['prompt']!=prompt:
        raise ValueError('Existing run configuration differs; choose a new YAM_RUN.')
    if 'lan_base_sha256' not in exp:
        if read(learner/'current.json')['version']!=0:
            raise ValueError('Cannot retrofit an old run')
        print('Fingerprinting base weights; this may take a minute. Wait for Online learner ready before starting other hosts.',flush=True)
        subprocess.run([sys.executable,__file__,'fingerprint','--root',str(root),
            '--checkpoint',str(checkpoint),'--tokenizer',str(tokenizer)],check=True)
    tokenpath=root/'control.token'
    if not tokenpath.exists():
        fd=os.open(tokenpath,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f: f.write(secrets.token_hex(32))
    payload={'root':str(root),'experiment':read(learner/'experiment.json'),
             'token':tokenpath.read_text().strip()}
    code="""import json,sys,os; from pathlib import Path
v=json.load(sys.stdin); root=Path(v['root'])
for name in ['learner','serve']:
 p=root/name; p.mkdir(parents=True,exist_ok=True)
 for sub in ['versions','sessions','replay']: (p/sub).mkdir(exist_ok=True)
 e=p/'experiment.json'
 if e.exists() and json.loads(e.read_text())!=v['experiment']: raise ValueError('Remote experiment mismatch')
 e.write_text(json.dumps(v['experiment']))
 c=p/'current.json'
 if not c.exists(): c.write_text(json.dumps(dict(version=0,episodes=[],checkpoint=None)))
t=root/'control.token'; t.write_text(v['token']); t.chmod(0o600)
"""
    remote(code,payload)
    nuc='yambox@192.168.0.121'
    token_dir='/home/yambox/yam-lan/runs/'+root.name
    subprocess.run(SSH+[nuc,'mkdir -p '+shlex.quote(token_dir)],check=True)
    subprocess.run(['scp','-q','-i',str(Path.home()/'.ssh/yam_lan_bench'),str(tokenpath),
        nuc+':'+token_dir+'/control.token'],check=True)
    subprocess.run(SSH+[nuc,'chmod 600 '+shlex.quote(token_dir+'/control.token')],check=True)
    # Compatibility for the earlier launcher; new tests use the run-scoped token.
    subprocess.run(['scp','-q','-i',str(Path.home()/'.ssh/yam_lan_bench'),str(tokenpath),
        nuc+':/home/yambox/yam-lan/control.token'],check=True)
    subprocess.run(SSH+[nuc,'chmod 600 /home/yambox/yam-lan/control.token'],check=True)



def sync_candidate(root):
    from expo_ft.yam.deployment import candidate_paths
    from expo_ft.yam.rounds import validate_version
    current=read(root/'learner/current.json')
    manifest_sha=None
    if current['checkpoint']:
        folder=Path(current['checkpoint'])
        validate_version(folder)
        manifest_sha=sha(folder/'manifest.json')
        stage, final=candidate_paths(root,current,manifest_sha)
        # Never rsync into a published version. Resume only in private staging.
        needed=json.loads(remote("""import json,sys
sys.path.insert(0,'/home/sra/yam-lan/runtime')
from expo_ft.yam.deployment import candidate_paths,publish_candidate
v=json.load(sys.stdin); stage,final=candidate_paths(v['root'],v['current'],v['manifest_sha'])
if final.exists(): publish_candidate(v['root'],v['current'],v['manifest_sha'])
else: stage.mkdir(parents=True,exist_ok=True)
print(json.dumps(not final.exists()))
""",{'root':str(root),'current':current,'manifest_sha':manifest_sha}))
        if not needed: return
        subprocess.run(['rsync','-rt','--checksum','--partial','--timeout=30','--bwlimit=20000',
            '-e',shlex.join(SSH),str(folder)+'/',REMOTE+':'+str(stage)+'/'],check=True,timeout=180)
    remote("""import json,sys
sys.path.insert(0,'/home/sra/yam-lan/runtime')
from expo_ft.yam.deployment import publish_candidate
v=json.load(sys.stdin); publish_candidate(v['root'],v['current'],v['manifest_sha'])
""",{'root':str(root),'current':current,'manifest_sha':manifest_sha})


async def coordinator(args):
    import msgpack
    from websockets.asyncio.server import serve
    from expo_ft.yam.online import Coordinator, CHUNK
    root=args.root; token=(root/'control.token').read_text().strip()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM,asyncio.current_task().cancel)
    if (root/'learner/online/STOPPED.json').exists(): raise ValueError('Run stopped; use a fresh YAM_RUN')
    class SplitCoordinator(Coordinator):
        def http(self,path,post=False):
            req=urllib.request.Request(self.policy_url+path,data=b'{}' if post else None,
                headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
            with urllib.request.urlopen(req,timeout=180) as r: return json.load(r)

        async def dispatch(self,message):
            # No existing lease may trigger publication, including retrying begin.
            if message.get('op')=='begin_episode' and not self.lease_path.exists():
                await asyncio.to_thread(sync_candidate,root)
            result=await super().dispatch(message)
            return result

        async def prepare_lease(self, health):
            if health.get('experiment_id') != read(self.root/'experiment.json')['experiment_id']:
                raise ValueError('Inference server belongs to another experiment')
            await asyncio.to_thread(subprocess.run,
                    ['rsync','-rt','--include=*/','--include=session.json','--exclude=*',
                     '-e',shlex.join(SSH),REMOTE+':'+str(root/'serve/sessions')+'/',
                     str(root/'learner/sessions')+'/'],check=True,timeout=30)
            session=read(self.root/'sessions'/health['session_id']/'session.json')
            if session != health and any(session.get(k)!=health.get(k) for k in ('experiment_id','policy_version','session_id','policy_weights_sha256')):
                raise ValueError('Serving session identity mismatch')
    c=SplitCoordinator(root/'learner',REPO,policy_url='http://192.168.0.119:18204',microbatch=8,candidate_batch=1)
    boundary=asyncio.Lock()
    from expo_ft.yam.online import handle_connection
    async def dispatch(message):
        async with boundary: return await c.dispatch(message)
    async def handle(ws):
        await handle_connection(ws, dispatch, token)
    (REPO/'logs').mkdir(exist_ok=True)
    async with serve(handle,'192.168.0.167',18208,max_size=CHUNK+4096,compression=None):
        print('Online learner ready: ws://192.168.0.167:18208; waiting for collector.',flush=True)
        await c.worker()


def inference(args):
    import numpy as np
    import torch
    import uvicorn
    from fastapi.responses import JSONResponse
    from expo_ft.yam.runtime import RoundPolicy
    from expo_ft.yam.rounds import lock
    sys.path.insert(0,str(REPO/'scripts/yam'))
    from serve_jax import build_app
    torch.set_num_threads(4)
    token=(args.root/'control.token').read_text().strip()
    os.environ['YAM_ONLINE_TRAINING_ROOT']=str(args.root/'learner')
    with lock(args.root/'serve'):
        policy=RoundPolicy(args.root/'serve',checkpoint=args.checkpoint,tokenizer=args.tokenizer)
        policy.metadata['run_name']=args.root.name
        actual=fingerprint(policy.base.policy,args.checkpoint,args.tokenizer)
        if actual!=policy.exp['lan_base_sha256']: raise ValueError('3060 base/tokenizer differs from 4090; refusing to serve')
        for name,expected in policy.exp.get('lan_normalization_files',{}).items():
            if sha(args.checkpoint/name)!=expected: raise ValueError('Normalization asset mismatch: '+name)
        frame=np.zeros((224,224,3),np.uint8)
        policy.predict({k:frame for k in ['top','left','right']},np.zeros(14,np.float32),policy.exp['prompt'])
        app=build_app(policy,args.checkpoint,dtype='bfloat16')
        @app.middleware('http')
        async def protect_reload(request,call_next):
            if request.url.path=='/online/reload' and not hmac.compare_digest(request.headers.get('Authorization',''),'Bearer '+token):
                return JSONResponse({'error':'Unauthorized'},status_code=401)
            return await call_next(request)
        print('3060 warmed; online policy ready on 192.168.0.119:18204',flush=True)
        uvicorn.run(app,host='192.168.0.119',port=18204)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=['4090','3060','fingerprint'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--prompt',default='fold the towel')
    a=p.parse_args(); a.root=a.root.resolve()
    if a.role=='fingerprint':
        from expo_ft.conversion.yam_loader import YamJaxPolicy
        policy=YamJaxPolicy(a.checkpoint,a.tokenizer,dtype='bfloat16')
        path=a.root/'learner/experiment.json'; exp=read(path)
        exp['lan_base_sha256']=fingerprint(policy,a.checkpoint,a.tokenizer)
        exp['lan_normalization_files']={p.name:sha(p) for p in sorted(a.checkpoint.glob('*processor*.safetensors'))}
        atomic_json(path,exp)
        return
    if a.role=='4090':
        a.root.mkdir(parents=True,exist_ok=True)
        with (a.root/'coordinator.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            bootstrap(a.root,a.checkpoint,a.tokenizer,a.prompt)
            try: asyncio.run(coordinator(a))
            except (KeyboardInterrupt,asyncio.CancelledError):
                print('Coordinator stopped; any interrupted job is marked failed.',flush=True)
    else: inference(a)


if __name__=='__main__': main()
