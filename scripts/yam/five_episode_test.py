#!/usr/bin/env python3
"""Persistent terminal launcher for a fresh five-by-five-minute EXPO experiment."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

DEFAULT_RUN='lan-five-20261007-001'
HOSTS={'4090':('192.168.0.167',18208),'3060':('192.168.0.119',18204)}


def run_root(role, run):
    return Path('/home/yambox/yam-lan/runs' if role=='nuc' else '/home/sra/yam-online-lan')/run


def save_new(path, data):
    if path.exists():
        if json.loads(path.read_text())!=data: raise ValueError('Existing test configuration differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())


def distribute(config):
    key=str(Path.home()/'.ssh/yam_lan_bench')
    for role,host in [('3060','sra@192.168.0.119'),('nuc','yambox@192.168.0.121')]:
        path=run_root(role,config['run'])/'test.json'
        code="""import json,sys,os; from pathlib import Path
v=json.load(sys.stdin); p=Path(v['path']); p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():
 if json.loads(p.read_text())!=v['config']: raise ValueError('Existing test configuration differs')
else:
 with p.open('x') as f: json.dump(v['config'],f,indent=2); f.flush(); os.fsync(f.fileno())
"""
        subprocess.run(['ssh','-i',key,'-o','BatchMode=yes','-o','ConnectTimeout=5',host,
            'python3 -c '+shlex.quote(code)],input=json.dumps({'path':str(path),'config':config}),text=True,check=True,timeout=20)


def validate(config):
    if config.get('episodes')!=5 or config.get('seconds')!=300 or config.get('initial_policy')!='base-only':
        raise ValueError('Expected fresh-base five-episode / 300-second test configuration')
    if not isinstance(config.get('prompt'),str) or not config['prompt'].strip():
        raise ValueError('Missing task string')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=['4090','3060','nuc'])
    p.add_argument('--run',default=DEFAULT_RUN)
    p.add_argument('--task',help='Exact task; otherwise requested once on the 4090')
    p.add_argument('--restart',action='store_true',help='Restart only an exited tmux pane; refuses a live process')
    p.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',args.run): p.error('Invalid run name')
    if not shutil.which('tmux'): p.error('tmux is required')
    root=run_root(args.role,args.run);configpath=root/'test.json'
    if args.role=='4090' and not configpath.exists():
        if (root/'learner').exists(): raise SystemExit('Refusing to attach a fresh test to an existing learner')
        task=(args.task if args.task is not None else input('Exact NEW task string for all five episodes: ')).strip()
        if not task: raise SystemExit('Task cannot be empty')
        config={'run':args.run,'prompt':task,'episodes':5,'seconds':300,'initial_policy':'base-only',
                'created_utc':datetime.now(timezone.utc).isoformat()}
        save_new(configpath,config)
    if not configpath.exists(): raise SystemExit('Start the 4090 launcher first to distribute this run configuration.')
    config=json.loads(configpath.read_text());validate(config)
    if config['run']!=args.run: raise SystemExit('Run identity mismatch')
    if args.task is not None and args.task!=config['prompt']: raise SystemExit('Task is pinned; choose a new --run to change it')
    if args.role=='4090' and not args.inside: distribute(config)
    name='yam-'+args.run+'-'+args.role
    if not args.inside:
        exists=subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if exists and args.restart:
            states=subprocess.check_output(['tmux','list-panes','-s','-t',name,'-F','#{pane_dead}'],text=True).splitlines()
            if not states or any(x!='1' for x in states): raise SystemExit('Session still has a live process; refusing restart. Attach normally.')
            subprocess.run(['tmux','kill-session','-t',name],check=True)
            exists=False
        if not exists:
            logs=root/'logs';logs.mkdir(exist_ok=True)
            stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
            log=logs/(args.role+'-'+stamp+'.log')
            subprocess.run(['tmux','new-session','-d','-s',name,'sleep 86400'],check=True)
            target=name+':0.0'
            subprocess.run(['tmux','set-option','-t',name,'remain-on-exit','on'],check=True)
            subprocess.run(['tmux','pipe-pane','-o','-t',target,'cat >> '+shlex.quote(str(log))],check=True)
            command=shlex.join([sys.executable,str(Path(__file__).resolve()),args.role,'--run',args.run,'--inside'])
            subprocess.run(['tmux','respawn-pane','-k','-t',target,command],check=True)
        # A dead pane is shown, never automatically restarted after an error.
        attach='switch-client' if os.environ.get('TMUX') else 'attach-session'
        os.execvp('tmux',['tmux',attach,'-t',name])
    print(f"Run {args.run}: 5 x 300 seconds; task {config['prompt']!r}",flush=True)
    print('Terminal closure detaches; it does NOT stop an active robot episode. Ctrl+C requests normal cleanup.',flush=True)
    print('Logs:',root/'logs',flush=True)
    if args.role=='3060':
        deadline=time.monotonic()+300
        while True:
            try:
                token=(root/'control.token').read_text().strip()
                exp=json.loads((root/'serve/experiment.json').read_text())
                json.loads((root/'serve/current.json').read_text())
                if token and exp.get('lan_base_sha256') and exp.get('prompt')==config['prompt']: break
            except (OSError,ValueError): pass
            if time.monotonic()>=deadline:
                raise SystemExit('4090 setup is not ready after 5 minutes. Check its logs; wait for Online learner ready, then rerun with --restart.')
            print('Waiting for the 4090 to finish fingerprinting and copy run credentials...',flush=True)
            time.sleep(5)
    if args.role in HOSTS:
        sock=socket.socket();sock.settimeout(2)
        try: occupied=sock.connect_ex(HOSTS[args.role])==0
        finally:sock.close()
        if occupied: raise SystemExit('Policy/coordinator port is occupied. Stop the old run normally first; nothing was killed.')
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
        if busy: raise SystemExit('GPU already has a compute process. Finish the other job first; nothing was killed.')
    else:
        # Refuse a competing robot controller, including another terminal's rollout.
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit(): continue
            try: argv=(entry/'cmdline').read_bytes().split(b'\0')
            except OSError: continue
            if any(a.endswith(b'/karma') for a in argv) and any(a in argv for a in [b'rollout',b'inference',b'hitl',b'teleop']):
                raise SystemExit('Another KARMA controller is running. Finish it before starting this test.')
    env=dict(os.environ,YAM_RUN=args.run,YAM_PROMPT=config['prompt'],YAM_EPISODES='5',YAM_SECONDS='300',
             YAM_EXPECTED_RUN=args.run,PYTHONUNBUFFERED='1')
    if args.role=='4090':
        repo=Path('/home/sra/tirth/expo-ft')
        env.update(YAM_REPO=str(repo),YAM_PYTHON=str(repo/'.venv-convert/bin/python'),
                   YAM_CHECKPOINT='/usr/local/models/sra-expo-ft/yam_pi05_jax',
                   YAM_TOKENIZER='/home/sra/molmoact2/outputs/models/paligemma-tokenizer')
        # Dedicated read-only monitor; survives this terminal and exits on summary.
        monitoring=root/'monitoring';monitoring.mkdir(exist_ok=True)
        with (monitoring/'monitor-service.log').open('ab') as log:
            subprocess.Popen(['python3',str(repo/'scripts/yam/monitor_online_run.py'),'--root',str(root),'--collect-logs'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        script=repo/'scripts/yam/start_4090.sh'
    elif args.role=='3060':
        env.update(YAM_REPO='/home/sra/yam-lan/runtime',YAM_PYTHON='/home/sra/expo-ft/.venv-yam/bin/python',
                   YAM_CHECKPOINT='/home/sra/expo-ft/yam-pi05',YAM_TOKENIZER='/home/sra/expo-ft/paligemma-tokenizer')
        script=Path('/home/sra/yam-lan/start_3060.sh')
    else:
        env['YAM_TOKEN_FILE']=str(root/'control.token')
        script=Path('/home/yambox/yam-lan/start_yam_nuc.sh')
    os.execvpe('bash',['bash',str(script)],env)


if __name__=='__main__': main()
