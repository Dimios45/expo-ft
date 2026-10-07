#!/usr/bin/env python3
"""Read-only five-minute monitoring of the active three-host YAM experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import urllib.request


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return {'read_error':str(exc)}


def snapshot(root, repo):
    learner=root/'learner'
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'learner_current':read(learner/'current.json'),
            'learner_status':read(learner/'online/learner.json'),
            'lease':read(learner/'online/lease.json') if (learner/'online/lease.json').exists() else None}
    jobs=[read(p) for p in sorted((learner/'online/queue').glob('*.json'))]
    result['coordinator_running']=False
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: argv=(p/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if b'4090' in argv and str(root).encode() in argv and any(a.endswith(b'/online_lan.py') or a==b'scripts/yam/online_lan.py' for a in argv):
            result['coordinator_running']=True
    result['jobs']=jobs
    result['job_counts']=dict(Counter(j.get('state','unknown') for j in jobs))
    try:
        with urllib.request.urlopen('http://192.168.0.119:18204/healthz',timeout=5) as response:
            result['inference']=json.load(response)
    except Exception as exc: result['inference']={'read_error':str(exc)}
    script="""import json; from pathlib import Path
root=Path('/home/yambox/yam-expo-data')/RUN
out={'run':json.loads((root/'run.json').read_text()),'episodes':[]}
for p in sorted(root.glob('episode-*')):
 if not p.is_dir(): continue
 e={'directory':p.name}
 for name,key in [('expo_session.json','session'),('meta/info.json','info')]:
  f=p/name
  if f.exists():
   try: e[key]=json.loads(f.read_text())
   except ValueError: e[key]={'read_error':'JSON not finalized'}
 out['episodes'].append(e)
if (root/'summary.json').exists(): out['summary']=json.loads((root/'summary.json').read_text())
print(json.dumps(out))
""".replace('RUN',repr(root.name))
    try:
        response=subprocess.run(['ssh','-i',str(Path.home()/'.ssh/yam_lan_bench'),'-o','BatchMode=yes',
            '-o','ConnectTimeout=5','yambox@192.168.0.121','python3 -'],input=script,text=True,
            capture_output=True,timeout=20,check=True)
        result['nuc']=json.loads(response.stdout)
    except Exception as exc: result['nuc']={'read_error':str(exc)}
    status=result['learner_status']; dataset=status.get('dataset')
    if dataset:
        key=Path(dataset).name
        result['training_log_tails']={}
        for phase in ['prepare','train']:
            path=repo/'logs'/f'online-{key}-{phase}.log'
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-8192))
                    result['training_log_tails'][phase]=stream.read().decode(errors='replace').splitlines()[-8:]
    try:
        result['gpu_4090']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu',
            '--format=csv,noheader'],text=True,timeout=5).strip()
    except Exception as exc: result['gpu_4090']=str(exc)
    return result


def collect_logs(root):
    key=str(Path.home()/'.ssh/yam_lan_bench')
    destinations=[('inference','sra@192.168.0.119',str(root/'logs')+'/'),
                  ('nuc','yambox@192.168.0.121','/home/yambox/yam-lan/runs/'+root.name+'/logs/')]
    errors={}
    local=root/'monitoring/learner-logs';local.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[2]
    for job in (root/'learner/online/queue').glob('*.json'):
        for source in (repo/'logs').glob('online-'+job.stem+'-*'):
            if source.is_file():
                try: shutil.copy2(source,local/source.name)
                except OSError as exc: errors[source.name]=str(exc)
    for name,host,source in destinations:
        dest=root/'monitoring/remote-logs'/name;dest.mkdir(parents=True,exist_ok=True)
        try:
            subprocess.run(['rsync','-rt','--timeout=10','--bwlimit=2000','-e',
                'ssh -i '+key+' -o BatchMode=yes -o ConnectTimeout=5',
                host+':'+source,str(dest)+'/'],capture_output=True,check=True,timeout=30)
        except Exception as exc: errors[name]=str(exc)
    return errors


def summarize(s):
    nuc=s.get('nuc',{}); run=nuc.get('run',{}); episodes=nuc.get('episodes',[])
    finalized=[e for e in episodes if 'reward' in e.get('session',{})]
    outcomes=Counter(e['session'].get('terminal','unknown') for e in finalized)
    lines=[f"[{s['timestamp_utc']}]",
           f"NUC: {len(run.get('episodes',{}))} episode slots started; {len(finalized)} finalized; outcomes={dict(outcomes)}",
           f"Serving version: {s.get('inference',{}).get('policy_version','unavailable')}; learner version: {s['learner_current'].get('version','unavailable')}; jobs: {s['job_counts']}"]
    for e in finalized:
        session=e['session']
        lines.append(f"  {e['directory']}: policy={session.get('policy_version')} reward={session['reward']} terminal={session.get('terminal')} frames={e.get('info',{}).get('total_frames')}")
    for name in ['nuc','inference','learner_current','learner_status']:
        if 'read_error' in s[name]: lines.append(f"READ ERROR {name}: {s[name]['read_error']}")
    if not s.get('coordinator_running',False):
        lines.append('COORDINATOR OFFLINE: job states above are persisted history, not proof of active training.')
    for job in s['jobs']:
        if job.get('state')=='failed': lines.append('TRAINING FAILURE: '+json.dumps(job))
    lines.append('4090 GPU (MiB, utilization, temperature): '+s['gpu_4090'])
    if 'summary' in nuc: lines.append('RUN COMPLETE: '+json.dumps(nuc['summary']))
    return '\n'.join(lines)+'\n\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--interval',type=int,default=300)
    p.add_argument('--once',action='store_true')
    p.add_argument('--collect-logs',action='store_true')
    args=p.parse_args()
    if args.interval<1: p.error('interval must be positive')
    folder=args.root/'monitoring';folder.mkdir(exist_ok=True)
    with (folder/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (folder/'monitor.pid').write_text(str(os.getpid())+'\n')
        repo=Path(__file__).resolve().parents[2]
        while True:
            started=time.monotonic(); s=snapshot(args.root,repo)
            if args.collect_logs: s['log_sync_errors']=collect_logs(args.root)
            with (folder/'progress.jsonl').open('a') as f:
                f.write(json.dumps(s)+'\n');f.flush();os.fsync(f.fileno())
            summary=summarize(s)
            with (folder/'progress.log').open('a') as f: f.write(summary);f.flush()
            print(summary,flush=True)
            if args.once or 'summary' in s.get('nuc',{}): break
            time.sleep(max(0,args.interval-(time.monotonic()-started)))


if __name__=='__main__': main()
