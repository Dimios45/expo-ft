#!/usr/bin/env python3
"""Inspect/requeue one verified prepared update, with both GPU services stopped.

No GPU allocation, model execution, robot commands, lease deletion or file removal.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from expo_ft.yam.rounds import read, sha, atomic_json, validate_version


def retry_prepared(root):
    learner=root/'learner'
    if (learner/'online/lease.json').exists():
        raise ValueError('An episode lease remains; establish robot inactivity and inspect it manually first')
    pending=read(learner/'pending.json');current=read(learner/'current.json')
    if pending.get('prepared') is not True or pending['parent']!=current['version']:
        raise ValueError('No prepared update on the current parent; inspect manually')
    if sha(Path(pending['dataset'])/'expo_session.json')!=pending['session_sha']:
        raise ValueError('Episode label changed')
    if current['checkpoint']:
        version=Path(current['checkpoint']); validate_version(version)
        if sha(version/'expo.msgpack')!=pending['parent_sha']: raise ValueError('Parent weights changed')
    jobs=[]
    for p in (learner/'online/queue').glob('*.json'):
        job=read(p)
        if job['dataset']==pending['dataset'] and job['state'] in ('failed','training'):
            jobs.append((p,job))
    if len(jobs)!=1: raise ValueError('Expected exactly one interrupted/failed job for prepared episode')
    path,job=jobs[0]
    history=job.get('recovery_history',[])+[{k:v for k,v in job.items() if k!='recovery_history'}]
    job.update(state='queued',prepared=True,requeued_at=time.time(),recovery_history=history)
    for key in ('error','exit_code','finished_at','worker_pid'): job.pop(key,None)
    atomic_json(path,job);atomic_json(learner/'online/learner.json',job)
    return job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--retry-prepared',action='store_true')
    args=p.parse_args();root=args.root.resolve()
    if not (root/'learner').is_dir(): p.error('Unknown run directory')
    with (root/'coordinator.lock').open('a') as c, (root/'learner/.round.lock').open('a') as l:
        fcntl.flock(c,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.retry_prepared:
            print(json.dumps(retry_prepared(root),indent=2))
        else:
            print(json.dumps({'current':read(root/'learner/current.json'),
                'status':read(root/'learner/online/learner.json'),
                'lease_present':(root/'learner/online/lease.json').exists(),
                'pending_present':(root/'learner/pending.json').exists()},indent=2))


if __name__=='__main__': main()
