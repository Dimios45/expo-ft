#!/usr/bin/env python3
"""NUC: supervised multi-episode recording with background WebSocket uploads.

This script invokes KARMA and moves robots. Keep the normal stop control available.
Requires websockets/msgpack, uv/KARMA, and record_round-a100.py beside this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tarfile
import threading
import time
import uuid

import msgpack
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(temporary,path)


def wait_for_recorder(child):
    """Foreground SIGINT also reaches KARMA; parents must await its cleanup."""
    interrupted = False
    while True:
        try:
            return child.wait(), interrupted
        except KeyboardInterrupt:
            interrupted = True
            print('\nStop requested. Waiting for KARMA cleanup and outcome prompts; no next episode will start.', flush=True)


def archive_discarded(folder):
    """Preserve a zero-episode dataset; never move finalized/saved recordings."""
    info_path = folder/'meta/info.json'
    if (folder/'expo_session.json').exists() or not info_path.exists():
        raise RuntimeError('Cannot establish that this recording was discarded: '+str(folder))
    info = json.loads(info_path.read_text())
    manifest_path = folder/'openpi_control_rollouts.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if info.get('total_episodes') != 0 or any(e.get('saved') for e in manifest.get('episodes', [])):
        raise RuntimeError('Recording contains saved episodes; inspect manually: '+str(folder))
    if folder.with_suffix('.tar').exists():
        raise RuntimeError('An upload archive exists; inspect before retrying')
    destination = folder.parent/'discarded-attempts'/f'{folder.name}-{uuid.uuid4().hex[:8]}'
    destination.parent.mkdir(exist_ok=True)
    folder.rename(destination)
    print('Preserved discarded attempt at',destination,flush=True)


class RPC:
    def __init__(self,url):
        self.url,self.socket=url,None

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket=None

    def call(self,**message):
        for attempt in range(3):
            try:
                if self.socket is None:
                    self.socket=connect(self.url,compression=None,open_timeout=15,ping_timeout=180,max_size=8*1024*1024)
                self.socket.send(msgpack.packb(message,use_bin_type=True))
                response=msgpack.unpackb(self.socket.recv(timeout=240),raw=False)
            except (OSError,TimeoutError,ConnectionClosed) as exc:
                self.close()
                if attempt==2: raise
                time.sleep(2)
                continue
            if not response.get('ok'):
                raise RuntimeError(response.get('error','Coordinator error'))
            return response['result']
        raise RuntimeError('No coordinator response')


def upload(rpc, folder):
    archive=folder.with_suffix('.tar')
    if not archive.exists():
        temp=archive.with_suffix('.partial')
        with tarfile.open(temp,'w:') as tar:
            for path in sorted(folder.rglob('*')):
                if path.is_symlink(): raise ValueError('Dataset contains a symlink')
                if path.is_file(): tar.add(path,arcname=str(path.relative_to(folder)),recursive=False)
        os.replace(temp,archive)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    digest=h.hexdigest()
    state=rpc.call(op='upload_begin',sha256=digest,size=archive.stat().st_size)
    if not state['complete']:
        with archive.open('rb') as f:
            offset=state['offset']; f.seek(offset)
            while data:=f.read(256*1024):
                reply=rpc.call(op='upload_chunk',sha256=digest,offset=offset,data=data)
                offset=reply['offset']
                time.sleep(.05)
    return rpc.call(op='upload_finish',sha256=digest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',default='ws://127.0.0.1:8208')
    p.add_argument('--server',default='http://127.0.0.1:8207')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--wait-for-learner',action='store_true',help='Finish queued uploads/training before each new episode')
    p.add_argument('--archive-discarded',action='store_true',help='Preserve and retry only verified zero-episode attempts')
    p.add_argument('--recorder',type=Path,default=Path(__file__).with_name('record_round-a100.py'))
    p.add_argument('karma_args',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.episodes < 1 or args.seconds <= 0: p.error('Positive episode count and duration required')
    args.root=args.root.expanduser().resolve();args.root.mkdir(parents=True,exist_ok=True)
    statefile=args.root/'run.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else dict(run_id=uuid.uuid4().hex,episodes={})
    save(statefile,state)
    extra=args.karma_args[1:] if args.karma_args[:1]==['--'] else args.karma_args
    rpc=RPC(args.control)
    work=queue.Queue()
    errors=[]
    uploaded={}
    def worker():
        channel=RPC(args.control)
        while True:
            item=work.get()
            try:
                if item is None:return
                i,folder=item
                receipt_path=args.root/f'upload-{i:03d}.json'
                if receipt_path.exists():
                    receipt=json.loads(receipt_path.read_text())
                else:
                    for retry in range(5):
                        try:
                            receipt=upload(channel,folder)
                            break
                        except Exception:
                            channel.close()
                            if retry==4:raise
                            time.sleep(3)
                    save(receipt_path,receipt)
                uploaded[i]=receipt
                print(f'\nEpisode {i+1}: uploaded and queued for training.',flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(f'\nUpload stopped: {exc}. Local recording retained.',flush=True)
            finally:work.task_done()
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    try:
        for i in range(args.episodes):
            if errors:raise RuntimeError(errors[0])
            folder=args.root/f'episode-{i:03d}'
            episode=f"{state['run_id']}-{i:03d}"
            if (folder/'expo_session.json').exists():
                lease=rpc.call(op='status')['lease']
                if lease and lease['episode']==episode:rpc.call(op='end_episode',episode=episode)
                work.put((i,folder));continue
            if folder.exists():
                if args.archive_discarded:
                    archive_discarded(folder)
                else:
                    raise RuntimeError(f'Incomplete episode at {folder}; inspect it before resuming')
            if args.wait_for_learner:
                work.join()
                if errors:raise RuntimeError(errors[0])
                while True:
                    jobs=rpc.call(op='status')['jobs']
                    if any(j['state']=='failed' for j in jobs):raise RuntimeError('Pod learner failed; inspect logs')
                    if all(j['state']=='ready' for j in jobs):break
                    print('Waiting for pending training before starting hardware...',flush=True)
                    time.sleep(5)
            input(f'\nEpisode {i+1}/{args.episodes}: reset scene, keep stop control ready, press Enter to start: ')
            lease=rpc.call(op='begin_episode',episode=episode)
            print('Recording policy version',lease['health']['policy_version'],flush=True)
            state['episodes'][str(i)]=dict(episode=episode,policy_version=lease['health']['policy_version'],started_at=time.time())
            save(statefile,state)
            command=[sys.executable,str(args.recorder),'--server',args.server,'--out',str(folder),
                     '--seconds',str(args.seconds),'--auto-upload','--',*extra]
            child=subprocess.Popen(command)
            try:
                code, interrupted=wait_for_recorder(child)
            finally:
                # wait() above guarantees no active rollout before release.
                if child.poll() is not None:rpc.call(op='end_episode',episode=episode)
            if code or not (folder/'expo_session.json').exists():
                raise RuntimeError(f'Episode {i+1} not finalized; collection stopped')
            state['episodes'][str(i)]['finished_at']=time.time();save(statefile,state)
            work.put((i,folder))
            if interrupted:
                print('Collection stopped after the interrupted episode. Finishing queued uploads; no next episode.',flush=True)
                work.join()
                if errors:raise RuntimeError(errors[0])
                return
        print('Collection finished. Waiting for remaining uploads and training...',flush=True)
        work.join()
        if errors:raise RuntimeError(errors[0])
        digests={r['sha256'] for r in uploaded.values()}
        while True:
            status=rpc.call(op='status')
            jobs=status['jobs']
            if any(j['state']=='failed' for j in jobs):raise RuntimeError('Pod learner failed; recordings retained. Inspect pod logs.')
            matching=[j for j in jobs if Path(j['dataset']).name in digests]
            if len(matching)==len(digests) and all(j['state']=='ready' for j in matching):break
            print('Learner:',status['learner'].get('state','waiting'),flush=True)
            time.sleep(15)
        # Publish the final candidate at an idle boundary; no robot is started.
        final_id=state['run_id']+'-final'
        health=rpc.call(op='begin_episode',episode=final_id)['health']
        rpc.call(op='end_episode',episode=final_id)
        outcomes=[json.loads((args.root/f'episode-{i:03d}'/'expo_session.json').read_text()) for i in range(args.episodes)]
        report=dict(episodes=len(outcomes),successes=sum(x['reward']==1 for x in outcomes),
                    truncated=sum(x['terminal']=='truncated' for x in outcomes),
                    final_policy_version=health['policy_version'],
                    behavior_versions=[x['policy_version'] for x in outcomes])
        save(args.root/'summary.json',report)
        print(json.dumps(report,indent=2),flush=True)
    finally:
        rpc.close()
        work.put(None)


if __name__=='__main__':
    main()
