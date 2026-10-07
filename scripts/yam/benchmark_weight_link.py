#!/usr/bin/env python3
"""Benchmark authenticated artifact bytes over a real LAN; never deploy weights."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from expo_ft.yam.lan_store import ArtifactServer,Client,Store
from expo_ft.yam.rounds import atomic_json,sha
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode',choices=['server','client'])
p.add_argument('--host',required=True)
p.add_argument('--port',type=int,default=18766)
p.add_argument('--token-file',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--bytes',type=int,default=151759800)
p.add_argument('--rate',type=float,default=20000000)
p.add_argument('--repeats',type=int,default=1)
a=p.parse_args();token=a.token_file.read_text().strip()
with tempfile.TemporaryDirectory(prefix='yam-weight-link-') as temp:
    root=Path(temp)
    if a.mode=='server':
        store=Store(root,run='benchmark',base_sha256='a'*64,preprocessing_sha256='b'*64)
        target=root/'bundles/1/weights.safetensors';target.parent.mkdir()
        with target.open('wb') as f:
            remaining=a.bytes
            while remaining:
                block=os.urandom(min(remaining,1024**2));f.write(block);remaining-=len(block)
            f.flush();os.fsync(f.fileno())
        meta=dict(bytes=a.bytes,sha256=sha(target),opaque_benchmark_bytes=True)
        atomic_json(root/'bundles/current.json',meta)
        server=ArtifactServer((a.host,a.port),store,token)
        def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        atomic_json(a.output,dict(ready=True,pid=os.getpid(),**meta))
        try:server.serve_forever()
        finally:server.server_close()
    else:
        client=Client(f'http://{a.host}:{a.port}',token,bytes_per_second=a.rate)
        meta=client.json('/current');times=[]
        for i in range(a.repeats):
            dest=root/f'weight-{i}'
            start=time.perf_counter()
            client.download('/bundles/1/weights.safetensors',dest,meta['sha256'],meta['bytes'])
            times.append(time.perf_counter()-start);dest.unlink()
        atomic_json(a.output,dict(bytes=meta['bytes'],seconds=times,rate_cap_bytes_s=a.rate,
                                  opaque_benchmark_bytes=True,published=False))
        print(json.dumps(times),flush=True)
