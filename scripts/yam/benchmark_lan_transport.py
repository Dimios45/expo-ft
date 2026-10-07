#!/usr/bin/env python3
"""Recorded-image codec and loopback transport costs; NOT a gigabit LAN test."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=100)
    a = p.parse_args()
    import av
    import numpy as np
    from PIL import Image
    from expo_ft.yam.lan_protocol import pack, unpack
    from expo_ft.yam.lan_store import ArtifactServer, Store, Client
    from expo_ft.yam.rounds import sha
    from websockets.sync.server import serve
    from websockets.sync.client import connect
    report = dict(scope='local CPU and loopback; not NUC or Ethernet measurements',
                  dataset=str(a.dataset), measurements={}, completed=False)
    def measure(label, fn, count=None):
        times=[]
        for _ in range(count or a.repeats):
            start=time.perf_counter(); value=fn(); times.append(time.perf_counter()-start)
        report['measurements'][label]=dict(median_ms=float(np.median(times)*1000),
            p95_ms=float(np.percentile(times,95)*1000), max_ms=max(times)*1000, samples=len(times))
        return value
    frames={}
    for role,key in [('top','top'),('left','left_wrist'),('right','right_wrist')]:
        video=sorted((a.dataset/'videos'/('observation.images.'+key)).rglob('*.mp4'))[0]
        with av.open(str(video)) as stream:
            frames[role]=next(stream.decode(video=0)).to_ndarray(format='rgb24')
    report['image_shapes']={k:list(v.shape) for k,v in frames.items()}
    report['raw_bytes_per_observation']=sum(v.nbytes for v in frames.values())
    encoded={}
    for quality in [85,95]:
        def encode():
            result={}
            for k,v in frames.items():
                buf=io.BytesIO();Image.fromarray(v).save(buf,format='JPEG',quality=quality)
                result[k]=buf.getvalue()
            return result
        encoded=measure('three_JPEG_encode_q'+str(quality),encode)
        measure('three_JPEG_decode_q'+str(quality),lambda:[np.asarray(Image.open(io.BytesIO(v)).convert('RGB')) for v in encoded.values()])
        report['jpeg_q'+str(quality)+'_bytes']=sum(map(len,encoded.values()))
    message=dict(schema=1,state=np.zeros(14,np.float32),images=encoded)
    blob=measure('msgpack_encode',lambda:pack(message))
    measure('msgpack_decode',lambda:unpack(blob))
    report['msgpack_bytes']=len(blob)
    def handler(ws):
        for raw in ws:
            unpack(raw)
            ws.send(pack(dict(schema=1,actions=np.zeros((8,14),np.float32))))
    with serve(handler,'127.0.0.1',0,compression=None,max_size=16*1024**2) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        with connect('ws://127.0.0.1:'+str(server.socket.getsockname()[1]),compression=None,max_size=16*1024**2) as ws:
            def roundtrip():
                ws.send(blob);return unpack(ws.recv(timeout=5))
            measure('websocket_roundtrip_JPEG_to_actions',roundtrip)
        server.shutdown();thread.join()
    with tempfile.TemporaryDirectory(prefix='yam-lan-benchmark-',dir='/tmp') as temp:
        root=Path(temp)
        store=Store(root/'store',run='benchmark',base_sha256='a'*64,preprocessing_sha256='b'*64)
        http=ArtifactServer(('127.0.0.1',0),store,'benchmark-only')
        thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        try:
            # Exact C8 FP32 inference tensor byte count, incompressible synthetic data.
            size=150454440
            source=store.root/'bundles/1/weights.safetensors';source.parent.mkdir()
            rng=np.random.default_rng(0)
            with source.open('wb') as f:
                remaining=size
                while remaining:
                    block=rng.bytes(min(remaining,1024**2));f.write(block);remaining-=len(block)
            checksum=measure('weight_SHA256',lambda:sha(source),3)
            for rate,label in [(1e12,'unlimited'),(20000000,'20MBps')]:
                client=Client('http://127.0.0.1:'+str(http.server_port),'benchmark-only',rate)
                def download():
                    dest=root/('download-'+label)
                    if dest.exists():dest.unlink()
                    return client.download('/bundles/1/weights.safetensors',dest,checksum,size)
                measure('HTTP_weight_download_'+label,download,1)
            client=Client('http://127.0.0.1:'+str(http.server_port),'benchmark-only',20000000)
            segment=root/'segment';segment.write_bytes(rng.bytes(8*1024**2))
            measure('resumable_HTTP_segment_upload_8MiB_20MBps',lambda:client.upload(segment),1)
            report['weight_bytes']=size
        finally:
            http.shutdown();http.server_close();thread.join()
    report['completed']=True
    report['bandwidth_model']=dict(fps=30,episode_seconds=300,window=8,
        replay_MB_per_second=report['jpeg_q95_bytes']*30/1e6,
        replay_MB_per_episode=report['jpeg_q95_bytes']*9000/1e6,
        two_phase_observations_MB_per_second=report['jpeg_q95_bytes']*(2*30/8)/1e6,
        note='Three representative recorded frames reused; excludes metadata and camera-scene variation')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
