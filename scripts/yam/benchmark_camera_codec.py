#!/usr/bin/env python3
"""CPU-only three-camera JPEG cost from an RGB NPZ fixture; no camera/robot I/O."""
import argparse
import io
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from expo_ft.yam.lan_protocol import pack, unpack

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=100)
a=p.parse_args()
frames=dict(np.load(a.fixture,allow_pickle=False))
report=dict(host=platform.node(),robot_accessed=False,measurements={},shapes={k:list(v.shape) for k,v in frames.items()})
def measure(name,fn):
    times=[]
    for _ in range(a.repeats):
        start=time.perf_counter();value=fn();times.append((time.perf_counter()-start)*1000)
    report['measurements'][name]=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),max_ms=max(times),samples=len(times))
    return value
for quality in [85,95]:
    def encode():
        out={}
        for k,v in frames.items():
            buf=io.BytesIO();Image.fromarray(v).save(buf,format='JPEG',quality=quality);out[k]=buf.getvalue()
        return out
    encoded=measure(f'encode_q{quality}',encode)
    measure(f'decode_q{quality}',lambda:[np.asarray(Image.open(io.BytesIO(v)).convert('RGB')) for v in encoded.values()])
    report[f'jpeg_q{quality}_bytes']=sum(map(len,encoded.values()))
blob=measure('msgpack_encode',lambda:pack(dict(schema=1,images=encoded,state=np.zeros(14,np.float32))))
measure('msgpack_decode',lambda:unpack(blob))
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
