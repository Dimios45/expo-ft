"""Pod-only integration check: upload an existing episode, train, reload, infer.

Uses recorded inputs and a diagnostic episode lease; never contacts robot hardware.
"""
import json
from pathlib import Path
import sys
import time
import urllib.request

import msgpack
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collect_online import RPC, upload

rpc=RPC('ws://127.0.0.1:8208')
receipt=upload(rpc,Path('artifacts/incoming/a100-round-0003'))
print('Upload receipt',receipt,flush=True)
fixture=msgpack.unpackb(Path('artifacts/yam-rtc/round3-probe.msgpack').read_bytes(),raw=False)[0]
samples=[]
start=time.time()
while time.time()-start < 1800:
    status=rpc.call(op='status')
    if any(j['state']=='failed' for j in status['jobs']):raise RuntimeError(str(status))
    if status['jobs'] and all(j['state']=='ready' for j in status['jobs']):break
    t=time.monotonic()
    req=urllib.request.Request('http://127.0.0.1:8204/act',data=fixture['observation_json'],headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:
        body=r.read()
    samples.append(time.monotonic()-t)
    print('Concurrent inference seconds',samples[-1],flush=True)
    time.sleep(5)
else:raise TimeoutError('Learner did not finish within diagnostic budget')
lease=rpc.call(op='begin_episode',episode='diagnostic-promotion-check')
print('Promoted',lease['health']['policy_version'],flush=True)
rpc.call(op='end_episode',episode='diagnostic-promotion-check')
Path('artifacts/yam-overlap-test/online-integration-report.json').write_text(json.dumps(dict(receipt=receipt,status=status,health=lease['health'],concurrent_inference_seconds=samples,hardware_accessed=False),indent=2)+'\n')
