#!/usr/bin/env python3
"""Check a base-only HTTP deployment using recorded observations; no robot access."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

import json_numpy
import msgpack
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8204')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.server + '/healthz', timeout=10) as response:
        health = json.load(response)
    if health['checkpoint'] != str(Path(args.checkpoint).resolve()):
        raise ValueError('Unexpected deployed checkpoint')
    if health.get('expo_enabled') is not False:
        raise ValueError('Expected base-only deployment')
    samples = []
    for fixture in msgpack.unpackb(args.fixture.read_bytes(), raw=False):
        payload = fixture['observation_json']
        if isinstance(payload, str):
            payload = payload.encode()
        started = time.monotonic()
        request = urllib.request.Request(args.server + '/act', data=payload,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json_numpy.loads(response.read().decode())
        actions = np.asarray(result['actions'])
        if actions.shape != (30, 14) or not np.isfinite(actions).all():
            raise ValueError('Invalid action output')
        samples.append(dict(frame=fixture['frame'], shape=list(actions.shape),
                            roundtrip_ms=(time.monotonic()-started)*1000,
                            server_ms=result['dt_ms'], minimum=float(actions.min()),
                            maximum=float(actions.max())))
    report = dict(health=health, samples=samples, passed=True, hardware_accessed=False,
                  limitation='Protocol and finite-output check, not a joint-limit or robot-success evaluation')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
