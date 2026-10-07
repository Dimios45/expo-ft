#!/usr/bin/env python3
"""Sequential, isolated GPU sweep. Outputs measurements only; no deployment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New results directory')
    p.add_argument('--scratch', type=Path, default=Path('/tmp'))
    p.add_argument('--prepare-episode', action='store_true', help='Also time 1125 uncached RTC preparations')
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    common = ['--checkpoint', str(a.checkpoint.resolve()), '--tokenizer', str(a.tokenizer.resolve()),
              '--scratch', str(a.scratch.resolve())]
    cases = [(f'train-m{m}', 'train', ['--microbatch', str(m), '--repeats', '3']) for m in (1,2,4,8)]
    cases += [(f'serve-n{n}-c{c}', 'serve', ['--candidates', str(n), '--window', str(c),
              '--delay', str(d), '--repeats', str(repeats)])
              for n,c,d,repeats in [(2,8,5,30),(4,8,5,30),(8,8,5,100),(32,8,5,30),(8,12,8,30),(8,15,10,30)]]
    if a.prepare_episode:
        cases.append(('episode-preparation', 'train', ['--microbatch', '8', '--repeats', '3',
                                                       '--prepare-transitions', '1125']))
    results = []
    for name, mode, options in cases:
        command = [sys.executable, str(ROOT/'scripts/yam/measure_gpu.py'), '--output',
                   str(a.output/(name+'-gpu.json')), '--', sys.executable,
                   str(ROOT/'scripts/yam/benchmark_lan_gpu.py'), mode, *common,
                   '--output', str(a.output/(name+'.json')), *options]
        print('START '+name, flush=True)
        with (a.output/(name+'.log')).open('w') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        results.append(dict(name=name, exit_code=process.returncode))
        temp = a.output/'suite.json.tmp'
        with temp.open('w') as f:
            json.dump(results, f, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(temp, a.output/'suite.json')
        print('DONE '+json.dumps(results[-1]), flush=True)
    raise SystemExit(int(any(x['exit_code'] for x in results)))


if __name__ == '__main__':
    main()
