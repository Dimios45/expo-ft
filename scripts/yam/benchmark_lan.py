#!/usr/bin/env python3
"""Read-only host inventory and opt-in pairwise network probes. No robot access."""
import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def command(args, timeout=40):
    if not shutil.which(args[0]):
        return {'available': False, 'command': args[0]}
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {'available': True, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except subprocess.TimeoutExpired:
        return {'available': True, 'error': 'timeout'}


def inventory():
    links = []
    for nic in sorted(Path('/sys/class/net').iterdir()):
        def read(name):
            try:
                return (nic / name).read_text().strip()
            except OSError:
                return None
        links.append(dict(interface=nic.name, wireless=(nic / 'wireless').exists(),
                          state=read('operstate'), speed_mbps=read('speed'), duplex=read('duplex'),
                          ethtool=command(['ethtool', nic.name])))
    return dict(schema=1, collected_at=time.time(), hostname=platform.node(),
                robot_accessed=False, cpu=command(['lscpu', '-J']),
                memory=Path('/proc/meminfo').read_text(), os=Path('/etc/os-release').read_text(),
                disk={str(p): dict(zip(('total', 'used', 'free'), shutil.disk_usage(p)))
                      for p in (Path.cwd(), Path('/tmp'))},
                gpu=command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv']),
                gpu_processes=command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv']),
                addresses=command(['ip', '-j', 'addr']), routes=command(['ip', '-j', 'route']), links=links)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--peer', help='Explicit wired IPv4 address; enables network probes')
    p.add_argument('--source', help='Local wired IPv4 address')
    p.add_argument('--interface')
    args = p.parse_args()
    result = inventory()
    if args.peer:
        import ipaddress
        if not args.source or not args.interface:
            p.error('--peer requires --source and --interface')
        for address in (args.peer, args.source):
            ipaddress.IPv4Address(address)
        if not any(x['interface'] == args.interface and not x['wireless'] for x in result['links']):
            p.error('source interface is missing or wireless')
        route = command(['ip', '-j', 'route', 'get', args.peer, 'from', args.source])
        result['peer_route'] = route
        entries = json.loads(route.get('stdout') or '[]')
        if not entries or entries[0].get('dev') != args.interface:
            p.error('peer route does not use the requested wired interface')
        result['network'] = {}
        base = ['iperf3', '-c', args.peer, '-B', args.source, '-t', '30', '-J']
        ping = ['ping', '-I', args.interface, '-i', '0.02', '-c', '1500', args.peer]
        result['network']['idle_ping'] = command(ping)
        for label, extra in [('forward', []), ('reverse', ['-R']), ('bidirectional', ['--bidir'])]:
            # The loaded ping must overlap the transfer to reveal queueing jitter.
            with ThreadPoolExecutor(max_workers=2) as pool:
                throughput = pool.submit(command, base + extra)
                jitter = pool.submit(command, ping)
                result['network'][label] = dict(iperf3=throughput.result(), loaded_ping=jitter.result())
    result['acceptance'] = dict(status='not_certified',
        required=['all three host inventories', 'three pairwise loaded network tests',
                  '3060 memory and RTC deadlines', '4090 cycle benchmark on idle GPU',
                  'frozen-base prefix behavior', 'autonomous reward/reset feasibility'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {args.output}; no hardware/network settings changed')


if __name__ == '__main__':
    main()
