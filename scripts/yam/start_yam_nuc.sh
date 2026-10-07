#!/usr/bin/env bash
# Starts supervised robot collection ONLY when the operator runs this script.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export YAM_RUN=${YAM_RUN:-lan-run-001}
run=$YAM_RUN
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid YAM_RUN'; exit 2; }
cd /home/yambox/karma
export TOP_SERIAL=${TOP_SERIAL:-}
export LEFT_WRIST_SERIAL=${LEFT_WRIST_SERIAL:-}
export RIGHT_WRIST_SERIAL=${RIGHT_WRIST_SERIAL:-}
.venv/bin/python - <<'CHECK'
import json, os, subprocess, urllib.request
from pathlib import Path
from openpi_control.cameras import sdk_present_asic_serials
for target in ['192.168.0.119','192.168.0.167']:
 r=json.loads(subprocess.check_output(['ip','-j','route','get',target]))[0]
 if r.get('dev')!='eno1' or r.get('prefsrc')!='192.168.0.121': raise SystemExit('Expected wired eno1 / .121')
if Path('/sys/class/net/eno1/speed').read_text().strip()!='1000': raise SystemExit('Expected gigabit Ethernet')
with urllib.request.urlopen('http://192.168.0.119:18204/healthz',timeout=5) as r: h=json.load(r)
if not h.get('experiment_id') or h.get('dtype')!='bfloat16': raise SystemExit('Wrong online policy endpoint')
if os.environ.get('YAM_EXPECTED_RUN') and h.get('run_name')!=os.environ['YAM_EXPECTED_RUN']: raise SystemExit('Inference belongs to another run')
asic_to_sdk=sdk_present_asic_serials()
available=set(asic_to_sdk)
def normalize(serial):
 matches={asic for asic,sdk in asic_to_sdk.items() if serial in (asic,sdk)}
 if len(matches)>1: raise SystemExit('Ambiguous camera identity: '+serial)
 return next(iter(matches)) if matches else serial
keys=['TOP_SERIAL','LEFT_WRIST_SERIAL','RIGHT_WRIST_SERIAL']
config=Path('/home/yambox/yam-lan/selected-cameras.json')
saved=json.loads(config.read_text()) if config.exists() else {}
defaults=dict(zip(keys,['348523020354','254623070863','254623070417']))
wanted=[normalize(os.environ[k] or saved.get(k,defaults[k])) for k in keys]
if len(set(wanted))!=3 or not set(wanted)<=available:
 print('Attached cameras (ASIC serial -> SDK device serial):',asic_to_sdk)
 print('Enter the PHYSICAL camera mapping; no robot control has started.')
 import sys
 try: sys.stdin=open('/dev/tty')
 except OSError: raise SystemExit('No terminal: set TOP_SERIAL, LEFT_WRIST_SERIAL, RIGHT_WRIST_SERIAL explicitly.')
 wanted=[normalize(input(role+' ASIC or SDK serial: ').strip()) for role in ['Overhead','Left wrist','Right wrist']]
 if len(set(wanted))!=3 or not set(wanted)<=available: raise SystemExit('Invalid or duplicate camera serial.')
config.write_text(json.dumps(dict(zip(keys,wanted))))
print('Camera mapping:',dict(zip(keys,wanted)))
print('Online policy version:',h['policy_version'],'Task:',h['prompt'])
CHECK
mapfile -t serials < <(.venv/bin/python -c 'import json; d=json.load(open("/home/yambox/yam-lan/selected-cameras.json")); print(d["TOP_SERIAL"]); print(d["LEFT_WRIST_SERIAL"]); print(d["RIGHT_WRIST_SERIAL"])')
export TOP_SERIAL="${serials[0]}" LEFT_WRIST_SERIAL="${serials[1]}" RIGHT_WRIST_SERIAL="${serials[2]}"
export YAM_CONTROL_TOKEN="$(cat "${YAM_TOKEN_FILE:-/home/yambox/yam-lan/control.token}")"
python=/home/yambox/.venvs/yam-rtc-py310/bin/python
"$python" - <<'CHECK'
import os,sys
sys.path.insert(0,'/home/yambox/yam-lan/online')
from collect_online import RPC
r=RPC('ws://192.168.0.167:18208')
try:
 status=r.call(op='status')
 if any(j['state']=='failed' for j in status['jobs']): raise SystemExit('Learner has a failed job; inspect 4090 before collection')
 print('4090 coordinator connected.')
finally: r.close()
CHECK
if [[ "${1:-}" == --check ]]; then exit 0; fi
if [[ $# != 0 ]]; then echo 'Usage: bash start_yam_nuc.sh [--check]' >&2; exit 2; fi
exec "$python" /home/yambox/yam-lan/online/collect_online.py \
  --recorder /home/yambox/yam-lan/online/record_round.py --fixed-prompt --strict-confirmation --archive-discarded \
  --control ws://192.168.0.167:18208 --server http://192.168.0.119:18204 \
  --root "/home/yambox/yam-expo-data/$run" \
  --episodes "${YAM_EPISODES:-10}" --seconds "${YAM_SECONDS:-180}" \
  -- --interface left=can_left --interface right=can_right \
  --camera-serial "top=$TOP_SERIAL" \
  --camera-serial "left_wrist=$LEFT_WRIST_SERIAL" \
  --camera-serial "right_wrist=$RIGHT_WRIST_SERIAL"
