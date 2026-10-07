#!/usr/bin/env bash
set -euo pipefail
export YAM_RUN=${YAM_RUN:-lan-run-001}
[[ "$YAM_RUN" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid YAM_RUN'; exit 2; }
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export YAM_CANDIDATE_BATCH=1
cd "${YAM_REPO:-/home/sra/tirth/expo-ft}"
python3 - <<'CHECK'
import json, subprocess
from pathlib import Path
for target in ['192.168.0.121','192.168.0.119']:
 r=json.loads(subprocess.check_output(['ip','-j','route','get',target]))[0]
 if r.get('dev')!='enp3s0' or r.get('prefsrc')!='192.168.0.167': raise SystemExit('Expected wired enp3s0 / 192.168.0.167')
if Path('/sys/class/net/enp3s0/speed').read_text().strip()!='1000': raise SystemExit('Expected gigabit Ethernet')
CHECK
exec "${YAM_PYTHON:-.venv-convert/bin/python}" scripts/yam/online_lan.py 4090 \
  --root "/home/sra/yam-online-lan/$YAM_RUN" \
  --checkpoint "${YAM_CHECKPOINT:-/usr/local/models/sra-expo-ft/yam_pi05_jax}" \
  --tokenizer "${YAM_TOKENIZER:-/home/sra/molmoact2/outputs/models/paligemma-tokenizer}" \
  --prompt "${YAM_PROMPT:-fold the towel}"
