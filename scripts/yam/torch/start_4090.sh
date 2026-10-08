#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${YAM_TORCH_PYTHON:-/home/sra/ksagar/lerobot/.venv/bin/python}"
RUN_ROOT="${YAM_TORCH_ROOT:?Set YAM_TORCH_ROOT to a new experiment directory}"
CHECKPOINT="${YAM_TORCH_CHECKPOINT:-/home/sra/ksagar/lerobot/yam_fold_70}"
mkdir -p "$(dirname -- "$RUN_ROOT")"
if [[ ! -e "$RUN_ROOT/experiment.json" ]]; then
  "$PYTHON" "$REPO/scripts/yam/torch/train.py" init --root "$RUN_ROOT" --checkpoint "$CHECKPOINT" --prompt "${YAM_TASK:-fold the towel}"
fi
if [[ ! -e "$RUN_ROOT/control.token" ]]; then
  (umask 077; "$PYTHON" -c 'import secrets; print(secrets.token_hex(32))' > "$RUN_ROOT/control.token")
fi
mkdir -p "$RUN_ROOT/logs"
cd "$REPO"
exec "$PYTHON" -u scripts/yam/torch/online.py learner --root "$RUN_ROOT" --token-file "$RUN_ROOT/control.token" \
  --host "${YAM_LEARNER_IP:-192.168.0.167}" --policy-url "${YAM_POLICY_URL:-http://192.168.0.119:18304}" \
  --microbatch "${YAM_MICROBATCH:-1}" > >(tee -a "$RUN_ROOT/logs/coordinator.log") 2>&1
