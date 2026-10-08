#!/usr/bin/env bash
# Refuses the stock fold-70 export on a 12GB GPU; see docs/yam_pytorch.md.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${YAM_TORCH_PYTHON:?Set YAM_TORCH_PYTHON to the pinned LeRobot environment Python}"
RUN_ROOT="${YAM_TORCH_SERVE_ROOT:?Set a dedicated inference run directory}"
TOKEN="${YAM_CONTROL_TOKEN_FILE:?Copy this experiment control.token from the learner and set its path}"
mkdir -p "$RUN_ROOT/logs"
cd "$REPO"
exec "$PYTHON" -u scripts/yam/torch/online.py inference --root "$RUN_ROOT" --token-file "$TOKEN" \
  --checkpoint "${YAM_TORCH_CHECKPOINT:?Set the local fold-70 checkpoint directory}" \
  --host "${YAM_INFERENCE_IP:-192.168.0.119}" --store-url "${YAM_STORE_URL:-http://192.168.0.167:18309}" \
  > >(tee -a "$RUN_ROOT/logs/inference.log") 2>&1
