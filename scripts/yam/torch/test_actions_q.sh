#!/usr/bin/env bash
# Offline only: no robot control, services, or checkpoint deployment.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${YAM_TORCH_PYTHON:-/home/sra/ksagar/lerobot/.venv/bin/python}"
CHECKPOINT="${YAM_TORCH_CHECKPOINT:-/home/sra/ksagar/lerobot/yam_fold_70}"
DATASET="${YAM_TEST_DATASET:-/home/sra/yam-online-lan/lan-five-20261007-001/learner/online/episodes/030438602f812308d7d33a269bf1a39e7fc80ecfcb910a1fe4bd156477e30076}"
OUTPUT="${YAM_TEST_OUTPUT:-$REPO/artifacts/torch-actions-q-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$(dirname -- "$OUTPUT")"
cd "$REPO"
exec "$PYTHON" -u scripts/yam/torch/test_actions_q.py \
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" --output "$OUTPUT" "$@" \
  > >(tee "${OUTPUT}.log") 2>&1
