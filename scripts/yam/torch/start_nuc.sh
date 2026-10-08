#!/usr/bin/env bash
# Operator-run only: this invokes KARMA robot control after explicit confirmation.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${YAM_COLLECT_PYTHON:-$HOME/.venvs/yam-rtc-py310/bin/python}"
DATA_ROOT="${YAM_DATA_ROOT:?Set a new dataset directory for this PyTorch experiment}"
export YAM_CONTROL_TOKEN="$(cat "${YAM_CONTROL_TOKEN_FILE:?Set path to this experiment control.token}")"
export PATH="$HOME/.local/bin:$PATH"
cd "${KARMA_ROOT:-$HOME/karma}"
exec "$PYTHON" -u "$REPO/scripts/yam/collect_online.py" \
  --recorder "$REPO/scripts/yam/record_round.py" \
  --control "${YAM_CONTROL_URL:-ws://192.168.0.167:18308}" \
  --server "${YAM_POLICY_URL:-http://192.168.0.119:18304}" \
  --root "$DATA_ROOT" --episodes 5 --seconds 300 --archive-discarded --fixed-prompt --strict-confirmation -- \
  --interface left=can_left --interface right=can_right \
  --camera-serial top=348523020354 \
  --camera-serial left_wrist=254623070863 \
  --camera-serial right_wrist=254623070417
