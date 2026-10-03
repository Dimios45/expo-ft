#!/usr/bin/env bash
# Run manually ON THE NUC from its configured Karma checkout. Moves robots.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ "${1:-}" == "--label-only" ]]; then
  exec python3 "$script_dir/label_hitl.py" "${2:?Supply the existing dataset directory}"
fi

if [[ $# -lt 1 || "$1" == "--help" ]]; then
  echo 'Usage: bash record_tshirt_hitl.sh NEW_DATASET_DIRECTORY [extra Karma HITL options]'
  echo 'Set LEFT_CAN, RIGHT_CAN, TOP_SERIAL, LEFT_WRIST_SERIAL, RIGHT_WRIST_SERIAL.'
  echo 'Existing recording: bash record_tshirt_hitl.sh --label-only DATASET_DIRECTORY'
  echo 'Enter "fold the t-shirt" at the Karma task prompt. This command moves robots.'
  exit 0
fi

: "${LEFT_CAN:?Set LEFT_CAN to your configured left-arm interface}"
: "${RIGHT_CAN:?Set RIGHT_CAN to your configured right-arm interface}"
: "${TOP_SERIAL:?Set TOP_SERIAL}"
: "${LEFT_WRIST_SERIAL:?Set LEFT_WRIST_SERIAL}"
: "${RIGHT_WRIST_SERIAL:?Set RIGHT_WRIST_SERIAL}"

out=$1
shift
if [[ -e "$out" ]]; then
  echo "Refusing to reuse existing dataset: $out" >&2
  exit 1
fi
if [[ ! -f pyproject.toml ]]; then
  echo 'Run this script from ~/karma on the NUC.' >&2
  exit 1
fi

echo 'Task prompt: fold the t-shirt'
echo 'Right B: intervene; grip: clutch; Left Y tap: return to policy; hold: end.'
echo 'Terminal review: y = keep success, n = keep failure, d = discard.'
echo 'Collection only: this dataset is not yet compatible with the EXPO round importer.'

if [[ ! -f "$script_dir/label_hitl.py" ]]; then
  echo 'Copy label_hitl.py alongside this script before recording.' >&2
  exit 1
fi

client_status=0
uv run karma hitl \
  --rig yam_bimanual \
  --server "${SERVER_URL:-http://192.168.0.167:8204}" \
  --root "$out" \
  --repo-id local/yam-tshirt-hitl \
  --episodes 1 --episode-seconds 300 \
  --fps 30 --speed 1 --chunk-size 30 --num-steps 10 --no-prefetch \
  --interface "left=$LEFT_CAN" --interface "right=$RIGHT_CAN" \
  --camera-serial "top=$TOP_SERIAL" \
  --camera-serial "left_wrist=$LEFT_WRIST_SERIAL" \
  --camera-serial "right_wrist=$RIGHT_WRIST_SERIAL" \
  --quest-transport usb --open-quest \
  "$@" || client_status=$?

if [[ $client_status -ne 0 ]]; then
  echo "Karma exited with status $client_status; checking whether an episode was saved."
fi
python3 "$script_dir/label_hitl.py" "$out"
exit "$client_status"
