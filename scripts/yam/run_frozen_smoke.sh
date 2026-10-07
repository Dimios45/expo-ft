#!/usr/bin/env bash
# Run at the YAM workstation after confirming camera placement and stop access.
set -euo pipefail
cd /home/yambox/karma
: "${LEFT_WRIST_SERIAL:?Set the confirmed left wrist camera serial}"
: "${RIGHT_WRIST_SERIAL:?Set the confirmed right wrist camera serial}"
if [[ "$LEFT_WRIST_SERIAL" == "$RIGHT_WRIST_SERIAL" ]]; then
  echo 'Wrist cameras must be different.' >&2
  exit 1
fi
output="/home/yambox/datasets/frozen-smoke-$(date +%Y%m%d-%H%M%S)"
echo "30-second frozen-policy rollout; dataset: $output"
echo 'At the task prompt enter: fold the t-shirt'
echo 'This command starts robot control. Keep the physical stop accessible.'
exec .venv/bin/karma rollout \
  --rig yam_bimanual --interface left=can_left --interface right=can_right \
  --camera-serial top=243622071623 \
  --camera-serial "left_wrist=$LEFT_WRIST_SERIAL" \
  --camera-serial "right_wrist=$RIGHT_WRIST_SERIAL" \
  --server http://192.168.0.119:8204 --norm-tag yam_dual_molmoact2 \
  --root "$output" --repo-id local/yam-frozen-smoke \
  --episodes 1 --episode-seconds 30 --fps 30 --speed 0.5 \
  --chunk-size 30 --num-steps 10 --no-prefetch
