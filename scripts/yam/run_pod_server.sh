#!/usr/bin/env bash
# Run from any directory; intended to be kept alive with tmux on the GPU pod.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
YAM_CHECKPOINT="${YAM_CHECKPOINT:-$ROOT/artifacts/yam_pi05_jax}"
YAM_TOKENIZER="${YAM_TOKENIZER:-$ROOT/artifacts/paligemma-tokenizer}"
if [[ ! -f "$YAM_CHECKPOINT/params/_METADATA" ]]; then
  echo "Checkpoint missing: $YAM_CHECKPOINT. Run scripts/yam/download_pod_assets.sh." >&2
  exit 1
fi
if [[ ! -f "$YAM_TOKENIZER/tokenizer_config.json" ]]; then
  echo "Tokenizer missing: $YAM_TOKENIZER. Log in with .venv-convert/bin/hf auth login, then run scripts/yam/download_pod_assets.sh." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROOT/artifacts/jax-cache}"
export PYTHONUNBUFFERED=1
# Use the CUDA libraries installed with JAX, not the pod's system CUDA path.
unset LD_LIBRARY_PATH
exec "$ROOT/.venv-convert/bin/python" "$ROOT/scripts/yam/serve_jax.py" \
  --checkpoint "$YAM_CHECKPOINT" --tokenizer "$YAM_TOKENIZER" \
  --host "${YAM_HOST:-127.0.0.1}" --port "${YAM_PORT:-8204}"
