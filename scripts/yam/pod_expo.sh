#!/usr/bin/env bash
# A100 entrypoint for sequential collection and training; never opens hardware.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROOT/artifacts/jax-cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMP_NUM_THREADS=4
unset LD_LIBRARY_PATH
PYTHON="$ROOT/.venv-convert/bin/python"
EXPERIMENT="${YAM_EXPERIMENT:-$ROOT/artifacts/yam-expo}"
COMMAND="${1:-help}"
if (( $# )); then shift; fi
case "$COMMAND" in
  init)
    MODE="${YAM_ACTOR_MODE:-frozen}"
    CHECKPOINT="${YAM_CHECKPOINT:-$ROOT/artifacts/yam_pi05_jax}"
    TOKENIZER="${YAM_TOKENIZER:-$ROOT/artifacts/paligemma-tokenizer}"
    PROMPT="${YAM_PROMPT:-fold the towel}"
    if [[ "$MODE" == frozen ]]; then
      exec "$PYTHON" scripts/yam/continue_stable.py init --root "$EXPERIMENT" \
        --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" --prompt "$PROMPT" "$@"
    elif [[ "$MODE" == expert ]]; then
      exec "$PYTHON" scripts/yam/round.py --root "$EXPERIMENT" init \
        --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" --prompt "$PROMPT" --actor-mode expert "$@"
    else
      echo 'YAM_ACTOR_MODE must be frozen or expert' >&2; exit 2
    fi
    ;;
  serve)
    export YAM_CANDIDATE_BATCH="${YAM_CANDIDATE_BATCH:-4}"
    exec "$PYTHON" scripts/yam/round.py --root "$EXPERIMENT" serve \
      --host 127.0.0.1 --port "${YAM_PORT:-8204}" "$@"
    ;;
  status)
    exec "$PYTHON" scripts/yam/round.py --root "$EXPERIMENT" status "$@"
    ;;
  prepare|train)
    PROFILE="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("profile", "legacy"))' "$EXPERIMENT/experiment.json")"
    if [[ "$PROFILE" == stable-v2 ]]; then
      if [[ "$COMMAND" == train ]]; then
        exec "$PYTHON" scripts/yam/continue_stable.py train --root "$EXPERIMENT" \
          --microbatch "${YAM_TRAIN_MICROBATCH:-8}" "$@"
      fi
      export YAM_CANDIDATE_BATCH="${YAM_CANDIDATE_BATCH:-4}"
      exec "$PYTHON" scripts/yam/continue_stable.py "$COMMAND" --root "$EXPERIMENT" "$@"
    fi
    if [[ "$COMMAND" == prepare ]]; then COMMAND=ingest; fi
    exec "$PYTHON" scripts/yam/round.py --root "$EXPERIMENT" "$COMMAND" "$@"
    ;;
  *)
    echo 'Usage: bash scripts/yam/pod_expo.sh {init|serve|status|prepare --dataset PATH|train}'
    echo 'init defaults to stable frozen-base EXPO; YAM_ACTOR_MODE=expert selects the legacy expert-update path.'
    exit 2
    ;;
esac
