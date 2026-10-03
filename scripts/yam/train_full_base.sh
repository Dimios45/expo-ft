#!/usr/bin/env bash
# Isolated all-parameter benchmark/training; never changes serving registries.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUTPUT="${1:?Usage: bash scripts/yam/train_full_base.sh artifacts/new-run [--steps 3 --batch-size 1 --save-weights]}"
shift
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec "$ROOT/.venv-convert/bin/python" scripts/yam/measure_gpu.py \
  --output "${OUTPUT}-gpu" -- \
  "$ROOT/.venv-convert/bin/python" scripts/yam/benchmark_base_update.py \
  --output "$OUTPUT" "$@"
