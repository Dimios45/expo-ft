#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
"$ROOT/.venv-convert/bin/hf" download sra-vjti/molmoact2-yam-pi05-jax \
  --local-dir "$ROOT/artifacts/yam_pi05_jax"
# Requires an HF login with access approved for this gated Google model.
"$ROOT/.venv-convert/bin/hf" download google/paligemma-3b-pt-224 \
  --include 'tokenizer*' 'special_tokens_map.json' 'added_tokens.json' 'config.json' \
  --local-dir "$ROOT/artifacts/paligemma-tokenizer"
