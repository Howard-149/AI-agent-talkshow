#!/usr/bin/env bash
# Run on GPU compute node (tmux). Requires: pip install "vllm[audio]" (cu129 wheel on Babel).
set -euo pipefail

export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_cache/hub}"
MODEL="${VLLM_MODEL:-google/gemma-4-E4B-it}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}'
