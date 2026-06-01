#!/usr/bin/env bash
# Run on GPU compute node (tmux). Requires vLLM cu129 + audio deps (see requirements.txt).
set -euo pipefail

# HF_HUB_CACHE / HF_DATASETS_CACHE: cluster ~/.bashrc
# Local copy (faster than AUTOFS hf_cache): export VLLM_MODEL_PATH=/data/user_data/$USER/models/gemma-4-E4B-it
MODEL="${VLLM_MODEL_PATH:-${VLLM_MODEL:-google/gemma-4-E4B-it}}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
LOG_DIR="${VLLM_LOG_DIR:-${HOME}/AI-agent-talkshow/logs}"
LOG_FILE="${LOG_DIR}/vllm-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"
echo "vLLM log → ${LOG_FILE}"
echo "  tail -f ${LOG_FILE}"

vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
  2>&1 | tee -a "${LOG_FILE}"
