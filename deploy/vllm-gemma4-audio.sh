#!/usr/bin/env bash
# Gemma / vLLM — default GPU 0.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/deploy/cuda_pin.sh"

_CVD="${CUDA_VISIBLE_DEVICES:-}"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
[[ -n "${_CVD}" ]] && export CUDA_VISIBLE_DEVICES="${_CVD}"

MODEL="${VLLM_MODEL_PATH:-${VLLM_MODEL:-google/gemma-4-E4B-it}}"
# API model id must match agent VLLM_MODEL (path ≠ HuggingFace id → 404 on chat).
SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-${VLLM_MODEL:-google/gemma-4-E4B-it}}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
LOG_DIR="${VLLM_LOG_DIR:-${HOME}/AI-agent-talkshow/logs}"
LOG_FILE="${LOG_DIR}/vllm-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "${LOG_DIR}"

: "${VLLM_CUDA_DEVICE:=0}"
talkshow_pin_cuda_device "${VLLM_CUDA_DEVICE}"

echo "vLLM log → ${LOG_FILE}"
echo "  model=${MODEL}"
echo "  served-model-name=${SERVED_NAME}"
vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME}" \
  --safetensors-load-strategy=prefetch\
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
  2>&1 | tee -a "${LOG_FILE}"
