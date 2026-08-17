#!/usr/bin/env bash
# CosyVoice sidecar — default GPU 2 (falls back to 1 then 0 if missing).
# Needs COSYVOICE_ROOT / COSYVOICE_MODEL_DIR (usually from .env).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/deploy/cuda_pin.sh"

_CVD="${CUDA_VISIBLE_DEVICES:-}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
[[ -n "${_CVD}" ]] && export CUDA_VISIBLE_DEVICES="${_CVD}"

: "${COSYVOICE_CUDA_DEVICE:=2}"
: "${COSYVOICE_SIDECAR_PORT:=8767}"
: "${COSYVOICE_SIDECAR_HOST:=127.0.0.1}"
: "${COSYVOICE_SIDECAR_WARM:=1}"

talkshow_pin_cuda_device "${COSYVOICE_CUDA_DEVICE}"

PYTHON="${COSYVOICE_PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "COSYVOICE_PYTHON not found: ${PYTHON}" >&2
  exit 1
fi

if [[ -z "${COSYVOICE_MODEL_DIR:-}" ]]; then
  echo "Set COSYVOICE_MODEL_DIR to Fun-CosyVoice3-0.5B checkpoint dir" >&2
  exit 1
fi

# Prefer absolute cluster paths (/data/...). Relative "data/..." is a common typo.
if [[ "${COSYVOICE_MODEL_DIR}" != /* ]]; then
  if [[ -d "/${COSYVOICE_MODEL_DIR}" ]]; then
    COSYVOICE_MODEL_DIR="/${COSYVOICE_MODEL_DIR}"
  elif [[ -d "${ROOT}/${COSYVOICE_MODEL_DIR}" ]]; then
    COSYVOICE_MODEL_DIR="${ROOT}/${COSYVOICE_MODEL_DIR}"
  else
    echo "COSYVOICE_MODEL_DIR is not absolute and not found: ${COSYVOICE_MODEL_DIR}" >&2
    echo "Use e.g. COSYVOICE_MODEL_DIR=/data/user_data/\$USER/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B" >&2
    exit 1
  fi
fi

# Infer clone root from .../CosyVoice/pretrained_models/<name> when unset.
if [[ -z "${COSYVOICE_ROOT:-}" ]]; then
  _parent="$(dirname "${COSYVOICE_MODEL_DIR}")"
  _grand="$(dirname "${_parent}")"
  if [[ "$(basename "${_parent}")" == "pretrained_models" && -d "${_grand}/cosyvoice" ]]; then
    COSYVOICE_ROOT="${_grand}"
  fi
fi

if [[ -z "${COSYVOICE_ROOT:-}" ]]; then
  echo "Set COSYVOICE_ROOT to the CosyVoice git clone (contains cosyvoice/ package)." >&2
  echo "  e.g. COSYVOICE_ROOT=/data/user_data/\$USER/CosyVoice" >&2
  exit 1
fi

if [[ "${COSYVOICE_ROOT}" != /* ]]; then
  if [[ -d "/${COSYVOICE_ROOT}" ]]; then
    COSYVOICE_ROOT="/${COSYVOICE_ROOT}"
  else
    echo "COSYVOICE_ROOT must be absolute: ${COSYVOICE_ROOT}" >&2
    exit 1
  fi
fi

if [[ ! -d "${COSYVOICE_ROOT}/cosyvoice" ]]; then
  echo "No package at ${COSYVOICE_ROOT}/cosyvoice — wrong COSYVOICE_ROOT?" >&2
  exit 1
fi

MATCHA_ROOT="${COSYVOICE_ROOT}/third_party/Matcha-TTS"
if [[ ! -f "${MATCHA_ROOT}/matcha/__init__.py" && ! -d "${MATCHA_ROOT}/matcha" ]]; then
  echo "Matcha-TTS submodule missing (needed as PYTHONPATH package 'matcha')." >&2
  echo "  Expected: ${MATCHA_ROOT}/matcha/" >&2
  echo "Fix on Babel:" >&2
  echo "  cd ${COSYVOICE_ROOT} && git submodule update --init --recursive" >&2
  echo "If third_party/Matcha-TTS is an empty dir, also:" >&2
  echo "  git clone https://github.com/shivammehta25/Matcha-TTS.git ${MATCHA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${COSYVOICE_MODEL_DIR}" ]]; then
  echo "Model dir missing: ${COSYVOICE_MODEL_DIR}" >&2
  exit 1
fi

export COSYVOICE_ROOT COSYVOICE_MODEL_DIR
export PYTHONPATH="${COSYVOICE_ROOT}:${MATCHA_ROOT}:${PYTHONPATH:-}"

echo "CosyVoice sidecar CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} http://${COSYVOICE_SIDECAR_HOST}:${COSYVOICE_SIDECAR_PORT}"
echo "  COSYVOICE_ROOT=${COSYVOICE_ROOT}"
echo "  model=${COSYVOICE_MODEL_DIR}"
echo "  Matcha-TTS=${MATCHA_ROOT}"
echo "  COSYVOICE_SIDECAR_WARM=${COSYVOICE_SIDECAR_WARM}"
echo "  COSYVOICE_FP16=${COSYVOICE_FP16:-1} COSYVOICE_CACHE_PROMPT=${COSYVOICE_CACHE_PROMPT:-1}"
echo "  COSYVOICE_SIDECAR_WARM_PROMPTS=${COSYVOICE_SIDECAR_WARM_PROMPTS:-1} COSYVOICE_SIDECAR_WARM_PROBE=${COSYVOICE_SIDECAR_WARM_PROBE:-1}"
echo "  health: curl -s http://${COSYVOICE_SIDECAR_HOST}:${COSYVOICE_SIDECAR_PORT}/health"
echo "Agent .env:"
echo "  COSYVOICE_SIDECAR_URL=http://${COSYVOICE_SIDECAR_HOST}:${COSYVOICE_SIDECAR_PORT}"
echo "  TALKSHOW_TTS_ENGINE=cosyvoice"
echo ""

exec "${PYTHON}" "${ROOT}/tts/cosyvoice_sidecar.py" \
  --host "${COSYVOICE_SIDECAR_HOST}" \
  --port "${COSYVOICE_SIDECAR_PORT}" \
  --model-dir "${COSYVOICE_MODEL_DIR}"
