#!/usr/bin/env bash
# DyStream sidecar — default GPU 1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/deploy/cuda_pin.sh"

# Keep SLURM's GPU list if .env tries to overwrite CUDA_VISIBLE_DEVICES.
_CVD="${CUDA_VISIBLE_DEVICES:-}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
[[ -n "${_CVD}" ]] && export CUDA_VISIBLE_DEVICES="${_CVD}"

: "${DYSTREAM_CUDA_DEVICE:=1}"
: "${DYSTREAM_SIDECAR_PORT:=8766}"
: "${DYSTREAM_SIDECAR_HOST:=127.0.0.1}"
: "${DYSTREAM_SIDECAR_WARM:=1}"

talkshow_pin_cuda_device "${DYSTREAM_CUDA_DEVICE}"

PYTHON="${DYSTREAM_PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "DYSTREAM_PYTHON not found: ${PYTHON}" >&2
  exit 1
fi

echo "DyStream CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} http://${DYSTREAM_SIDECAR_HOST}:${DYSTREAM_SIDECAR_PORT}"
exec "${PYTHON}" -m avatar.dystream_sidecar \
  --host "${DYSTREAM_SIDECAR_HOST}" \
  --port "${DYSTREAM_SIDECAR_PORT}"
