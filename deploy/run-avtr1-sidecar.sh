#!/usr/bin/env bash
# AVTR-1 avatar sidecar in place of DyStream (TALKSHOW_AVATAR_ENGINE=avtr1): same GPU slot,
# port and HTTP API, so the agent needs no change. Install first: deploy/avtr1/install.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${ROOT}/deploy/cuda_pin.sh"
_CVD="${CUDA_VISIBLE_DEVICES:-}"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
[[ -n "${_CVD}" ]] && export CUDA_VISIBLE_DEVICES="${_CVD}"
: "${DYSTREAM_CUDA_DEVICE:=1}"
: "${DYSTREAM_SIDECAR_PORT:=8766}"
: "${DYSTREAM_SIDECAR_HOST:=127.0.0.1}"
talkshow_pin_cuda_device "${DYSTREAM_CUDA_DEVICE}"
source "${ROOT}/deploy/avtr1/env.sh"
avtr1_select_storage
echo "AVTR-1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} storage=${AVTR1_LOCAL_STORAGE} http://${DYSTREAM_SIDECAR_HOST}:${DYSTREAM_SIDECAR_PORT}"
cd "${AVTR1_REPO}"
export PYTHONUNBUFFERED=1
exec pixi run python "${ROOT}/avatar/avtr1_sidecar.py" \
  --host "${DYSTREAM_SIDECAR_HOST}" \
  --port "${DYSTREAM_SIDECAR_PORT}"
