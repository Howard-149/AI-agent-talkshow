#!/usr/bin/env bash
# tmux 3 — DyStream sidecar on GPU 1 (models stay loaded between bakes).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${DYSTREAM_CUDA_DEVICE:=1}"
: "${DYSTREAM_SIDECAR_PORT:=8766}"
: "${DYSTREAM_SIDECAR_HOST:=127.0.0.1}"

export CUDA_VISIBLE_DEVICES="${DYSTREAM_CUDA_DEVICE}"
export DYSTREAM_SIDECAR_PORT DYSTREAM_SIDECAR_HOST
export DYSTREAM_SIDECAR_WARM="${DYSTREAM_SIDECAR_WARM:-1}"

PYTHON="${DYSTREAM_PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "DYSTREAM_PYTHON not found: ${PYTHON}" >&2
  echo "Set DYSTREAM_PYTHON in .env to dystream conda python" >&2
  exit 1
fi

echo "DyStream sidecar GPU=${DYSTREAM_CUDA_DEVICE} http://${DYSTREAM_SIDECAR_HOST}:${DYSTREAM_SIDECAR_PORT}"
echo "  DYSTREAM_SIDECAR_WARM=${DYSTREAM_SIDECAR_WARM} — models load BEFORE HTTP accepts traffic"
echo "  DYSTREAM_SIDECAR_WARM_ROLES=${DYSTREAM_SIDECAR_WARM_ROLES:-1} — prebuild per-persona portrait caches"
echo "  DYSTREAM_SIDECAR_WARM_PROBE=${DYSTREAM_SIDECAR_WARM_PROBE:-1} — short silence AR probe"
echo "  health: curl -s http://${DYSTREAM_SIDECAR_HOST}:${DYSTREAM_SIDECAR_PORT}/health"
echo "Agent .env: DYSTREAM_SIDECAR_URL=http://${DYSTREAM_SIDECAR_HOST}:${DYSTREAM_SIDECAR_PORT}"
echo ""
echo "Loading models now (several minutes). Wait for log: models ready ... sidecar can accept /bake"

exec "${PYTHON}" -m avatar.dystream_sidecar \
  --host "${DYSTREAM_SIDECAR_HOST}" \
  --port "${DYSTREAM_SIDECAR_PORT}"
