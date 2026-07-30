#!/usr/bin/env bash
# Create conda env `dystream` and install curated inference deps (separate from talkshow).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${DYSTREAM_CONDA_ENV:-dystream}"
TARGET="${DYSTREAM_ROOT:-/data/user_data/${USER}/dystream}"
INFERENCE_REQ="${ROOT}/deploy/dystream-requirements-inference.txt"
UPSTREAM_REQ="${TARGET}/requirements.txt"

usage() {
  echo "Usage: bash deploy/install-dystream-env.sh [--full]"
  echo "  Creates conda env '${ENV_NAME}' if missing, then installs DyStream deps."
  echo "  default   curated inference deps (recommended)"
  echo "  --full    upstream requirements.txt + mediapipee typo fix (often fails on py3.11)"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists — installing/updating deps"
else
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"

pip install --upgrade pip wheel

if [[ "${1:-}" == "--full" ]]; then
  if [[ ! -f "${UPSTREAM_REQ}" ]]; then
    echo "Missing ${UPSTREAM_REQ} — run deploy/clone-dystream.sh first" >&2
    exit 1
  fi
  PATCHED="$(mktemp)"
  trap 'rm -f "${PATCHED}"' EXIT
  sed 's/mediapipee/mediapipe/g' "${UPSTREAM_REQ}" > "${PATCHED}"
  echo "Installing upstream ${UPSTREAM_REQ} (typo patched) — may fail on numpy/opencv conflict"
  pip install -r "${PATCHED}"
else
  echo "Installing curated DyStream inference deps → ${INFERENCE_REQ}"
  pip install -r "${INFERENCE_REQ}"
fi

PY="$(which python)"
conda deactivate

echo ""
echo "Done. Add to .env:"
echo "  DYSTREAM_ROOT=${TARGET}"
echo "  DYSTREAM_PYTHON=${PY}"
echo "  DYSTREAM_CUDA_DEVICE=1"
