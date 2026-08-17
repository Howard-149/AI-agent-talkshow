#!/usr/bin/env bash
# Install CosyVoice deps for talkshow TTS sidecar (skip gradio / deepspeed).
# openai-whisper is required for clone (mel features); install unpinned (not 20231117).
# Optional accel (default on): vLLM 0.11 + TensorRT into the SAME conda env.
#
#   bash deploy/install-cosyvoice-env.sh
#   COSYVOICE_INSTALL_ACCEL=0 bash deploy/install-cosyvoice-env.sh   # base only
#
# Expects CosyVoice cloned at COSYVOICE_ROOT (default /data/user_data/$USER/CosyVoice).
# Creates/uses conda env `cosyvoice` (Python 3.10 — CosyVoice upstream default).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${COSYVOICE_CONDA_ENV:-cosyvoice}"
COSY_ROOT="${COSYVOICE_ROOT:-/data/user_data/${USER}/CosyVoice}"
INFERENCE_REQ="${ROOT}/deploy/cosyvoice-requirements-inference.txt"
INSTALL_ACCEL="${COSYVOICE_INSTALL_ACCEL:-1}"

usage() {
  echo "Usage: bash deploy/install-cosyvoice-env.sh"
  echo "  COSYVOICE_ROOT=${COSY_ROOT}"
  echo "  conda env: ${ENV_NAME}"
  echo "  installs: ${INFERENCE_REQ}"
  echo "  accel (vLLM 0.11 + tensorrt): COSYVOICE_INSTALL_ACCEL=${INSTALL_ACCEL}"
  echo "Do NOT pip install CosyVoice's full requirements.txt (pinned whisper-20231117 breaks)."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 1
fi

if [[ ! -d "${COSY_ROOT}/cosyvoice" ]]; then
  echo "CosyVoice package missing at ${COSY_ROOT}/cosyvoice" >&2
  echo "Clone first:" >&2
  echo "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git ${COSY_ROOT}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' exists — installing/updating inference deps"
else
  echo "Creating conda env '${ENV_NAME}' (python=3.10)"
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

PIP_MIRROR=(-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com)

pip install --upgrade "pip" "setuptools>=69,<81" wheel packaging "${PIP_MIRROR[@]}"
# Whisper before the rest so clone path works; unpinned avoids 20231117 sdist failure.
pip install "openai-whisper>=20240930" "${PIP_MIRROR[@]}" || \
  pip install "openai-whisper" "${PIP_MIRROR[@]}"
pip install -r "${INFERENCE_REQ}" "${PIP_MIRROR[@]}"

# Matcha-TTS is a CosyVoice submodule; package import name is `matcha`.
MATCHA="${COSY_ROOT}/third_party/Matcha-TTS"
if [[ ! -d "${MATCHA}/matcha" ]]; then
  echo "Initializing Matcha-TTS submodule…"
  (cd "${COSY_ROOT}" && git submodule update --init --recursive) || true
fi
if [[ ! -d "${MATCHA}/matcha" ]]; then
  echo "ERROR: ${MATCHA}/matcha still missing after submodule update." >&2
  echo "  cd ${COSY_ROOT} && git submodule update --init --recursive" >&2
  echo "  # or: git clone https://github.com/shivammehta25/Matcha-TTS.git ${MATCHA}" >&2
  exit 1
fi

if [[ "${INSTALL_ACCEL}" == "1" || "${INSTALL_ACCEL}" == "true" || "${INSTALL_ACCEL}" == "yes" ]]; then
  echo ""
  echo "Installing CosyVoice accel into env '${ENV_NAME}' (vLLM 0.11 + TensorRT)…"
  # Pins match CosyVoice README vllm>=0.11 path; upgrades transformers past inference req 4.51.3.
  pip install \
    "vllm==0.11.0" \
    "transformers==4.57.1" \
    "numpy==1.26.4" \
    "${PIP_MIRROR[@]}"
  # Meta package pulls tensorrt-libs / bindings for the local CUDA stack.
  python -m pip install tensorrt "${PIP_MIRROR[@]}"
  echo "Accel packages installed. Sidecar .env typically:"
  echo "  COSYVOICE_LOAD_VLLM=1"
  echo "  COSYVOICE_LOAD_TRT=0    # CosyVoice convert_onnx_to_trt breaks on TRT10 EXPLICIT_BATCH; use vLLM-only until patched"
  echo "  COSYVOICE_FP16=0"
else
  echo "Skipping accel (COSYVOICE_INSTALL_ACCEL=0). Later:"
  echo "  COSYVOICE_INSTALL_ACCEL=1 bash deploy/install-cosyvoice-env.sh"
fi

echo ""
echo "OK. Verify:"
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"import whisper; print('whisper ok', whisper.__file__)\""
if [[ "${INSTALL_ACCEL}" == "1" || "${INSTALL_ACCEL}" == "true" || "${INSTALL_ACCEL}" == "yes" ]]; then
  echo "  python -c \"import vllm, tensorrt; print('vllm', vllm.__version__, 'tensorrt ok')\""
fi
echo ""
echo "Then: bash deploy/run-cosyvoice-sidecar.sh"
echo "  COSYVOICE_PYTHON=\$(which python)"
echo "  COSYVOICE_ROOT=${COSY_ROOT}"
echo "  COSYVOICE_MODEL_DIR=${COSY_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"
