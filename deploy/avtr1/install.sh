#!/usr/bin/env bash
# Install AVTR-1 (github.com/avaturn-live/avtr-1) for the talkshow, on a GPU node of a job:
#   srun --jobid=<job> --overlap --gpus=1 bash -lc "bash deploy/avtr1/install.sh"
# Needs the gated Hugging Face weights accepted for the account of HF_TOKEN. Builds the
# TensorRT engines for this GPU type, then pads the role portraits (prepare_portraits.py).
# Rerun on each new GPU type: weights are copied, engines rebuilt.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${ROOT}/deploy/avtr1/env.sh"
if [ ! -x "${PIXI_HOME}/bin/pixi" ]; then
  # pixi.sh's installer redirect is unreachable from Babel compute nodes; GitHub is not.
  mkdir -p "${PIXI_HOME}/bin"
  curl -fsSL -L https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl.tar.gz \
    | tar -xz -C "${PIXI_HOME}/bin"
fi
[ -d "${AVTR1_REPO}" ] || git clone https://github.com/avaturn-live/avtr-1.git "${AVTR1_REPO}"
cd "${AVTR1_REPO}"
pixi install
avtr1_select_storage
if [ ! -d "${AVTR1_LOCAL_STORAGE}/main" ]; then
  other=$(ls -d "${AVTR1_HOME}"/avtr1_storage_sm* 2>/dev/null | grep -v "${AVTR1_LOCAL_STORAGE}$" | head -1 || true)
  if [ -n "${other}" ]; then
    mkdir -p "${AVTR1_LOCAL_STORAGE}" && rsync -a --exclude '*.engine' "${other}/" "${AVTR1_LOCAL_STORAGE}/"
  else
    # The pixi "download" task forces an interactive login; HF_TOKEN from the env is enough.
    pixi run python scripts/download_artifacts.py
  fi
fi
ls "${AVTR1_LOCAL_STORAGE}"/main/*/*.engine >/dev/null 2>&1 || pixi run build-trt-engines
pixi run python "${ROOT}/deploy/avtr1/prepare_portraits.py" \
  "${AVTR1_LOCAL_STORAGE}/main/avatars_artifacts/reference_frames"
echo "AVTR-1 ready: ${AVTR1_LOCAL_STORAGE}"
