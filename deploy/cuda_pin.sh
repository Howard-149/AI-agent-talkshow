#!/usr/bin/env bash
# Pin CUDA_VISIBLE_DEVICES to one GPU from the current allocation.
#
# Usage (from deploy/*.sh after sourcing .env and restoring SLURM CVD):
#   source "${ROOT}/deploy/cuda_pin.sh"
#   talkshow_pin_cuda_device "${VLLM_CUDA_DEVICE}"   # 0-based index into CVD list
#
# If CUDA_VISIBLE_DEVICES is unset, treats the argument as a physical GPU id.
# Logs a GPU_PIN line consumed by deploy/slurm-talkshow-3gpu.sh report_gpu_bind.

talkshow_pin_cuda_device() {
  local slot="${1:?usage: talkshow_pin_cuda_device <slot-or-id>}"
  local before="${CUDA_VISIBLE_DEVICES:-}"
  local after

  if [[ -z "${before}" ]]; then
    after="${slot}"
  else
    local -a gpus=()
    IFS=',' read -r -a gpus <<< "${before}"
    if ! [[ "${slot}" =~ ^[0-9]+$ ]]; then
      echo "FATAL: CUDA slot must be an integer, got: ${slot}" >&2
      return 1
    fi
    if ((slot < 0 || slot >= ${#gpus[@]})); then
      echo "FATAL: CUDA slot ${slot} out of range for CUDA_VISIBLE_DEVICES=${before} (n=${#gpus[@]})" >&2
      return 1
    fi
    after="${gpus[slot]}"
  fi

  export CUDA_VISIBLE_DEVICES="${after}"
  echo "GPU_PIN before_CVD=${before:-<unset>} slot=${slot} after_CVD=${after}"
}
