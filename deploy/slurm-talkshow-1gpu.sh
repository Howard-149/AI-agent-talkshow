#!/usr/bin/env bash

#SBATCH --job-name=talkshow
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/username/AI-agent-talkshow
#SBATCH --output=/home/username/AI-agent-talkshow/slurm-logs/slurm-talkshow-text-%j.out
#SBATCH --error=/home/username/AI-agent-talkshow/slurm-logs/slurm-talkshow-text-%j.err

# ============================================================
# Text-only talkshow
#
# GPU:
#   slot 0 -> vLLM
#
# Processes:
#   1. vLLM
#   2. agent.main with TEXT_ONLY=1
#
# No DyStream.
# No CosyVoice.
#
# Submit:
#   mkdir -p ~/AI-agent-talkshow/slurm-logs
#   sbatch ~/AI-agent-talkshow/deploy/slurm-talkshow-text.sh
# ============================================================

set -euo pipefail

echo "=== text-only talkshow start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} ==="

ROOT="${HOME}/AI-agent-talkshow"
if [[ ! -d "${ROOT}" ]]; then
  echo "FATAL: missing ${ROOT}" >&2
  exit 1
fi

cd "${ROOT}"
mkdir -p slurm-logs logs
echo "ROOT=${ROOT} pwd=$(pwd)"

# Do not source ~/.bashrc under set -e.
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/data/user_data/${USER}/miniconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="/data/user_data/${USER}/miniconda3/etc/profile.d/conda.sh"
else
  echo "FATAL: conda.sh not found under ~/miniconda3 or /data/user_data/${USER}/miniconda3" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"

export HF_HOME="${HF_HOME:-/data/user_data/${USER}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_cache/datasets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.6}"

# Preserve the GPU allocation that Slurm assigned.
SLURM_CVD="${CUDA_VISIBLE_DEVICES:-}"
echo "CUDA_VISIBLE_DEVICES(slurm)=${SLURM_CVD:-<unset>}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "sourced .env"
else
  echo "WARN: no .env in ${ROOT}" >&2
fi

# .env must not replace Slurm's GPU allocation.
if [[ -n "${SLURM_CVD}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_CVD}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "FATAL: CUDA_VISIBLE_DEVICES empty — not a GPU allocation?" >&2
  exit 1
fi

IFS=',' read -r -a ALLOC_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if ((${#ALLOC_GPUS[@]} < 1)); then
  echo "FATAL: need 1 GPU; got ${#ALLOC_GPUS[@]} (${CUDA_VISIBLE_DEVICES})" >&2
  exit 1
fi

# Text-only mode is forced here even if .env contains another value.
export TEXT_ONLY=1

# One GPU means vLLM uses slot 0 inside the Slurm allocation.
export VLLM_CUDA_DEVICE="${VLLM_CUDA_DEVICE:-0}"

ENV_TALKSHOW="${TALKSHOW_CONDA_ENV:-talkshow}"
conda activate "${ENV_TALKSHOW}"

echo "talkshow env=${ENV_TALKSHOW}"
echo "python=$(command -v python) $(python -V 2>&1)"
echo "TEXT_ONLY=${TEXT_ONLY}"
echo "vLLM slot=${VLLM_CUDA_DEVICE}"
echo "physical GPU=${ALLOC_GPUS[0]}"

JOB_TAG="${SLURM_JOB_ID:-local-$$}"

# Per-job localhost port for vLLM.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  PORT_SLOT=$((SLURM_JOB_ID % 10000))
else
  PORT_SLOT=$$
  PORT_SLOT=$((PORT_SLOT % 10000))
fi

VLLM_PORT=$((20000 + PORT_SLOT * 4))
export VLLM_PORT
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

LOG_DIR="${ROOT}/logs/slurm-text-${JOB_TAG}"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "text-only talkshow job ${JOB_TAG}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  physical GPU=${ALLOC_GPUS[0]}"
echo "  conda=${ENV_TALKSHOW}"
echo "  TEXT_ONLY=${TEXT_ONLY}"
echo "  vLLM port=${VLLM_PORT}"
echo "  logs -> ${LOG_DIR}"
echo "=============================================="

PIDS=()

cleanup() {
  echo "Stopping children: ${PIDS[*]:-}"
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

assert_alive() {
  local label="$1"
  local pid="$2"
  local log_file="${3:-}"

  if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
    echo "FATAL: ${label} pid=${pid} died" >&2
    if [[ -n "${log_file}" && -f "${log_file}" ]]; then
      echo "---- last 40 lines of ${log_file} ----" >&2
      tail -n 40 "${log_file}" >&2 || true
    fi
    return 1
  fi
}

wait_http() {
  local name="$1"
  local url="$2"
  local timeout_s="${3:-900}"
  local watch_pid="${4:-}"
  local log_file="${5:-}"

  local t0 now last_hb
  t0=$(date +%s)
  last_hb=${t0}

  echo "Waiting for ${name}: ${url}"

  while true; do
    now=$(date +%s)

    if ((now - t0 > timeout_s)); then
      echo "Timeout waiting for ${name} (${timeout_s}s)" >&2
      if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        echo "---- last 40 lines of ${log_file} ----" >&2
        tail -n 40 "${log_file}" >&2 || true
      fi
      return 1
    fi

    assert_alive "${name}" "${watch_pid}" "${log_file}" || return 1

    if ((now - last_hb >= 30)); then
      echo "  ... still waiting for ${name} ($((now - t0))s)"
      last_hb=${now}
    fi

    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "${name} ready ($((now - t0))s)"
      return 0
    fi

    sleep 5
  done
}

start_vllm() {
  (
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${ENV_TALKSHOW}"

    export VLLM_LOG_DIR="${LOG_DIR}"
    export VLLM_CUDA_DEVICE

    # Keep the existing vLLM launcher so this text-only deployment
    # uses exactly the same model/config as the current talkshow.
    bash "${ROOT}/deploy/vllm-gemma4-audio.sh"
  ) >"${LOG_DIR}/vllm.log" 2>&1 &

  VLLM_PID=$!
  PIDS+=("${VLLM_PID}")

  echo "vLLM pid=${VLLM_PID} -> ${LOG_DIR}/vllm.log"
}

start_vllm
wait_http \
  "vLLM" \
  "http://127.0.0.1:${VLLM_PORT}/v1/models" \
  900 \
  "${VLLM_PID}" \
  "${LOG_DIR}/vllm.log"

echo "vLLM is ready."
echo "Starting text-only agent.main ..."

# Keep the Slurm GPU visible to the agent process.
export CUDA_VISIBLE_DEVICES="${SLURM_CVD}"

# agent.main should branch on TEXT_ONLY=1 and must not initialize
# DyStream/CosyVoice in text-only mode.
python -m agent.main dev 2>&1 | tee "${LOG_DIR}/agent.log"
