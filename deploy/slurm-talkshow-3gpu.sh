#!/usr/bin/env bash
#SBATCH --job-name=talkshow
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=3
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/hsuanhal/AI-agent-talkshow
# Absolute paths — relative slurm-logs/ fails when that dir is missing in submit cwd.
#SBATCH --output=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-talkshow-%j.out
#SBATCH --error=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-talkshow-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hsuanhal@andrew.cmu.edu
#
# 3-GPU talkshow:
#   GPU 0 → Gemma/vLLM (conda talkshow)
#   GPU 1 → DyStream     (conda dystream)
#   GPU 2 → CosyVoice    (conda cosyvoice_vllm)
#   then agent           (conda talkshow)
#
# Default: start vLLM + DyStream + CosyVoice in parallel (128G host RAM).
# If host OOM returns:  TALKSHOW_STAGGER_START=1 sbatch ...
#
# First time:  mkdir -p ~/AI-agent-talkshow/slurm-logs
# Submit:      sbatch ~/AI-agent-talkshow/deploy/slurm-talkshow-3gpu.sh
# After ready: cat logs/slurm-<jobid>/gpu-bind.txt
#
set -euo pipefail

echo "=== talkshow job start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} ==="

ROOT="${HOME}/AI-agent-talkshow"
if [[ ! -d "${ROOT}" ]]; then
  echo "FATAL: missing ${ROOT}" >&2
  exit 1
fi
cd "${ROOT}"
mkdir -p slurm-logs logs
echo "ROOT=${ROOT} pwd=$(pwd)"

# Do NOT source ~/.bashrc here (interactive guards / early exit under set -e).
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

conda_env_python() {
  local env="$1" p
  for base in "${HOME}/miniconda3" "/data/user_data/${USER}/miniconda3"; do
    p="${base}/envs/${env}/bin/python"
    if [[ -x "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

export HF_HOME="${HF_HOME:-/data/user_data/${USER}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_cache/datasets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.6}"

# Preserve SLURM GPU bind — .env must not clobber CUDA_VISIBLE_DEVICES.
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

if [[ -n "${SLURM_CVD}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_CVD}"
fi

# Intended slots (override only if you know what you are doing).
export VLLM_CUDA_DEVICE="${VLLM_CUDA_DEVICE:-0}"
export DYSTREAM_CUDA_DEVICE="${DYSTREAM_CUDA_DEVICE:-1}"
export COSYVOICE_CUDA_DEVICE="${COSYVOICE_CUDA_DEVICE:-2}"

# Per-service conda envs (.env may set TALKSHOW_CONDA_ENV / DYSTREAM_CONDA_ENV / COSYVOICE_CONDA_ENV).
ENV_TALKSHOW="${TALKSHOW_CONDA_ENV:-talkshow}"
ENV_DYSTREAM="${DYSTREAM_CONDA_ENV:-dystream}"
ENV_COSYVOICE="${COSYVOICE_CONDA_ENV:-cosyvoice_vllm}"

conda activate "${ENV_TALKSHOW}"
echo "agent/vLLM env=${ENV_TALKSHOW} python=$(command -v python) $(python -V 2>&1)"

# Fill *_PYTHON from conda envs when .env left them unset.
if [[ -z "${DYSTREAM_PYTHON:-}" ]]; then
  DYSTREAM_PYTHON="$(conda_env_python "${ENV_DYSTREAM}")" \
    || { echo "FATAL: no python for conda env ${ENV_DYSTREAM}; set DYSTREAM_PYTHON in .env" >&2; exit 1; }
fi
if [[ -z "${COSYVOICE_PYTHON:-}" ]]; then
  COSYVOICE_PYTHON="$(conda_env_python "${ENV_COSYVOICE}")" \
    || { echo "FATAL: no python for conda env ${ENV_COSYVOICE}; set COSYVOICE_PYTHON in .env" >&2; exit 1; }
fi
export DYSTREAM_PYTHON COSYVOICE_PYTHON
echo "DyStream  env=${ENV_DYSTREAM}  python=${DYSTREAM_PYTHON}"
echo "CosyVoice env=${ENV_COSYVOICE} python=${COSYVOICE_PYTHON}"

echo "CUDA_VISIBLE_DEVICES(after .env restore)=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "FATAL: CUDA_VISIBLE_DEVICES empty — not a GPU allocation?" >&2
  exit 1
fi
IFS=',' read -r -a ALLOC_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if ((${#ALLOC_GPUS[@]} < 3)); then
  echo "FATAL: need 3 GPUs; got ${#ALLOC_GPUS[@]} (${CUDA_VISIBLE_DEVICES})" >&2
  exit 1
fi

# Catch .env foot-guns that force CosyVoice onto DyStream's slot.
if [[ "${DYSTREAM_CUDA_DEVICE}" == "${COSYVOICE_CUDA_DEVICE}" ]] \
  || [[ "${VLLM_CUDA_DEVICE}" == "${DYSTREAM_CUDA_DEVICE}" ]] \
  || [[ "${VLLM_CUDA_DEVICE}" == "${COSYVOICE_CUDA_DEVICE}" ]]; then
  echo "FATAL: GPU slots must be distinct:" >&2
  echo "  VLLM_CUDA_DEVICE=${VLLM_CUDA_DEVICE}" >&2
  echo "  DYSTREAM_CUDA_DEVICE=${DYSTREAM_CUDA_DEVICE}" >&2
  echo "  COSYVOICE_CUDA_DEVICE=${COSYVOICE_CUDA_DEVICE}" >&2
  exit 1
fi

JOB_TAG="${SLURM_JOB_ID:-local-$$}"
LOG_DIR="${ROOT}/logs/slurm-${JOB_TAG}"
mkdir -p "${LOG_DIR}"
GPU_BIND_LOG="${LOG_DIR}/gpu-bind.txt"

echo "=============================================="
echo "talkshow SLURM job ${JOB_TAG}"
echo "  SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  slots: vLLM=${VLLM_CUDA_DEVICE} DyStream=${DYSTREAM_CUDA_DEVICE} CosyVoice=${COSYVOICE_CUDA_DEVICE}"
echo "  physical map: slot0=${ALLOC_GPUS[0]} slot1=${ALLOC_GPUS[1]} slot2=${ALLOC_GPUS[2]}"
echo "  envs: ${ENV_TALKSHOW} / ${ENV_DYSTREAM} / ${ENV_COSYVOICE}"
echo "  stagger=${TALKSHOW_STAGGER_START:-0} (1 = vLLM first, then sidecars)"
echo "  child logs → ${LOG_DIR}"
echo "  GPU bind report → ${GPU_BIND_LOG}"
echo "=============================================="

{
  echo "# talkshow GPU bind plan $(date -Is) job=${JOB_TAG} host=$(hostname)"
  echo "SLURM_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "slot_intent vLLM=${VLLM_CUDA_DEVICE} DyStream=${DYSTREAM_CUDA_DEVICE} CosyVoice=${COSYVOICE_CUDA_DEVICE}"
  echo "slot_physical 0→${ALLOC_GPUS[0]} 1→${ALLOC_GPUS[1]} 2→${ALLOC_GPUS[2]}"
  nvidia-smi -L 2>/dev/null || true
} | tee "${GPU_BIND_LOG}"

PIDS=()
cleanup() {
  echo "Stopping children: ${PIDS[*]:-}"
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# If any watched sibling dies (often host-RAM OOM while models load), fail fast.
assert_alive() {
  local label="$1" pid="$2" log_file="${3:-}"
  if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
    echo "FATAL: ${label} pid=${pid} died (often host OOM / cgroup kill — not necessarily GPU OOM)" >&2
    echo "  dmesg | grep -i 'oom\|killed process' | tail -20" >&2
    [[ -n "${log_file}" && -f "${log_file}" ]] && {
      echo "---- last 40 lines of ${log_file} ----" >&2
      tail -n 40 "${log_file}" >&2 || true
    }
    return 1
  fi
  return 0
}

wait_http() {
  local name="$1" url="$2" extra_grep="${3:-}" timeout_s="${4:-900}"
  local watch_pid="${5:-}" log_file="${6:-}"
  local t0 now last_hb
  t0=$(date +%s)
  last_hb=${t0}
  echo "Waiting for ${name}: ${url}"
  while true; do
    now=$(date +%s)
    if ((now - t0 > timeout_s)); then
      echo "Timeout waiting for ${name} (${timeout_s}s)" >&2
      [[ -n "${log_file}" && -f "${log_file}" ]] && {
        echo "---- last 40 lines of ${log_file} ----" >&2
        tail -n 40 "${log_file}" >&2 || true
      }
      return 1
    fi
    assert_alive "${name}" "${watch_pid}" "${log_file}" || return 1
    # Also notice siblings dying while we wait on someone else.
    if [[ -n "${DYSTREAM_PID:-}" ]]; then
      assert_alive "DyStream" "${DYSTREAM_PID}" "${LOG_DIR}/dystream.log" || return 1
    fi
    if [[ -n "${COSYVOICE_PID:-}" ]]; then
      assert_alive "CosyVoice" "${COSYVOICE_PID}" "${LOG_DIR}/cosyvoice.log" || return 1
    fi
    if [[ -n "${VLLM_PID:-}" ]]; then
      assert_alive "vLLM" "${VLLM_PID}" "${LOG_DIR}/vllm.log" || return 1
    fi
    if ((now - last_hb >= 30)); then
      echo "  … still waiting for ${name} ($((now - t0))s)"
      last_hb=${now}
    fi
    if body=$(curl -fsS --max-time 2 "${url}" 2>/dev/null); then
      if [[ -z "${extra_grep}" ]] || grep -q "${extra_grep}" <<<"${body}"; then
        echo "${name} ready ($((now - t0))s)"
        return 0
      fi
    fi
    sleep 5
  done
}

# Pin happens inside run-*.sh / vllm-*.sh (child of the wrapper subshell), so read
# GPU_PIN after_CVD from each log — that is the authoritative single-GPU id.
pin_cvd_from_log() {
  local log="$1"
  grep -m1 'GPU_PIN before_CVD' "${log}" 2>/dev/null \
    | sed -n 's/.*after_CVD=\([^ ]*\).*/\1/p' || true
}

# After services are up: prove the three services sit on distinct physical GPUs.
report_gpu_bind() {
  local cvd_v cvd_d cvd_c
  cvd_v="$(pin_cvd_from_log "${LOG_DIR}/vllm.log")"
  cvd_d="$(pin_cvd_from_log "${LOG_DIR}/dystream.log")"
  cvd_c="$(pin_cvd_from_log "${LOG_DIR}/cosyvoice.log")"

  {
    echo ""
    echo "# runtime bind check $(date -Is)"
    echo "wrapper_PIDs VLLM=${VLLM_PID:-} DyStream=${DYSTREAM_PID:-} CosyVoice=${COSYVOICE_PID:-}"
    echo "--- GPU_PIN lines from child logs ---"
    grep -h '^GPU_PIN ' "${LOG_DIR}/vllm.log" "${LOG_DIR}/dystream.log" "${LOG_DIR}/cosyvoice.log" 2>/dev/null || true
    echo "resolved_CVD vLLM=${cvd_v:-?} DyStream=${cvd_d:-?} CosyVoice=${cvd_c:-?}"
    echo "--- nvidia-smi ---"
    nvidia-smi 2>/dev/null || true
    echo "--- compute apps ---"
    nvidia-smi --query-compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_gpu_memory --format=csv 2>/dev/null || true
  } | tee -a "${GPU_BIND_LOG}"

  if [[ -z "${cvd_v}" || -z "${cvd_d}" || -z "${cvd_c}" ]]; then
    echo "WARN: could not resolve all three CVDs — check ${GPU_BIND_LOG}" | tee -a "${GPU_BIND_LOG}"
    return 0
  fi
  if [[ "${cvd_v}" == "${cvd_d}" || "${cvd_v}" == "${cvd_c}" || "${cvd_d}" == "${cvd_c}" ]]; then
    echo "FATAL: GPU sharing detected — two services have the same CUDA_VISIBLE_DEVICES" | tee -a "${GPU_BIND_LOG}" >&2
    echo "  vLLM=${cvd_v} DyStream=${cvd_d} CosyVoice=${cvd_c}" | tee -a "${GPU_BIND_LOG}" >&2
    echo "  Expected distinct ids from SLURM list ${CUDA_VISIBLE_DEVICES}" | tee -a "${GPU_BIND_LOG}" >&2
    return 1
  fi
  echo "OK: three distinct CUDA_VISIBLE_DEVICES (vLLM=${cvd_v} DyStream=${cvd_d} CosyVoice=${cvd_c})" | tee -a "${GPU_BIND_LOG}"
  return 0
}

start_vllm() {
  (
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${ENV_TALKSHOW}"
    export VLLM_LOG_DIR="${LOG_DIR}"
    export VLLM_CUDA_DEVICE
    bash "${ROOT}/deploy/vllm-gemma4-audio.sh"
  ) >"${LOG_DIR}/vllm.log" 2>&1 &
  VLLM_PID=$!
  PIDS+=("${VLLM_PID}")
  echo "vLLM pid=${VLLM_PID} → ${LOG_DIR}/vllm.log"
}

start_dystream() {
  (
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${ENV_DYSTREAM}"
    export DYSTREAM_PYTHON="${DYSTREAM_PYTHON:-$(command -v python)}"
    export DYSTREAM_CUDA_DEVICE
    bash "${ROOT}/deploy/run-dystream-sidecar.sh"
  ) >"${LOG_DIR}/dystream.log" 2>&1 &
  DYSTREAM_PID=$!
  PIDS+=("${DYSTREAM_PID}")
  echo "DyStream pid=${DYSTREAM_PID} → ${LOG_DIR}/dystream.log"
}

start_cosyvoice() {
  (
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${ENV_COSYVOICE}"
    export COSYVOICE_PYTHON="${COSYVOICE_PYTHON:-$(command -v python)}"
    export COSYVOICE_CUDA_DEVICE
    bash "${ROOT}/deploy/run-cosyvoice-sidecar.sh"
  ) >"${LOG_DIR}/cosyvoice.log" 2>&1 &
  COSYVOICE_PID=$!
  PIDS+=("${COSYVOICE_PID}")
  echo "CosyVoice pid=${COSYVOICE_PID} → ${LOG_DIR}/cosyvoice.log"
}

if [[ "${TALKSHOW_STAGGER_START:-0}" == "1" ]]; then
  echo "TALKSHOW_STAGGER_START=1 — vLLM first, then DyStream+CosyVoice"
  start_vllm
  wait_http "vLLM" "http://127.0.0.1:${VLLM_PORT:-8000}/v1/models" "" 900 "${VLLM_PID}" "${LOG_DIR}/vllm.log"
  start_dystream
  start_cosyvoice
else
  echo "Parallel start: vLLM + DyStream + CosyVoice (128G mem; set TALKSHOW_STAGGER_START=1 if OOM)"
  start_vllm
  start_dystream
  start_cosyvoice
fi

wait_http "vLLM" "http://127.0.0.1:${VLLM_PORT:-8000}/v1/models" "" 900 "${VLLM_PID}" "${LOG_DIR}/vllm.log"
wait_http "DyStream" "${DYSTREAM_SIDECAR_URL:-http://127.0.0.1:8766}/health" "models_warmed" 900 "${DYSTREAM_PID}" "${LOG_DIR}/dystream.log"
wait_http "CosyVoice" "${COSYVOICE_SIDECAR_URL:-http://127.0.0.1:8767}/health" "" 900 "${COSYVOICE_PID}" "${LOG_DIR}/cosyvoice.log"

report_gpu_bind

echo "Starting agent.main …"
export CUDA_VISIBLE_DEVICES="${SLURM_CVD}"
python -m agent.main dev 2>&1 | tee "${LOG_DIR}/agent.log"
