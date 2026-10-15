#!/usr/bin/env bash
#SBATCH --job-name=talkshow
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=3
#SBATCH --mem=128G
#SBATCH --time=16:00:00
# Absolute paths — relative slurm-logs/ fails when that dir is missing in submit cwd.
#SBATCH --output=/home/%u/AI-agent-talkshow/slurm-logs/slurm-talkshow-%j.out
#SBATCH --error=/home/%u/AI-agent-talkshow/slurm-logs/slurm-talkshow-%j.err
#SBATCH --mail-type=END
# %u = submitting user. Mail: add --mail-user=<you>@andrew.cmu.edu when submitting.
#
# 3-GPU talkshow on preempt (requires --gpus=3):
#
#   Slot (in CUDA_VISIBLE_DEVICES)   Conda env              Process
#   ------------------------------   --------------------   ----------------------
#   VLLM_CUDA_DEVICE (.env/0)        talkshow               vLLM Gemma
#   DYSTREAM_CUDA_DEVICE (.env/1)    dystream               DyStream sidecar
#   COSYVOICE_CUDA_DEVICE (.env/2)   cosyvoice_vllm         CosyVoice sidecar
#   (all visible)                    talkshow               agent.main
#
# Values come from .env when set; otherwise the defaults above.
# Job fails if the three CUDA slots or the three Python interpreters collide.
#
# Preempt jobs can be killed/requeued — walltime 8h; --requeue above.
# Override partition:  sbatch -p <name> deploy/slurm-talkshow-3gpu.sh
#
# Default: start vLLM + DyStream + CosyVoice in parallel (128G host RAM).
# If host OOM:  TALKSHOW_STAGGER_START=1 sbatch --export=ALL ...
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

# Avatar off (TALKSHOW_AVATAR_ENABLED unset/0) on 2 GPUs → skip DyStream:
#   sbatch --gpus=2 deploy/slurm-talkshow-3gpu.sh
# On 3 GPUs DyStream starts either way (TALKSHOW_DYSTREAM_START=auto), so the agent can
# switch the avatar on/off with a restart (deploy/talkshow-ctl.sh) instead of a resubmit.
case "${TALKSHOW_AVATAR_ENABLED:-0}" in
  1|true|yes|on) AVATAR_ON=1 ;;
  *) AVATAR_ON=0 ;;
esac
IFS=',' read -r -a _SLURM_GPUS <<< "${SLURM_CVD}"
case "${TALKSHOW_DYSTREAM_START:-auto}" in
  1|true|yes|on) DYSTREAM_ON=1 ;;
  0|false|no|off) DYSTREAM_ON=0 ;;
  *) DYSTREAM_ON=$((AVATAR_ON == 1 || ${#_SLURM_GPUS[@]} >= 3 ? 1 : 0)) ;;
esac
if [[ "${AVATAR_ON}" == "1" && "${DYSTREAM_ON}" != "1" ]]; then
  echo "FATAL: TALKSHOW_AVATAR_ENABLED=1 needs DyStream (TALKSHOW_DYSTREAM_START=${TALKSHOW_DYSTREAM_START:-auto})" >&2
  exit 1
fi
echo "avatar=${AVATAR_ON} (TALKSHOW_AVATAR_ENABLED=${TALKSHOW_AVATAR_ENABLED:-<unset>}) dystream=${DYSTREAM_ON}"

# Intended slots WITHIN the SLURM allocation (indices into CUDA_VISIBLE_DEVICES).
# Prefer .env; fall back to 0/1/2 only when unset.
export VLLM_CUDA_DEVICE="${VLLM_CUDA_DEVICE:-0}"
export DYSTREAM_CUDA_DEVICE="${DYSTREAM_CUDA_DEVICE:-1}"
if [[ "${DYSTREAM_ON}" == "1" ]]; then
  export COSYVOICE_CUDA_DEVICE="${COSYVOICE_CUDA_DEVICE:-2}"
else
  export COSYVOICE_CUDA_DEVICE="${COSYVOICE_CUDA_DEVICE:-1}"
fi

# Per-service conda envs (.env may set TALKSHOW_CONDA_ENV / DYSTREAM_CONDA_ENV / COSYVOICE_CONDA_ENV).
ENV_TALKSHOW="${TALKSHOW_CONDA_ENV:-talkshow}"
ENV_DYSTREAM="${DYSTREAM_CONDA_ENV:-dystream}"
ENV_COSYVOICE="${COSYVOICE_CONDA_ENV:-cosyvoice_vllm}"

echo "=== 3-GPU / 3-env contract (from .env or defaults) ==="
echo "  vLLM      slot=${VLLM_CUDA_DEVICE}  conda=${ENV_TALKSHOW}"
echo "  DyStream  slot=${DYSTREAM_CUDA_DEVICE}  conda=${ENV_DYSTREAM}"
echo "  CosyVoice slot=${COSYVOICE_CUDA_DEVICE}  conda=${ENV_COSYVOICE}"
echo "  agent     all slots visible  conda=${ENV_TALKSHOW}"

conda activate "${ENV_TALKSHOW}"
echo "agent/vLLM env=${ENV_TALKSHOW} python=$(command -v python) $(python -V 2>&1)"

# Fill *_PYTHON from conda envs when .env left them unset.
if [[ "${DYSTREAM_ON}" == "1" && -z "${DYSTREAM_PYTHON:-}" ]]; then
  DYSTREAM_PYTHON="$(conda_env_python "${ENV_DYSTREAM}")" \
    || { echo "FATAL: no python for conda env ${ENV_DYSTREAM}; set DYSTREAM_PYTHON in .env" >&2; exit 1; }
fi
[[ "${DYSTREAM_ON}" == "1" ]] || DYSTREAM_PYTHON="${DYSTREAM_PYTHON:-<dystream-off>}"
if [[ -z "${COSYVOICE_PYTHON:-}" ]]; then
  COSYVOICE_PYTHON="$(conda_env_python "${ENV_COSYVOICE}")" \
    || { echo "FATAL: no python for conda env ${ENV_COSYVOICE}; set COSYVOICE_PYTHON in .env" >&2; exit 1; }
fi
export DYSTREAM_PYTHON COSYVOICE_PYTHON
echo "DyStream  env=${ENV_DYSTREAM}  python=${DYSTREAM_PYTHON}"
echo "CosyVoice env=${ENV_COSYVOICE} python=${COSYVOICE_PYTHON}"

# Refuse same-interpreter foot-gun (sidecars must not share talkshow python).
if [[ "${DYSTREAM_PYTHON}" == "$(command -v python)" ]] \
  || [[ "${COSYVOICE_PYTHON}" == "$(command -v python)" ]] \
  || [[ "${DYSTREAM_PYTHON}" == "${COSYVOICE_PYTHON}" ]]; then
  echo "FATAL: DyStream / CosyVoice / talkshow must use three distinct Python interpreters" >&2
  echo "  talkshow=$(command -v python)" >&2
  echo "  DYSTREAM_PYTHON=${DYSTREAM_PYTHON}" >&2
  echo "  COSYVOICE_PYTHON=${COSYVOICE_PYTHON}" >&2
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES(after .env restore)=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "FATAL: CUDA_VISIBLE_DEVICES empty — not a GPU allocation?" >&2
  exit 1
fi
IFS=',' read -r -a ALLOC_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NEED_GPUS=$((DYSTREAM_ON == 1 ? 3 : 2))
if ((${#ALLOC_GPUS[@]} < NEED_GPUS)); then
  echo "FATAL: need ${NEED_GPUS} GPUs; got ${#ALLOC_GPUS[@]} (${CUDA_VISIBLE_DEVICES})" >&2
  exit 1
fi

# Catch .env foot-guns that force CosyVoice onto DyStream's slot.
if [[ "${VLLM_CUDA_DEVICE}" == "${COSYVOICE_CUDA_DEVICE}" ]] \
  || { [[ "${DYSTREAM_ON}" == "1" ]] && {
    [[ "${DYSTREAM_CUDA_DEVICE}" == "${COSYVOICE_CUDA_DEVICE}" ]] \
      || [[ "${VLLM_CUDA_DEVICE}" == "${DYSTREAM_CUDA_DEVICE}" ]]; }; }; then
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
echo "  physical map: slot0=${ALLOC_GPUS[0]} slot1=${ALLOC_GPUS[1]} slot2=${ALLOC_GPUS[2]:-none}"
echo "  envs: ${ENV_TALKSHOW} / ${ENV_DYSTREAM} / ${ENV_COSYVOICE}"
echo "  stagger=${TALKSHOW_STAGGER_START:-0} (1 = vLLM first, then sidecars)"
echo "  child logs → ${LOG_DIR}"
echo "  GPU bind report → ${GPU_BIND_LOG}"
echo "=============================================="

{
  echo "# talkshow GPU bind plan $(date -Is) job=${JOB_TAG} host=$(hostname)"
  echo "SLURM_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "slot_intent vLLM=${VLLM_CUDA_DEVICE} DyStream=${DYSTREAM_CUDA_DEVICE} CosyVoice=${COSYVOICE_CUDA_DEVICE}"
  echo "slot_physical 0→${ALLOC_GPUS[0]} 1→${ALLOC_GPUS[1]} 2→${ALLOC_GPUS[2]:-none}"
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

  if [[ "${DYSTREAM_ON}" != "1" ]]; then
    cvd_d="off"
  fi
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
  [[ "${DYSTREAM_ON}" == "1" ]] && start_dystream
  start_cosyvoice
else
  echo "Parallel start: vLLM + CosyVoice$([[ "${DYSTREAM_ON}" == "1" ]] && echo " + DyStream") (128G mem; set TALKSHOW_STAGGER_START=1 if OOM)"
  start_vllm
  [[ "${DYSTREAM_ON}" == "1" ]] && start_dystream
  start_cosyvoice
fi

wait_http "vLLM" "http://127.0.0.1:${VLLM_PORT:-8000}/v1/models" "" 900 "${VLLM_PID}" "${LOG_DIR}/vllm.log"
if [[ "${DYSTREAM_ON}" == "1" ]]; then
  wait_http "DyStream" "${DYSTREAM_SIDECAR_URL:-http://127.0.0.1:8766}/health" "models_warmed" 900 "${DYSTREAM_PID}" "${LOG_DIR}/dystream.log"
fi
wait_http "CosyVoice" "${COSYVOICE_SIDECAR_URL:-http://127.0.0.1:8767}/health" "" 900 "${COSYVOICE_PID}" "${LOG_DIR}/cosyvoice.log"

report_gpu_bind

# ---------------------------------------------------------------------------
# Supervisor: the models above stay loaded while the agent (or one sidecar) restarts.
#   deploy/talkshow-ctl.sh restart agent       # new code / .env, ~20 s, no resubmit
#   deploy/talkshow-ctl.sh restart cosyvoice   # sidecar code change; reloads that model
# Requests are files in ${LOG_DIR}/control/ (home is shared, so no srun is needed).
# ---------------------------------------------------------------------------
CTL_DIR="${LOG_DIR}/control"
mkdir -p "${CTL_DIR}"
rm -f "${CTL_DIR}"/restart-*  # stale requests from before a requeue
EVENTS_LOG="${CTL_DIR}/events.log"

ctl_event() {
  echo "$(date -Is) $*" | tee -a "${EVENTS_LOG}"
}

code_version() {
  local rev
  rev="$(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  git -C "${ROOT}" diff --quiet HEAD -- 2>/dev/null || rev="${rev}-dirty"
  echo "${rev}"
}

# Every process under a wrapper pid (run-*.sh → python → vLLM EngineCore …).
descendants() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    descendants "${child}"
    echo "${child}"
  done
}

stop_tree() {
  local root="$1" pids p alive
  [[ -n "${root}" ]] || return 0
  pids="$(descendants "${root}") ${root}"
  # shellcheck disable=SC2086
  kill -TERM ${pids} 2>/dev/null || true
  for _ in $(seq 1 30); do
    alive=0
    for p in ${pids}; do
      kill -0 "${p}" 2>/dev/null && alive=1
    done
    ((alive)) || break
    sleep 1
  done
  # shellcheck disable=SC2086
  kill -KILL ${pids} 2>/dev/null || true
  wait "${root}" 2>/dev/null || true
  local keep=() p2
  for p2 in "${PIDS[@]}"; do
    [[ "${p2}" == "${root}" ]] || keep+=("${p2}")
  done
  PIDS=("${keep[@]}")
}

start_agent() {
  (
    # Fresh .env on every start: agent-only settings (TALKSHOW_AVATAR_ENABLED,
    # TALKSHOW_PAD_*, TALKSHOW_HAND_RAISE_*, …) change with a restart.
    if [[ -f "${ROOT}/.env" ]]; then
      set -a
      # shellcheck disable=SC1091
      source "${ROOT}/.env"
      set +a
    fi
    export CUDA_VISIBLE_DEVICES="${SLURM_CVD}"
    case "${TALKSHOW_AVATAR_ENABLED:-0}" in
      1|true|yes|on)
        [[ "${DYSTREAM_ON}" == "1" ]] \
          || echo "WARN: TALKSHOW_AVATAR_ENABLED=1 but DyStream is not running in this job" >&2
        ;;
    esac
    python -m agent.main dev 2>&1 | tee -a "${LOG_DIR}/agent.log"
  ) &
  AGENT_PID=$!
  PIDS+=("${AGENT_PID}")
  ctl_event "agent started pid=${AGENT_PID} code=$(code_version)"
}

# start_* truncate <svc>.log; keep the previous run's log next to it.
keep_log() {
  local log="${LOG_DIR}/$1.log"
  [[ -f "${log}" ]] && mv "${log}" "${LOG_DIR}/$1-$(date +%Y%m%d-%H%M%S).log"
  return 0
}

restart_service() {
  case "$1" in
    agent)
      stop_tree "${AGENT_PID:-}"
      start_agent
      ;;
    cosyvoice)
      stop_tree "${COSYVOICE_PID:-}"
      keep_log cosyvoice
      start_cosyvoice
      wait_http "CosyVoice" "${COSYVOICE_SIDECAR_URL:-http://127.0.0.1:8767}/health" "" 900 "${COSYVOICE_PID}" "${LOG_DIR}/cosyvoice.log"
      ;;
    dystream)
      if [[ "${DYSTREAM_ON}" != "1" ]]; then
        ctl_event "DyStream is not running in this job; ignoring restart"
        return 0
      fi
      stop_tree "${DYSTREAM_PID:-}"
      keep_log dystream
      start_dystream
      wait_http "DyStream" "${DYSTREAM_SIDECAR_URL:-http://127.0.0.1:8766}/health" "models_warmed" 900 "${DYSTREAM_PID}" "${LOG_DIR}/dystream.log"
      ;;
    vllm)
      stop_tree "${VLLM_PID:-}"
      keep_log vllm
      start_vllm
      wait_http "vLLM" "http://127.0.0.1:${VLLM_PORT:-8000}/v1/models" "" 900 "${VLLM_PID}" "${LOG_DIR}/vllm.log"
      ;;
    *)
      ctl_event "unknown service '$1' (agent|cosyvoice|dystream|vllm)"
      ;;
  esac
}

# Latency A/B: with TALKSHOW_ARMS_FILE set, run ghost-session arms instead of the
# supervised agent. deploy/ghost-arms.sh restarts only the agent per arm (models stay
# loaded) and the job ends when the arms are done.
if [[ -n "${TALKSHOW_ARMS_FILE:-}" ]]; then
  ctl_event "running ghost arms from ${TALKSHOW_ARMS_FILE}"
  ( export CUDA_VISIBLE_DEVICES="${SLURM_CVD}"; bash deploy/ghost-arms.sh "${TALKSHOW_ARMS_FILE}" ) 2>&1 | tee "${LOG_DIR}/arms.log"
  rc="${PIPESTATUS[0]}"
  ctl_event "ghost arms finished rc=${rc}"
  exit "${rc}"
fi

echo "Starting agent.main …"
start_agent
while true; do
  for req in "${CTL_DIR}"/restart-*; do
    [[ -e "${req}" ]] || continue
    svc="${req##*/restart-}"
    rm -f "${req}"
    ctl_event "restart ${svc}: requested"
    restart_service "${svc}" || { ctl_event "restart ${svc}: FAILED"; exit 1; }
    ctl_event "restart ${svc}: done"
  done
  if ! kill -0 "${AGENT_PID}" 2>/dev/null; then
    ctl_event "agent exited unexpectedly; restarting in 10 s (see ${LOG_DIR}/agent.log)"
    sleep 10
    start_agent
  fi
  # A dead sidecar is reported once and left for `talkshow-ctl.sh restart <svc>`.
  for svc_pid in "vllm:${VLLM_PID:-}" "cosyvoice:${COSYVOICE_PID:-}" "dystream:${DYSTREAM_PID:-}"; do
    svc="${svc_pid%%:*}"
    pid="${svc_pid#*:}"
    [[ -n "${pid}" ]] || continue
    if ! kill -0 "${pid}" 2>/dev/null && [[ "${REPORTED_DEAD:-}" != *" ${pid} "* ]]; then
      ctl_event "${svc} pid=${pid} died; see ${LOG_DIR}/${svc}.log, then: deploy/talkshow-ctl.sh restart ${svc}"
      REPORTED_DEAD="${REPORTED_DEAD:- } ${pid} "
    fi
  done
  sleep 2
done
