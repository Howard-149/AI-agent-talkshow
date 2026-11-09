#!/usr/bin/env bash
#SBATCH --job-name=ts-eval-ghost
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-logs/slurm-eval-ghost-%j.out
#SBATCH --error=slurm-logs/slurm-eval-ghost-%j.err
#
# Listening test, live part: replay ghost sessions against a headless text-only show on one GPU.
# vLLM Gemma + a local livekit-server --dev + the agent worker (Piper TTS on CPU, avatar off) + ghost clients that
# replay scripted human turns. No LiveKit Cloud, CosyVoice or DyStream needed. Partition / QOS: see eval/README.md.
#
#   GHOST_SCRIPTS="$(ls eval/data/minimal_pairs/cond/*.yaml)" GHOST_OUT=logs/listen \
#     GHOST_CLIENT_ARGS="--stop-after-panel 4" sbatch --export=ALL deploy/slurm-eval-ghost.sh
#
# Env: GHOST_SCRIPTS (default eval/data/minimal_pairs/cond/*.yaml), GHOST_OUT (default logs/ghost-batch),
#      GHOST_PARALLEL (sessions at once, default 4), GHOST_CLIENT_ARGS (extra eval.ghost_session flags),
#      VLLM_MODEL (read by deploy/vllm-gemma4-audio.sh), LIVEKIT_SERVER_BIN (default: livekit-server on PATH),
#      VLLM_TF_OVERLAY (see deploy/slurm_common.sh).
# Output: <GHOST_OUT>/session-*.jsonl (agent logs) + <GHOST_OUT>/done/<script> markers, so a requeued job skips
# finished scripts.
#
set -euo pipefail

# shellcheck disable=SC1091
source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"

echo "=== talkshow eval ghost $(date -Is) ==="
talkshow_cd
talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"
SLURM_CVD="${CUDA_VISIBLE_DEVICES:-}"
talkshow_source_env
[[ -n "${SLURM_CVD}" ]] && export CUDA_VISIBLE_DEVICES="${SLURM_CVD}"
talkshow_hf_cache

# Ports offset by job id so two jobs on one node do not collide.
OFF=$(( ${SLURM_JOB_ID:-$$} % 400 * 10 ))
LK_PORT=$((17000 + OFF))
export VLLM_PORT=$((18000 + OFF))
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

# Local LiveKit (dev keys): overrides the cloud credentials in .env.
export LIVEKIT_URL="ws://127.0.0.1:${LK_PORT}"
export LIVEKIT_API_KEY="ghostdevkey"
export LIVEKIT_API_SECRET="ghost-dev-secret-0123456789abcdef0123456789"

# Text-only: no avatar, Piper TTS on CPU (keeps the GPU for vLLM). Replay prefixes on.
export TALKSHOW_AVATAR_ENABLED=0
export NEXT_PUBLIC_DYSTREAM_ENABLED=0
export TALKSHOW_TTS_ENGINE=piper
export TALKSHOW_PIPER_CUDA=0
export TALKSHOW_GHOST_TURNS=1
export TALKSHOW_GHOST_REPLAY="${TALKSHOW_GHOST_REPLAY:-1}"

echo "MODEL=${VLLM_MODEL:-google/gemma-4-E4B-it} (deploy/vllm-gemma4-audio.sh)"

OUT="${GHOST_OUT:-${PWD}/logs/ghost-batch}"
JOBDIR="${OUT}/job-${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}/done" "${JOBDIR}"
export LOG_DIR="${OUT}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
nvidia-smi -L 2>/dev/null || true

PIDS=()
cleanup() {
  echo "cleanup: ${PIDS[*]:-}"
  for pid in "${PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cat >"${JOBDIR}/livekit.yaml" <<EOF
port: ${LK_PORT}
bind_addresses: ["127.0.0.1"]
rtc:
  tcp_port: $((LK_PORT + 1))
  port_range_start: $((LK_PORT + 2))
  port_range_end: $((LK_PORT + 9))
  use_external_ip: false
keys:
  ${LIVEKIT_API_KEY}: ${LIVEKIT_API_SECRET}
logging:
  level: info
EOF
LK_BIN="${LIVEKIT_SERVER_BIN:-$(command -v livekit-server || echo "${HOME}/bin/livekit-server")}"
"${LK_BIN}" --config "${JOBDIR}/livekit.yaml" >"${JOBDIR}/livekit.log" 2>&1 &
PIDS+=($!)

PYTHONPATH="$(talkshow_tf_overlay_pythonpath)" VLLM_CUDA_DEVICE=0 VLLM_LOG_DIR="${JOBDIR}" \
  bash deploy/vllm-gemma4-audio.sh >"${JOBDIR}/vllm.log" 2>&1 &
VLLM_PID=$!
PIDS+=("${VLLM_PID}")

t0=$(date +%s)
until curl -fsS --max-time 2 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; do
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "FATAL: vLLM died"; tail -40 "${JOBDIR}/vllm.log"; exit 1; }
  (( $(date +%s) - t0 > 1500 )) && { echo "FATAL: vLLM timeout"; tail -40 "${JOBDIR}/vllm.log"; exit 1; }
  sleep 5
done
echo "vLLM ready ($(( $(date +%s) - t0 ))s)"

# `dev`, not `start`: the production load threshold uses host CPU and rejects rooms on a shared node
# ("no workers with sufficient capacity"). Job processes import agent/ when spawned, so do not edit agent/ mid-run.
export TALKSHOW_IDLE_PROCESSES="${GHOST_PARALLEL:-4}"
python -m agent.main dev >"${JOBDIR}/agent.log" 2>&1 &
AGENT_PID=$!
PIDS+=("${AGENT_PID}")
t0=$(date +%s)
until grep -qi 'registered worker' "${JOBDIR}/agent.log" 2>/dev/null; do
  kill -0 "${AGENT_PID}" 2>/dev/null || { echo "FATAL: agent died"; tail -60 "${JOBDIR}/agent.log"; exit 1; }
  (( $(date +%s) - t0 > 600 )) && { echo "FATAL: agent register timeout"; tail -60 "${JOBDIR}/agent.log"; exit 1; }
  sleep 3
done
echo "agent registered ($(( $(date +%s) - t0 ))s)"

run_one() {
  local script="$1" slug room
  slug="$(basename "${script}" .yaml)"
  [[ -f "${OUT}/done/${slug}" ]] && { echo "skip ${slug} (done)"; return 0; }
  # The room name carries the script name: eval.responsiveness collect reads topic / condition / k / arm from it.
  room="ghost-${slug}-${SLURM_JOB_ID:-local}-$(date +%s)"
  echo "start ${slug} room=${room}"
  if timeout 1500 python -m eval.ghost_session --room "${room}" --script "${script}" \
      --hold-sec 20 --floor-timeout 300 --log-dir "${OUT}" ${GHOST_CLIENT_ARGS:-} >"${JOBDIR}/ghost-${slug}.log" 2>&1; then
    echo "${room}" >"${OUT}/done/${slug}"
    echo "done ${slug}"
  else
    echo "FAIL ${slug} rc=$? (see ${JOBDIR}/ghost-${slug}.log)"
  fi
}
export -f run_one
export OUT JOBDIR GHOST_CLIENT_ARGS

SCRIPTS="${GHOST_SCRIPTS:-$(ls eval/data/minimal_pairs/cond/*.yaml)}"
printf '%s\n' ${SCRIPTS} | xargs -P "${GHOST_PARALLEL:-4}" -I{} bash -c 'run_one "$@"' _ {}

echo "done markers: $(ls "${OUT}/done" | wc -l)"
echo "=== eval ghost done $(date -Is) ==="
