#!/usr/bin/env bash
#SBATCH --job-name=ts-voice-emo
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/hsuanhal/AI-agent-talkshow
#SBATCH --output=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-voice-emotion-%j.out
#SBATCH --error=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-voice-emotion-%j.err
#
# Voice check (eval/voice_emotion.py): CosyVoice sidecar → synth → stop sidecar → SER score → report.
# VOICE_STAGES picks steps (default synth,msp,dim,cat,report); e.g. VOICE_STAGES=msp,dim,cat,report
# adds the real-speech reference and categorical scores to an existing run without re-synthesizing.
#
# Submit:  sbatch deploy/slurm-voice-emotion.sh
# Options: VOICE_OUT=eval/results/voice_emotion/<name>  VOICE_REPEATS=2
#          VOICE_VARIANTS=live,zh,strong_en,strong_zh  (instruct wordings to compare)
#          VOICE_SESSIONS="logs/session-1.jsonl logs/session-2.jsonl"  (adds live lines)
#          Re-running with the same VOICE_OUT resumes (synth and score skip finished clips).
#
set -euo pipefail

source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"
talkshow_cd
talkshow_source_env

OUT="${VOICE_OUT:-eval/results/voice_emotion/run-${SLURM_JOB_ID:-local}}"
LOG_DIR="logs/slurm-${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}" "${LOG_DIR}"
export HF_HOME="${HF_HOME:-/data/user_data/${USER}/.hf_cache}"
# A free port: shared nodes may already run someone else's sidecar on 8767.
export COSYVOICE_SIDECAR_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
export COSYVOICE_SIDECAR_URL="http://127.0.0.1:${COSYVOICE_SIDECAR_PORT}"
export COSYVOICE_CUDA_DEVICE=0
export TALKSHOW_TTS_ENGINE=cosyvoice

STAGES=",${VOICE_STAGES:-synth,msp,dim,cat,report},"
has() { [[ "${STAGES}" == *",$1,"* ]]; }
echo "=== voice emotion $(date -Is) out=${OUT} stages=${STAGES} ==="
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

if has synth; then
  (
    talkshow_activate "${COSYVOICE_CONDA_ENV:-cosyvoice_vllm}"
    export COSYVOICE_PYTHON="${COSYVOICE_PYTHON:-$(command -v python)}"
    bash deploy/run-cosyvoice-sidecar.sh
  ) >"${LOG_DIR}/cosyvoice.log" 2>&1 &
  SIDECAR_PID=$!
  trap 'kill ${SIDECAR_PID} 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    curl -fsS --max-time 2 "${COSYVOICE_SIDECAR_URL}/health" >/dev/null 2>&1 && break
    kill -0 "${SIDECAR_PID}" 2>/dev/null || { echo "sidecar died"; tail -40 "${LOG_DIR}/cosyvoice.log"; exit 1; }
    sleep 5
  done
  echo "sidecar ready ${COSYVOICE_SIDECAR_URL}"
  talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"
  SESSION_ARGS=()
  for s in ${VOICE_SESSIONS:-}; do SESSION_ARGS+=(--session "${s}"); done
  python eval/voice_emotion.py synth --out "${OUT}" --repeats "${VOICE_REPEATS:-2}" \
    --variants "${VOICE_VARIANTS:-live}" "${SESSION_ARGS[@]}"
  # Free the GPU for the SER models.
  kill "${SIDECAR_PID}" 2>/dev/null || true
  pkill -f "port ${COSYVOICE_SIDECAR_PORT}" 2>/dev/null || true
  sleep 10
fi

talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"
if has msp; then
  python eval/voice_emotion.py msp --out "${OUT}" --per-label "${VOICE_MSP_PER_LABEL:-60}"
fi
if has dim; then
  python eval/voice_emotion.py score --out "${OUT}" --model dim
fi
if has cat; then
  # The categorical model's remote code needs transformers 4.x.
  talkshow_activate "${VOICE_CAT_ENV:-cosyvoice}"
  python eval/voice_emotion.py score --out "${OUT}" --model cat || echo "WARN: categorical scoring failed"
  talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"
fi
if has report; then
  for src in session msp; do
    python eval/voice_emotion.py report --out "${OUT}" --source "${src}" || true
  done
  for v in $(tr ',' ' ' <<<"${VOICE_VARIANTS:-live}"); do
    python eval/voice_emotion.py report --out "${OUT}" --source controlled --variant "${v}" || true
  done
fi
echo "=== done $(date -Is) ==="
