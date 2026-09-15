#!/usr/bin/env bash
#SBATCH --job-name=ts-dl
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/hsuanhal/AI-agent-talkshow
#SBATCH --output=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-download-%j.out
#SBATCH --error=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-download-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hsuanhal@andrew.cmu.edu
#
# Asset downloads on a preempt GPU node (1 GPU required by cluster policy).
# Modes (DOWNLOAD_MODE env, default: piper_livekit):
#   piper_livekit  — Piper voices (en+zh) + LiveKit agent ONNX
#   piper          — Piper only
#   livekit        — LiveKit ONNX only
#   dystream       — DyStream weights (needs HF_TOKEN / .env)
#
# Submit:  sbatch deploy/slurm-download-assets.sh
#          DOWNLOAD_MODE=dystream sbatch --export=ALL deploy/slurm-download-assets.sh
#
set -euo pipefail

# SLURM copies this script to /var/spool/slurmd/… — do not resolve helpers via BASH_SOURCE.
# shellcheck disable=SC1091
source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"

MODE="${DOWNLOAD_MODE:-piper_livekit}"

echo "=== talkshow download-assets mode=${MODE} $(date -Is) ==="
talkshow_cd
talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"
talkshow_source_env

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L 2>/dev/null || true

case "${MODE}" in
  piper_livekit)
    bash deploy/download-piper-voices.sh en zh
    bash deploy/download-livekit-agent-models.sh
    ;;
  piper)
    bash deploy/download-piper-voices.sh en zh
    ;;
  livekit)
    bash deploy/download-livekit-agent-models.sh
    ;;
  dystream)
    if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
      echo "FATAL: set HF_TOKEN in .env for dystream weights" >&2
      exit 1
    fi
    bash deploy/download-dystream-weights.sh
    ;;
  *)
    echo "FATAL: unknown DOWNLOAD_MODE=${MODE}" >&2
    exit 1
    ;;
esac

echo "=== download-assets done $(date -Is) ==="
