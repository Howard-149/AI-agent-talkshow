#!/usr/bin/env bash
#SBATCH --job-name=ts-msp-anc
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/AI-agent-talkshow/slurm-logs/slurm-msp-anchors-%j.out
#SBATCH --error=/home/%u/AI-agent-talkshow/slurm-logs/slurm-msp-anchors-%j.err
#SBATCH --mail-type=END
# %u = submitting user. Mail: add --mail-user=<you>@andrew.cmu.edu when submitting.
#
# Inspect MSP-PODCAST labels + build PAD KNN anchors (needs pandas on talkshow env).
#
# Submit:  sbatch deploy/slurm-build-msp-anchors.sh
# Override CSV:  MSP_LABELS_CSV=/path/to/other.csv sbatch --export=ALL ...
#
set -euo pipefail

source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"

echo "=== build MSP PAD anchors $(date -Is) ==="
talkshow_cd
talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"

# pandas may already be present via vLLM stack; install if missing.
python -c "import pandas" 2>/dev/null || pip install -q pandas

CSV="${MSP_LABELS_CSV:-/data/user_data/${USER}/MSP-PODCAST-Publish-2.0/Labels/labels_consensus.csv}"
OUT="${MSP_ANCHORS_OUT:-agent/emotion/data/msp_pad_anchors.npz}"

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
python eval/msp_podcast/inspect_and_build_anchors.py --csv "${CSV}" --out "${OUT}"

echo "=== done $(date -Is) ==="
