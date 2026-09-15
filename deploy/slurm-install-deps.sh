#!/usr/bin/env bash
#SBATCH --job-name=ts-deps
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/hsuanhal/AI-agent-talkshow
#SBATCH --output=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-deps-%j.out
#SBATCH --error=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-deps-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hsuanhal@andrew.cmu.edu
#
# Install / refresh talkshow conda deps (incl. langgraph) on a preempt GPU node.
# Requests 1 GPU (cluster policy: no CPU-only preempt jobs).
#
# Submit:  sbatch deploy/slurm-install-deps.sh
#
set -euo pipefail

# SLURM copies this script to /var/spool/slurmd/… — do not resolve helpers via BASH_SOURCE.
# shellcheck disable=SC1091
source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"

echo "=== talkshow install-deps $(date -Is) ==="
talkshow_cd
talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L 2>/dev/null || true

echo "pip install -r requirements.txt …"
pip install -r requirements.txt

echo "verify langgraph …"
python -c "from langgraph.graph import StateGraph, END, START; print('langgraph OK')"

echo "=== install-deps done $(date -Is) ==="
