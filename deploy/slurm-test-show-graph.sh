#!/usr/bin/env bash
#SBATCH --job-name=ts-sg-test
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/hsuanhal/AI-agent-talkshow
#SBATCH --output=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-show-graph-%j.out
#SBATCH --error=/home/hsuanhal/AI-agent-talkshow/slurm-logs/slurm-show-graph-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hsuanhal@andrew.cmu.edu
#
# LangGraph floor planner parity tests (no LiveKit).
# Requests 1 GPU (cluster policy: no CPU-only preempt jobs).
#
# Submit:  sbatch deploy/slurm-test-show-graph.sh
# Optional JSONL compare:
#   FLOOR_JSONL_REF=logs/session-REF.jsonl FLOOR_JSONL_NEW=logs/session-NEW.jsonl \
#     sbatch --export=ALL deploy/slurm-test-show-graph.sh
#
set -euo pipefail

# SLURM copies this script to /var/spool/slurmd/… — do not resolve helpers via BASH_SOURCE.
# shellcheck disable=SC1091
source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"

echo "=== talkshow show_graph tests $(date -Is) ==="
talkshow_cd
talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L 2>/dev/null || true

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

echo "unittest tests.show_graph.test_parity …"
python -m unittest tests.show_graph.test_parity -v

if [[ -n "${FLOOR_JSONL_REF:-}" && -n "${FLOOR_JSONL_NEW:-}" ]]; then
  echo "JSONL floor event order: ${FLOOR_JSONL_REF} vs ${FLOOR_JSONL_NEW}"
  python tests/show_graph/check_floor_jsonl_order.py \
    "${FLOOR_JSONL_REF}" "${FLOOR_JSONL_NEW}"
else
  echo "Skip JSONL compare (set FLOOR_JSONL_REF + FLOOR_JSONL_NEW to enable)"
fi

echo "=== show_graph tests done $(date -Is) ==="
