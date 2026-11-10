#!/usr/bin/env bash
#SBATCH --job-name=ts-eval-ruler
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/slurm-eval-ruler-%j.out
#SBATCH --error=slurm-logs/slurm-eval-ruler-%j.err
#
# Quality ruler, one stage of `python -m eval.quality_ruler.ruler` per job (GPU stages: generate, embed, jury,
# rewrite, invert). Partition / QOS and the full command list: eval/README.md.
#
#   RULER_CMD="generate --model google/gemma-4-E4B-it -k 2 --out eval/results/ruler-x/production.jsonl" \
#     sbatch --gpus=1 --export=ALL deploy/slurm-eval-ruler.sh
#   RULER_CMD="embed --episodes eval/results/ruler-x/production.jsonl" sbatch --gpus=1 --export=ALL deploy/slurm-eval-ruler.sh
#   RULER_CMD="jury --model casperhansen/llama-3.3-70b-instruct-awq --tag llama70 --dims specific,argument,fidelity,distinct,host \
#     --repeat 0 --episodes eval/results/ruler-x/production.jsonl" sbatch --export=ALL deploy/slurm-eval-ruler.sh
#
# Pass every model on the command line: talkshow_resolve_models only rewrites names it can see in RULER_CMD.
# Env: RULER_CMD (required), RULER_OVERLAY=1 for Gemma generation (transformers overlay, deploy/slurm_common.sh).
set -euo pipefail
# shellcheck disable=SC1091
source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"
talkshow_cd
talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"
talkshow_source_env
talkshow_hf_cache
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"  # local snapshots only (unauthenticated hub calls hit rate limits)
export NCCL_P2P_DISABLE=1                     # tensor parallel > 1 hung at init on some nodes without it
export VLLM_USE_FLASHINFER_SAMPLER=0          # some nodes have no nvcc for the FlashInfer JIT sampler (logprobs only)
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${RULER_OVERLAY:-0}" == "1" ]]; then
  PYTHONPATH="$(talkshow_tf_overlay_pythonpath)"
  export PYTHONPATH
fi
nvidia-smi -L 2>/dev/null || true
echo "=== ruler ${RULER_CMD:?} $(date -Is) ==="
# shellcheck disable=SC2046
python -m eval.quality_ruler.ruler $(talkshow_resolve_models "${RULER_CMD}")
echo "=== ruler done $(date -Is) ==="
