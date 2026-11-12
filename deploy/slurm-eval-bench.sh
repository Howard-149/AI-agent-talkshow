#!/usr/bin/env bash
#SBATCH --job-name=ts-eval-bench
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/slurm-eval-bench-%j.out
#SBATCH --error=slurm-logs/slurm-eval-bench-%j.err
#
# Benchmark runs (eval/bench/README.md): one or more `python -m <module> <args>` per job, separated by ";".
# Partition / QOS come from the caller (sbatch -p / --qos, as for the other wrappers in eval/README.md).
#
#   BENCH_CMD="eval.bench.run --system plain --model Qwen/Qwen2.5-7B-Instruct --split test --out eval/results/bench/plain-qwen25-7b; \
#              eval.bench.run --system plain --model Qwen/Qwen3-8B --split test --out eval/results/bench/plain-qwen3-8b" \
#     sbatch --export=ALL deploy/slurm-eval-bench.sh
#   BENCH_CMD="eval.responsiveness --out eval/results/bench/plain-qwen3-8b/score/R/T1 embed" sbatch --export=ALL deploy/slurm-eval-bench.sh
#
# Pass every model on the command line: talkshow_resolve_models only rewrites names it can see in BENCH_CMD.
# Env: BENCH_CMD (required), BENCH_OVERLAY=1 for Gemma-4 models (transformers overlay, deploy/slurm_common.sh).
set -euo pipefail
# shellcheck disable=SC1091
source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"
talkshow_cd
talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"
talkshow_source_env
talkshow_hf_cache
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"  # local snapshots only
export NCCL_P2P_DISABLE=1                     # tensor parallel > 1 hung at init on some nodes without it
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${BENCH_OVERLAY:-0}" == "1" ]]; then
  PYTHONPATH="$(talkshow_tf_overlay_pythonpath)"
  export PYTHONPATH
fi
nvidia-smi -L 2>/dev/null || true
failed=0
IFS=';' read -r -a CMDS <<< "${BENCH_CMD:?}"
for cmd in "${CMDS[@]}"; do
  [[ -z "${cmd// /}" ]] && continue
  echo "=== bench ${cmd} $(date -Is) ==="
  # shellcheck disable=SC2046
  # Keep going if one command fails (e.g. a crash during vLLM shutdown after outputs are written); report at the end.
  if ! python -m $(talkshow_resolve_models "${cmd}"); then
    echo "=== bench FAILED: ${cmd} ==="
    failed=$((failed + 1))
  fi
done
echo "=== bench done $(date -Is), ${failed} failed ==="
[[ "${failed}" == "0" ]]
