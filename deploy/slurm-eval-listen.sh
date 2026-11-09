#!/usr/bin/env bash
#SBATCH --job-name=ts-eval-listen
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/slurm-eval-listen-%j.out
#SBATCH --error=slurm-logs/slurm-eval-listen-%j.err
#
# Listening test, scoring part: GPU stages on one results dir, in order. Run `python -m eval.responsiveness --out DIR
# collect <log dirs>` on the login node first (or use the generate stage for the model-only version).
# Partition / QOS: see eval/README.md.
#
#   LISTEN_OUT=eval/results/listen sbatch --export=ALL deploy/slurm-eval-listen.sh                  # embed logprob report
#   LISTEN_OUT=eval/results/listen LISTEN_STAGES=judge sbatch --gpus=2 --export=ALL deploy/slurm-eval-listen.sh
#   LISTEN_OUT=eval/results/listen-model LISTEN_STAGES="generate embed logprob report" \
#     sbatch --export=ALL deploy/slurm-eval-listen.sh                                             # model-only version
#
# Stages: generate (eval.model_only, Gemma), embed + logprob + report (eval.responsiveness), judge (eval.judge_context,
# 70B-class model, tensor parallel 2: submit with --gpus=2).
# Env: LISTEN_OUT (results dir), LISTEN_STAGES (default "embed logprob report"), LISTEN_K (generate, default 5),
#      LISTEN_SCAFFOLD (generate: system | minimal | both), LISTEN_GEN_ARGS (extra eval.model_only generate flags,
#      e.g. "--arm concrete"), LISTEN_RESP_ARGS (extra
#      eval.responsiveness global flags), JUDGE_MODEL (default Qwen/Qwen2.5-72B-Instruct-AWQ), JUDGE_ARGS.
#
set -euo pipefail

# shellcheck disable=SC1091
source "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}/deploy/slurm_common.sh"

echo "=== talkshow eval listen $(date -Is) ==="
talkshow_cd
talkshow_activate "${TALKSHOW_CONDA_ENV:-talkshow}"
talkshow_source_env
talkshow_hf_cache
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
nvidia-smi -L 2>/dev/null || true

OUT="${LISTEN_OUT:?set LISTEN_OUT to the results dir}"
for stage in ${LISTEN_STAGES:-embed logprob report}; do
  echo "--- ${stage} $(date -Is)"
  case "${stage}" in
    generate)
      PYTHONPATH="$(talkshow_tf_overlay_pythonpath)" \
        python -m eval.model_only --out "${OUT}" --scaffold "${LISTEN_SCAFFOLD:-system}" generate -k "${LISTEN_K:-5}" \
        --model "${VLLM_MODEL:-google/gemma-4-E4B-it}" ${LISTEN_GEN_ARGS:-}
      ;;
    embed|logprob|report)
      python -m eval.responsiveness --out "${OUT}" ${LISTEN_RESP_ARGS:-} "${stage}"
      ;;
    judge)
      python -m eval.judge_context --resp "${OUT}" \
        --model "$(talkshow_resolve_models "${JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}")" --tp 2 ${JUDGE_ARGS:-}
      ;;
    *)
      echo "unknown stage ${stage}" >&2
      exit 2
      ;;
  esac
done
echo "=== eval listen done $(date -Is) ==="
