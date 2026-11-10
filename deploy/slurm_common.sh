#!/usr/bin/env bash
# Shared helpers for SLURM batch jobs (source from deploy/slurm-*.sh).
# Not submitted directly.

set -euo pipefail

talkshow_root() {
  echo "${TALKSHOW_ROOT:-${HOME}/AI-agent-talkshow}"
}

talkshow_cd() {
  local root
  root="$(talkshow_root)"
  if [[ ! -d "${root}" ]]; then
    echo "FATAL: missing ${root}" >&2
    exit 1
  fi
  cd "${root}"
  mkdir -p slurm-logs logs
  echo "ROOT=${root} pwd=$(pwd) host=$(hostname) job=${SLURM_JOB_ID:-local}"
}

talkshow_conda_sh() {
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    echo "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "/data/user_data/${USER}/miniconda3/etc/profile.d/conda.sh" ]]; then
    echo "/data/user_data/${USER}/miniconda3/etc/profile.d/conda.sh"
  else
    echo "FATAL: conda.sh not found under ~/miniconda3 or /data/user_data/${USER}/miniconda3" >&2
    exit 1
  fi
}

talkshow_activate() {
  local env_name="${1:-${TALKSHOW_CONDA_ENV:-talkshow}}"
  local conda_sh
  conda_sh="$(talkshow_conda_sh)"
  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${env_name}"
  echo "conda env=${env_name} python=$(command -v python) $(python -V 2>&1)"
}

talkshow_source_env() {
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "sourced .env"
  else
    echo "WARN: no .env in $(pwd)" >&2
  fi
}

# Hugging Face cache for eval jobs: HF_HUB_CACHE if set, else ${HF_HOME:-~/.cache/huggingface}/hub.
talkshow_hf_cache() {
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"
  mkdir -p "${HF_HUB_CACHE}"
  echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
}

# vLLM + Gemma-4 needs transformers 5.13 while the env may have a newer one. Prints a PYTHONPATH with
# VLLM_TF_OVERLAY (default ~/.local/vllm-tf-overlay, made with `pip install --target <dir> transformers==5.13.*`)
# in front when that directory exists, else the current PYTHONPATH. Use it for the vLLM process only.
talkshow_tf_overlay_pythonpath() {
  local overlay="${VLLM_TF_OVERLAY:-${HOME}/.local/vllm-tf-overlay}"
  if [[ -d "${overlay}" ]]; then
    echo "${overlay}${PYTHONPATH:+:${PYTHONPATH}}"
  else
    echo "${PYTHONPATH:-}"
  fi
}

# With HF_HUB_OFFLINE=1, loading by repo id fails on snapshots that lack metadata files
# (IncompleteSnapshotError) even though the weights are cached. Replace every "org/name" word that has a
# local snapshot in HF_HUB_CACHE with that snapshot's path (the first one holding config.json).
talkshow_resolve_models() {
  local out=() w d snap s
  for w in $1; do
    if [[ "${HF_HUB_OFFLINE:-0}" == "1" && "$w" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
      d="${HF_HUB_CACHE:-}/models--${w//\//--}/snapshots"
      snap=""
      if [[ -d "$d" ]]; then
        for s in "$d"/*/; do
          if [[ -f "${s}config.json" ]]; then snap="${s%/}"; break; fi
        done
      fi
      if [[ -n "$snap" ]]; then w="$snap"; fi
    fi
    out+=("$w")
  done
  echo "${out[*]}"
}
