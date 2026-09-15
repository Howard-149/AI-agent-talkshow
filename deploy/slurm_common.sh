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
