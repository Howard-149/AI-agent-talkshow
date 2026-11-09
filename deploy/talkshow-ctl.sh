#!/usr/bin/env bash
# Control a running talkshow SLURM job without resubmitting (login node is fine).
#
#   deploy/talkshow-ctl.sh restart agent [JOBID]      # new code or .env; models stay loaded
#   deploy/talkshow-ctl.sh restart cosyvoice [JOBID]  # also: dystream | vllm (reloads that model)
#   deploy/talkshow-ctl.sh events [JOBID]             # supervisor log (starts, restarts, exits)
#
# JOBID defaults to your running "talkshow" job. The job's supervisor polls
# logs/slurm-<JOBID>/control/ every 2 s; agent restarts re-read .env.
# The agent restart drops any session in progress — run it between ghost runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmd="${1:-}"
shift || true

job_id() {
  if [[ -n "${1:-}" ]]; then
    echo "$1"
    return
  fi
  local ids
  ids="$(squeue -u "${USER}" -n talkshow -t RUNNING -h -o %i)"
  if [[ -z "${ids}" ]]; then
    echo "no running talkshow job; pass JOBID" >&2
    exit 1
  fi
  if [[ "$(wc -l <<<"${ids}")" -gt 1 ]]; then
    echo "several running talkshow jobs (${ids//$'\n'/ }); pass JOBID" >&2
    exit 1
  fi
  echo "${ids}"
}

case "${cmd}" in
  restart)
    svc="${1:-}"
    case "${svc}" in
      agent|cosyvoice|dystream|vllm) ;;
      *)
        echo "usage: $0 restart agent|cosyvoice|dystream|vllm [JOBID]" >&2
        exit 2
        ;;
    esac
    job="$(job_id "${2:-}")"
    ctl="${ROOT}/logs/slurm-${job}/control"
    if [[ ! -d "${ctl}" ]]; then
      echo "${ctl} missing: job ${job} is still starting, or predates the supervisor" >&2
      exit 1
    fi
    events="${ctl}/events.log"
    agent_log="${ROOT}/logs/slurm-${job}/agent.log"
    n_events="$(wc -l <"${events}" 2>/dev/null || echo 0)"
    n_registered="$(grep -c "registered worker" "${agent_log}" 2>/dev/null || true)"
    touch "${ctl}/restart-${svc}"
    echo "requested restart of ${svc} in job ${job}; waiting …"
    # Model restarts take minutes (CosyVoice ~5–7, DyStream ~3, vLLM up to 15).
    wait_s=1200
    [[ "${svc}" == agent ]] && wait_s=180
    deadline=$((SECONDS + wait_s))
    while ((SECONDS < deadline)); do
      if tail -n +"$((n_events + 1))" "${events}" 2>/dev/null | grep -q "restart ${svc}: FAILED"; then
        echo "restart failed; the job is stopping. See ${events}" >&2
        exit 1
      fi
      if tail -n +"$((n_events + 1))" "${events}" 2>/dev/null | grep -q "restart ${svc}: done"; then
        if [[ "${svc}" != agent ]]; then
          echo "${svc} is back up"
          exit 0
        fi
        n_now="$(grep -c "registered worker" "${agent_log}" 2>/dev/null || true)"
        if ((${n_now:-0} > ${n_registered:-0})); then
          tail -n +"$((n_events + 1))" "${events}" | grep "agent started" | tail -1
          echo "agent registered"
          exit 0
        fi
      fi
      sleep 2
    done
    echo "timed out waiting for ${svc}; see ${events}" >&2
    exit 1
    ;;
  events)
    job="$(job_id "${1:-}")"
    cat "${ROOT}/logs/slurm-${job}/control/events.log"
    ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
