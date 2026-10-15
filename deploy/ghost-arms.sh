#!/usr/bin/env bash
# Run ghost-session "arms" (a baseline plus latency flags) against one running stack.
#
# For each arm this restarts only the agent worker with that arm's env flags, runs
# ARMS_RUNS ghost sessions (each in a fresh room), and appends the session logs to a
# manifest. vLLM, CosyVoice and DyStream keep running, so every arm shares one node
# and GPU and the arms can be compared directly (eval/compare_arms.py).
#
# Usually started by deploy/slurm-talkshow-3gpu.sh when TALKSHOW_ARMS_FILE is set:
#   sbatch --export=ALL,TALKSHOW_ARMS_FILE=deploy/arms/latency.arms deploy/slurm-talkshow-3gpu.sh
# or by hand on a node where the sidecars are already up (no agent.main running):
#   bash deploy/ghost-arms.sh deploy/arms/latency.arms
#
# Arms file: one arm per line, "NAME [VAR=value ...]"; # starts a comment.
# Env: ARMS_RUNS (default 3), ARMS_TAG (default the SLURM job id), GHOST_ARGS (extra
#      eval.ghost_session flags), ARMS_READY_TIMEOUT (seconds, default 300).
# Re-running with the same ARMS_TAG (a requeued job keeps its id) skips finished runs.
#
# The ghost session cannot record video on Babel; to review an arm visually, run the
# stack normally with that arm's flags in .env and `python -m eval.ghost_session --record`
# from the laptop.
set -uo pipefail

ARMS_FILE="${1:?usage: ghost-arms.sh ARMS_FILE}"
RUNS="${ARMS_RUNS:-3}"
TAG="${ARMS_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M)}}"
READY_TIMEOUT="${ARMS_READY_TIMEOUT:-300}"
SESSION_DIR="${TALKSHOW_SESSION_DIR:-logs}"
OUT="logs/arms/${TAG}"
MANIFEST="${OUT}/manifest.tsv"
mkdir -p "${OUT}"
[[ -f "${MANIFEST}" ]] || printf 'arm\trun\tsession\troom\tflags\n' > "${MANIFEST}"

# On/off latency flags. Arms that don't set one get it explicitly off, so a value in
# .env (which the agent loads without overriding the environment) can't leak into
# the baseline. Pacing values (TALKSHOW_HAND_RAISE_*_SEC) are left to .env defaults.
BOOL_FLAGS=(TALKSHOW_PRERENDER_CANNED TALKSHOW_POLL_DURING_OPEN_FLOOR TALKSHOW_DRAFT_AHEAD)

newest_session() {
  ls -1t "${SESSION_DIR}"/session-*.jsonl 2>/dev/null | head -1
}

runs_done() {
  awk -F'\t' -v a="$1" 'NR > 1 && $1 == a' "${MANIFEST}" | wc -l | tr -d ' '
}

wait_ready() {  # $1 = agent log, $2 = agent pid
  local t=0
  while (( t < READY_TIMEOUT )); do
    grep -q "registered worker" "$1" 2>/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 2; t=$(( t + 2 ))
  done
  return 1
}

if grep -qE '^[^#]*TALKSHOW_PRERENDER_CANNED=1' "${ARMS_FILE}"; then
  echo "== warming the canned-line cache"
  python -m agent.session.canned_warm < /dev/null || echo "WARN: canned_warm reported failures"
fi

status=0
while read -r name flags <&3; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  flags="${flags%%#*}"
  done_n=$(runs_done "${name}")
  if (( done_n >= RUNS )); then
    echo "== arm ${name}: ${done_n}/${RUNS} runs already in the manifest, skipping"
    continue
  fi
  echo "== arm ${name}: ${flags:-(no flags)}"
  (
    for f in "${BOOL_FLAGS[@]}"; do export "${f}=0"; done
    for kv in ${flags}; do export "${kv}"; done
    agent_log="${OUT}/agent-${name}.log"
    # Own process group (setsid) so stopping the worker also stops its job processes.
    if command -v setsid > /dev/null 2>&1; then
      setsid python -m agent.main dev < /dev/null >> "${agent_log}" 2>&1 &
    else
      python -m agent.main dev < /dev/null >> "${agent_log}" 2>&1 &
    fi
    pid=$!
    trap 'kill -TERM -- -"${pid}" 2>/dev/null || { pkill -TERM -P "${pid}"; kill -TERM "${pid}"; } 2>/dev/null; wait "${pid}" 2>/dev/null' EXIT
    if ! wait_ready "${agent_log}" "${pid}"; then
      if kill -0 "${pid}" 2>/dev/null; then
        echo "ERROR: arm ${name}: worker not ready after ${READY_TIMEOUT}s (see ${agent_log})"
      else
        echo "ERROR: arm ${name}: worker exited before it was ready (see ${agent_log})"
      fi
      exit 1
    fi
    for run in $(seq $(( done_n + 1 )) "${RUNS}"); do
      room="talkshow-${USER}-${TAG}-${name}-${run}"
      before=$(newest_session)
      echo "-- arm ${name} run ${run}: room ${room}"
      # shellcheck disable=SC2086
      python -m eval.ghost_session --room "${room}" ${GHOST_ARGS:-} < /dev/null >> "${OUT}/ghost-${name}.log" 2>&1 \
        || echo "WARN: ghost session exited non-zero (arm ${name} run ${run})"
      sleep 5  # let the worker close the session log
      after=$(newest_session)
      if [[ -n "${after}" && "${after}" != "${before}" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${run}" "${after}" "${room}" "${flags:-}" >> "${MANIFEST}"
        echo "   session ${after}"
      else
        echo "WARN: arm ${name} run ${run}: no new session log; not recorded"
      fi
    done
  ) || status=1
done 3< "${ARMS_FILE}"

echo "== manifest ${MANIFEST}"
column -t -s $'\t' "${MANIFEST}" 2>/dev/null || cat "${MANIFEST}"
echo "compare with: python -m eval.compare_arms ${MANIFEST}"
exit "${status}"
