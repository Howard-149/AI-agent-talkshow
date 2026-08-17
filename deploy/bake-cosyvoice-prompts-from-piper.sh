#!/usr/bin/env bash
# Bake CosyVoice clone prompt WAVs from Piper voices (Lessac/Amy/Ryan).
#
#   bash deploy/bake-cosyvoice-prompts-from-piper.sh
#   bash deploy/bake-cosyvoice-prompts-from-piper.sh --dest /tmp/prompts  # optional
#
# Output: <repo>/cosyvoice-prompts/{role}.wav (+ legacy en_{role}.wav copy).
# One ref per role — language-agnostic for instruct2 (zh/en share the same timbre).
# Requires talkshow env with piper-tts (or piper CLI). Reads PIPER_MODEL_PATH* from .env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON="${COSYVOICE_PROMPT_BAKE_PYTHON:-${TALKSHOW_PYTHON:-python}}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON}" >&2
  exit 1
fi

exec "${PYTHON}" "${ROOT}/deploy/bake_cosyvoice_prompts_from_piper.py" "$@"
