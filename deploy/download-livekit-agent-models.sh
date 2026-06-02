#!/usr/bin/env bash
# LiveKit agent runtime assets (Silero VAD + turn-detector ONNX e.g. model_q8.onnx).
# Run once on cluster after: pip install -r requirements.txt
#
# Stores turn-detector (and related agent ONNX) in TALKSHOW_TURN_DETECTOR_CACHE
# (private user_data), not the shared /data/hf_cache that gets purged.
#
#   cd ~/AI-agent-talkshow && source .env && bash deploy/download-livekit-agent-models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
  echo "Loaded ${ROOT}/.env"
fi

USER_NAME="${USER:-}"
TURN_CACHE="${TALKSHOW_TURN_DETECTOR_CACHE:-}"
if [[ -z "${TURN_CACHE}" && -d /data/user_data && -n "${USER_NAME}" ]]; then
  TURN_CACHE="/data/user_data/${USER_NAME}/livekit-turn-detector/hub"
fi

if [[ -z "${TURN_CACHE}" ]]; then
  echo "Set TALKSHOW_TURN_DETECTOR_CACHE in .env (e.g. /data/user_data/\$USER/livekit-turn-detector/hub)" >&2
  exit 1
fi

TURN_CACHE="${TURN_CACHE//\$\{USER\}/${USER_NAME}}"
TURN_CACHE="${TURN_CACHE//\$USER/${USER_NAME}}"
mkdir -p "${TURN_CACHE}"

if ! python -c "import livekit.agents" 2>/dev/null; then
  echo "Activate conda env talkshow and: pip install -r requirements.txt" >&2
  exit 1
fi

echo "TALKSHOW_TURN_DETECTOR_CACHE=${TURN_CACHE}"
echo "Downloading LiveKit agent models into private cache (not shared /data/hf_cache)..."
HF_HUB_CACHE="${TURN_CACHE}" HUGGINGFACE_HUB_CACHE="${TURN_CACHE}" \
  python -m livekit.agents download-files

echo ""
echo "Verify turn-detector ONNX:"
FOUND=()
while IFS= read -r f; do
  FOUND+=("$f")
done < <(find "${TURN_CACHE}" -name 'model_q8.onnx' 2>/dev/null | head -5)

if ((${#FOUND[@]} == 0)); then
  echo "ERROR: model_q8.onnx not found under ${TURN_CACHE}" >&2
  echo "  Re-run: source .env && bash deploy/download-livekit-agent-models.sh" >&2
  echo "  Or bypass: add TALKSHOW_TURN_DETECTOR=vad to .env (VAD-only, less accurate)" >&2
  exit 1
fi

printf '  %s\n' "${FOUND[@]}"

echo ""
echo "Done. Agent will use this cache via bootstrap (TALKSHOW_TURN_DETECTOR_CACHE)."
echo "Start agent: source .env && python -m agent.main dev"
