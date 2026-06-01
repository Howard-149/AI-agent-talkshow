#!/usr/bin/env bash
# LiveKit agent runtime assets (Silero VAD + turn-detector ONNX e.g. model_q8.onnx).
# Run once on cluster after: pip install -r requirements.txt
set -euo pipefail
# HF_HUB_CACHE / HF_DATASETS_CACHE: cluster ~/.bashrc (do not duplicate here)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! python -c "import livekit.agents" 2>/dev/null; then
  echo "Activate conda env talkshow and: pip install -r requirements.txt" >&2
  exit 1
fi

echo "HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"
echo "Downloading LiveKit agent models (turn detector, silero, etc.)..."
python -m livekit.agents download-files

echo ""
echo "Verify turn-detector ONNX:"
if [[ -n "${HF_HUB_CACHE:-}" ]]; then
  find "${HF_HUB_CACHE}" -name 'model_q8.onnx' 2>/dev/null | head -3 || true
else
  find "${HOME}/.cache/huggingface" -name 'model_q8.onnx' 2>/dev/null | head -3 || true
fi

if [[ -z "${HF_HUB_CACHE:-}" ]]; then
  echo "WARNING: HF_HUB_CACHE unset — models may be in ~/.cache; set ~/.bashrc or agent will use /data/hf_cache/hub" >&2
fi

echo ""
echo "Done. Start agent: source .env && python -m agent.main dev"
