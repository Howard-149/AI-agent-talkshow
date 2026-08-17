#!/usr/bin/env bash
# Download DyStream checkpoints + tools from Hugging Face (avoids git-lfs rate limits).
#
# Prerequisite: GitHub code clone
#   export DYSTREAM_ROOT=/data/user_data/$USER/dystream
#
# Auth (required on Babel shared egress IPs):
#   export HF_TOKEN=hf_...          # https://huggingface.co/settings/tokens (read)
#   bash deploy/verify-hf-token.sh
#
# Usage:
#   bash deploy/download-dystream-weights.sh
set -euo pipefail

REPO="${DYSTREAM_HF_REPO:-robinwitch/DyStream}"
TARGET="${DYSTREAM_ROOT:-/data/user_data/${USER}/dystream}"

if [[ ! -f "$TARGET/app.py" ]]; then
  echo "DyStream code missing at $TARGET — run: bash deploy/clone-dystream.sh" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN not set — may hit rate limits on shared cluster IPs" >&2
fi

echo "Downloading DyStream weights → $TARGET"
echo "  repo: $REPO"
echo "  cache: ${HF_HUB_CACHE:-~/.cache/huggingface/hub}"

export DYSTREAM_ROOT="$TARGET"
export DYSTREAM_HF_REPO="$REPO"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

python3 <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

target = Path(os.environ["DYSTREAM_ROOT"])
repo = os.environ.get("DYSTREAM_HF_REPO", "robinwitch/DyStream")
token = os.environ.get("HF_TOKEN") or None

print(f"Using huggingface_hub snapshot_download (repo={repo})")
try:
    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        allow_patterns=["checkpoints/*", "tools/*"],
        token=token,
    )
except TypeError:
    # older huggingface_hub without local_dir + allow_patterns combo
    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        allow_patterns=["checkpoints/*", "tools/*"],
        token=token,
        local_dir_use_symlinks=False,
    )

ckpt = target / "checkpoints" / "last.ckpt"
tools = target / "tools"
if not ckpt.is_file():
    print(f"Missing: {ckpt}", file=sys.stderr)
    if (target / "checkpoints").is_dir():
        print("checkpoints contents:", list((target / "checkpoints").iterdir()), file=sys.stderr)
    sys.exit(1)
if not tools.is_dir() or not any(tools.iterdir()):
    print(f"Missing or empty: {tools}", file=sys.stderr)
    sys.exit(1)

print(f"OK {ckpt} ({ckpt.stat().st_size // (1024*1024)} MiB)")
print(f"OK {tools}/ ({len(list(tools.rglob('*')))} files)")
PY

echo "Done. DYSTREAM_ROOT=$TARGET"
