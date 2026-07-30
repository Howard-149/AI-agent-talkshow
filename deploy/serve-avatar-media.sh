#!/usr/bin/env bash
# Serve avatar assets (portraits, idle loops) + runtime speech clips on Babel.
# Laptop talkshow-web loads URLs from panel_roster (AVATAR_*_BASE_URL), not local files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${AVATAR_MEDIA_PORT:=8765}"
export AVATAR_MEDIA_PORT

exec python -m avatar.serve_media --host 0.0.0.0 --port "$AVATAR_MEDIA_PORT"
