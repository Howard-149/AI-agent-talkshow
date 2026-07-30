#!/usr/bin/env bash
# Pull DyStream idle loops + portraits from Babel into the single source tree:
#   avatar/assets/{loops,portraits}/
# talkshow-web/public/avatars/{loops,portraits} are symlinks — no second copy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${BABEL_HOST:-babel}"
REMOTE="${BABEL_REPO:-~/AI-agent-talkshow}"

LOOPS_DST="$ROOT/avatar/assets/loops"
PORTRAITS_DST="$ROOT/avatar/assets/portraits"
mkdir -p "$LOOPS_DST" "$PORTRAITS_DST"

echo "From ${HOST}:${REMOTE}/avatar/assets → ${ROOT}/avatar/assets"
scp "${HOST}:${REMOTE}/avatar/assets/loops/"*-idle.mp4 "$LOOPS_DST/"
scp "${HOST}:${REMOTE}/avatar/assets/portraits/"{lessac,ryan,amy}.png "$PORTRAITS_DST/" || \
  scp "${HOST}:${REMOTE}/avatar/assets/portraits/"*.png "$PORTRAITS_DST/"

echo "Done. Frontend: talkshow-web/public/avatars/{loops,portraits} → avatar/assets (symlink)."
ls -la "$LOOPS_DST"/*-idle.mp4 "$PORTRAITS_DST"/{lessac,ryan,amy}.png 2>/dev/null || true
