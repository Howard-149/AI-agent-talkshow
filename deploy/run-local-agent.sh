#!/usr/bin/env bash
set -euo pipefail
DEMO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DEMO_ROOT"
# Python.org's macOS interpreter may not have system CA roots installed.
# Use a verified CA bundle, never disable certificate validation.
export SSL_CERT_FILE="$("$DEMO_ROOT/.venv/bin/python" -m certifi)"
export PYTHONUNBUFFERED=1
export TALKSHOW_REACTIVE="${TALKSHOW_REACTIVE:-1}"
exec "$DEMO_ROOT/.venv/bin/python" -m agent.main dev
