#!/usr/bin/env bash
# Convenience: download default host voice (en_US-lessac-medium).
# For multiple voices use: bash deploy/download-piper-voices.sh ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "${ROOT}/download-piper-voices.sh" "$@" lessac
