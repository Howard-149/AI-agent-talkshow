#!/usr/bin/env bash
# Download one or more Piper voices from Hugging Face (rhasspy/piper-voices).
#
# Examples:
#   bash deploy/download-piper-voices.sh lessac amy ryan
#   bash deploy/download-piper-voices.sh --dest /data/user_data/$USER/piper en_US-lessac-medium en_US-amy-medium
#   PIPER_DIR=/other/path bash deploy/download-piper-voices.sh --list
#
# Short names (en_US, medium) map to: en_US-<name>-medium
# Full IDs: en_US-lessac-medium, en_GB-alan-medium, ...
#
# After download, set in cluster .env:
#   PIPER_MODEL_PATH=.../en_US-lessac-medium.onnx
#   PIPER_MODEL_PATH_GUEST=.../en_US-amy-medium.onnx
#   PIPER_MODEL_PATH_COMMENTATOR=.../en_US-ryan-medium.onnx
set -euo pipefail

REPO="rhasspy/piper-voices"
DEST="${PIPER_DIR:-/data/user_data/${USER}/piper}"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --dest DIR     Output directory (default: $PIPER_DIR or /data/user_data/$USER/piper)
  --repo REPO    Hugging Face repo (default: rhasspy/piper-voices)
  --list         Print common voice IDs and exit
  -h, --help     Show this help

Voice arguments (one or more):
  lessac                    → en_US-lessac-medium
  amy                       → en_US-amy-medium
  en_US-ryan-medium         full Piper voice id (see --list / HF repo)

HF hub cache: uses your ~/.bashrc HF_HUB_CACHE (not set in this script).
EOF
}

COMMON_VOICES=(
  "en_US-lessac-medium   # host (male, US)"
  "en_US-amy-medium       # guest (female, US)"
  "en_US-ryan-medium      # commentator (male, US)"
  "en_US-kristin-medium"
  "en_US-joe-medium"
  "en_GB-alan-medium"
  "en_GB-southern_english_female-low"
)

# Resolve short name → full voice id (locale-speaker-quality)
normalize_voice_id() {
  local v="$1"
  v="${v%.onnx}"
  if [[ "$v" == *-*-* ]]; then
    echo "$v"
    return
  fi
  echo "en_US-${v}-medium"
}

# voice id → paths inside HF repo
voice_hf_paths() {
  local id="$1"
  if [[ ! "$id" =~ ^([a-z]{2}_[A-Z]{2})-(.+)-([^-]+)$ ]]; then
    echo "Invalid voice id: $id (expected e.g. en_US-lessac-medium)" >&2
    return 1
  fi
  local locale="${BASH_REMATCH[1]}"
  local speaker="${BASH_REMATCH[2]}"
  local quality="${BASH_REMATCH[3]}"
  local lang="${locale%%_*}"
  echo "${lang}/${locale}/${speaker}/${quality}/${id}.onnx"
  echo "${lang}/${locale}/${speaker}/${quality}/${id}.onnx.json"
}

hf_download_cmd() {
  if command -v hf &>/dev/null; then
    echo "hf download"
  elif command -v huggingface-cli &>/dev/null; then
    echo "Warning: huggingface-cli is deprecated; use 'hf' from huggingface_hub." >&2
    echo "huggingface-cli download"
  else
    echo "Install: pip install -U huggingface_hub" >&2
    return 1
  fi
}

download_voice() {
  local voice_id="$1"
  local paths onnx_rel json_rel
  mapfile -t paths < <(voice_hf_paths "$voice_id") || return 1
  onnx_rel="${paths[0]}"
  json_rel="${paths[1]}"

  local onnx_out="${DEST}/${voice_id}.onnx"
  local json_out="${DEST}/${voice_id}.onnx.json"

  if [[ -f "$onnx_out" && -f "$json_out" ]]; then
    echo "Already present: $onnx_out"
    return 0
  fi

  local hf_cmd
  hf_cmd=$(hf_download_cmd) || return 1
  read -r -a HF_CMD <<< "$hf_cmd"

  local tmp="${DEST}/.download_tmp_${voice_id}"
  rm -rf "$tmp"
  mkdir -p "$tmp" "$DEST"

  echo "Downloading ${voice_id} from ${REPO} → ${DEST}"
  "${HF_CMD[@]}" "$REPO" "$onnx_rel" "$json_rel" --local-dir "$tmp"

  if [[ -f "${tmp}/${onnx_rel}" ]]; then
    cp -f "${tmp}/${onnx_rel}" "$onnx_out"
    cp -f "${tmp}/${json_rel}" "$json_out"
  else
    find "$tmp" -name "${voice_id}.onnx" -exec cp -f {} "$DEST/" \;
    find "$tmp" -name "${voice_id}.onnx.json" -exec cp -f {} "$DEST/" \;
  fi
  rm -rf "$tmp"

  if [[ ! -f "$onnx_out" ]]; then
    echo "Download failed: missing $onnx_out" >&2
    echo "Check voice id at https://huggingface.co/${REPO}/tree/main" >&2
    return 1
  fi
  echo "  OK: $onnx_out"
}

# --- parse args ---
VOICES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --list)
      echo "Common Piper voice IDs (${REPO}):"
      printf '  %s\n' "${COMMON_VOICES[@]}"
      echo ""
      echo "Default dest: ${PIPER_DIR:-/data/user_data/\$USER/piper}"
      exit 0
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      VOICES+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      VOICES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#VOICES[@]} -eq 0 ]]; then
  echo "No voices specified." >&2
  usage >&2
  exit 1
fi

mkdir -p "$DEST"
echo "DEST=${DEST}"
echo "REPO=${REPO}"
echo ""

FAILED=0
for raw in "${VOICES[@]}"; do
  vid=$(normalize_voice_id "$raw") || { FAILED=1; continue; }
  download_voice "$vid" || FAILED=1
  echo ""
done

echo "Done. Files in ${DEST}:"
ls -lh "${DEST}"/*.onnx 2>/dev/null || true
echo ""
echo "Example .env:"
for raw in "${VOICES[@]}"; do
  vid=$(normalize_voice_id "$raw")
  case "$vid" in
    *lessac*) echo "PIPER_MODEL_PATH=${DEST}/${vid}.onnx" ;;
    *amy*) echo "PIPER_MODEL_PATH_GUEST=${DEST}/${vid}.onnx" ;;
    *ryan*) echo "PIPER_MODEL_PATH_COMMENTATOR=${DEST}/${vid}.onnx" ;;
    *) echo "# ${DEST}/${vid}.onnx" ;;
  esac
done

exit "$FAILED"
