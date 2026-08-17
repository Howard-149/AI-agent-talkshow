#!/usr/bin/env bash
# Download Piper voices from Hugging Face (rhasspy/piper-voices).
#
# Quick start (on Babel):
#   bash deploy/download-piper-voices.sh en           # lessac + amy + ryan
#   bash deploy/download-piper-voices.sh zh           # huayan
#   bash deploy/download-piper-voices.sh en zh        # both defaults
#   bash deploy/download-piper-voices.sh --list
#
# Multiple voices (all positional args are presets / short names / full ids):
#   bash deploy/download-piper-voices.sh lessac amy ryan
#   bash deploy/download-piper-voices.sh huayan chaowen xiao_ya
#   bash deploy/download-piper-voices.sh en_US-kristin-medium zh_CN-xiao_ya-medium
#
# Pick any voice yourself:
#   1. Browse https://huggingface.co/rhasspy/piper-voices  (en/… or zh/zh_CN/…)
#   2. Voice id = {locale}-{speaker}-{quality}
#      e.g. en_US-kristin-medium   zh_CN-xiao_ya-medium   zh_CN-huayan-x_low
#   3. Pass one or more ids / short names / presets on the same line.
#   4. Point .env at the printed .onnx path(s) (see “After download” below).
#
# Presets:     en → lessac amy ryan    zh → huayan
# Short names: lessac amy ryan huayan chaowen xiao_ya  (see --list)
#
# After download, set in cluster .env:
#   English roles:
#     PIPER_MODEL_PATH=.../en_US-lessac-medium.onnx
#     PIPER_MODEL_PATH_GUEST=.../en_US-amy-medium.onnx
#     PIPER_MODEL_PATH_COMMENTATOR=.../en_US-ryan-medium.onnx
#   Chinese (used only when a zh viewer is in the room):
#     PIPER_MODEL_PATH_ZH=.../zh_CN-huayan-medium.onnx
#     # optional per-role Chinese voices:
#     PIPER_MODEL_PATH_ZH_HOST=...
#     PIPER_MODEL_PATH_ZH_GUEST=...
#     PIPER_MODEL_PATH_ZH_COMMENTATOR=...
set -euo pipefail

REPO="rhasspy/piper-voices"
DEST="${PIPER_DIR:-/data/user_data/${USER}/piper}"

usage() {
  sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --dest DIR     Output directory (default: $PIPER_DIR or /data/user_data/$USER/piper)
  --repo REPO    Hugging Face repo (default: rhasspy/piper-voices)
  --list         Print short names + common voice IDs and exit
  -h, --help     Show this help

How to choose a voice
  Official catalog: https://huggingface.co/rhasspy/piper-voices
  Folder layout:    {lang}/{locale}/{speaker}/{quality}/{id}.onnx
  Voice id format:  {locale}-{speaker}-{quality}

  English examples (https://huggingface.co/rhasspy/piper-voices/tree/main/en):
    en_US-lessac-medium              host (male, US) — default
    en_US-amy-medium                 guest (female, US) — default
    en_US-ryan-medium                commentator (male, US) — default
    en_US-kristin-medium             female, US
    en_US-joe-medium                 male, US
    en_GB-alan-medium                male, UK

  Chinese examples (https://huggingface.co/rhasspy/piper-voices/tree/main/zh/zh_CN):
    zh_CN-huayan-medium              female — default for all ZH roles
    zh_CN-huayan-x_low               smaller / faster
    zh_CN-chaowen-medium
    zh_CN-xiao_ya-medium

  Then wire the .onnx into .env (PIPER_MODEL_PATH* / PIPER_MODEL_PATH_ZH*).
  Each .onnx needs a sibling .onnx.json (this script downloads both).

  Multiple ids on one line are all downloaded (not only the first argument).

HF hub cache: uses your ~/.bashrc HF_HUB_CACHE (not set in this script).
EOF
}

# short name → full voice id
resolve_short_name() {
  local key
  key="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$key" in
    lessac) echo "en_US-lessac-medium" ;;
    amy) echo "en_US-amy-medium" ;;
    ryan) echo "en_US-ryan-medium" ;;
    kristin) echo "en_US-kristin-medium" ;;
    joe) echo "en_US-joe-medium" ;;
    alan) echo "en_GB-alan-medium" ;;
    huayan) echo "zh_CN-huayan-medium" ;;
    chaowen) echo "zh_CN-chaowen-medium" ;;
    xiao_ya | xiaoya | xiao-ya) echo "zh_CN-xiao_ya-medium" ;;
    *) echo "" ;;
  esac
}

COMMON_VOICES=(
  "Presets"
  "  en         lessac + amy + ryan"
  "  zh         huayan"
  ""
  "English (short → full id)"
  "  lessac     en_US-lessac-medium     # host (male, US)"
  "  amy        en_US-amy-medium        # guest (female, US)"
  "  ryan       en_US-ryan-medium       # commentator (male, US)"
  "  kristin    en_US-kristin-medium"
  "  joe        en_US-joe-medium"
  "  alan       en_GB-alan-medium"
  "  en_GB-southern_english_female-low"
  ""
  "Chinese (short → full id)"
  "  huayan     zh_CN-huayan-medium     # default ZH (female)"
  "  chaowen    zh_CN-chaowen-medium"
  "  xiao_ya    zh_CN-xiao_ya-medium"
  "  zh_CN-huayan-x_low                 # smaller / faster"
)

# Preset / short name / full id → one or more tokens for normalize_voice_id
expand_voice_arg() {
  local key
  key="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$key" in
    en | english)
      printf '%s\n' lessac amy ryan
      ;;
    zh | chinese | cn)
      printf '%s\n' huayan
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

# Resolve short name or full voice id (locale-speaker-quality)
normalize_voice_id() {
  local v="$1"
  v="${v%.onnx}"
  local mapped
  mapped="$(resolve_short_name "$v")"
  if [[ -n "$mapped" ]]; then
    echo "$mapped"
    return
  fi
  if [[ "$v" == *-*-* ]]; then
    echo "$v"
    return
  fi
  echo "Unknown short name '$v'. Use a full id (e.g. en_US-kristin-medium) or --list." >&2
  return 1
}

# voice id → paths inside HF repo
voice_hf_paths() {
  local id="$1"
  if [[ ! "$id" =~ ^([a-z]{2}_[A-Z]{2})-(.+)-([^-]+)$ ]]; then
    echo "Invalid voice id: $id (expected e.g. en_US-lessac-medium or zh_CN-huayan-medium)" >&2
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

env_hint_for_voice() {
  local vid="$1"
  case "$vid" in
    *lessac*) echo "PIPER_MODEL_PATH=${DEST}/${vid}.onnx" ;;
    *amy*) echo "PIPER_MODEL_PATH_GUEST=${DEST}/${vid}.onnx" ;;
    *ryan*) echo "PIPER_MODEL_PATH_COMMENTATOR=${DEST}/${vid}.onnx" ;;
    zh_CN-huayan-medium)
      echo "PIPER_MODEL_PATH_ZH=${DEST}/${vid}.onnx"
      echo "# optional per-role ZH (else all ZH roles use PIPER_MODEL_PATH_ZH / locales.zh default):"
      echo "# PIPER_MODEL_PATH_ZH_HOST=${DEST}/${vid}.onnx"
      echo "# PIPER_MODEL_PATH_ZH_GUEST=${DEST}/${vid}.onnx"
      echo "# PIPER_MODEL_PATH_ZH_COMMENTATOR=${DEST}/${vid}.onnx"
      ;;
    zh_CN-*)
      echo "# Chinese voice — assign to a ZH env var, e.g.:"
      echo "# PIPER_MODEL_PATH_ZH=${DEST}/${vid}.onnx"
      echo "# PIPER_MODEL_PATH_ZH_HOST=${DEST}/${vid}.onnx"
      echo "# PIPER_MODEL_PATH_ZH_GUEST=${DEST}/${vid}.onnx"
      echo "# PIPER_MODEL_PATH_ZH_COMMENTATOR=${DEST}/${vid}.onnx"
      ;;
    *)
      echo "# ${DEST}/${vid}.onnx"
      echo "# Point PIPER_MODEL_PATH / _GUEST / _COMMENTATOR (en) or PIPER_MODEL_PATH_ZH* (zh) at this file."
      ;;
  esac
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
      echo "Common Piper voices (${REPO})"
      echo "Catalog: https://huggingface.co/${REPO}"
      echo ""
      printf '%s\n' "${COMMON_VOICES[@]}"
      echo ""
      echo "Default dest: ${PIPER_DIR:-/data/user_data/\$USER/piper}"
      echo "Examples:     bash deploy/download-piper-voices.sh en zh"
      echo "              bash deploy/download-piper-voices.sh huayan chaowen"
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

EXPANDED=()
for raw in "${VOICES[@]}"; do
  while IFS= read -r item; do
    [[ -n "$item" ]] && EXPANDED+=("$item")
  done < <(expand_voice_arg "$raw")
done
VOICES=("${EXPANDED[@]}")

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
  vid=$(normalize_voice_id "$raw") || continue
  env_hint_for_voice "$vid"
done

exit "$FAILED"
