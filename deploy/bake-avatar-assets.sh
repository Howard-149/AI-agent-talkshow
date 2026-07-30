#!/usr/bin/env bash
# Pre-bake idle motion loops (fixed assets) into AVATAR_ASSETS_DIR/loops/.
# Reads DYSTREAM_ROOT, AVATAR_ASSETS_DIR, DYSTREAM_PYTHON from repo .env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON="${DYSTREAM_PYTHON:-python}"
: "${DYSTREAM_CUDA_DEVICE:=1}"
export CUDA_VISIBLE_DEVICES="${DYSTREAM_CUDA_DEVICE}"

read -r ASSETS LOOPS DYSTREAM <<EOF
$("$PYTHON" - <<'PY'
from avatar.paths import avatar_assets_dir, dystream_root, load_repo_env

load_repo_env()
root = dystream_root()
assets = avatar_assets_dir()
print(assets, assets / "loops", root)
PY
)
EOF

mkdir -p "$ASSETS/portraits" "$LOOPS"

echo "DyStream root: $DYSTREAM"
echo "Assets:        $ASSETS"
echo "Idle loops:    $LOOPS"
echo ""

bake_idle() {
  local name="$1"
  local portrait="$ASSETS/portraits/${name}.png"
  local out="$LOOPS/${name}-idle.mp4"
  if [[ ! -f "$portrait" ]]; then
    echo "skip $name — missing $portrait" >&2
    return 0
  fi
  if [[ -f "$out" ]]; then
    echo "exists $out"
    return 0
  fi
  echo "baking idle → $out"
  "$PYTHON" -m avatar.bake idle \
    --portrait "$portrait" \
    --output "$out" \
    --duration 3.0 \
    --steps 5
}

for role in lessac ryan amy; do
  bake_idle "$role"
done

echo "Done. Idle loops in $LOOPS"
