# Shared settings for the AVTR-1 scripts (source it). Everything large lives under
# AVTR1_HOME (default /data/user_data/$USER), not in home.
: "${AVTR1_HOME:=/data/user_data/${USER}}"
: "${AVTR1_REPO:=${AVTR1_HOME}/avtr-1}"
export AVTR1_HOME AVTR1_REPO
export PIXI_HOME="${AVTR1_HOME}/pixi" PIXI_CACHE_DIR="${AVTR1_HOME}/pixi-cache"
export RATTLER_CACHE_DIR="${AVTR1_HOME}/pixi-cache" XDG_CACHE_HOME="${AVTR1_HOME}/cache/.cache"
export TMPDIR="${AVTR1_HOME}/tmp" PATH="${AVTR1_HOME}/pixi/bin:${PATH}"
mkdir -p "${TMPDIR}"

# TensorRT engines are built per GPU type: one storage per compute capability
# (avtr1_storage_sm80 for A100, _sm120 for RTX PRO 6000, ...). Weights are shared by copy.
avtr1_select_storage() {
  local cc
  cc=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    || nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  cc=$(echo "${cc}" | head -1 | tr -d '. ')
  export AVTR1_LOCAL_STORAGE="${AVTR1_STORAGE:-${AVTR1_HOME}/avtr1_storage_sm${cc}}"
}
