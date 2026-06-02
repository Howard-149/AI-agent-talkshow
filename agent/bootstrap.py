"""Set HF cache env before LiveKit spawns inference subprocesses."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_HUB = "/data/hf_cache/hub"
_DEFAULT_DATASETS = "/data/hf_cache/datasets"


def _expand_user(path: str) -> str:
    return path.replace("${USER}", os.environ.get("USER", ""))


def _default_turn_detector_cache() -> str | None:
    user = os.environ.get("USER", "").strip()
    if not user or not Path("/data/user_data").is_dir():
        return None
    return str(Path("/data/user_data") / user / "livekit-turn-detector" / "hub")


def ensure_onnx_thread_env() -> None:
    """
    Babel/Slurm cgroup: ORT auto CPU affinity → pthread_setaffinity_np EINVAL.
    Set thread counts before LiveKit spawns inference (turn-detector / Silero).
    """
    n = os.environ.get("TALKSHOW_ORT_NUM_THREADS", "1").strip() or "1"
    for key in (
        "ORT_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, n)
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    logger.info("ONNX/thread env ORT_NUM_THREADS=%s (Slurm-safe)", n)


def ensure_cluster_runtime_env() -> None:
    """Call once at process start, before livekit inference subprocesses."""
    ensure_onnx_thread_env()
    ensure_hf_hub_env()


def ensure_hf_hub_env() -> None:
    """
    Inference workers may not reload ~/.bashrc (spawn).

    Turn-detector ONNX (model_q8.onnx) can live in a private cache
    (TALKSHOW_TURN_DETECTOR_CACHE) so public /data/hf_cache purges do not break the agent.
    When set, agent inference uses that path as HF_HUB_CACHE; global HF cache is unchanged.
    """
    td_cache = _expand_user(os.environ.get("TALKSHOW_TURN_DETECTOR_CACHE", "").strip())
    if not td_cache:
        td_cache = _default_turn_detector_cache() or ""

    if td_cache:
        Path(td_cache).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = td_cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = td_cache
        logger.info(
            "HF_HUB_CACHE=%s (turn-detector private cache; run deploy/download-livekit-agent-models.sh)",
            td_cache,
        )
        return

    hub = os.environ.get("HF_HUB_CACHE", "").strip()
    if not hub and Path(_DEFAULT_HUB).is_dir():
        os.environ["HF_HUB_CACHE"] = _DEFAULT_HUB
        hub = _DEFAULT_HUB

    if hub:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub)

    os.environ.setdefault("HF_DATASETS_CACHE", _DEFAULT_DATASETS)

    logger.info(
        "HF_HUB_CACHE=%s (turn-detector ONNX must exist here; run deploy/download-livekit-agent-models.sh)",
        os.environ.get("HF_HUB_CACHE", "<unset>"),
    )
