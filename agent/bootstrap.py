"""Set HF cache env before LiveKit spawns inference subprocesses."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_HUB = "/data/hf_cache/hub"
_DEFAULT_DATASETS = "/data/hf_cache/datasets"


def ensure_hf_hub_env() -> None:
    """
    Inference workers may not reload ~/.bashrc (spawn).
    Mirror cluster defaults when HF_HUB_CACHE is unset.
    """
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
