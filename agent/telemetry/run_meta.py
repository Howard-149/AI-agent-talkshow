"""Run metadata for session_start: code version, GPU, and latency-relevant settings."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Env knobs that change latency; logged so runs can be compared without guessing.
_LATENCY_ENV = (
    "TALKSHOW_TTS_ENGINE",
    "TALKSHOW_AVATAR_ENABLED",
    "TALKSHOW_AVATAR_CHUNK_STREAM",
    "TALKSHOW_COSYVOICE_CHUNK_MODE",
    "TALKSHOW_TURN_DETECTOR",
    "TALKSHOW_VAD_MIN_SILENCE_SEC",
    "TALKSHOW_MIN_ENDPOINTING_DELAY",
    "TALKSHOW_MAX_ENDPOINTING_DELAY",
    "TALKSHOW_HAND_RAISE_WAIT_SEC",
    "TALKSHOW_HAND_RAISE_FLASH_SEC",
    "TALKSHOW_HAND_RAISE_GRANT_PAUSE_SEC",
    "TALKSHOW_HISTORY_MAX_LINES",
    "TALKSHOW_PANEL_MAX_TURNS",
    "COSYVOICE_MODE",
    "COSYVOICE_LOAD_VLLM",
    "COSYVOICE_LOAD_TRT",
    "COSYVOICE_FP16",
    "COSYVOICE_SPEED",
    "VLLM_MODEL",
)


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def run_metadata() -> dict[str, Any]:
    """Best-effort; missing tools (git, nvidia-smi) just leave fields empty."""
    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain", "--untracked-files=no"]))
    gpus = _run(["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"])
    return {
        "git_commit": commit + ("-dirty" if commit and dirty else ""),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "gpus": [g.strip() for g in gpus.splitlines() if g.strip()],
        "env": {k: os.environ[k] for k in _LATENCY_ENV if k in os.environ},
    }
