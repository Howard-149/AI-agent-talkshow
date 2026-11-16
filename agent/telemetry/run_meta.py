"""Run metadata for session_start: code version, GPU, and latency-relevant settings."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Env knobs that change latency or what the panel says; logged so runs can be compared
# without guessing. An arms preset or a ghost replay that sets a flag needs it here too
# (tests/session/test_run_meta.py).
_LATENCY_ENV = (
    "TALKSHOW_TTS_ENGINE",
    "TALKSHOW_AVATAR_ENABLED",
    "TALKSHOW_AVATAR_CHUNK_STREAM",
    "TALKSHOW_AVATAR_STEPS",
    "TALKSHOW_COSYVOICE_CHUNK_MODE",
    "TALKSHOW_TURN_DETECTOR",
    "TALKSHOW_VAD_MIN_SILENCE_SEC",
    "TALKSHOW_MIN_ENDPOINTING_DELAY",
    "TALKSHOW_MAX_ENDPOINTING_DELAY",
    "TALKSHOW_VAD_MIN_SPEECH_SEC",
    "TALKSHOW_AVATAR_SKIP_UNUSED_WAV2VEC",
    "TALKSHOW_PRERENDER_CANNED",
    "TALKSHOW_SEED",
    # Think & Action pacing: how often listeners think, how far text runs ahead of
    # playback, the pause between speakers, cuts, and where the decisions run.
    "TALKSHOW_TEXT_TOKEN_INTERVAL",
    "TALKSHOW_TEXT_INTERRUPTION_THRESHOLD",
    "TALKSHOW_MAX_LOOKAHEAD_TURNS",
    "TALKSHOW_TURN_GAP_S",
    "TALKSHOW_TURN_GAP_JITTER_S",
    "TALKSHOW_FLOOR_RULES",
    "TALKSHOW_FLOOR_HOST_COOLDOWN_S",
    "TALKSHOW_CUT_OVERLAP_S",
    "TALKSHOW_HUMAN_CALL_LIMIT",
    "TALKSHOW_DECISIONS_ON_MAIN_LLM",
    "TALKSHOW_THINK_BRIEF",
    # Thinking while the human is still talking.
    "TALKSHOW_SPECULATE_HUMAN",
    "TALKSHOW_SPECULATE_INTERVAL_S",
    "TALKSHOW_SPECULATE_MAX",
    "TALKSHOW_SPECULATE_MAX_MISSING_FRACTION",
    "TALKSHOW_SPECULATE_MAX_MISSING_WORDS",
    "TALKSHOW_SPECULATE_MIN_SPEECH_S",
    "TALKSHOW_SPECULATE_ROLES",
    "TALKSHOW_HISTORY_MAX_LINES",
    "TALKSHOW_PANEL_MAX_TURNS",
    "TALKSHOW_UPTAKE_PLANNER",
    "TALKSHOW_PANEL_SCAFFOLD_FIX",
    "TALKSHOW_ADDRESSEE_FLOOR",
    "TALKSHOW_DIALOGUE_LIBRARY",
    "TALKSHOW_SHOW_FRAME",
    "TALKSHOW_HOST_CARD",
    "TALKSHOW_OPENER_VARIETY",
    "TALKSHOW_FLOOR_JEV",
    "TALKSHOW_FLOOR_JEV_MIN_CONF",
    "TALKSHOW_MAX_SENTENCES",
    "TALKSHOW_PANEL_SELECT",
    "TALKSHOW_PANEL_CONCRETE",
    "TALKSHOW_PANEL_BEST_OF",
    "TALKSHOW_THINK_LEAN",
    "TALKSHOW_THINK_PERSONA_NEAR",
    "TALKSHOW_THINK_FOCUS",
    "TALKSHOW_PERSONA_LIBRARY",
    "TALKSHOW_PERSONA_LIBRARY_NAMES",
    "TALKSHOW_PERSONA_COMMENTATOR",
    "TALKSHOW_PERSONA_GUEST",
    "TALKSHOW_FACT_PACK",
    "TALKSHOW_FACT_PACK_K",
    "TALKSHOW_FACT_PACK_TOPIC",
    "COSYVOICE_MODE",
    "COSYVOICE_LOAD_VLLM",
    "COSYVOICE_LOAD_TRT",
    "COSYVOICE_FP16",
    "COSYVOICE_SPEED",
    "COSYVOICE_EMOTION_REFS",
    "COSYVOICE_EMOTION_REFS_DIR",
    "VLLM_MODEL",
    "TALKSHOW_TEXT_LLM_MODEL",
    "TALKSHOW_TEXT_LLM_BASE_URL",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_USE_DEFAULT_SAMPLING",
    "OPENAI_USE_MAX_COMPLETION_TOKENS",
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
