"""CosyVoice mode + built-in speaker ID helpers (no prompt wav required)."""

from __future__ import annotations

import os
from pathlib import Path

# CosyVoice-300M-SFT / Instruct typical spk ids (confirm via sidecar GET /spks).
# They cover zh + en (and often ja/yue/ko depending on checkpoint).
DEFAULT_SPK_BY_ROLE: dict[str, str] = {
    "host": "英文男",
    "guest": "英文女",
    "commentator": "中文男",
}

DEFAULT_SPK_BY_ROLE_ZH: dict[str, str] = {
    "host": "中文男",
    "guest": "中文女",
    "commentator": "粤语女",
}


def cosyvoice_mode() -> str:
    """Default ``instruct`` = CosyVoice-300M-Instruct (spk + emotion, no wav)."""
    raw = os.environ.get("COSYVOICE_MODE", "instruct").strip()
    if "#" in raw:
        raw = raw.split("#", 1)[0].strip()
    raw = (raw.split() or ["instruct"])[0].lower()
    if raw in ("instruct", "sft", "instruct2", "zero_shot"):
        return raw
    return "instruct"


def mode_needs_prompt_wav(mode: str | None = None) -> bool:
    m = (mode or cosyvoice_mode()).lower()
    return m in ("instruct2", "zero_shot")


def resolve_spk_id(
    *,
    role: str = "",
    locale: str = "en",
    model_path: str = "",
) -> str:
    """
    Resolve built-in speaker id.

    Priority: model_path if it is NOT an existing file path → env COSYVOICE_SPK_* → defaults.
    """
    cand = (model_path or "").strip()
    if cand and not Path(cand).expanduser().is_file():
        # Treat as spk id (e.g. 英文男) rather than a missing wav path.
        if not cand.endswith((".wav", ".mp3", ".flac", ".onnx")):
            return cand

    role_key = (role or "host").strip().lower() or "host"
    loc = (locale or "en").strip().lower()
    loc_key = "ZH" if loc.startswith("zh") else loc.upper()
    role_env = role_key.upper()

    for key in (
        f"COSYVOICE_SPK_{loc_key}_{role_env}",
        f"COSYVOICE_SPK_{role_env}" if not loc.startswith("zh") else "",
        f"COSYVOICE_SPK_{loc_key}" if role_key == "host" else "",
        "COSYVOICE_SPK" if role_key == "host" and not loc.startswith("zh") else "",
    ):
        if not key:
            continue
        val = os.environ.get(key, "").strip()
        if val:
            return val

    table = DEFAULT_SPK_BY_ROLE_ZH if loc.startswith("zh") else DEFAULT_SPK_BY_ROLE
    return table.get(role_key, table["host"])
