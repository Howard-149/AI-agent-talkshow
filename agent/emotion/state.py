"""Per-role emotion state + [emotion]/[pad] tag helpers for talk-show turns.

Live path (``TALKSHOW_EMOTION_SOURCE=pad``, default): ``role_emotion`` holds the
MSP kNN primary label (e.g. Anger, Concerned). Legacy closed-set tags remain
only for ``TALKSHOW_EMOTION_SOURCE=llm`` rollback.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

# Legacy CosyVoice closed set (llm rollback only).
LEGACY_EMOTIONS = frozenset(
    {
        "neutral",
        "amused",
        "curious",
        "serious",
        "skeptical",
        "warm",
        "surprised",
    }
)
# Back-compat alias
EMOTIONS = LEGACY_EMOTIONS
DEFAULT_EMOTION = "Neutral"
LEGACY_DEFAULT_EMOTION = "neutral"
EMOTION_CHOICES = " | ".join(sorted(LEGACY_EMOTIONS))

_EMOTION_RE = re.compile(
    r"\[emotion\]\s*:\s*([A-Za-z][\w-]*)\b"
    r"|\[emotion\s*:\s*([A-Za-z][\w-]*)\s*\]",
    re.IGNORECASE,
)

_PAD_RE = re.compile(
    r"\[pad\]\s*:?\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s*,\s*|\s+)"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s*,\s*|\s+)"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)


def emotion_source() -> str:
    """``pad`` (default) = PAD→kNN MSP tags; ``llm`` = legacy [emotion] closed set."""
    return (os.environ.get("TALKSHOW_EMOTION_SOURCE") or "pad").strip().lower()


def normalize_emotion(raw: str | None) -> str | None:
    """Return a usable emotion label, or None if missing/invalid."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    key = s.lower()
    if key in LEGACY_EMOTIONS:
        # Preserve lowercase closed-set codes for llm / instruct overrides.
        return key
    from agent.emotion.msp_anchors import canonicalize_emotion_label

    label = canonicalize_emotion_label(s)
    if not label or label.lower() in {"none", "nan"}:
        return None
    return label


def parse_emotion_tag(text: str) -> str | None:
    m = _EMOTION_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    return normalize_emotion(raw)


def strip_emotion_tag(text: str) -> str:
    cleaned = _EMOTION_RE.sub("", text or "")
    cleaned = re.sub(r"\[emotion\s*:?\s*\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def parse_pad_delta(text: str) -> tuple[float, float, float] | None:
    m = _PAD_RE.search(text or "")
    if not m:
        return None
    try:
        p, a, d = float(m.group(1)), float(m.group(2)), float(m.group(3))
    except ValueError:
        return None
    def _clip(x: float) -> float:
        return max(-1.0, min(1.0, x))

    return (_clip(p), _clip(a), _clip(d))


def strip_pad_tag(text: str) -> str:
    cleaned = _PAD_RE.sub("", text or "")
    cleaned = re.sub(r"\[pad\s*:?\s*\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def get_role_emotion(data: TalkShowData, role: str) -> str:
    role = (role or "").strip().lower()
    default = LEGACY_DEFAULT_EMOTION if emotion_source() == "llm" else DEFAULT_EMOTION
    if not role or role == "human":
        return default
    current = data.role_emotion.get(role)
    return normalize_emotion(current) or default


def set_role_emotion(data: TalkShowData, role: str, emotion: str | None) -> str:
    """Update role emotion when tag is valid; otherwise keep previous. Returns current."""
    role = (role or "").strip().lower()
    default = LEGACY_DEFAULT_EMOTION if emotion_source() == "llm" else DEFAULT_EMOTION
    if not role or role == "human":
        return default
    normalized = normalize_emotion(emotion)
    if normalized is None:
        return get_role_emotion(data, role)
    prev = data.role_emotion.get(role)
    data.role_emotion[role] = normalized
    if normalized != prev:
        logger.info("role_emotion role=%s → %s (was %s)", role, normalized, prev)
    return normalized


def apply_emotion_from_parsed(
    data: TalkShowData, role: str, emotion: str | None
) -> str:
    """Apply parsed emotion tag (or keep prior). Returns the role's current emotion."""
    return set_role_emotion(data, role, emotion)


def emotion_prompt_block(data: TalkShowData, role: str) -> str:
    """Legacy llm-mode prompt: current mood + closed-set output contract."""
    current = get_role_emotion(data, role)
    others: list[str] = []
    for other_role, emo in sorted(data.role_emotion.items()):
        if other_role == role:
            continue
        code = normalize_emotion(emo) or LEGACY_DEFAULT_EMOTION
        if code == LEGACY_DEFAULT_EMOTION or code == DEFAULT_EMOTION:
            continue
        from agent.config import load_persona_name

        others.append(f"{load_persona_name(other_role)}={code}")
    others_line = ""
    if others:
        others_line = f"\nOther panel moods (for context): {', '.join(others)}."

    return (
        f"Mood state: you are currently **{current}**.{others_line}\n"
        f"- Keep [reply] wording and tone consistent with that mood.\n"
        f"- Update [emotion] only when the conversation warrants a gradual shift; "
        f"do not jump randomly.\n"
        f"- Output exactly one tag from: {EMOTION_CHOICES}."
    )


def emotion_output_lines() -> str:
    return f"[emotion]: {EMOTION_CHOICES}"


def pad_prompt_block(data: TalkShowData, role: str) -> str:
    """PAD mode: show current PAD + MSP tag (read-only); ask for [pad] deltas."""
    from agent.emotion.role_pad import get_role_pad

    p, a, d = get_role_pad(data, role)
    tag = get_role_emotion(data, role)
    others: list[str] = []
    for other_role, emo in sorted(getattr(data, "role_emotion", {}).items()):
        if other_role == role:
            continue
        code = normalize_emotion(emo)
        if not code or code in {DEFAULT_EMOTION, LEGACY_DEFAULT_EMOTION}:
            continue
        from agent.config import load_persona_name

        others.append(f"{load_persona_name(other_role)}={code}")
    others_line = ""
    if others:
        others_line = f"\nOther panel moods (for context): {', '.join(others)}."

    return (
        f"Affect state (PAD in [-1,1]): P={p:.2f} A={a:.2f} D={d:.2f}; "
        f"current mood tag **{tag}** (derived — do not invent a tag).{others_line}\n"
        f"- Keep [reply] wording consistent with that affect.\n"
        f"- You MUST output [pad]: ΔP ΔA ΔD on its own last line — never skip it. "
        f"Three numbers in [-1,1] (spaces, not commas). Use 0 0 0 if affect is unchanged. "
        f"Example: [pad]: 0.12 -0.05 0.00 — small gradual shifts only. "
        f"Do not output [emotion]; the tag is derived from PAD."
    )


def pad_output_lines() -> str:
    return "[pad]: <ΔP> <ΔA> <ΔD>   (REQUIRED last line; 0 0 0 if unchanged)"


def mood_prompt_block(data: TalkShowData, role: str) -> str:
    if emotion_source() == "llm":
        return emotion_prompt_block(data, role)
    return pad_prompt_block(data, role)


def mood_output_lines() -> str:
    if emotion_source() == "llm":
        return emotion_output_lines()
    return pad_output_lines()
