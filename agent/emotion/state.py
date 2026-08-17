"""Per-role emotion state + [emotion] tag parse/strip for talk-show turns."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

EMOTIONS = frozenset(
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
DEFAULT_EMOTION = "neutral"
EMOTION_CHOICES = " | ".join(sorted(EMOTIONS))

_EMOTION_RE = re.compile(
    r"\[emotion\]\s*:\s*(\w+)\b"
    r"|\[emotion\s*:\s*(\w+)\s*\]",
    re.IGNORECASE,
)


def normalize_emotion(raw: str | None) -> str | None:
    """Return a valid emotion code, or None if missing/invalid."""
    if not raw:
        return None
    key = raw.strip().lower()
    if key in EMOTIONS:
        return key
    return None


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


def get_role_emotion(data: TalkShowData, role: str) -> str:
    role = (role or "").strip().lower()
    if not role or role == "human":
        return DEFAULT_EMOTION
    current = data.role_emotion.get(role)
    return normalize_emotion(current) or DEFAULT_EMOTION


def set_role_emotion(data: TalkShowData, role: str, emotion: str | None) -> str:
    """Update role emotion when tag is valid; otherwise keep previous. Returns current."""
    role = (role or "").strip().lower()
    if not role or role == "human":
        return DEFAULT_EMOTION
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
    """Inject into speaking prompts: current mood + output contract."""
    current = get_role_emotion(data, role)
    others: list[str] = []
    for other_role, emo in sorted(data.role_emotion.items()):
        if other_role == role:
            continue
        code = normalize_emotion(emo) or DEFAULT_EMOTION
        if code == DEFAULT_EMOTION:
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
