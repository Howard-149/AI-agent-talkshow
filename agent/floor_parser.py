from __future__ import annotations

import re
from dataclasses import dataclass

from agent.config import load_persona_name
from agent.data import TalkShowData

_RAISE_RE = re.compile(r"\[raise\]\s*:\s*(yes|no)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"\[reason\]\s*:\s*(.+?)(?=\[|$)", re.IGNORECASE | re.DOTALL)
_TOPIC_RE = re.compile(r"\[topic\]\s*:\s*(.+?)(?=\[|$)", re.IGNORECASE | re.DOTALL)
_NEXT_RE = re.compile(
    r"\[next\]\s*:\s*(commentator|guest|close|host|human)\b"
    r"|\[next\s*:\s*(commentator|guest|close|host|human)\s*\]",
    re.IGNORECASE,
)
_VALID_NEXT = frozenset({"commentator", "guest", "close", "host", "human"})
_NEXT_ALIASES: dict[str, str] = {
    "ryan": "commentator",
    "amy": "guest",
    "lessac": "host",
}


@dataclass(frozen=True)
class HandRaiseResult:
    role: str
    raised: bool
    reason: str = ""
    topic: str = ""


@dataclass(frozen=True)
class FloorDecision:
    next_role: str  # commentator | guest | close | host | human
    reason: str = ""


def parse_hand_raise(role: str, text: str) -> HandRaiseResult:
    text = text.strip()
    raised = False
    m = _RAISE_RE.search(text)
    if m and m.group(1).lower() == "yes":
        raised = True
    reason_m = _REASON_RE.search(text)
    reason = reason_m.group(1).strip() if reason_m else ""
    topic_m = _TOPIC_RE.search(text)
    topic = topic_m.group(1).strip() if topic_m else ""
    return HandRaiseResult(role=role, raised=raised, reason=reason, topic=topic)


def normalize_floor_role(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in _VALID_NEXT:
        return key
    return _NEXT_ALIASES.get(key)


def _normalize_next(raw: str) -> str | None:
    return normalize_floor_role(raw)


def parse_next_speaker_tag(text: str) -> str | None:
    m = _NEXT_RE.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    return normalize_floor_role(raw) if raw else None


def strip_next_tag(text: str) -> str:
    cleaned = _NEXT_RE.sub("", text)
    cleaned = re.sub(r"\[next\s*:?\s*\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def parse_floor_decision(text: str) -> FloorDecision | None:
    text = text.strip()
    role = parse_next_speaker_tag(text)
    if not role:
        return None
    reason_m = _REASON_RE.search(text)
    reason = reason_m.group(1).strip() if reason_m else ""
    return FloorDecision(next_role=role, reason=reason)


def fallback_floor_decision(
    *,
    panel_roles: list[str],
    raises: dict[str, HandRaiseResult],
    spoken_roles: set[str],
) -> str:
    """When host LLM output is invalid — prefer sole raiser, then raised queue, then unspoken."""
    raised = [r for r in panel_roles if raises.get(r) and raises[r].raised]
    if len(raised) == 1:
        return raised[0]
    if len(raised) > 1:
        for r in panel_roles:
            if r in raised:
                return r
    unspoken = [r for r in panel_roles if r not in spoken_roles]
    if unspoken:
        return unspoken[0]
    return "close"


def infer_host_called_role(data: TalkShowData) -> str | None:
    """
    When the host tee-up names Ryan/Amy, pick them for direct host calls (empty queue).
    Returns role id: commentator | guest | None
    """
    tee = (data.last_host_panel_tee or "").strip()
    if not tee:
        for line in reversed(data.show_history.lines):
            if line.role_id == "host":
                tee = line.text
                break
    if not tee:
        return None

    t = tee.lower()
    ryan_name = load_persona_name("commentator").lower()
    amy_name = load_persona_name("guest").lower()
    ryan = re.escape(ryan_name)
    amy = re.escape(amy_name)

    call_patterns = (
        rf"\b(?:over to|turn to|go to|i'?ll hand|give the floor to|call on)\s+({ryan}|{amy})\b",
        rf"\blet'?s hear(?:\s+from)?\s+({ryan}|{amy})\b",
        rf"\b({ryan}|{amy})\s+(?:to speak|you have the floor|take it)\b",
    )
    for pat in call_patterns:
        m = re.search(pat, t, re.I)
        if m:
            name = m.group(1).lower()
            if name == ryan_name:
                return "commentator"
            if name == amy_name:
                return "guest"

    ryan_pos = t.find(ryan_name)
    amy_pos = t.find(amy_name)
    if ryan_pos >= 0 and (amy_pos < 0 or ryan_pos <= amy_pos):
        if re.search(rf"\b{ryan}\b", t):
            return "commentator"
    if amy_pos >= 0 and re.search(rf"\b{amy}\b", t):
        return "guest"
    return None
