"""Parse [raise]/[reason]/[topic]/[next] tags from host/panelist model output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from agent.config import load_persona_name
from agent.data import TalkShowData

_RAISE_RE = re.compile(r"\[raise\]\s*:\s*(yes|no)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"\[reason\]\s*:\s*(.+?)(?=\[|$)", re.IGNORECASE | re.DOTALL)
_TOPIC_RE = re.compile(r"\[topic\]\s*:\s*(.+?)(?=\[|$)", re.IGNORECASE | re.DOTALL)
_NEXT_RE = re.compile(
    r"\[next\]\s*:\s*(\w+)\b"
    r"|\[next\s*:\s*(\w+)\s*\]",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _floor_role_lookup() -> tuple[frozenset[str], dict[str, str]]:
    from agent.panel.panel_context import floor_valid_next_roles, role_name_aliases

    valid = floor_valid_next_roles()
    aliases = role_name_aliases(include_host=True)
    return valid, aliases


@dataclass(frozen=True)
class HandRaiseResult:
    role: str
    raised: bool
    reason: str = ""
    topic: str = ""


@dataclass(frozen=True)
class FloorDecision:
    next_role: str
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
    valid, aliases = _floor_role_lookup()
    if key in valid:
        return key
    return aliases.get(key)


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


def strip_speech_control_tags(text: str) -> str:
    """Remove [next] and [emotion] tags from spoken text."""
    from agent.emotion import strip_emotion_tag

    return strip_emotion_tag(strip_next_tag(text)).strip()


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
    When the host tee-up names a panelist, return their role id for direct host calls.
    """
    from agent.panel.panel_context import panel_speaker_roles, name_to_panel_role

    panel_roles = panel_speaker_roles(data.scenario)
    tee = (data.last_host_panel_tee or "").strip()
    if not tee:
        for line in reversed(data.show_history.lines):
            if line.role_id == "host":
                tee = line.text
                break
    if not tee:
        return None

    t = tee.lower()
    names_by_role = {r: load_persona_name(r).lower() for r in panel_roles}
    escaped = {r: re.escape(n) for r, n in names_by_role.items()}

    call_patterns: list[str] = []
    for role in panel_roles:
        n = escaped[role]
        call_patterns.extend(
            [
                rf"\b(?:over to|turn to|go to|i'?ll hand|give the floor to|call on)\s+({n})\b",
                rf"\blet'?s hear(?:\s+from)?\s+({n})\b",
                rf"\b({n})\s+(?:to speak|you have the floor|take it)\b",
            ]
        )
    for pat in call_patterns:
        m = re.search(pat, t, re.I)
        if m:
            resolved = name_to_panel_role(m.group(1), panel_roles)
            if resolved:
                return resolved

    positions: list[tuple[int, str]] = []
    for role, name in names_by_role.items():
        pos = t.find(name)
        if pos >= 0:
            positions.append((pos, role))
    if positions:
        positions.sort(key=lambda x: x[0])
        return positions[0][1]
    return None
