"""Parse [next] tags from host/panelist model output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from agent.config import load_persona_name
from agent.data import TalkShowData

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


def normalize_floor_role(raw: str) -> str | None:
    key = raw.strip().lower()
    valid, aliases = _floor_role_lookup()
    if key in valid:
        return key
    return aliases.get(key)


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


_REPLY_RE = re.compile(r"\[reply\]\s*:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)
_HEARD_RE = re.compile(r"\[heard\]\s*:.*?(?=\[reply\]\s*:|\Z)", re.IGNORECASE | re.DOTALL)


def spoken_reply_text(raw: str) -> str:
    """Spoken words from a free-form host turn.

    The host persona prompt teaches the [heard]/[reply] format, so models may echo
    the request under [heard]; keep only the reply and never speak the [heard] part."""
    text = str(raw or "")
    reply = _REPLY_RE.search(text)
    if reply:
        text = reply.group(1)
    return strip_speech_control_tags(_HEARD_RE.sub("", text))


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
