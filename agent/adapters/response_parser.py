from __future__ import annotations

import re
from dataclasses import dataclass

_HEARD_RE = re.compile(r"\[heard\]\s*:\s*(.*?)(?=\[reply\]\s*:|$)", re.IGNORECASE | re.DOTALL)
_REPLY_RE = re.compile(r"\[reply\]\s*:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)
_HANDOFF_TAG_RE = re.compile(r"\[handoff:([^\]]+)\]", re.IGNORECASE)
_VALID_HANDOFF = frozenset({"host", "guest", "commentator"})
# Model sometimes uses display names instead of role ids
_HANDOFF_ALIASES: dict[str, str] = {
    "lessac": "host",
    "ryan": "commentator",
    "amy": "guest",
}


@dataclass(frozen=True)
class ParsedTurn:
    heard: str
    reply: str
    handoff_to: str | None = None


def _normalize_handoff_role(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in _VALID_HANDOFF:
        return key
    return _HANDOFF_ALIASES.get(key)


def _strip_handoff_tags(reply: str) -> tuple[str, str | None]:
    """Remove all [handoff:…] tags; never speak them. Last valid tag wins for tooling."""
    handoff_to: str | None = None
    for m in _HANDOFF_TAG_RE.finditer(reply):
        role = _normalize_handoff_role(m.group(1))
        if role:
            handoff_to = role
    clean = _HANDOFF_TAG_RE.sub("", reply)
    clean = re.sub(r"\s+", " ", clean).strip(" .")
    return clean, handoff_to


def parse_heard_reply(text: str) -> ParsedTurn:
    text = text.strip()
    heard_m = _HEARD_RE.search(text)
    reply_m = _REPLY_RE.search(text)

    heard = heard_m.group(1).strip() if heard_m else ""
    reply = reply_m.group(1).strip() if reply_m else ""

    if not reply:
        # Model ignored format — treat full output as reply for MVP robustness
        reply = text
    if not heard:
        heard = "(audio understood; no transcript line)"

    reply, handoff_to = _strip_handoff_tags(reply)
    return ParsedTurn(heard=heard, reply=reply, handoff_to=handoff_to)
