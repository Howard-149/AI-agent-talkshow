from __future__ import annotations

import re
from dataclasses import dataclass

_HEARD_RE = re.compile(r"\[heard\]\s*:\s*(.*?)(?=\[reply\]\s*:|$)", re.IGNORECASE | re.DOTALL)
_REPLY_RE = re.compile(r"\[reply\]\s*:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedTurn:
    heard: str
    reply: str


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

    return ParsedTurn(heard=heard, reply=reply)
