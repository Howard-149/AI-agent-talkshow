"""Parse talkshow/control data-channel payloads (hand-raise, topic, locale)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CONTROL_TOPIC = "talkshow/control"


@dataclass(frozen=True)
class ControlEvent:
    type: str
    raised: bool = False
    topic: str = ""
    reason: str = ""
    locale: str = ""
    text: str = ""
    # ghost_human_turn only, honored when TALKSHOW_GHOST_REPLAY=1: history / mood
    # state to swap in before the turn (see agent/session/ghost_replay.py).
    prefix: dict[str, Any] | None = None


def parse_control_payload(raw: bytes) -> ControlEvent | None:
    try:
        text = raw.decode("utf-8")
        data: dict[str, Any] = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.debug("control_events: invalid payload", exc_info=True)
        return None

    ev_type = str(data.get("type", "")).strip()
    if ev_type == "hand_raise":
        return ControlEvent(
            type=ev_type,
            raised=bool(data.get("raised", False)),
            topic=str(data.get("topic", "") or "").strip(),
            reason=str(data.get("reason", "") or "").strip(),
        )
    if ev_type == "set_locale":
        return ControlEvent(
            type=ev_type,
            locale=str(data.get("locale", "") or "").strip(),
        )
    if ev_type == "ghost_human_turn":
        prefix = data.get("prefix")
        return ControlEvent(
            type=ev_type,
            text=str(data.get("text", "") or "").strip(),
            prefix=prefix if isinstance(prefix, dict) else None,
        )
    logger.debug("control_events: unknown type=%s", ev_type)
    return None
