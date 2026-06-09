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
    logger.debug("control_events: unknown type=%s", ev_type)
    return None
