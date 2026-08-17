"""Wait until LiveKit room.local_participant is usable after connect."""

from __future__ import annotations

import asyncio
import logging

from livekit import rtc

logger = logging.getLogger(__name__)


def local_participant_ready(room: rtc.Room) -> bool:
    """True when room.local_participant can be read without raising."""
    try:
        _ = room.local_participant  # noqa: B018
        return True
    except Exception:
        return False


async def wait_for_local_participant(
    room: rtc.Room, *, timeout_sec: float = 60.0
) -> bool:
    """Block until room.local_participant is usable (or timeout)."""
    deadline = asyncio.get_running_loop().time() + timeout_sec
    while asyncio.get_running_loop().time() < deadline:
        if local_participant_ready(room):
            logger.info("room local participant ready room=%s", room.name)
            return True
        await asyncio.sleep(0.05)
    logger.error(
        "wait_for_local_participant timed out after %.0fs room=%s",
        timeout_sec,
        room.name,
    )
    return False
