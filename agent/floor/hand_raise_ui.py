"""Sync hand-raise flash/queue UI events with floor poll and dequeue."""

from __future__ import annotations

import asyncio
import logging
import os

from agent.data import TalkShowData
from agent.floor.floor_parser import HandRaiseResult
from agent.ui.ui_events import emit_hand_raise

logger = logging.getLogger(__name__)


async def flash_poll_raises(
    data: TalkShowData,
    *,
    panel_roles: list[str],
    poll: dict[str, HandRaiseResult],
    flash_sec: float | None = None,
) -> None:
    """Briefly show all poll yes hands before tie-break enqueue clears losers."""
    if flash_sec is None:
        flash_sec = float(os.environ.get("TALKSHOW_HAND_RAISE_FLASH_SEC", "0.5"))
    yes_roles = [r for r in panel_roles if poll.get(r) and poll[r].raised]
    if not yes_roles or flash_sec <= 0:
        return
    for role in yes_roles:
        hr = poll[role]
        await emit_hand_raise(
            role,
            True,
            reason=hr.reason,
            topic=hr.topic,
        )
    logger.info(
        "hand_raise poll flash roles=%s sec=%.2f",
        yes_roles,
        flash_sec,
    )
    if data.turn_log:
        data.turn_log.log(
            "hand_raise_poll_flash",
            roles=yes_roles,
            flash_sec=flash_sec,
            room=data.room_name,
        )
    await asyncio.sleep(flash_sec)


async def sync_hand_raise_ui(
    data: TalkShowData,
    *,
    panel_roles: list[str],
    phase: str = "sync",
) -> None:
    """Publish UI hand state from the queue (source of truth)."""
    from agent.ui.ui_events import emit_queue_state

    roles = list(panel_roles) + ["human"]
    for role in roles:
        entry = data.hand_raise_queue.get(role)
        await emit_hand_raise(
            role,
            entry is not None,
            reason=entry.reason if entry else "",
            topic=entry.topic if entry else "",
        )
    await emit_queue_state(data, phase=phase)


async def dequeue_hand_raise(data: TalkShowData, role: str) -> None:
    """Remove role from queue after grant or hand down."""
    popped = data.hand_raise_queue.pop(role)
    if popped is None:
        return
    await emit_hand_raise(role, False)
    from agent.ui.ui_events import emit_queue_state

    await emit_queue_state(data, phase=f"dequeue:{role}")
    if data.turn_log:
        data.turn_log.log(
            "hand_raise_dequeue",
            role=role,
            room=data.room_name,
            queue_remaining=data.hand_raise_queue.roles(),
        )


async def clear_hand_raise_queue(
    data: TalkShowData, *, panel_roles: list[str]
) -> None:
    """Clear queue when host bypasses moderation with a direct grant."""
    for role in data.hand_raise_queue.roles():
        await emit_hand_raise(role, False)
    data.hand_raise_queue.clear_roles(list(panel_roles) + ["human"])
    if data.turn_log:
        data.turn_log.log(
            "hand_raise_clear",
            room=data.room_name,
            panel_roles=panel_roles,
        )


async def wait_for_hand_raises(
    data: TalkShowData,
    *,
    panel_roles: list[str],
    wait_sec: float | None = None,
) -> None:
    """
    Host needs a speaker but the queue is empty — wait up to ``wait_sec`` for
    anyone (human UI or later AI poll) to raise. Exit early when someone joins.
    Call only when ``data.hand_raise_queue.roles()`` is empty.
    """
    if wait_sec is None:
        wait_sec = float(os.environ.get("TALKSHOW_HAND_RAISE_WAIT_SEC", "10"))
    if wait_sec <= 0:
        return

    tick = 0.5
    elapsed = 0.0
    logger.info("hand_raise wait start sec=%.1f queue=empty", wait_sec)
    if data.turn_log:
        data.turn_log.log(
            "hand_raise_wait_start",
            wait_sec=wait_sec,
            room=data.room_name,
            queue=[],
        )
    while elapsed < wait_sec:
        if data.shutdown_event.is_set():
            logger.info("hand_raise wait aborted — session shutting down")
            return
        if data.hand_raise_queue.roles():
            logger.info(
                "hand_raise wait early exit at %.1fs queue=%s",
                elapsed,
                data.hand_raise_queue.roles(),
            )
            if data.turn_log:
                data.turn_log.log(
                    "hand_raise_wait_early",
                    elapsed_sec=elapsed,
                    room=data.room_name,
                    queue=data.hand_raise_queue.roles(),
                )
            return
        await asyncio.sleep(tick)
        elapsed += tick

    logger.info("hand_raise wait done queue=%s", data.hand_raise_queue.roles())
    if data.turn_log:
        data.turn_log.log(
            "hand_raise_wait_done",
            wait_sec=wait_sec,
            room=data.room_name,
            queue=data.hand_raise_queue.roles(),
        )
