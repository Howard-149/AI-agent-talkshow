from __future__ import annotations

import asyncio
import json
import logging
from collections import deque

from agent.config import ScenarioConfig, load_persona_name
from agent.panel_roster import panel_roster_entries
from agent.room_connect import wait_for_local_participant

logger = logging.getLogger(__name__)

UI_TOPIC = "talkshow/ui"

_pending: deque[tuple[bytes, str]] = deque()
_flush_task: asyncio.Task[None] | None = None


def _local_participant_ready(room: object) -> bool:
    try:
        _ = room.local_participant  # noqa: B018
        return True
    except Exception:
        return False


def _enqueue(payload: bytes, event_type: str) -> None:
    _pending.append((payload, event_type))


def _schedule_flush() -> None:
    global _flush_task
    if _flush_task is not None and not _flush_task.done():
        return
    _flush_task = asyncio.create_task(flush_pending_ui_events())


async def flush_pending_ui_events(*, max_wait_sec: float = 15.0) -> None:
    """Publish UI events queued before the room local participant was ready."""
    if not _pending:
        return
    try:
        from livekit.agents.job import get_job_context

        ctx = get_job_context()
        room = ctx.room
    except Exception:
        logger.debug("flush_pending_ui_events: no job context")
        return

    if not await wait_for_local_participant(room, timeout_sec=max_wait_sec):
        logger.warning(
            "flush_pending_ui_events: gave up with %d queued events", len(_pending)
        )
        return

    while _pending:
        payload, event_type = _pending.popleft()
        try:
            await room.local_participant.publish_data(
                payload,
                reliable=True,
                topic=UI_TOPIC,
            )
            logger.debug("flushed deferred ui event type=%s", event_type)
        except Exception as exc:
            msg = str(exc).lower()
            if "before connecting" in msg:
                _pending.appendleft((payload, event_type))
                await asyncio.sleep(0.05)
                if not await wait_for_local_participant(room, timeout_sec=2.0):
                    break
            elif "room closed" in msg or "channel closed" in msg:
                logger.debug(
                    "flush_pending_ui_events skipped type=%s (room closed)", event_type
                )
                break
            else:
                logger.warning(
                    "flush_pending_ui_events failed type=%s: %s", event_type, exc
                )


async def publish_ui_event(event_type: str, **fields: object) -> None:
    """Broadcast panel state to talkshow-web (and any client on UI_TOPIC)."""
    payload = json.dumps({"type": event_type, **fields}, ensure_ascii=False).encode(
        "utf-8"
    )
    try:
        from livekit.agents.job import get_job_context

        ctx = get_job_context()
        room = ctx.room
        if not _local_participant_ready(room):
            _enqueue(payload, event_type)
            _schedule_flush()
            logger.debug("publish_ui_event deferred type=%s (not connected)", event_type)
            return
        await room.local_participant.publish_data(
            payload,
            reliable=True,
            topic=UI_TOPIC,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "before connecting" in msg or "local participant" in msg:
            _enqueue(payload, event_type)
            _schedule_flush()
            logger.debug("publish_ui_event deferred type=%s (%s)", event_type, exc)
        elif "room closed" in msg or "channel closed" in msg:
            logger.debug("publish_ui_event skipped type=%s (room closed)", event_type)
        else:
            logger.warning("publish_ui_event failed type=%s: %s", event_type, exc)


async def emit_panel_roster(scenario: ScenarioConfig) -> None:
    """Register AI panelists for frontend (from scenario + persona yaml)."""
    await publish_ui_event(
        "panel_roster",
        scenario=scenario.id,
        members=panel_roster_entries(scenario),
    )


async def emit_role_idle() -> None:
    await publish_ui_event("role_idle")


async def emit_role_active(role: str) -> None:
    await publish_ui_event(
        "role_active",
        role=role,
        name=load_persona_name(role),
    )


async def emit_transcript(
    role: str,
    text: str,
    *,
    final: bool = True,
    step: str = "",
) -> None:
    text = text.strip()
    if not text:
        return
    await publish_ui_event(
        "transcript",
        role=role,
        speaker=load_persona_name(role) if role != "human" else "Human guest",
        text=text,
        final=final,
        step=step,
    )


async def emit_hand_raise(
    role: str,
    raised: bool,
    *,
    reason: str = "",
    topic: str = "",
) -> None:
    speaker = "Human guest" if role == "human" else load_persona_name(role)
    await publish_ui_event(
        "hand_raise",
        role=role,
        name=speaker,
        raised=raised,
        reason=reason.strip(),
        topic=topic.strip(),
    )


async def emit_queue_state(data: object, *, phase: str = "") -> None:
    """Full FIFO queue snapshot for UI + debug panel."""
    from agent.data import TalkShowData

    assert isinstance(data, TalkShowData)
    queue: list[dict[str, str]] = []
    for role in data.hand_raise_queue.roles():
        entry = data.hand_raise_queue.get(role)
        queue.append(
            {
                "role": role,
                "name": "Human guest" if role == "human" else load_persona_name(role),
                "reason": entry.reason if entry else "",
                "topic": entry.topic if entry else "",
            }
        )
    await publish_ui_event("queue_state", queue=queue, phase=phase.strip())


async def emit_clear_hand_raises(
    panel_roles: list[str], *, data: TalkShowData | None = None
) -> None:
    """Legacy alias — prefer clear_hand_raise_queue(data, panel_roles=...)."""
    if data is not None:
        from agent.hand_raise_ui import clear_hand_raise_queue

        await clear_hand_raise_queue(data, panel_roles=panel_roles)
        return
    for role in panel_roles:
        await emit_hand_raise(role, False)
    await emit_hand_raise("human", False)


async def emit_floor_pending(*, active: bool = True) -> None:
    await publish_ui_event("floor_pending", active=active)


async def emit_floor_grant(role: str, *, reason: str = "") -> None:
    name = "Human guest" if role == "human" else load_persona_name(role)
    await publish_ui_event(
        "floor_grant",
        role=role,
        name=name,
        reason=reason.strip(),
    )
