"""Session active checks, human presence, and shutdown-when-alone handling."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from livekit import rtc
from livekit.agents import AgentSession, JobContext

from agent.data import TalkShowData

logger = logging.getLogger(__name__)


def session_is_active(session: AgentSession) -> bool:
    """False while AgentSession is stopped or shutting down."""
    running = getattr(session, "is_running", None)
    if running is not None and not running:
        return False
    if getattr(session, "_activity", None) is None:
        return False
    close_task = getattr(session, "_close_session_atask", None)
    if close_task is not None and close_task.done():
        return False
    return True


def room_has_humans(room: rtc.Room) -> bool:
    """Any remote participant that is not another talkshow agent."""
    for participant in room.remote_participants.values():
        if not _is_talkshow_agent(participant):
            return True
    return False


def _is_talkshow_agent(participant: rtc.RemoteParticipant) -> bool:
    raw = participant.metadata or ""
    if not raw:
        return False
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return bool(meta.get("talkshowAgent"))


def _drain_on_human_leave() -> bool:
    return os.environ.get("TALKSHOW_SESSION_DRAIN_ON_LEAVE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def should_stop_session_work(
    data: TalkShowData, session: AgentSession | None = None
) -> bool:
    """True after human left / session shutdown — background loops must exit.

    Do not treat between-utterance idle (``AgentSession._activity is None``) as
    stopped; that incorrectly aborted the post-welcome hand-raise round.
    """
    if data.shutdown_event.is_set():
        return True
    if session is not None:
        running = getattr(session, "is_running", None)
        if running is not None and not running:
            return True
        close_task = getattr(session, "_close_session_atask", None)
        if close_task is not None and close_task.done():
            return True
    return False


async def shutdown_session_when_alone(
    session: AgentSession,
    data: TalkShowData,
    *,
    room_name: str,
    reason: str,
    job_ctx: JobContext | None = None,
) -> None:
    if data.shutdown_event.is_set():
        return
    data.shutdown_event.set()
    data.panel_chain_running = False
    data.panel_followup_pending = False
    data.speak_line_busy = False
    data.panel_followup_deferred = False
    if not session_is_active(session):
        if job_ctx is not None:
            try:
                job_ctx.shutdown(reason=reason)
            except Exception:
                logger.debug("job_ctx.shutdown skipped room=%s", room_name, exc_info=True)
        return
    drain = _drain_on_human_leave()
    logger.info(
        "shutting down agent session room=%s reason=%s drain=%s",
        room_name,
        reason,
        drain,
    )
    try:
        session.shutdown(drain=drain)
    except Exception:
        logger.debug("session.shutdown skipped room=%s", room_name, exc_info=True)
    if job_ctx is not None:
        try:
            job_ctx.shutdown(reason=reason)
        except Exception:
            logger.debug("job_ctx.shutdown skipped room=%s", room_name, exc_info=True)


async def _alone_watchdog(
    session: AgentSession,
    data: TalkShowData,
    room: rtc.Room,
    *,
    job_ctx: JobContext | None,
) -> None:
    """Fallback when participant_disconnected is missed (tab kill, network drop)."""
    interval = 5.0
    while not data.shutdown_event.is_set():
        try:
            await asyncio.wait_for(data.shutdown_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass
        if not session_is_active(session):
            break
        if room_has_humans(room):
            continue
        logger.info("alone_watchdog: no humans in room=%s — shutting down", room.name)
        await shutdown_session_when_alone(
            session,
            data,
            room_name=room.name,
            reason="alone_watchdog",
            job_ctx=job_ctx,
        )
        break


def register_session_lifecycle(
    session: AgentSession,
    data: TalkShowData,
    room: rtc.Room,
    *,
    job_ctx: JobContext | None = None,
) -> None:
    """Stop background loops when the session closes or the last human leaves."""

    @session.on("close")
    def _on_session_close(ev) -> None:  # type: ignore[no-untyped-def]
        data.shutdown_event.set()
        reason = getattr(getattr(ev, "reason", None), "value", ev)
        logger.info("AgentSession closed room=%s reason=%s", room.name, reason)

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        if room_has_humans(room):
            return
        logger.info(
            "last human left room=%s participant=%s — stopping background work",
            room.name,
            participant.identity,
        )
        asyncio.create_task(
            shutdown_session_when_alone(
                session,
                data,
                room_name=room.name,
                reason="last_human_left",
                job_ctx=job_ctx,
            )
        )

    asyncio.create_task(
        _alone_watchdog(session, data, room, job_ctx=job_ctx)
    )
