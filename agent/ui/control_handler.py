"""Handle talkshow/control events and drive idle-topic / queue moderation loops."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from livekit import rtc
from livekit.agents import AgentSession

from agent.ui.control_events import CONTROL_TOPIC, parse_control_payload
from agent.data import TalkShowData
from agent.floor.floor_control import apply_floor_next, peek_floor_next
from agent.floor.hand_raise_ui import dequeue_hand_raise
from agent.floor.host_floor import run_host_moderation_from_queue
from agent.session.session_lifecycle import (
    room_has_humans,
    session_is_active,
    shutdown_session_when_alone,
)
from agent.floor import TurnController
from agent.ui.ui_events import emit_hand_raise, emit_queue_state
from agent.locale.viewer_locales import recompute_needed_locales, set_viewer_locale

logger = logging.getLogger(__name__)


def register_control_channel(
    room: rtc.Room,
    *,
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Listen for human UI events on talkshow/control."""

    async def _apply_human_hand(*, raised: bool, topic: str, reason: str) -> None:
        if raised:
            data.set_human_hand(raised=True, topic=topic, reason=reason)
            await emit_hand_raise("human", True, reason=reason, topic=topic)
        else:
            await dequeue_hand_raise(data, "human")
        await emit_queue_state(data, phase="human_ui")

    async def _apply_set_locale(*, identity: str, locale: str) -> None:
        needed = set_viewer_locale(
            data, room, identity=identity, locale=locale
        )
        logger.info(
            "set_locale identity=%s locale=%s needed=%s",
            identity,
            locale,
            sorted(needed),
        )

    @room.on("data_received")
    def _on_data_received(ev: rtc.DataPacket) -> None:
        if ev.topic != CONTROL_TOPIC:
            return
        parsed = parse_control_payload(ev.data)
        if parsed is None:
            return
        if parsed.type == "hand_raise":
            asyncio.create_task(
                _apply_human_hand(
                    raised=parsed.raised,
                    topic=parsed.topic,
                    reason=parsed.reason,
                )
            )
            return
        if parsed.type == "set_locale":
            participant = ev.participant
            identity = (
                participant.identity
                if participant is not None
                else ""
            )
            if not identity or not parsed.locale:
                return
            asyncio.create_task(
                _apply_set_locale(identity=identity, locale=parsed.locale)
            )

    @room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        recompute_needed_locales(data, room)

    @room.on("participant_disconnected")
    def _on_participant_disconnected_locales(
        participant: rtc.RemoteParticipant,
    ) -> None:
        identity = participant.identity or ""
        data.viewer_locale_by_identity.pop(identity, None)
        recompute_needed_locales(data, room)

    # Seed from anyone already in the room
    recompute_needed_locales(data, room)


async def idle_topic_loop(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    room: rtc.Room,
) -> None:
    """When quiet, host opens the hand-raise queue — no auto topic or direct call-outs."""
    if not controller.is_host_moderated_mode():
        return
    interval = 5
    idle_sec = int(os.environ.get("TALKSHOW_IDLE_TOPIC_SEC", "60"))
    while not data.shutdown_event.is_set():
        try:
            await asyncio.wait_for(data.shutdown_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass

        if not session_is_active(session):
            break
        if not room_has_humans(room):
            await shutdown_session_when_alone(
                session,
                data,
                room_name=room.name,
                reason="idle_no_humans",
            )
            break
        if data.panel_chain_running or data.panel_followup_pending:
            continue
        if peek_floor_next(data) == "human":
            continue
        if data.active_role != controller.listen_role():
            continue
        if data.last_activity_ts <= 0:
            continue
        if time.time() - data.last_activity_ts < idle_sec:
            continue

        apply_floor_next(data, "host")
        data.panel_chain_running = True
        try:
            await run_host_moderation_from_queue(
                session, data, controller, trigger="idle_wait"
            )
        except RuntimeError as exc:
            if "isn't running" in str(exc):
                logger.info("idle_topic_loop stopped — session no longer running")
                break
            raise
        finally:
            data.panel_chain_running = False
        data.touch_activity()
