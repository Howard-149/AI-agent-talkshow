from __future__ import annotations

import asyncio
import logging
import os
import time

from livekit import rtc
from livekit.agents import AgentSession

from agent.control_events import CONTROL_TOPIC, parse_control_payload
from agent.data import TalkShowData
from agent.floor_control import apply_floor_next, peek_floor_next
from agent.hand_raise_ui import dequeue_hand_raise
from agent.host_floor import run_host_moderation_from_queue
from agent.supervisor import TurnController
from agent.ui_events import emit_hand_raise, emit_queue_state

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

    @room.on("data_received")
    def _on_data_received(ev: rtc.DataPacket) -> None:
        if ev.topic != CONTROL_TOPIC:
            return
        parsed = parse_control_payload(ev.data)
        if parsed is None or parsed.type != "hand_raise":
            return
        asyncio.create_task(
            _apply_human_hand(
                raised=parsed.raised,
                topic=parsed.topic,
                reason=parsed.reason,
            )
        )


async def idle_topic_loop(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """When quiet, host opens the hand-raise queue — no auto topic or direct call-outs."""
    if not controller.is_host_moderated_mode():
        return
    interval = 5
    idle_sec = int(os.environ.get("TALKSHOW_IDLE_TOPIC_SEC", "60"))
    while True:
        await asyncio.sleep(interval)
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
        finally:
            data.panel_chain_running = False
        data.touch_activity()
