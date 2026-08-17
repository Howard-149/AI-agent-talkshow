"""Switch to listen role and run post-human floor open/poll when the queue is empty."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from agent.data import TalkShowData
    from agent.telemetry.turn_jsonl_logger import TurnJsonlLogger

logger = logging.getLogger(__name__)


async def ensure_listen_role_for_human(
    session: AgentSession,
    data: TalkShowData,
    *,
    reason: str = "panel_human_turn",
    turn_log: TurnJsonlLogger | None = None,
    room: str = "",
) -> None:
    """Switch to host/listen role before Gemma human turn — must complete before LLM/TTS."""
    from agent.floor import TurnController
    from agent.session.session_handoff import switch_to_role

    listen = TurnController(data.scenario, data).listen_role()
    if data.active_role == listen:
        return
    entry_from = data.active_role
    await switch_to_role(session, data, listen, reason=reason)
    logger.info("human turn listen ready %s -> %s", entry_from, listen)
    if turn_log is not None:
        turn_log.log(
            "handoff",
            text=f"{entry_from} -> {listen}",
            room=room,
            active_role=listen,
            reason=reason,
        )
