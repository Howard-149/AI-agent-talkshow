from __future__ import annotations

import logging
import os

from livekit.agents import AgentSession

from agent.data import TalkShowData
from agent.host_floor import (
    host_speak_session_welcome,
    run_host_moderation_from_queue,
)
from agent.supervisor import TurnController

logger = logging.getLogger(__name__)


async def run_session_opening(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Welcome + open hand-raise queue — host does not seed a topic or call on anyone."""
    if data.silent_handoff or data.panel_chain_running:
        return
    if os.environ.get("TALKSHOW_SKIP_GREETING", "").lower() in ("1", "true", "yes"):
        return
    if not controller.is_host_moderated_mode():
        return

    await host_speak_session_welcome(session, data, controller)
    await run_host_moderation_from_queue(
        session,
        data,
        controller,
        trigger="session_start",
        skip_open_floor=True,
    )
