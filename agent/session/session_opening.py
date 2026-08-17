"""Session welcome and initial open floor / hand-raise queue for host-moderated mode."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from livekit.agents import AgentSession

from agent.data import TalkShowData
from agent.floor.host_floor import (
    host_speak_session_welcome,
    run_host_moderation_from_queue,
)
from agent.floor import TurnController

if TYPE_CHECKING:
    from livekit import rtc

logger = logging.getLogger(__name__)


async def run_session_opening(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    room: rtc.Room | None = None,
) -> None:
    """Welcome + open hand-raise queue — host does not seed a topic or call on anyone."""
    if data.silent_handoff or data.panel_chain_running:
        return
    if os.environ.get("TALKSHOW_SKIP_GREETING", "").lower() in ("1", "true", "yes"):
        return
    if not controller.is_host_moderated_mode():
        return

    # Need the joining human present so token metadata locale is visible (no set_locale wait).
    if room is not None:
        from agent.locale.viewer_locales import (
            recompute_needed_locales,
            wait_for_remote_humans,
        )

        await wait_for_remote_humans(room)
        needed = recompute_needed_locales(data, room)
        logger.info("session opening locales=%s", sorted(needed))

    await host_speak_session_welcome(session, data, controller)
    await run_host_moderation_from_queue(
        session,
        data,
        controller,
        trigger="session_start",
        skip_open_floor=True,
    )
