from __future__ import annotations

import asyncio
import logging

from livekit.agents import AgentSession, JobContext

from agent.config import ScenarioConfig
from agent.control_handler import idle_topic_loop, register_control_channel
from agent.data import TalkShowData
from agent.participant_display import publish_agent_panel_metadata, set_agent_display_name
from agent.room_connect import wait_for_local_participant
from agent.session_opening import run_session_opening
from agent.supervisor import TurnController
from agent.adapters.avatar_video import get_avatar_video_publisher
from agent.ui_events import emit_panel_roster, flush_pending_ui_events

logger = logging.getLogger(__name__)


async def bootstrap_after_connect(
    ctx: JobContext,
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    scenario: ScenarioConfig,
) -> None:
    """
    Runs in parallel with session.start() (which blocks for the whole session).
    Publishes roster/metadata and plays opening once the RTC room is connected.
    """
    try:
        if not await wait_for_local_participant(ctx.room):
            return

        video_pub = get_avatar_video_publisher()
        if video_pub is not None:
            await video_pub.bind_room(ctx.room)
            logger.info("avatar video track bound to room")

        register_control_channel(
            ctx.room, session=session, data=data, controller=controller
        )
        asyncio.create_task(
            idle_topic_loop(session, data, controller, room=ctx.room)
        )

        await set_agent_display_name("host")
        await publish_agent_panel_metadata(scenario)
        await emit_panel_roster(scenario)
        await flush_pending_ui_events()

        if controller.is_panel_mode():
            await run_session_opening(session, data, controller)
            await flush_pending_ui_events()

        data.touch_activity()
        logger.info("bootstrap_after_connect complete room=%s", ctx.room.name)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("bootstrap_after_connect failed room=%s", ctx.room.name)
