"""Set RTC display name and publish virtual panel metadata for the UI."""

from __future__ import annotations

import json
import logging

from agent.config import ScenarioConfig
from agent.panel.panel_roster import panel_roster_entries
from agent.ui.room_connect import local_participant_ready, wait_for_local_participant

logger = logging.getLogger(__name__)

async def set_agent_display_name(role: str) -> None:
    """One RTC participant; UI shows virtual panel — keep RTC name stable."""
    try:
        from livekit.agents.job import get_job_context

        ctx = get_job_context()
        room = ctx.room
        if not local_participant_ready(room):
            if not await wait_for_local_participant(room, timeout_sec=10.0):
                logger.debug(
                    "set_agent_display_name skipped role=%s (room not connected)",
                    role,
                )
                return
        lp = room.local_participant
        name = "Talkshow Studio"
        if hasattr(lp, "set_name"):
            await lp.set_name(name)
            logger.debug("participant display name → %s (active role=%s)", name, role)
    except Exception as exc:
        msg = str(exc).lower()
        if "before connecting" in msg or "local participant" in msg:
            logger.debug("set_agent_display_name skipped role=%s (not connected)", role)
            return
        logger.warning("set_agent_display_name failed role=%s: %s", role, exc)


async def publish_agent_panel_metadata(scenario: ScenarioConfig) -> None:
    """Expose roster on agent RTC participant for clients that join late."""
    try:
        from livekit.agents.job import get_job_context

        ctx = get_job_context()
        room = ctx.room
        if not local_participant_ready(room):
            if not await wait_for_local_participant(room, timeout_sec=10.0):
                logger.debug(
                    "publish_agent_panel_metadata skipped (room not connected)"
                )
                return
        lp = room.local_participant
        payload = json.dumps(
            {
                "talkshowAgent": True,
                "scenario": scenario.id,
                "panelRoster": panel_roster_entries(scenario),
            },
            ensure_ascii=False,
        )
        if hasattr(lp, "set_metadata"):
            await lp.set_metadata(payload)
            logger.info("agent participant metadata → scenario=%s", scenario.id)
    except Exception as exc:
        msg = str(exc).lower()
        if "before connecting" in msg or "local participant" in msg:
            logger.debug("publish_agent_panel_metadata skipped (not connected)")
            return
        logger.warning("publish_agent_panel_metadata failed: %s", exc)
