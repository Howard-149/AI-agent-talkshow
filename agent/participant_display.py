from __future__ import annotations

import json
import logging

from agent.config import ScenarioConfig, load_persona_name
from agent.panel_roster import panel_roster_entries

logger = logging.getLogger(__name__)

_ROLE_LABELS = {
    "host": "Host",
    "guest": "Guest",
    "commentator": "Commentator",
}


def role_display_name(role: str) -> str:
    label = _ROLE_LABELS.get(role, role.title())
    return f"{load_persona_name(role)} ({label})"


async def set_agent_display_name(role: str) -> None:
    """One RTC participant; UI shows virtual panel — keep RTC name stable."""
    try:
        from livekit.agents.job import get_job_context

        ctx = get_job_context()
        lp = ctx.room.local_participant
        name = "Talkshow Studio"
        if hasattr(lp, "set_name"):
            await lp.set_name(name)
            logger.debug("participant display name → %s (active role=%s)", name, role)
    except Exception:
        logger.debug("set_agent_display_name skipped", exc_info=True)


async def publish_agent_panel_metadata(scenario: ScenarioConfig) -> None:
    """Expose roster on agent RTC participant for clients that join late."""
    try:
        from livekit.agents.job import get_job_context

        ctx = get_job_context()
        lp = ctx.room.local_participant
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
    except Exception:
        logger.debug("publish_agent_panel_metadata skipped", exc_info=True)
