from __future__ import annotations

import logging

from agent.config import load_persona_name

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
    """One RTC participant; update visible name on handoff (Playground participant list)."""
    try:
        from livekit.agents.job import get_job_context

        ctx = get_job_context()
        lp = ctx.room.local_participant
        name = role_display_name(role)
        if hasattr(lp, "set_name"):
            await lp.set_name(name)
            logger.info("participant display name → %s", name)
    except Exception:
        logger.debug("set_agent_display_name skipped", exc_info=True)
