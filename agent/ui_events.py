from __future__ import annotations

import json
import logging

from agent.config import ScenarioConfig, load_persona_name
from agent.panel_roster import panel_roster_entries

logger = logging.getLogger(__name__)

UI_TOPIC = "talkshow/ui"


async def publish_ui_event(event_type: str, **fields: object) -> None:
    """Broadcast panel state to talkshow-web (and any client on UI_TOPIC)."""
    try:
        from livekit.agents.job import get_job_context

        ctx = get_job_context()
        payload = json.dumps({"type": event_type, **fields}, ensure_ascii=False).encode(
            "utf-8"
        )
        await ctx.room.local_participant.publish_data(
            payload,
            reliable=True,
            topic=UI_TOPIC,
        )
    except Exception:
        logger.debug("publish_ui_event skipped type=%s", event_type, exc_info=True)


async def emit_panel_roster(scenario: ScenarioConfig) -> None:
    """Register AI panelists for frontend (from scenario + persona yaml)."""
    await publish_ui_event(
        "panel_roster",
        scenario=scenario.id,
        members=panel_roster_entries(scenario),
    )


async def emit_role_active(role: str) -> None:
    await publish_ui_event(
        "role_active",
        role=role,
        name=load_persona_name(role),
    )


async def emit_transcript(
    role: str,
    text: str,
    *,
    final: bool = True,
    step: str = "",
) -> None:
    text = text.strip()
    if not text:
        return
    await publish_ui_event(
        "transcript",
        role=role,
        speaker=load_persona_name(role) if role != "human" else "Human guest",
        text=text,
        final=final,
        step=step,
    )
