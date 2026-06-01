from __future__ import annotations

import logging

from livekit.agents import AgentSession

from agent.config import load_persona_name
from agent.data import TalkShowData
from agent.panel_prompts import panel_speech_prompt, panelist_system_for_text
from agent.session_handoff import speak_panel_line
from agent.panel_voice import normalize_panelist_speech
from agent.show_history import append_role
from agent.supervisor import TurnController

logger = logging.getLogger(__name__)

PANEL_HOST_CLOSE = (
    "That wraps our panel on this round. Back to you — "
    "what would you like to add or ask?"
)


async def run_panel_round(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Host tee is already in show_history; Ryan → Amy → host close."""
    listen = controller.listen_role()
    host_name = load_persona_name("host")

    for role in controller.panel_followup_roles():
        name = load_persona_name(role)
        hist = data.show_history.prior_messages()
        latest_human = data.show_history.latest_human_text()
        prompt = panel_speech_prompt(
            role=role,
            name=name,
            host_name=host_name,
            latest_human=latest_human,
        )
        speech = await data.runtime.gemma_client.complete_text(
            prompt,
            system_prompt=panelist_system_for_text(role),
            history_messages=hist,
        )
        if not speech.strip():
            logger.warning("empty panel speech for role=%s", role)
            speech = (
                "I want to make sure we speak to what our guest raised — "
                "let me add my view on that."
            )
        raw = speech.strip()
        speech = normalize_panelist_speech(role, raw)
        if speech != raw:
            logger.info("panel voice normalized for role=%s", role)
        append_role(data, role, speech)
        await speak_panel_line(
            session,
            data,
            speak_role=role,
            text=speech,
            step=f"speech_{role}",
        )

    append_role(data, listen, PANEL_HOST_CLOSE)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=PANEL_HOST_CLOSE,
        step="close_round",
    )
