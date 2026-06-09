from __future__ import annotations

import logging

from livekit.agents import AgentSession

from agent.adapters.response_parser import parse_host_speech
from agent.config import load_persona_name
from agent.data import TalkShowData
from agent.floor_control import apply_floor_next
from agent.floor_parser import strip_next_tag
from agent.panel_prompts import panel_speech_prompt, panelist_system_for_text
from agent.session_handoff import speak_panel_line, switch_to_role
from agent.panel_voice import normalize_panelist_speech
from agent.show_history import append_role
from agent.supervisor import TurnController

logger = logging.getLogger(__name__)

PANEL_HOST_CLOSE = (
    "That wraps our panel on this round. Back to you — "
    "what would you like to add or ask?"
)


async def speak_one_panelist(
    session: AgentSession,
    data: TalkShowData,
    *,
    speak_role: str,
    step: str,
) -> str:
    """
    Generate and speak one panelist line (Ryan / Amy).
    Returns consumed [next] role for floor routing (commentator|guest|host|human|close).
    """
    name = load_persona_name(speak_role)
    host_name = load_persona_name("host")
    hist = data.show_history.prior_messages()
    latest_human = data.show_history.latest_human_text()
    guest_spoken = data.show_history.role_has_spoken("guest")
    commentator_spoken = data.show_history.role_has_spoken("commentator")
    other_has_spoken = guest_spoken if speak_role == "commentator" else commentator_spoken

    if speak_role != data.active_role:
        await switch_to_role(session, data, speak_role, reason=f"panel:{step}")

    prompt = panel_speech_prompt(
        role=speak_role,
        name=name,
        host_name=host_name,
        latest_human=latest_human,
    )
    raw = await data.runtime.gemma_client.complete_text(
        prompt,
        system_prompt=panelist_system_for_text(
            speak_role, other_has_spoken=other_has_spoken
        ),
        history_messages=hist,
    )
    parsed = parse_host_speech(raw)
    speech = strip_next_tag(parsed.reply.strip())
    if not speech:
        logger.warning("empty panel speech for role=%s", speak_role)
        speech = (
            "I want to make sure we speak to what our guest raised — "
            "let me add my view on that."
        )
    normalized = normalize_panelist_speech(
        speak_role, speech, guest_has_spoken=guest_spoken
    )
    if normalized != speech:
        logger.info("panel voice normalized for role=%s", speak_role)
        speech = normalized

    if parsed.next_speaker in ("commentator", "guest", "human", "close"):
        next_role = parsed.next_speaker
    else:
        next_role = "host"

    append_role(data, speak_role, speech)
    await speak_panel_line(
        session,
        data,
        speak_role=speak_role,
        text=speech,
        step=step,
    )
    apply_floor_next(data, next_role)
    logger.info("panelist spoke role=%s tagged next=%s", speak_role, next_role)
    return next_role


async def run_panel_round(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Dispatch fixed-order panel or host-moderated floor control."""
    if controller.is_host_moderated_mode():
        from agent.host_floor import run_host_moderated_panel

        await run_host_moderated_panel(session, data, controller)
        return

    listen = controller.listen_role()

    for role in controller.panel_followup_roles():
        await speak_one_panelist(
            session,
            data,
            speak_role=role,
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
