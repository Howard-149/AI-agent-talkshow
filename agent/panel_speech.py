from __future__ import annotations

import logging
import os
import time

from livekit.agents import AgentSession

from agent.adapters.response_parser import parse_host_speech
from agent.config import load_persona_name
from agent.data import TalkShowData
from agent.floor_control import apply_floor_next
from agent.floor_parser import strip_next_tag
from agent.panel_context import panel_speaker_roles
from agent.panel_prompts import panel_speech_prompt, panelist_system_prompt
from agent.dialogue_library import format_dialogue_hint
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
    Generate and speak one panelist line.
    Returns consumed [next] role for floor routing.
    """
    controller = TurnController(data.scenario, data)
    panel_roles = controller.panel_speaker_roles()
    name = load_persona_name(speak_role)
    host_name = load_persona_name("host")
    hist = data.show_history.prior_messages()
    latest_human = data.show_history.latest_human_text()

    if speak_role != data.active_role:
        await switch_to_role(session, data, speak_role, reason=f"panel:{step}")

    dialogue_hint = ""
    dlg = data.scenario.dialogue
    if dlg and dlg.library:
        dialogue_hint, data.dialogue_example_index = format_dialogue_hint(
            dlg.library,
            role=speak_role,
            pick=dlg.pick,
            index=data.dialogue_example_index,
        )
    else:
        logger.info("dialogue hint role=%s skipped (no library in scenario)", speak_role)

    prompt = panel_speech_prompt(
        role=speak_role,
        name=name,
        host_name=host_name,
        scenario=data.scenario,
        panel_roles=panel_roles,
        history=data.show_history,
        latest_human=latest_human,
    )
    system_prompt = panelist_system_prompt(
        speak_role,
        scenario=data.scenario,
        panel_roles=panel_roles,
        history=data.show_history,
        dialogue_hint=dialogue_hint,
    )
    logger.info(
        "panel speech prompt role=%s step=%s system_chars=%d user_chars=%d hint_chars=%d pick=%s",
        speak_role,
        step,
        len(system_prompt),
        len(prompt),
        len(dialogue_hint),
        dlg.pick if dlg and dlg.library else "none",
    )
    if os.environ.get("TALKSHOW_LOG_PROMPTS", "").lower() in ("1", "true", "yes"):
        logger.info(
            "TALKSHOW_LOG_PROMPTS role=%s\n--- system ---\n%s\n--- user ---\n%s",
            speak_role,
            system_prompt,
            prompt,
        )
    t0 = time.monotonic()
    raw = await data.runtime.gemma_client.complete_text(
        prompt,
        system_prompt=system_prompt,
        history_messages=hist,
    )
    model_latency_s = time.monotonic() - t0
    parsed = parse_host_speech(raw)
    speech = strip_next_tag(parsed.reply.strip())
    if not speech:
        logger.warning("empty panel speech for role=%s", speak_role)
        speech = (
            "I want to make sure we speak to what the room is discussing — "
            "let me add my view on that."
        )
    normalized = normalize_panelist_speech(
        speak_role,
        speech,
        panel_roles=panel_roles,
        history=data.show_history,
    )
    if normalized != speech:
        logger.info("panel voice normalized for role=%s", speak_role)
        speech = normalized

    valid_next = set(panel_roles) | {"host", "human", "close"}
    if parsed.next_speaker in valid_next:
        next_role = parsed.next_speaker
    else:
        next_role = "host"

    if getattr(data, "turn_log", None) is not None:
        data.turn_log.log(
            "panel_model_done",
            role=speak_role,
            step=step,
            reply=speech,
            reply_len=len(speech),
            next_tag=next_role,
            model_latency_s=round(model_latency_s, 3),
            dialogue_library=dlg.library if dlg and dlg.library else None,
            dialogue_pick=dlg.pick if dlg and dlg.library else None,
            hint_chars=len(dialogue_hint),
            system_chars=len(system_prompt),
            user_chars=len(prompt),
            room=getattr(data, "room_name", ""),
        )

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
