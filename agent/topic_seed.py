from __future__ import annotations

import logging

from livekit.agents import AgentSession

from agent.adapters.response_parser import parse_host_speech
from agent.config import load_persona_instructions, load_persona_name
from agent.data import TalkShowData
from agent.floor_control import apply_floor_next, resolve_floor_after_host_speech
from agent.floor_parser import strip_next_tag
from agent.hand_raise_ui import dequeue_hand_raise
from agent.panel_context import (
    format_name_list,
    next_tag_options,
    panel_role_name_map,
    panel_speaker_names,
)
from agent.host_lines import host_direct_call_fallback_line, host_intro_speaker_line
from agent.session_handoff import speak_panel_line
from agent.show_history import append_role
from agent.supervisor import TurnController

logger = logging.getLogger(__name__)


async def host_open_topic(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    trigger: str,
) -> bool:
    """
    Host proactively opens a discussion topic (no human input required).
    Returns True if a topic line was spoken.
    """
    listen = controller.listen_role()
    host = load_persona_name("host")
    panel_roles = controller.panel_speaker_roles()
    panel = format_name_list(panel_speaker_names(data.scenario))
    tag_opts = next_tag_options(panel_roles, include_host=True) + " | human | close"
    has_human = any(line.role_id == "human" for line in data.show_history.lines)

    if has_human:
        situation = (
            "The panel paused — introduce a fresh angle, follow-up, or debate question "
            "based on what the human guest already raised."
        )
    else:
        situation = (
            "The human guest has not spoken yet — open the show with one engaging "
            "topic question the panel can discuss."
        )

    prompt = f"""You are {host}, talk-show host. Trigger: {trigger}.

{situation}

Speak 2–4 sentences to the ROOM in [reply].
Floor rules for [next] — CRITICAL:
- Do NOT call on panelists ({panel}) by name — use [next:host] so the hand-raise queue decides
- Brief angle or follow-up question, then always [next:host]
Do not ask the human a direct question while their floor is frozen.

Output exactly:
[reply]: <spoken host line>
[next]: {tag_opts}
"""
    hist = data.show_history.prior_messages()
    system = load_persona_instructions("host", panel_mode=controller.is_panel_mode())
    raw = await data.runtime.gemma_client.complete_text(
        prompt,
        system_prompt=system,
        history_messages=hist,
    )
    parsed = parse_host_speech(raw)
    text = strip_next_tag(parsed.reply.strip())
    if not text or len(text) < 12:
        logger.info("host_open_topic: empty skip trigger=%s", trigger)
        return False

    resolved = resolve_floor_after_host_speech(
        data,
        spoken=text,
        tagged_next=parsed.next_speaker,
        tee_fallback=False,
    )
    if resolved in panel_roles:
        apply_floor_next(data, "host")
        resolved = "host"
    logger.info(
        "host_open_topic next=%s tag=%r trigger=%s",
        resolved,
        parsed.next_speaker,
        trigger,
    )

    append_role(data, listen, text)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=text,
        step=f"topic_{trigger}",
    )
    logger.info("host_open_topic trigger=%s text=%.80r", trigger, text)
    return True


async def host_direct_call_when_no_raises(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    panel_roles: list[str],
    spoken_roles: set[str],
    trigger: str,
) -> tuple[str | None, str]:
    """
    Queue empty after open floor — host names a panelist and grants direct [next].
    """
    from agent.floor_control import is_direct_next
    from agent.floor_parser import fallback_floor_decision

    listen = controller.listen_role()
    host = load_persona_name("host")
    unspoken = [r for r in panel_roles if r not in spoken_roles]
    if not unspoken:
        return "close", "no_panelists_left"

    role_map = panel_role_name_map(panel_roles)
    names_hint = ", ".join(load_persona_name(r) for r in unspoken)
    tag_opts = next_tag_options(panel_roles, include_host=False)
    prompt = f"""You are {host}, talk-show host. Trigger: {trigger}.

You opened the floor for hand raises but nobody raised.
Call on ONE panelist directly — use their name in [reply] and set matching [next]:
{role_map}
Prefer someone not heard yet this round: {names_hint}.
Speak 2–3 sentences with a question or topic angle FOR THEM.

Output exactly:
[reply]: <spoken host line naming them>
[next]: {tag_opts}
"""
    hist = data.show_history.prior_messages()
    system = load_persona_instructions("host", panel_mode=controller.is_panel_mode())
    raw = await data.runtime.gemma_client.complete_text(
        prompt,
        system_prompt=system,
        history_messages=hist,
    )
    parsed = parse_host_speech(raw)
    text = strip_next_tag(parsed.reply.strip())
    if not text or len(text) < 8:
        pick = unspoken[0]
        name = load_persona_name(pick)
        text = host_direct_call_fallback_line(name)
        resolved = pick
    else:
        resolved = resolve_floor_after_host_speech(
            data,
            spoken=text,
            tagged_next=parsed.next_speaker,
        )
        if not is_direct_next(resolved):
            fb = fallback_floor_decision(
                panel_roles=panel_roles,
                raises=data.hand_raise_queue.snapshot(panel_roles),
                spoken_roles=spoken_roles,
            )
            resolved = fb if is_direct_next(fb) else unspoken[0]
            name = load_persona_name(resolved)
            if name.lower() not in text.lower():
                text = (
                    f"{name}, {text[0].lower()}{text[1:]}"
                    if text
                    else host_intro_speaker_line(name)
                )

    apply_floor_next(data, resolved)
    logger.info(
        "host_direct_call next=%s tag=%r trigger=%s",
        resolved,
        parsed.next_speaker,
        trigger,
    )
    append_role(data, listen, text)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=text,
        step=f"direct_call_{trigger}",
    )
    await dequeue_hand_raise(data, resolved)
    return resolved, "host_direct_no_raises"
