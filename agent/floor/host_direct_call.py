"""When the hand-raise queue is empty, host names one panelist and grants the floor."""

from __future__ import annotations

import logging

from livekit.agents import AgentSession

from agent.adapters.response_parser import parse_host_speech
from agent.config import load_persona_instructions, load_persona_name
from agent.data import TalkShowData
from agent.emotion import (
    apply_mood_from_parsed,
    mood_output_block,
    mood_prompt_block,
)
from agent.floor.floor_control import apply_floor_next, resolve_floor_after_host_speech
from agent.floor.floor_parser import strip_speech_control_tags
from agent.panel.panel_context import (
    next_tag_options,
    panel_role_name_map,
)
from agent.floor.host_lines import host_direct_call_fallback_line, host_intro_speaker_line
from agent.floor import TurnController

logger = logging.getLogger(__name__)


def _host_call_candidates(
    panel_roles: list[str], spoken_roles: set[str], *, allow_repeat: bool
) -> list[str]:
    unspoken = [role for role in panel_roles if role not in spoken_roles]
    return unspoken or (list(panel_roles) if allow_repeat else [])


async def plan_host_direct_call(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    panel_roles: list[str],
    spoken_roles: set[str],
    trigger: str,
) -> tuple[str | None, str, str | None]:
    """
    Queue empty after open floor — host names a panelist and grants direct [next].

    Returns ``(resolved_role, reason, host_line)``. The show graph speaks ``host_line``
    (so listeners appraise it) and dequeues the role; nothing is spoken here.
    """
    from agent.floor.floor_control import is_direct_next
    from agent.floor.floor_parser import fallback_floor_decision

    host = load_persona_name("host")
    text_only = session is None and getattr(data, "floor_interaction", None) is not None
    unspoken = [role for role in panel_roles if role not in spoken_roles]
    call_candidates = _host_call_candidates(
        panel_roles, spoken_roles, allow_repeat=text_only
    )
    if not call_candidates:
        return "close", "no_panelists_left", None

    from agent.emotion.appraisal import await_pending_appraisal

    await await_pending_appraisal(data, "host")
    role_map = panel_role_name_map(panel_roles)
    names_hint = ", ".join(load_persona_name(r) for r in call_candidates)
    tag_opts = next_tag_options(panel_roles, include_host=False)
    call_instruction = (
        (
            "Continue naturally from the topic and latest conversation. Choose ONE panelist for "
            "the next turn; use their name in [reply] and set matching [next]. Do not mention "
            "hand raising, an open floor, or that nobody volunteered."
            if unspoken
            else "Everyone has spoken once, but the conversation can continue. Choose ONE panelist "
            "to speak again based on the latest exchange; use their name in [reply] and set "
            "matching [next]. Do not mention hand raising or an open floor."
        )
        if text_only
        else "You opened the floor for hand raises but nobody raised. Call on ONE panelist directly "
        "— use their name in [reply] and set matching [next]."
    )
    prompt = f"""You are {host}, talk-show host. Trigger: {trigger}.

{call_instruction}
{role_map}
{"Prefer someone not heard yet this round" if unspoken else "Choose from the panelists"}: {names_hint}.
Speak 2–3 natural sentences with a specific question or topic angle FOR THEM.

{mood_prompt_block(data, "host")}

Output exactly:
[reply]: <spoken host line naming them>
[next]: {tag_opts}
{mood_output_block()}"""
    hist = data.show_history.prior_messages(perspective=controller.listen_role())
    system = load_persona_instructions("host", panel_mode=controller.is_panel_mode())
    raw = await data.runtime.gemma_client.complete_text(
        prompt,
        system_prompt=system,
        history_messages=hist,
        raw=True,
        output_role="host",
        reasoning_effort="low",
    )
    parsed = parse_host_speech(raw)
    emotion = apply_mood_from_parsed(
        data,
        "host",
        emotion=parsed.emotion,
    )
    if data.turn_log is not None:
        data.turn_log.log(
            "host_direct_call",
            role="host",
            trigger=trigger,
            reply=(parsed.reply or "")[:400],
            next_tag=parsed.next_speaker,
            emotion=emotion,
            raw=(raw or "")[:1500] or None,
            room=data.room_name,
        )
    text = strip_speech_control_tags(parsed.reply.strip())
    if not text or len(text) < 8:
        pick = call_candidates[0]
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
            resolved = fb if is_direct_next(fb) else call_candidates[0]
            name = load_persona_name(resolved)
            if name.lower() not in text.lower():
                text = (
                    f"{name}, {text[0].lower()}{text[1:]}"
                    if text
                    else host_intro_speaker_line(name)
                )

    apply_floor_next(data, resolved)
    logger.info(
        "host_direct_call next=%s tag=%r emotion=%s trigger=%s",
        resolved,
        parsed.next_speaker,
        emotion,
        trigger,
    )
    return resolved, "host_direct_no_raises", text
