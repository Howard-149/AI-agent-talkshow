"""Listen-role handoff and shared post-heard commit for mic + ghost human turns."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from agent.adapters.response_parser import ParsedTurn
    from agent.data import TalkShowData
    from agent.telemetry.turn_jsonl_logger import TurnJsonlLogger

logger = logging.getLogger(__name__)


async def ensure_listen_role_for_human(
    session: AgentSession,
    data: TalkShowData,
    *,
    reason: str = "panel_human_turn",
    turn_log: TurnJsonlLogger | None = None,
    room: str = "",
) -> None:
    """Switch to host/listen role before Gemma human turn — must complete before LLM/TTS."""
    from agent.floor import TurnController
    from agent.session.session_handoff import switch_to_role

    listen = TurnController(data.scenario, data).listen_role()
    if data.active_role == listen:
        return
    entry_from = data.active_role
    await switch_to_role(session, data, listen, reason=reason)
    logger.info("human turn listen ready %s -> %s", entry_from, listen)
    if turn_log is not None:
        turn_log.log(
            "handoff",
            text=f"{entry_from} -> {listen}",
            room=room,
            active_role=listen,
            reason=reason,
        )


async def finalize_human_turn(
    data: TalkShowData,
    *,
    heard: str,
    parsed: ParsedTurn,
    panel_turn: bool,
    model_latency_s: float,
    done_event: str = "gemma_stt_done",
    raw: str | None = None,
) -> str:
    """Sanitize host reply, apply floor / PAD / history, and queue host TTS.

    Shared by Gemma audio-in and the ghost-session text inject path.
    Returns the spoken host reply (also stored on ``data.pending_host_speak``).
    """
    from agent.emotion import apply_mood_from_parsed, get_role_emotion

    reply = parsed.reply
    resolved_next = parsed.next_speaker
    emotion = apply_mood_from_parsed(
        data,
        "host",
        emotion=parsed.emotion,
        pad_delta=parsed.pad_delta,
    )
    if panel_turn:
        from agent.floor.floor_control import resolve_floor_after_host_speech
        from agent.show.show_context import host_used_tee_fallback, sanitize_host_panel_reply

        fitted = sanitize_host_panel_reply(reply, heard, data.scenario)
        if fitted != reply:
            logger.info(
                "host reply reshaped for panel floor (was %r)",
                reply[:80],
            )
        reply = fitted
        from agent.floor.floor_parser import strip_speech_control_tags

        reply = strip_speech_control_tags(reply)
        resolved_next = resolve_floor_after_host_speech(
            data,
            spoken=reply,
            tagged_next=parsed.next_speaker,
            tee_fallback=host_used_tee_fallback(reply, heard, data.scenario),
            after_human_turn=True,
        )
        logger.info(
            "human turn floor next=%s tag=%r emotion=%s",
            resolved_next,
            parsed.next_speaker,
            emotion,
        )
    else:
        from agent.floor.floor_parser import strip_speech_control_tags

        reply = strip_speech_control_tags(reply)

    from agent.locale.localize import texts_for_needed_locales
    from agent.show.show_history import append_human, append_role
    from agent.ui.ui_events import emit_floor_grant, emit_transcript

    append_human(data, heard)
    append_role(data, "host", reply)
    human_texts = await texts_for_needed_locales(data, heard)
    await emit_transcript("human", heard, step="human_turn", texts=human_texts)
    if data.human_hand_raised:
        await emit_floor_grant("human", reason="human_spoke_hand_up")
    # Spoken via speak_panel_line (TalkShowAgent.on_user_turn_completed or ghost_turn).
    data.pending_host_speak = reply
    data.last_human_heard = heard
    if panel_turn:
        data.last_host_panel_tee = reply
        data.panel_followup_pending = True

    data.runtime.turn_store.set_turn(heard, reply, handoff_to=parsed.handoff_to)

    if data.turn_log is not None:
        data.turn_log.log(
            done_event,
            heard=heard,
            reply=reply,
            model_latency_s=round(model_latency_s, 3),
            floor_next=data.floor_next_speaker,
            tagged_next=parsed.next_speaker,
            resolved_next=resolved_next,
            emotion=get_role_emotion(data, "host"),
            pad_delta=list(parsed.pad_delta) if parsed.pad_delta is not None else None,
            has_pad_tag="[pad]" in (raw or "").lower(),
            raw=(raw or "")[:1500] or None,
            room=data.room_name,
            active_role=data.active_role,
            panel_followup_pending=data.panel_followup_pending,
        )

    logger.info(
        "heard=%r reply_len=%d floor_next=%s emotion=%s",
        heard[:80],
        len(reply),
        data.floor_next_speaker,
        get_role_emotion(data, "host"),
    )
    return reply
