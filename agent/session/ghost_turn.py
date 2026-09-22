"""Inject a scripted human turn without a microphone (ghost session).

The ghost client joins LiveKit as a real remote human so welcome / locale /
TTS / DyStream still run. Canned lines arrive on ``talkshow/control`` and
reuse the same post-heard commit + ``speak_panel_line`` path as mic turns.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from agent.floor.floor_control import peek_floor_next
from agent.session.ghost_flags import ghost_turns_allowed
from agent.session.human_turn import ensure_listen_role_for_human, finalize_human_turn

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from agent.data import TalkShowData
    from agent.floor import TurnController

logger = logging.getLogger(__name__)


async def run_ghost_human_turn(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    text: str,
) -> bool:
    """Treat ``text`` as the human transcript and run host + panel actuators."""
    heard = (text or "").strip()
    if not ghost_turns_allowed():
        logger.warning("ghost_human_turn ignored — TALKSHOW_GHOST_TURNS=0")
        return False
    if not heard:
        logger.info("ghost_human_turn skip — empty text")
        return False
    if data.panel_chain_running:
        logger.info("ghost_human_turn skip — panel chain running")
        return False
    if peek_floor_next(data) != "human":
        logger.info(
            "ghost_human_turn skip — human has no floor (floor_next=%s)",
            peek_floor_next(data),
        )
        if data.turn_log is not None:
            data.turn_log.log(
                "ghost_human_skip",
                reason="no_human_floor",
                room=data.room_name,
                floor_next=peek_floor_next(data),
            )
        return False

    async with data.human_turn_lock:
        return await _run_ghost_human_turn_locked(
            session, data, controller, heard=heard
        )


async def _run_ghost_human_turn_locked(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    heard: str,
) -> bool:
    panel_turn = controller.is_panel_mode()
    if panel_turn:
        controller.apply_listen_persona()
        await ensure_listen_role_for_human(
            session,
            data,
            reason="ghost_human_turn",
            turn_log=data.turn_log,
            room=data.room_name,
        )

    if data.turn_log is not None:
        data.turn_log.log(
            "ghost_human_start",
            room=data.room_name,
            active_role=data.active_role,
            panel_turn=panel_turn,
            heard=heard[:200],
        )

    from agent.adapters.response_parser import ParsedTurn, parse_host_speech
    from agent.emotion import mood_prompt_block
    from agent.show.show_context import host_text_user_hint

    user_hint = f"{host_text_user_hint(heard)}\n\n{mood_prompt_block(data, 'host')}"
    hist = data.show_history.prior_messages()
    t0 = time.monotonic()
    raw = await data.runtime.gemma_client.complete_text(
        user_hint,
        history_messages=hist,
    )
    model_latency_s = time.monotonic() - t0
    parsed_raw = parse_host_speech(raw)
    parsed = ParsedTurn(
        heard=heard,
        reply=parsed_raw.reply,
        handoff_to=parsed_raw.handoff_to,
        next_speaker=parsed_raw.next_speaker,
        emotion=parsed_raw.emotion,
        pad_delta=parsed_raw.pad_delta,
    )

    reply = await finalize_human_turn(
        data,
        heard=heard,
        parsed=parsed,
        panel_turn=panel_turn,
        model_latency_s=model_latency_s,
        done_event="ghost_human_done",
        raw=raw,
    )
    if not reply.strip():
        logger.warning("ghost_human_turn — empty host reply after finalize")
        return False

    from agent.session.session_handoff import speak_panel_line

    spoken = (data.pending_host_speak or reply).strip()
    data.pending_host_speak = ""
    data.runtime.turn_store.consume_turn()
    logger.info("ghost host_reply via speak_panel_line chars=%d", len(spoken))
    await speak_panel_line(
        session,
        data,
        speak_role=controller.listen_role() if panel_turn else data.active_role,
        text=spoken,
        step="host_reply",
    )

    if data.panel_followup_pending and not data.panel_chain_running:
        runner = data.panel_followup_runner
        if runner is not None:
            data.panel_followup_pending = False
            if data.turn_log is not None:
                data.turn_log.log(
                    "panel_trigger",
                    reason="ghost_host_reply_done",
                    floor_next=data.floor_next_speaker or "host",
                    room=data.room_name,
                )
            runner()

    data.touch_activity()
    return True
