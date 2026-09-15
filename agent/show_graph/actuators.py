"""Execute Commands against LiveKit session using existing floor helpers."""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import AgentSession

from agent.data import TalkShowData
from agent.floor import TurnController
from agent.floor.floor_control import (
    apply_floor_next,
    consume_floor_next,
    is_direct_next,
    is_hand_raise_pending,
    peek_floor_next,
)
from agent.floor.hand_raise_ui import dequeue_hand_raise
from agent.floor.host_floor import (
    _grant_panelist_turn,
    _run_hand_raise_round,
    grant_human_floor,
    host_speak_open_floor,
    return_floor_to_host,
)
from agent.panel.panel_speech import PANEL_HOST_CLOSE, speak_one_panelist
from agent.session.session_handoff import speak_panel_line
from agent.session.session_lifecycle import should_stop_session_work
from agent.show.show_history import append_role
from agent.show_graph.commands import Command
from agent.ui.ui_events import emit_floor_grant

logger = logging.getLogger(__name__)


async def execute_command(
    command: Command,
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    spoken_roles: set[str],
) -> dict[str, Any]:
    """
    Run one command. Returns optional result fields for ShowState
    (next_role, grant_reason, nest_pending, …).
    """
    op = command["op"]
    result: dict[str, Any] = {}

    if should_stop_session_work(data, session) and op not in (
        "open_floor_skip_log",
    ):
        result["stopped"] = True
        return result

    if op == "open_floor_speak":
        await host_speak_open_floor(
            session, data, controller, trigger=command["trigger"] or "host_moderate"
        )
        return result

    if op == "open_floor_skip_log":
        queue_before = data.hand_raise_queue.roles()
        logger.info(
            "open floor skip queue=%s trigger=%s",
            queue_before,
            command["trigger"],
        )
        if data.turn_log:
            data.turn_log.log(
                "open_floor_skip",
                reason="queue_nonempty",
                trigger=command["trigger"],
                room=data.room_name,
                queue=queue_before,
            )
        return result

    if op == "hand_raise_round":
        panel_roles = controller.panel_speaker_roles()
        next_role, grant_reason = await _run_hand_raise_round(
            session,
            data,
            controller,
            panel_roles=panel_roles,
            spoken_roles=spoken_roles,
            turn_idx=command.get("turn_idx", 0),
        )
        result["next_role"] = next_role or ""
        result["grant_reason"] = grant_reason or ""
        return result

    if op == "grant_panelist":
        await _grant_panelist_turn(
            session,
            data,
            controller,
            command["role"],
            grant_reason=command["reason"] or "queue_fifo",
            trigger=command["trigger"] or "host_moderate",
            step=command["step"] or f"queue_{command['role']}",
            skip_intro=bool(command.get("skip_intro")),
        )
        spoken_roles.add(command["role"])
        return result

    if op == "panelist_beat":
        # grant + chain direct [next] + return host (parity with host_floor)
        role = command["role"]
        trigger = command["trigger"] or "host_moderate"
        await _grant_panelist_turn(
            session,
            data,
            controller,
            role,
            grant_reason=command["reason"] or "queue_fifo",
            trigger=trigger,
            step=command["step"] or f"queue_{role}_{trigger}",
            skip_intro=bool(command.get("skip_intro")),
        )
        spoken_roles.add(role)
        while is_direct_next(peek_floor_next(data)):
            chained = consume_floor_next(data)
            await emit_floor_grant(chained, reason="next_tag")
            await dequeue_hand_raise(data, chained)
            await speak_one_panelist(
                session,
                data,
                speak_role=chained,
                step=f"chain_{chained}_{trigger}",
            )
            spoken_roles.add(chained)
        await return_floor_to_host(
            session, data, controller, trigger=f"queue_{trigger}"
        )
        result["nest_pending"] = is_hand_raise_pending(peek_floor_next(data))
        return result

    if op == "chain_panelist":
        chained = command["role"]
        await emit_floor_grant(chained, reason=command["reason"] or "next_tag")
        await dequeue_hand_raise(data, chained)
        await speak_one_panelist(
            session,
            data,
            speak_role=chained,
            step=command["step"] or f"chain_{chained}",
        )
        spoken_roles.add(chained)
        return result

    if op == "grant_human":
        await grant_human_floor(
            session, data, controller, reason=command["reason"] or "queue_fifo"
        )
        return result

    if op == "return_floor_host":
        await return_floor_to_host(
            session,
            data,
            controller,
            trigger=command["trigger"] or "queue",
        )
        return result

    if op == "speak_direct_panelist":
        role = command["role"]
        await dequeue_hand_raise(data, role)
        await emit_floor_grant(role, reason=command["reason"] or "next_tag")
        await speak_one_panelist(
            session,
            data,
            speak_role=role,
            step=command["step"] or f"speech_{role}",
        )
        spoken_roles.add(role)
        return result

    if op == "host_close":
        listen = controller.listen_role()
        append_role(data, listen, PANEL_HOST_CLOSE)
        await speak_panel_line(
            session,
            data,
            speak_role=listen,
            text=PANEL_HOST_CLOSE,
            step="close_round",
        )
        return result

    logger.warning("unknown floor command op=%s", op)
    return result
