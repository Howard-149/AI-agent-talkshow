"""LiveKit runner — drives LangGraph floor plans and executes Commands."""

from __future__ import annotations

import logging
import os
from typing import Any

from livekit.agents import AgentSession

from agent.data import TalkShowData
from agent.floor import TurnController
from agent.floor.floor_control import apply_floor_next, consume_floor_next, is_direct_next, peek_floor_next
from agent.session.session_lifecycle import should_stop_session_work
from agent.show_graph.actuators import execute_command
from agent.show_graph.graph import plan_moderation_step, plan_panel_step
from agent.show_graph.state import ShowState, base_state

logger = logging.getLogger(__name__)


def _participants(controller: TurnController) -> list[str]:
    listen = controller.listen_role()
    panel = controller.panel_speaker_roles()
    return list(dict.fromkeys([listen, *panel, "human"]))


def _sync_from_data(state: ShowState, data: TalkShowData) -> None:
    state["queue"] = data.hand_raise_queue.roles()
    state["panel_priority"] = list(data.panel_priority)
    state["active_role"] = data.active_role
    state["role_emotion"] = dict(getattr(data, "role_emotion", {}) or {})


def _state_from_session(
    data: TalkShowData,
    controller: TurnController,
    *,
    phase: str,
    trigger: str,
    skip_open_floor: bool = False,
    depth: int = 0,
    spoken_roles: set[str] | None = None,
    turn_idx: int = 0,
) -> ShowState:
    max_depth = int(os.environ.get("TALKSHOW_HOST_MODERATE_MAX_DEPTH", "8"))
    max_turns = int(os.environ.get("TALKSHOW_PANEL_MAX_TURNS", "12"))
    panel_roles = controller.panel_speaker_roles()
    listen = controller.listen_role()
    return base_state(
        phase=phase,  # type: ignore[arg-type]
        trigger=trigger,
        panel_roles=panel_roles,
        listen_role=listen,
        participants=_participants(controller),
        skip_open_floor=skip_open_floor,
        depth=depth,
        max_depth=max_depth,
        max_turns=max_turns,
        turn_idx=turn_idx,
        active_role=data.active_role,
        floor_next=peek_floor_next(data) or "",
        queue=data.hand_raise_queue.roles(),
        panel_priority=list(data.panel_priority),
        role_emotion=dict(getattr(data, "role_emotion", {}) or {}),
        spoken_roles=list(spoken_roles or ()),
    )


async def _run_commands(
    state: ShowState,
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    spoken: set[str],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for command in list(state.get("pending_commands") or []):
        if should_stop_session_work(data, session):
            merged["stopped"] = True
            break
        part = await execute_command(
            command, session, data, controller, spoken_roles=spoken
        )
        merged.update(part)
    state["pending_commands"] = []
    state["barrier"] = False
    _sync_from_data(state, data)
    state["spoken_roles"] = list(spoken)
    return merged


async def run_host_moderation_from_queue(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    trigger: str = "host_moderate",
    skip_open_floor: bool = False,
    depth: int = 0,
    spoken_roles: set[str] | None = None,
) -> str | None:
    """
    LangGraph-orchestrated host moderation beat.
    Returns granted role id, ``human``, ``close``, or None.
    """
    if should_stop_session_work(data, session):
        return None

    spoken = set(spoken_roles or ())
    state = _state_from_session(
        data,
        controller,
        phase="mod_start",
        trigger=trigger,
        skip_open_floor=skip_open_floor,
        depth=depth,
        spoken_roles=spoken,
    )

    while True:
        if should_stop_session_work(data, session):
            return None
        _sync_from_data(state, data)
        state = plan_moderation_step(state)
        result = await _run_commands(state, session, data, controller, spoken)

        if result.get("stopped"):
            return None

        if "next_role" in result:
            state["next_role"] = result.get("next_role") or ""
            state["grant_reason"] = result.get("grant_reason") or ""

        if result.get("nest_pending"):
            nested = await run_host_moderation_from_queue(
                session,
                data,
                controller,
                trigger=f"after_{trigger}",
                depth=depth + 1,
                spoken_roles=spoken,
            )
            if nested in ("human", "close"):
                return nested
            granted = state.get("outcome") or ""
            if granted and is_direct_next(granted, data.scenario):
                return granted
            return nested

        if state["phase"] == "done":
            outcome = state.get("outcome") or ""
            if outcome == "close":
                return "close"
            if outcome == "human":
                return "human"
            if outcome and is_direct_next(outcome, data.scenario):
                return outcome
            return outcome or None


async def run_host_moderated_panel(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """LangGraph-orchestrated panel loop (parity with legacy host_floor)."""
    spoken: set[str] = set()
    state = _state_from_session(
        data,
        controller,
        phase="panel_start",
        trigger="after_human",
        spoken_roles=spoken,
    )
    state = plan_panel_step(state)  # panel_start → panel_consume

    while True:
        if should_stop_session_work(data, session):
            return

        if state["phase"] == "panel_consume":
            consumed = consume_floor_next(data)
            state["floor_next"] = consumed or ""
            state["pending_commands"] = []
            state["barrier"] = False
            _sync_from_data(state, data)
            state = plan_panel_step(state)

        if state["phase"] == "panel_moderate":
            apply_floor_next(data, state.get("floor_next") or "host")
            outcome = await run_host_moderation_from_queue(
                session,
                data,
                controller,
                trigger=state.get("trigger") or f"turn_{state['turn_idx']}",
                spoken_roles=spoken,
            )
            state["outcome"] = outcome or ""
            state["spoken_roles"] = list(spoken)
            state["phase"] = "panel_after_moderate"
            state["pending_commands"] = []
            state["barrier"] = False
            state = plan_panel_step(state)
            continue

        if state["phase"] == "panel_close":
            state = plan_panel_step(state)
            await _run_commands(state, session, data, controller, spoken)
            return

        if state["phase"] == "done":
            await _run_commands(state, session, data, controller, spoken)
            return

        # Barrier (e.g. direct panelist speak or grant_human)
        result = await _run_commands(state, session, data, controller, spoken)
        if result.get("stopped"):
            return
        if state["phase"] == "done":
            return
        # Direct grant path stays on panel_consume for next consume
        if state["phase"] != "panel_consume":
            logger.warning(
                "show_graph panel unexpected phase=%s after barrier", state["phase"]
            )
            return
