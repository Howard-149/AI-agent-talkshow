"""ShowState — decision snapshot for LangGraph floor control."""

from __future__ import annotations

from typing import Literal, TypedDict

from agent.show_graph.commands import Command

Trigger = Literal[
    "session_start",
    "after_human",
    "idle_wait",
    "after_line",
    "host_moderate",
    "panel_turn",
]

Phase = Literal[
    # moderation_from_queue
    "mod_start",
    "open_floor",
    "hand_raise",
    "after_raise",
    # panel outer loop
    "panel_start",
    "panel_consume",
    "panel_moderate",
    "panel_after_moderate",
    "panel_close",
    "done",
]


class ShowState(TypedDict):
    """Extensible floor state. Write-through to TalkShowData after actuators."""

    phase: Phase
    trigger: str
    skip_open_floor: bool
    depth: int
    max_depth: int
    max_turns: int
    turn_idx: int
    participants: list[str]
    panel_roles: list[str]
    listen_role: str
    active_role: str
    floor_next: str
    queue: list[str]
    panel_priority: list[str]
    role_emotion: dict[str, str]
    role_pad: dict[str, tuple[float, float, float]]
    spoken_roles: list[str]
    pending_commands: list[Command]
    # Set by hand_raise_round actuator
    next_role: str
    grant_reason: str
    outcome: str  # last moderation outcome for panel loop
    close_round: bool
    interrupt_requested: bool
    # When True, runner stops invoke and executes pending_commands
    barrier: bool


def empty_commands() -> list[Command]:
    return []


def base_state(
    *,
    phase: Phase,
    trigger: str,
    panel_roles: list[str],
    listen_role: str,
    participants: list[str],
    skip_open_floor: bool = False,
    depth: int = 0,
    max_depth: int = 8,
    max_turns: int = 12,
    turn_idx: int = 0,
    active_role: str = "host",
    floor_next: str = "",
    queue: list[str] | None = None,
    panel_priority: list[str] | None = None,
    role_emotion: dict[str, str] | None = None,
    role_pad: dict[str, tuple[float, float, float]] | None = None,
    spoken_roles: list[str] | None = None,
) -> ShowState:
    return {
        "phase": phase,
        "trigger": trigger,
        "skip_open_floor": skip_open_floor,
        "depth": depth,
        "max_depth": max_depth,
        "max_turns": max_turns,
        "turn_idx": turn_idx,
        "participants": list(participants),
        "panel_roles": list(panel_roles),
        "listen_role": listen_role,
        "active_role": active_role,
        "floor_next": floor_next,
        "queue": list(queue or ()),
        "panel_priority": list(panel_priority or ()),
        "role_emotion": dict(role_emotion or {}),
        "role_pad": dict(role_pad or {}),
        "spoken_roles": list(spoken_roles or ()),
        "pending_commands": [],
        "next_role": "",
        "grant_reason": "",
        "outcome": "",
        "close_round": False,
        "interrupt_requested": False,
        "barrier": False,
    }
