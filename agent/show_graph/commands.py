"""Stable command bus between graph nodes and LiveKit actuators."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


CommandOp = Literal[
    "open_floor_speak",
    "open_floor_skip_log",
    "hand_raise_round",
    "grant_panelist",
    "panelist_beat",
    "chain_panelist",
    "grant_human",
    "return_floor_host",
    "speak_direct_panelist",
    "host_close",
]


class Command(TypedDict):
    op: CommandOp
    # Optional fields used by specific ops (role, reason, trigger, step, …).
    role: str
    reason: str
    trigger: str
    step: str
    skip_intro: bool
    turn_idx: int


def cmd(
    op: CommandOp,
    *,
    role: str = "",
    reason: str = "",
    trigger: str = "",
    step: str = "",
    skip_intro: bool = False,
    turn_idx: int = 0,
) -> Command:
    return {
        "op": op,
        "role": role,
        "reason": reason,
        "trigger": trigger,
        "step": step,
        "skip_intro": skip_intro,
        "turn_idx": turn_idx,
    }


def command_ops(commands: list[Command]) -> list[str]:
    """Golden-test helper: list of op names in order."""
    return [c["op"] for c in commands]
