"""Pure decision nodes for host-moderated floor (parity with host_floor.py)."""

from __future__ import annotations

from agent.show_graph.commands import Command, cmd
from agent.show_graph.state import ShowState


def _is_panelist(role: str, panel_roles: list[str]) -> bool:
    return bool(role) and role in panel_roles


def node_mod_start(state: ShowState) -> dict:
    if state.get("interrupt_requested"):
        return {"phase": "done", "pending_commands": [], "barrier": False}
    if state["depth"] >= state["max_depth"]:
        return {"phase": "done", "pending_commands": [], "barrier": False}
    return {"phase": "open_floor", "pending_commands": [], "barrier": False}


def node_open_floor(state: ShowState) -> dict:
    commands: list[Command] = []
    queue = state.get("queue") or []
    if not state["skip_open_floor"] and not queue:
        commands.append(cmd("open_floor_speak", trigger=state["trigger"]))
    elif queue:
        commands.append(cmd("open_floor_skip_log", trigger=state["trigger"]))
    return {
        "phase": "hand_raise",
        "pending_commands": commands,
        "barrier": True,
    }


def node_hand_raise(state: ShowState) -> dict:
    return {
        "phase": "after_raise",
        "pending_commands": [
            cmd(
                "hand_raise_round",
                trigger=state["trigger"],
                turn_idx=state.get("turn_idx", 0),
            )
        ],
        "barrier": True,
    }


def node_after_raise(state: ShowState) -> dict:
    next_role = state.get("next_role") or ""
    grant_reason = state.get("grant_reason") or ""
    panel_roles = state["panel_roles"]
    trigger = state["trigger"]

    if next_role == "close":
        return {
            "phase": "done",
            "outcome": "close",
            "pending_commands": [],
            "barrier": False,
        }
    if next_role == "human":
        return {
            "phase": "done",
            "outcome": "human",
            "pending_commands": [
                cmd("grant_human", reason=grant_reason or "queue_fifo")
            ],
            "barrier": True,
        }
    if _is_panelist(next_role, panel_roles):
        spoken = list(state.get("spoken_roles") or [])
        if next_role not in spoken:
            spoken.append(next_role)
        return {
            "phase": "done",
            "spoken_roles": spoken,
            "outcome": next_role,
            "pending_commands": [
                cmd(
                    "panelist_beat",
                    role=next_role,
                    reason=grant_reason,
                    trigger=trigger,
                    step=f"queue_{next_role}_{trigger}",
                    skip_intro=grant_reason == "host_direct_no_raises",
                )
            ],
            "barrier": True,
        }
    return {
        "phase": "done",
        "outcome": "",
        "pending_commands": [],
        "barrier": False,
    }


def node_panel_start(state: ShowState) -> dict:
    return {
        "phase": "panel_consume",
        "pending_commands": [],
        "barrier": False,
        "close_round": False,
    }


def node_panel_consume(state: ShowState) -> dict:
    if state["turn_idx"] >= state["max_turns"] or state.get("close_round"):
        return {"phase": "panel_close", "pending_commands": [], "barrier": False}

    floor_next = state.get("floor_next") or ""
    panel_roles = state["panel_roles"]
    turn_idx = state["turn_idx"]

    if _is_panelist(floor_next, panel_roles):
        spoken = list(state.get("spoken_roles") or [])
        if floor_next not in spoken:
            spoken.append(floor_next)
        return {
            "phase": "panel_consume",
            "spoken_roles": spoken,
            "turn_idx": turn_idx + 1,
            "pending_commands": [
                cmd(
                    "speak_direct_panelist",
                    role=floor_next,
                    reason="next_tag",
                    step=f"speech_{floor_next}_{turn_idx}",
                )
            ],
            "barrier": True,
            "floor_next": "",
        }

    if floor_next == "human":
        return {
            "phase": "done",
            "outcome": "human",
            "pending_commands": [cmd("grant_human", reason="next_tag")],
            "barrier": True,
        }

    if floor_next == "close":
        return {
            "phase": "panel_close",
            "close_round": True,
            "pending_commands": [],
            "barrier": False,
        }

    # host / unset → runner runs moderation subgraph
    return {
        "phase": "panel_moderate",
        "pending_commands": [],
        "barrier": True,
        "trigger": f"turn_{turn_idx}",
        "skip_open_floor": False,
        "depth": 0,
        "floor_next": floor_next or "host",
    }


def node_panel_after_moderate(state: ShowState) -> dict:
    outcome = state.get("outcome") or ""
    if outcome == "human":
        return {"phase": "done", "pending_commands": [], "barrier": False}
    if outcome == "close":
        return {
            "phase": "panel_close",
            "close_round": True,
            "pending_commands": [],
            "barrier": False,
        }
    spoken = list(state.get("spoken_roles") or [])
    if _is_panelist(outcome, state["panel_roles"]) and outcome not in spoken:
        spoken.append(outcome)
    return {
        "phase": "panel_consume",
        "spoken_roles": spoken,
        "turn_idx": state["turn_idx"] + 1,
        "pending_commands": [],
        "barrier": False,
        "floor_next": "",
    }


def node_panel_close(state: ShowState) -> dict:
    return {
        "phase": "done",
        "pending_commands": [cmd("host_close")],
        "barrier": True,
    }
