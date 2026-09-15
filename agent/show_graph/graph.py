"""LangGraph wiring for moderation-from-queue and host-moderated panel loop."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.show_graph import nodes as N
from agent.show_graph.state import ShowState


def _route(state: ShowState) -> str:
    return state["phase"]


def _passthrough(state: ShowState) -> dict:
    return {}


def build_moderation_graph() -> Any:
    """Host-moderation beat with phase router for barrier re-entry."""
    g: StateGraph = StateGraph(ShowState)

    g.add_node("router", _passthrough)
    g.add_node("mod_start", N.node_mod_start)
    g.add_node("open_floor", N.node_open_floor)
    g.add_node("hand_raise", N.node_hand_raise)
    g.add_node("after_raise", N.node_after_raise)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _route,
        {
            "mod_start": "mod_start",
            "open_floor": "open_floor",
            "hand_raise": "hand_raise",
            "after_raise": "after_raise",
            "done": END,
        },
    )
    g.add_conditional_edges(
        "mod_start",
        _route,
        {"open_floor": "open_floor", "done": END},
    )
    # open_floor / hand_raise set next phase then END (barrier)
    g.add_edge("open_floor", END)
    g.add_edge("hand_raise", END)
    g.add_edge("after_raise", END)
    return g.compile()


def build_panel_graph() -> Any:
    g: StateGraph = StateGraph(ShowState)

    g.add_node("router", _passthrough)
    g.add_node("panel_start", N.node_panel_start)
    g.add_node("panel_consume", N.node_panel_consume)
    g.add_node("panel_after_moderate", N.node_panel_after_moderate)
    g.add_node("panel_close", N.node_panel_close)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _route,
        {
            "panel_start": "panel_start",
            "panel_consume": "panel_consume",
            "panel_after_moderate": "panel_after_moderate",
            "panel_close": "panel_close",
            "panel_moderate": END,
            "done": END,
        },
    )
    g.add_conditional_edges(
        "panel_start",
        _route,
        {"panel_consume": "panel_consume", "done": END},
    )
    g.add_edge("panel_consume", END)
    g.add_conditional_edges(
        "panel_after_moderate",
        _route,
        {
            "panel_consume": "panel_consume",
            "panel_close": "panel_close",
            "done": END,
        },
    )
    g.add_edge("panel_close", END)
    return g.compile()


_moderation_app = None
_panel_app = None


def moderation_app() -> Any:
    global _moderation_app
    if _moderation_app is None:
        _moderation_app = build_moderation_graph()
    return _moderation_app


def panel_app() -> Any:
    global _panel_app
    if _panel_app is None:
        _panel_app = build_panel_graph()
    return _panel_app


def plan_moderation_step(state: ShowState) -> ShowState:
    return moderation_app().invoke(state)  # type: ignore[return-value]


def plan_panel_step(state: ShowState) -> ShowState:
    return panel_app().invoke(state)  # type: ignore[return-value]
