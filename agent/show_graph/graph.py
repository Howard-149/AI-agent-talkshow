"""LangGraph wiring for one show beat (human turn, host moderation, or panel round)."""

from __future__ import annotations

from typing import Any

from langgraph.graph import START, StateGraph

from agent.show_graph import nodes as N
from agent.show_graph.state import ShowState

# Node routing is dynamic (nodes return ``Command(goto=...)``); only the fan-in
# from the parallel speak/appraise branches and the terminal edge are static.
_COMMAND_NODES = (
    "human_turn",
    "panel_consume",
    "moderate",
    "hand_raise",
    "panelist_line",
    "grant_human",
    "close",
    "prepare_speak",
    "after_speak",
    "wait_event",
    "finish",
)


def build_show_graph(checkpointer: Any = None) -> Any:
    """One-shot beat graph, or — with a checkpointer — the session graph that
    pauses in ``wait_event`` (``interrupt``) between beats."""
    g: StateGraph = StateGraph(ShowState)
    for name in _COMMAND_NODES:
        g.add_node(name, getattr(N, name))
    g.add_node("speak", N.speak)
    g.add_node("appraise", N.appraise)

    g.add_conditional_edges(
        START,
        N.route_entry,
        {
            "human_turn": "human_turn",
            "moderate": "moderate",
            "panel_consume": "panel_consume",
            "wait_event": "wait_event",
        },
    )
    # Join: after_speak runs once both the playout and all appraisals finished.
    g.add_edge("speak", "after_speak")
    g.add_edge("appraise", "after_speak")
    return g.compile(checkpointer=checkpointer)


_app: Any = None


def show_app() -> Any:
    global _app
    if _app is None:
        _app = build_show_graph()
    return _app
