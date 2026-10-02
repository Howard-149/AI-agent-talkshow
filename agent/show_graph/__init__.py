"""
LangGraph show graph — the runtime for host-moderated shows.

One invocation runs one show beat (a human turn and the panel round after it, or a
host-moderation beat at session start / idle). Nodes are async and act through
``Actuators`` (``LiveActuators`` in production, fakes in tests). Every spoken line
goes through ``prepare_speak``, which fans out TTS and the listeners' PAD
appraisals in parallel; ``after_speak`` joins them, so a speaker's next prompt
always sees an up-to-date mood.

Not yet in the graph (next phases): a session-long graph that waits for human /
idle / UI events via ``interrupt``, and an LLM host node that picks speakers.
Non-host-moderated turn modes keep the legacy path in agent/panel/panel_speech.py.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "run_host_moderated_panel",
    "run_host_moderation_from_queue",
    "run_show_beat",
    "submit_human_turn",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from agent.show_graph import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
