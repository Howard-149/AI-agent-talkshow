"""
LangGraph show graph — the runtime for host-moderated shows.

One long-lived graph per session (``ShowSession``, agent/show_graph/session.py):
it pauses in ``wait_event`` (LangGraph ``interrupt``) and each event — the
opening, a human turn, an idle/moderation trigger — resumes it for one show beat
(host reply, moderation, panel lines, grant human / close) before it pauses again.
``TALKSHOW_SHOW_SESSION_GRAPH=0`` falls back to one graph invocation per beat.

Nodes are async and act through ``Actuators`` (``LiveActuators`` in production,
fakes in tests). Every spoken line goes through ``prepare_speak``, which fans out
TTS and the listeners' PAD appraisals in parallel; ``after_speak`` joins them, so
a speaker's next prompt always sees an up-to-date mood.

Not yet: an LLM host node that picks speakers (phase 4). Non-host-moderated turn
modes keep the legacy path in agent/panel/panel_speech.py.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "open_show_session",
    "run_host_moderated_panel",
    "run_host_moderation_from_queue",
    "run_show_beat",
    "start_show_session",
    "submit_human_turn",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from agent.show_graph import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
