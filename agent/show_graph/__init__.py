"""
LangGraph show controller — host-moderated floor orchestration.

Phase 1: rule nodes emit Commands; the LiveKit runner executes actuators
(speak / poll / UI). Behavior matches the pre-migration host_floor loops.

Future swap points (do not require another rewrite):
- Host chooses speaker: replace grant node with tool-calling LLM; mask with
  legal set from queue / participants.
- Interrupt / yield: runner cancels actuator; set interrupt_requested; edge
  back to grant / open_floor.
- Real multi-agent: same Command bus; multiple workers execute speak_*;
  graph stays coordinator.
- Emotion per participant: nodes read/write role_emotion; optional VAD
  updater before introduce_and_speak.
- Role enter/exit: mutate participants + UI roster reload; poll/grant iterate
  that list.

LiveKit-only fields stay on TalkShowData — not duplicated into ShowState.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "run_host_moderated_panel",
    "run_host_moderation_from_queue",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from agent.show_graph import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
