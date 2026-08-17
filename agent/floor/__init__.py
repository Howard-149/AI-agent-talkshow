"""Floor control: hand-raise queue, host moderation, and turn controller."""

from __future__ import annotations

from typing import Any

__all__ = ["TurnController"]


def __getattr__(name: str) -> Any:
    # Lazy export — avoid circular import with talkshow_data → hand_raise_queue.
    if name == "TurnController":
        from agent.floor.turn_controller import TurnController

        return TurnController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
