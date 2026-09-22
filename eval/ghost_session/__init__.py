"""Ghost session: join LiveKit as a silent human and inject scripted turns."""

from __future__ import annotations

__all__ = ["run_ghost_session"]


def __getattr__(name: str):
    if name == "run_ghost_session":
        from eval.ghost_session.client import run_ghost_session

        return run_ghost_session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
