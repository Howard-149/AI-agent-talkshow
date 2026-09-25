"""Data structures for multi-agent persona drafting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDraft:
    role: str
    speech: str
    next_role: str
    emotion: str | None
    dialogue_example_index: int | None
    history_version: int
    model_latency_s: float = 0.0
