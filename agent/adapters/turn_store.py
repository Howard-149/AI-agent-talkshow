"""Thread-safe handoff of one Gemma turn result between custom STT and StoredReplyLLM."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class TurnResult:
    heard: str
    reply: str


class TurnStore:
    """Shares one Gemma multimodal result between custom STT and passthrough LLM."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: TurnResult | None = None
        self._last_handoff_to: str | None = None

    def set_turn(
        self, heard: str, reply: str, *, handoff_to: str | None = None
    ) -> None:
        with self._lock:
            self._pending = TurnResult(heard=heard, reply=reply)
            self._last_handoff_to = handoff_to

    def consume_turn(self) -> TurnResult | None:
        with self._lock:
            turn = self._pending
            self._pending = None
            return turn

    def consume_handoff_to(self) -> str | None:
        with self._lock:
            target = self._last_handoff_to
            self._last_handoff_to = None
            return target
