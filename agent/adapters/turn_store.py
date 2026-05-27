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

    def set_turn(self, heard: str, reply: str) -> None:
        with self._lock:
            self._pending = TurnResult(heard=heard, reply=reply)

    def consume_turn(self) -> TurnResult | None:
        with self._lock:
            turn = self._pending
            self._pending = None
            return turn
