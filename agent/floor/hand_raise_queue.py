"""Strict FIFO hand-raise queue for humans and AI panelists."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.floor.floor_parser import HandRaiseResult


def _queue_eligible(role: str) -> bool:
    from agent.config import load_scenario
    from agent.panel.panel_context import panel_speaker_roles

    key = role.strip().lower()
    if key == "human":
        return True
    return key in panel_speaker_roles(load_scenario())


@dataclass
class HandRaiseEntry:
    role: str
    reason: str = ""
    topic: str = ""
    enqueued_at: float = field(default_factory=time.time)


class HandRaiseQueue:
    """FIFO queue of panelists / human waiting for the host to grant the floor."""

    def __init__(self) -> None:
        self._entries: deque[HandRaiseEntry] = deque()

    def enqueue(self, role: str, *, reason: str = "", topic: str = "") -> None:
        role = role.strip().lower()
        if not _queue_eligible(role):
            return
        self.remove(role)
        self._entries.append(
            HandRaiseEntry(
                role=role,
                reason=reason.strip(),
                topic=topic.strip(),
            )
        )

    def remove(self, role: str) -> bool:
        role = role.strip().lower()
        before = len(self._entries)
        self._entries = deque(e for e in self._entries if e.role != role)
        return len(self._entries) < before

    def pop(self, role: str) -> HandRaiseEntry | None:
        role = role.strip().lower()
        for idx, entry in enumerate(self._entries):
            if entry.role == role:
                del self._entries[idx]
                return entry
        return None

    def clear_all(self) -> None:
        self._entries.clear()

    def clear_roles(self, roles: list[str]) -> None:
        drop = {r.strip().lower() for r in roles}
        self._entries = deque(e for e in self._entries if e.role not in drop)

    def has(self, role: str) -> bool:
        role = role.strip().lower()
        return any(e.role == role for e in self._entries)

    def get(self, role: str) -> HandRaiseEntry | None:
        role = role.strip().lower()
        for entry in self._entries:
            if entry.role == role:
                return entry
        return None

    def roles(self) -> list[str]:
        return [e.role for e in self._entries]

    def snapshot(self, panel_roles: list[str]) -> dict[str, HandRaiseResult]:
        """Build raise results for host moderation from the live queue."""
        from agent.floor.floor_parser import HandRaiseResult

        out: dict[str, HandRaiseResult] = {}
        for role in panel_roles:
            entry = self.get(role)
            out[role] = HandRaiseResult(
                role=role,
                raised=entry is not None,
                reason=entry.reason if entry else "",
                topic=entry.topic if entry else "",
            )
        return out
