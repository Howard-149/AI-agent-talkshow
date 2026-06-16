from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.floor_parser import HandRaiseResult

VALID_QUEUE_ROLES = frozenset({"commentator", "guest", "human"})


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
        if role not in VALID_QUEUE_ROLES:
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

    def peek_first(self) -> HandRaiseEntry | None:
        if not self._entries:
            return None
        return self._entries[0]

    def snapshot(self, panel_roles: list[str]) -> dict[str, HandRaiseResult]:
        """Build raise results for host moderation from the live queue."""
        from agent.floor_parser import HandRaiseResult

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

    def raised_roles(self, panel_roles: list[str]) -> list[str]:
        allowed = set(panel_roles) | {"human"}
        return [r for r in self.roles() if r in allowed]

    def peek_first_eligible(
        self,
        panel_roles: list[str],
    ) -> HandRaiseEntry | None:
        """FIFO head among human + panel roles."""
        panel = set(panel_roles)
        for entry in self._entries:
            if entry.role == "human":
                return entry
            if entry.role in panel:
                return entry
        return None

    def merge_poll_raise(self, role: str, *, reason: str = "", topic: str = "") -> None:
        """Apply AI poll yes — new roles append; existing roles keep their FIFO slot."""
        role = role.strip().lower()
        if role not in VALID_QUEUE_ROLES or role == "human":
            return
        existing = self.get(role)
        if existing is not None:
            existing.reason = reason.strip()
            existing.topic = topic.strip()
            return
        self._entries.append(
            HandRaiseEntry(
                role=role,
                reason=reason.strip(),
                topic=topic.strip(),
            )
        )

    def apply_poll_batch(
        self,
        raises: dict[str, HandRaiseResult],
        panel_priority: list[str],
    ) -> tuple[str | None, list[str], bool]:
        """
        Apply AI poll: drop nos; among yes pick one by ``panel_priority``; enqueue winner only.
        Human queue entries are never touched. Returns (winner, yes_roles, had_tie).
        """
        yes_roles: list[str] = []
        for role, hr in raises.items():
            if role == "human":
                continue
            if hr.raised:
                yes_roles.append(role)
            else:
                self.remove(role)

        if not yes_roles:
            return None, [], False

        yes_set = set(yes_roles)
        winner: str | None = None
        for role in panel_priority:
            if role in yes_set:
                winner = role
                break
        if winner is None:
            winner = yes_roles[0]

        hr = raises[winner]
        self.merge_poll_raise(winner, reason=hr.reason, topic=hr.topic)
        return winner, yes_roles, len(yes_roles) >= 2
