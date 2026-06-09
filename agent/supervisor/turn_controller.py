from __future__ import annotations

import logging

from agent.config import ScenarioConfig
from agent.data.talkshow import TalkShowData

logger = logging.getLogger(__name__)

PANEL_MODES = frozenset({"panel_round_robin", "host_moderated"})


class TurnController:
    """Who speaks next: tool handoffs, scenario rotation, or host-moderated floor."""

    def __init__(self, scenario: ScenarioConfig, data: TalkShowData) -> None:
        self._scenario = scenario
        self._data = data

    def record_handoff(self, *, to_role: str, reason: str) -> None:
        entry = {
            "from": self._data.active_role,
            "to": to_role,
            "reason": reason,
        }
        self._data.handoff_history.append(entry)
        self._data.last_handoff_reason = reason
        self._data.active_role = to_role
        order = self._scenario.turn_control.order
        if to_role in order:
            self._data.rotation_index = order.index(to_role)
        logger.info("handoff %s → %s (%s)", entry["from"], to_role, reason)

    def apply_persona_for_role(self, role: str) -> None:
        from agent.config import load_persona_instructions

        panel = self.is_panel_mode()
        instructions = load_persona_instructions(role, panel_mode=panel and role == "host")
        self._data.runtime.gemma_client.set_persona(role, instructions)

    def next_role_after_user_turn(self) -> str | None:
        tc = self._scenario.turn_control
        if tc.mode != "rotate_after_user":
            return None
        order = tc.order
        if not order:
            return None
        idx = (self._data.rotation_index + 1) % len(order)
        return order[idx]

    def advance_rotation_after_assistant(self) -> str | None:
        """After the active agent finishes replying to the user, optionally cycle role."""
        next_role = self.next_role_after_user_turn()
        if not next_role or next_role == self._data.active_role:
            return None
        self.record_handoff(to_role=next_role, reason="rotate_after_user")
        self.apply_persona_for_role(next_role)
        return next_role

    def turn_mode(self) -> str:
        return self._scenario.turn_control.mode

    def is_panel_mode(self) -> bool:
        return self.turn_mode() in PANEL_MODES

    def is_host_moderated_mode(self) -> bool:
        return self.turn_mode() == "host_moderated"

    def listen_role(self) -> str:
        return self._scenario.turn_control.listen_role

    def apply_listen_persona(self) -> None:
        """Human turn: host persona with panel moderator rules when applicable."""
        role = self.listen_role()
        from agent.config import load_persona_instructions

        panel = self.is_panel_mode()
        instructions = load_persona_instructions(
            role, panel_mode=panel and role == "host"
        )
        self._data.runtime.gemma_client.set_persona(role, instructions)

    def panel_speaker_roles(self) -> list[str]:
        """AI panelists who may take the floor (excludes human and listen/host role)."""
        listen = self.listen_role()
        return [
            r
            for r in self._scenario.turn_control.order
            if r not in ("human", listen)
        ]

    def panel_followup_roles(self) -> list[str]:
        """Fixed-order panel speakers (panel_round_robin only)."""
        return self.panel_speaker_roles()
