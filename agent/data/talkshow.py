from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent.config import ScenarioConfig
from agent.show_history import ShowHistory

if TYPE_CHECKING:
    from agent.runtime import TalkShowRuntime


@dataclass
class TalkShowData:
    """Shared session state for multi-agent handoff and turn control."""

    scenario: ScenarioConfig
    runtime: TalkShowRuntime
    active_role: str = "host"
    rotation_index: int = 0
    last_handoff_reason: str = ""
    handoff_history: list[dict[str, str]] = field(default_factory=list)
    silent_handoff: bool = False
    user_turn_pending_rotation: bool = False
    user_turn_pending_panel: bool = False
    panel_chain_running: bool = False
    last_human_heard: str = ""
    last_host_panel_tee: str = ""
    show_history: ShowHistory = field(default_factory=ShowHistory)
