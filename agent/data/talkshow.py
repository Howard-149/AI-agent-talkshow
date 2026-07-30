from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent.config import ScenarioConfig
from agent.hand_raise_queue import HandRaiseQueue
from agent.show_history import ShowHistory

if TYPE_CHECKING:
    from agent.runtime import TalkShowRuntime
    from livekit.agents import AgentSession

    from agent.hooks.logging import TurnJsonlLogger


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
    user_turn_pending_panel: bool = False  # legacy; use panel_followup_pending
    panel_followup_pending: bool = False  # set after host tee-up TTS; triggers panel on listening
    # True while speak_panel_line is in progress (incl. gaps between avatar chunks).
    speak_line_busy: bool = False
    # Listening fired mid speak_panel_line — start panel after the line finishes.
    panel_followup_deferred: bool = False
    # Wired in main.entrypoint — create_task(_run_panel_followups)
    panel_followup_runner: object | None = field(default=None, repr=False)
    panel_chain_running: bool = False
    last_human_heard: str = ""
    last_host_panel_tee: str = ""
    floor_next_speaker: str = ""  # Gemma [next]: — direct grant or host=pending
    hand_raise_queue: HandRaiseQueue = field(default_factory=HandRaiseQueue)
    # Poll tie-break order among AI panelists (rotates after multi-yes polls).
    panel_priority: list[str] = field(default_factory=list)
    last_activity_ts: float = 0.0
    show_history: ShowHistory = field(default_factory=ShowHistory)
    # (role, text, step) — popped when agent enters "speaking" (audio playout)
    pending_speech_ui: deque[tuple[str, str, str]] = field(
        default_factory=deque
    )
    # Host reply after human STT — spoken via speak_panel_line (not StoredReplyLLM+TTS).
    pending_host_speak: str = ""
    # Wired in main.entrypoint — used to hand off before human-turn LLM/TTS
    agent_session: AgentSession | None = field(default=None, repr=False)
    turn_log: TurnJsonlLogger | None = field(default=None, repr=False)
    room_name: str = ""
    human_turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    dialogue_example_index: int = 0

    @property
    def human_hand_raised(self) -> bool:
        return self.hand_raise_queue.has("human")

    @property
    def human_hand_reason(self) -> str:
        entry = self.hand_raise_queue.get("human")
        return entry.reason if entry else ""

    @property
    def human_hand_topic(self) -> str:
        entry = self.hand_raise_queue.get("human")
        return entry.topic if entry else ""

    def queue_speech_ui(self, role: str, text: str, *, step: str = "") -> None:
        # Empty text allowed: re-assert role_active between avatar chunks without
        # duplicating transcript (emit skips blank text).
        self.pending_speech_ui.append((role, text.strip(), step))

    def pop_pending_speech_ui(self) -> tuple[str, str, str] | None:
        if self.pending_speech_ui:
            return self.pending_speech_ui.popleft()
        return None

    def set_human_hand(
        self,
        *,
        raised: bool,
        reason: str = "",
        topic: str = "",
    ) -> None:
        import time

        if raised:
            self.hand_raise_queue.enqueue("human", reason=reason, topic=topic)
        else:
            self.hand_raise_queue.remove("human")
        self.last_activity_ts = time.time()
        if self.turn_log:
            self.turn_log.log(
                "hand_raise_enqueue" if raised else "hand_raise_remove",
                role="human",
                reason=reason,
                topic=topic,
                room=self.room_name,
                queue=self.hand_raise_queue.roles(),
            )

    def clear_human_hand(self) -> None:
        self.hand_raise_queue.remove("human")

    def touch_activity(self) -> None:
        import time

        self.last_activity_ts = time.time()

    def ensure_panel_priority(self, order: list[str], listen_role: str) -> None:
        """Initialize poll tie-break order from scenario turn order (excludes human/host)."""
        if self.panel_priority:
            return
        self.panel_priority = [
            r for r in order if r not in ("human", listen_role)
        ]

    def rotate_panel_priority(self, winner: str) -> None:
        """Move poll winner to tail so the next tie favors other panelists."""
        winner = winner.strip().lower()
        if winner not in self.panel_priority:
            return
        rest = [r for r in self.panel_priority if r != winner]
        self.panel_priority = rest + [winner]
        if self.turn_log:
            self.turn_log.log(
                "panel_priority_rotate",
                winner=winner,
                priority=list(self.panel_priority),
                room=self.room_name,
            )
