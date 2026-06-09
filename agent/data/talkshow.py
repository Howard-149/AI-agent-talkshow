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
    panel_chain_running: bool = False
    last_human_heard: str = ""
    last_host_panel_tee: str = ""
    floor_next_speaker: str = ""  # Gemma [next]: — direct grant or host=pending
    hand_raise_queue: HandRaiseQueue = field(default_factory=HandRaiseQueue)
    last_activity_ts: float = 0.0
    show_history: ShowHistory = field(default_factory=ShowHistory)
    # (role, text, step) — popped on speech_created; transcript + role_active sync with TTS
    pending_transcripts: deque[tuple[str, str, str]] = field(
        default_factory=deque
    )
    # Wired in main.entrypoint — used to hand off before human-turn LLM/TTS
    agent_session: AgentSession | None = field(default=None, repr=False)
    turn_log: TurnJsonlLogger | None = field(default=None, repr=False)
    room_name: str = ""
    human_turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

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

    def queue_transcript(self, role: str, text: str, *, step: str = "") -> None:
        text = text.strip()
        if text:
            self.pending_transcripts.append((role, text, step))

    def pop_pending_transcript(self) -> tuple[str, str, str] | None:
        if self.pending_transcripts:
            return self.pending_transcripts.popleft()
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
