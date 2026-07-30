from __future__ import annotations

import logging

from agent.adapters import PiperTTS
from agent.agents.handoff import HandoffToolsMixin
from agent.agents.talkshow_agent import TalkShowAgent
from agent.config import load_persona_name
from agent.data import TalkShowData
from agent.session_handoff import speak_panel_line

logger = logging.getLogger(__name__)


class GuestAgent(HandoffToolsMixin, TalkShowAgent):
    _role = "guest"

    def __init__(self, *, instructions: str, tts: PiperTTS, data: TalkShowData) -> None:
        self._data = data
        super().__init__(role="guest", instructions=instructions, tts=tts, data=data)

    async def on_enter(self) -> None:
        logger.info("GuestAgent entered session")
        if self._data.silent_handoff:
            return
        name = load_persona_name("guest")
        await speak_panel_line(
            self.session,
            self._data,
            speak_role="guest",
            text=f"Thanks for having me — {name} here.",
            step="guest_enter",
        )
