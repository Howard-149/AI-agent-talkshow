from __future__ import annotations

import logging

from agent.adapters import PiperTTS
from agent.agents.handoff import HandoffToolsMixin
from agent.config import load_persona_name
from agent.agents.talkshow_agent import TalkShowAgent
from agent.data import TalkShowData

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
        await self.session.generate_reply(
            instructions=(
                f"You were just brought in as guest {name}. "
                "Respond briefly in character (1–2 sentences) to continue the show."
            )
        )
