from __future__ import annotations

import logging

from agent.adapters import PiperTTS
from agent.agents.handoff import HandoffToolsMixin
from agent.config import load_persona_name
from agent.agents.talkshow_agent import TalkShowAgent
from agent.data import TalkShowData

logger = logging.getLogger(__name__)


class CommentatorAgent(HandoffToolsMixin, TalkShowAgent):
    _role = "commentator"

    def __init__(self, *, instructions: str, tts: PiperTTS, data: TalkShowData) -> None:
        self._data = data
        super().__init__(role="commentator", instructions=instructions, tts=tts, data=data)

    async def on_enter(self) -> None:
        logger.info("CommentatorAgent entered session")
        if self._data.silent_handoff:
            return
        name = load_persona_name("commentator")
        await self.session.generate_reply(
            instructions=(
                f"You were just brought in as commentator {name}. "
                "Add a short colorful aside (1–2 sentences) in character."
            )
        )
