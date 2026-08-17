"""Commentator panelist agent — enters floor via handoff and speaks panel lines."""

from __future__ import annotations

import logging

from livekit.agents import tts as agents_tts

from agent.agents.handoff import HandoffToolsMixin
from agent.agents.talkshow_agent import TalkShowAgent
from agent.config import load_persona_name
from agent.data import TalkShowData
from agent.session.session_handoff import speak_panel_line

logger = logging.getLogger(__name__)


class CommentatorAgent(HandoffToolsMixin, TalkShowAgent):
    _role = "commentator"

    def __init__(
        self, *, instructions: str, tts: agents_tts.TTS, data: TalkShowData
    ) -> None:
        self._data = data
        super().__init__(role="commentator", instructions=instructions, tts=tts, data=data)

    async def on_enter(self) -> None:
        logger.info("CommentatorAgent entered session")
        if self._data.silent_handoff:
            return
        name = load_persona_name("commentator")
        await speak_panel_line(
            self.session,
            self._data,
            speak_role="commentator",
            text=f"{name} on the panel — glad to be here.",
            step="commentator_enter",
        )
