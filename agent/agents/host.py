from __future__ import annotations

import logging
import os

from agent.adapters import PiperTTS
from agent.agents.handoff import HandoffToolsMixin
from agent.agents.talkshow_agent import TalkShowAgent
from agent.data import TalkShowData

logger = logging.getLogger(__name__)


class HostAgent(HandoffToolsMixin, TalkShowAgent):
    _role = "host"

    def __init__(self, *, instructions: str, tts: PiperTTS, data: TalkShowData) -> None:
        self._data = data
        super().__init__(role="host", instructions=instructions, tts=tts, data=data)

    async def on_enter(self) -> None:
        logger.info("HostAgent entered session")
        if self._data.silent_handoff or self._data.panel_chain_running:
            return
        if os.environ.get("TALKSHOW_SKIP_GREETING", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        if self._data.scenario.turn_control.mode in (
            "panel_round_robin",
            "host_moderated",
        ):
            # Opening runs from main.entrypoint after room connect (session_opening.py)
            return
        await self.session.generate_reply(
            instructions="Greet the user briefly in English as the talk-show host and invite them to speak."
        )
