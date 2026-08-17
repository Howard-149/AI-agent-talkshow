"""Host agent — listen role, moderation entry, and panel line speaking."""

from __future__ import annotations

import logging
import os

from livekit.agents import tts as agents_tts

from agent.agents.handoff import HandoffToolsMixin
from agent.agents.talkshow_agent import TalkShowAgent
from agent.config import load_persona_name
from agent.data import TalkShowData
from agent.session.session_handoff import speak_panel_line

logger = logging.getLogger(__name__)


class HostAgent(HandoffToolsMixin, TalkShowAgent):
    _role = "host"

    def __init__(
        self, *, instructions: str, tts: agents_tts.TTS, data: TalkShowData
    ) -> None:
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
        host = load_persona_name("host")
        greeting = (
            f"Hey there! I'm {host}, your host. "
            "What would you like to talk about today?"
        )
        await speak_panel_line(
            self.session,
            self._data,
            speak_role="host",
            text=greeting,
            step="host_greeting",
        )
