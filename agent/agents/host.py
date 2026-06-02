from __future__ import annotations

import logging
import os

from agent.adapters import PiperTTS
from agent.agents.handoff import HandoffToolsMixin
from agent.agents.talkshow_agent import TalkShowAgent
from agent.data import TalkShowData
from agent.participant_display import set_agent_display_name

logger = logging.getLogger(__name__)


class HostAgent(HandoffToolsMixin, TalkShowAgent):
    _role = "host"

    def __init__(self, *, instructions: str, tts: PiperTTS, data: TalkShowData) -> None:
        self._data = data
        super().__init__(role="host", instructions=instructions, tts=tts, data=data)

    async def on_enter(self) -> None:
        logger.info("HostAgent entered session")
        await set_agent_display_name("host")
        if self._data.silent_handoff or self._data.panel_chain_running:
            return
        if os.environ.get("TALKSHOW_SKIP_GREETING", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        if self._data.scenario.turn_control.mode == "panel_round_robin":
            from agent.panel_prompts import PANEL_OPENING_LINE

            from agent.ui_events import emit_role_active

            await emit_role_active("host")
            self._data.queue_transcript("host", PANEL_OPENING_LINE, step="opening")
            handle = self.session.say(
                PANEL_OPENING_LINE,
                allow_interruptions=False,
            )
            await handle.wait_for_playout()
            from agent.show_history import append_role

            append_role(self._data, "host", PANEL_OPENING_LINE)
            return
        await self.session.generate_reply(
            instructions="Greet the user briefly in English as the talk-show host and invite them to speak."
        )
