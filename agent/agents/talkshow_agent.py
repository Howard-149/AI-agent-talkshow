"""Base LiveKit Agent for talk-show roles with shared human-turn handling."""

from __future__ import annotations

import logging

from livekit.agents import Agent
from livekit.agents import tts as agents_tts
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse

from agent.data import TalkShowData

logger = logging.getLogger(__name__)


class TalkShowAgent(Agent):
    """Base voice agent with per-role TTS (Piper or CosyVoice)."""

    def __init__(
        self,
        *,
        role: str,
        instructions: str,
        tts: agents_tts.TTS,
        data: TalkShowData,
    ) -> None:
        self._role = role
        self._data = data
        super().__init__(instructions=instructions, tts=tts)

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Speak Gemma host reply via speak_panel_line; never StoredReplyLLM→TTS."""
        reply = (self._data.pending_host_speak or "").strip()
        self._data.pending_host_speak = ""
        if not reply:
            # Empty / noise STT — do not fall through to canned StoredReply TTS.
            raise StopResponse()

        # Drop stored reply so an accidental generate_reply cannot TTS the same line.
        self._data.runtime.turn_store.consume_turn()

        from agent.session.session_handoff import speak_panel_line

        logger.info("host_reply via speak_panel_line chars=%d", len(reply))
        await speak_panel_line(
            self.session,
            self._data,
            speak_role="host",
            text=reply,
            step="host_reply",
        )
        raise StopResponse()
