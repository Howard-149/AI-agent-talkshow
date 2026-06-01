from __future__ import annotations

from livekit.agents import Agent

from agent.adapters import PiperTTS
from agent.data import TalkShowData


class TalkShowAgent(Agent):
    """Base voice agent with per-role Piper TTS."""

    def __init__(
        self,
        *,
        role: str,
        instructions: str,
        tts: PiperTTS,
        data: TalkShowData,
    ) -> None:
        self._role = role
        self._data = data
        super().__init__(instructions=instructions, tts=tts)
