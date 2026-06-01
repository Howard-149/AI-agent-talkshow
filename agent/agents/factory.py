from __future__ import annotations

from livekit.agents import Agent

from agent.adapters import PiperTTS
from agent.config import load_persona_instructions, load_persona_tts
from agent.data import TalkShowData

ROLES = ("host", "guest", "commentator")


def build_agent(role: str, data: TalkShowData) -> Agent:
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    instructions = load_persona_instructions(role)
    tts = PiperTTS(load_persona_tts(role, data.runtime.config))
    if role == "host":
        from agent.agents.host import HostAgent

        return HostAgent(instructions=instructions, tts=tts, data=data)
    if role == "guest":
        from agent.agents.guest import GuestAgent

        return GuestAgent(instructions=instructions, tts=tts, data=data)
    from agent.agents.commentator import CommentatorAgent

    return CommentatorAgent(instructions=instructions, tts=tts, data=data)
