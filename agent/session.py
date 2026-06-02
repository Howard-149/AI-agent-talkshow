from __future__ import annotations

import os

from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.adapters import GemmaAudioSTT, PiperTTS, StoredReplyLLM
from agent.data import TalkShowData
from agent.runtime import TalkShowRuntime


def build_turn_handling() -> TurnHandlingOptions:
    """Default: MultilingualModel (needs deploy/download-livekit-agent-models.sh → TALKSHOW_TURN_DETECTOR_CACHE). TALKSHOW_TURN_DETECTOR=vad to skip."""
    mode = os.environ.get("TALKSHOW_TURN_DETECTOR", "multilingual").lower()
    if mode in ("vad", "0", "false", "none", "off"):
        return TurnHandlingOptions()
    return TurnHandlingOptions(turn_detection=MultilingualModel())


def build_agent_session(runtime: TalkShowRuntime, userdata: TalkShowData) -> AgentSession:
    locale = runtime.config.locale()
    stt = GemmaAudioSTT(
        client=runtime.gemma_client,
        turn_store=runtime.turn_store,
        talkshow_data=userdata,
    )
    llm = StoredReplyLLM(runtime.turn_store)
    # Session default TTS = host voice; each Agent can override with its own PiperTTS
    tts = PiperTTS(locale.tts)

    return AgentSession[TalkShowData](
        vad=silero.VAD.load(),
        stt=stt,
        llm=llm,
        tts=tts,
        turn_handling=build_turn_handling(),
        userdata=userdata,
    )
