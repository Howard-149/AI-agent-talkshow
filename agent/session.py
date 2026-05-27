from __future__ import annotations

from dataclasses import dataclass

from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.adapters import GemmaAudioSTT, PiperTTS, StoredReplyLLM, TurnStore
from agent.adapters.gemma_mm_client import GemmaMMClient
from agent.config import AppConfig, load_persona_instructions


@dataclass
class TalkShowRuntime:
    turn_store: TurnStore
    gemma_client: GemmaMMClient
    config: AppConfig


def build_runtime(config: AppConfig | None = None) -> TalkShowRuntime:
    from agent.config import load_config

    app_cfg = config or load_config()
    locale = app_cfg.locale()
    turn_store = TurnStore()
    persona = load_persona_instructions("host")
    gemma_client = GemmaMMClient(locale.llm, system_prompt=persona)
    return TalkShowRuntime(turn_store=turn_store, gemma_client=gemma_client, config=app_cfg)


def build_agent_session(runtime: TalkShowRuntime) -> AgentSession:
    locale = runtime.config.locale()
    stt = GemmaAudioSTT(client=runtime.gemma_client, turn_store=runtime.turn_store)
    llm = StoredReplyLLM(runtime.turn_store)
    tts = PiperTTS(locale.tts)

    return AgentSession(
        vad=silero.VAD.load(),
        stt=stt,
        llm=llm,
        tts=tts,
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
        ),
    )
