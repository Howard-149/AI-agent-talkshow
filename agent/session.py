from __future__ import annotations

import logging
import os
from typing import Any

from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.adapters import GemmaAudioSTT, PiperTTS, StoredReplyLLM
from agent.data import TalkShowData
from agent.runtime import TalkShowRuntime

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def load_vad(prewarmed: Any | None = None) -> Any:
    """Silero VAD — longer min_silence keeps the human turn open through brief pauses."""
    if prewarmed is not None:
        logger.info("AgentSession using prewarmed Silero VAD")
        return prewarmed

    min_silence = _env_float("TALKSHOW_VAD_MIN_SILENCE_SEC", 1.0)
    logger.info("AgentSession loading Silero VAD min_silence_duration=%.2fs", min_silence)
    return silero.VAD.load(min_silence_duration=min_silence)


def build_turn_handling() -> TurnHandlingOptions:
    """Default: MultilingualModel + generous endpointing for panel-style pauses."""
    mode = os.environ.get("TALKSHOW_TURN_DETECTOR", "multilingual").lower()
    min_delay = _env_float("TALKSHOW_MIN_ENDPOINTING_DELAY", 0.8)
    max_delay = _env_float("TALKSHOW_MAX_ENDPOINTING_DELAY", 12.0)

    endpointing: dict[str, object] = {
        "mode": "dynamic",
        "min_delay": min_delay,
        "max_delay": max_delay,
    }

    if mode in ("vad", "0", "false", "none", "off"):
        logger.info(
            "turn_handling vad-only endpointing min=%.2fs max=%.2fs",
            min_delay,
            max_delay,
        )
        return TurnHandlingOptions(endpointing=endpointing)

    logger.info(
        "turn_handling multilingual endpointing min=%.2fs max=%.2fs",
        min_delay,
        max_delay,
    )
    return TurnHandlingOptions(
        turn_detection=MultilingualModel(),
        endpointing=endpointing,
    )


def build_agent_session(
    runtime: TalkShowRuntime,
    userdata: TalkShowData,
    *,
    prewarmed_vad: Any | None = None,
) -> AgentSession:
    locale = runtime.config.locale()
    stt = GemmaAudioSTT(
        client=runtime.gemma_client,
        turn_store=runtime.turn_store,
        talkshow_data=userdata,
    )
    llm = StoredReplyLLM(runtime.turn_store)
    tts = PiperTTS(locale.tts)

    return AgentSession[TalkShowData](
        vad=load_vad(prewarmed_vad),
        stt=stt,
        llm=llm,
        tts=tts,
        turn_handling=build_turn_handling(),
        userdata=userdata,
    )
