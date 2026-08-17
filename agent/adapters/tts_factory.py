"""Build LiveKit TTS instances for the active engine (piper | cosyvoice)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from livekit.agents import tts

from agent.adapters.tts_synthesize import resolve_tts_engine
from agent.config import LocaleTTSConfig, load_persona_tts
from agent.data import TalkShowData

if TYPE_CHECKING:
    from agent.telemetry.turn_jsonl_logger import TurnJsonlLogger

logger = logging.getLogger(__name__)


def build_role_tts(
    role: str,
    data: TalkShowData,
    *,
    locale: str | None = None,
    emotion: str | None = None,
) -> tts.TTS:
    cfg = load_persona_tts(role, data.runtime.config, locale=locale)
    return build_tts_from_config(
        cfg,
        emotion=emotion,
        locale=locale or "en",
        role=role,
        turn_log=getattr(data, "turn_log", None),
        room_name=getattr(data, "room_name", ""),
    )


def build_tts_from_config(
    cfg: LocaleTTSConfig,
    *,
    emotion: str | None = None,
    locale: str = "en",
    role: str = "host",
    turn_log: TurnJsonlLogger | None = None,
    room_name: str = "",
) -> tts.TTS:
    engine = resolve_tts_engine(cfg)
    if engine == "cosyvoice":
        from agent.adapters.cosyvoice_client import cosyvoice_enabled
        from agent.adapters.cosyvoice_tts import CosyVoiceTTS

        if cosyvoice_enabled():
            return CosyVoiceTTS(
                cfg,
                emotion=emotion,
                locale=locale,
                role=role,
                turn_log=turn_log,
                room_name=room_name,
            )
        logger.warning(
            "TALKSHOW_TTS_ENGINE=cosyvoice but no COSYVOICE_SIDECAR_URL — using Piper"
        )

    from agent.adapters.piper_tts import PiperTTS

    return PiperTTS(cfg, turn_log=turn_log, room_name=room_name)
