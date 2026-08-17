"""Engine-agnostic one-shot PCM synthesis for avatar / locale tracks."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from agent.config import LocaleTTSConfig
from agent.emotion.tts import emotion_instruct

if TYPE_CHECKING:
    from agent.telemetry.turn_jsonl_logger import TurnJsonlLogger

logger = logging.getLogger(__name__)


def resolve_tts_engine(cfg: LocaleTTSConfig | None = None) -> str:
    """``TALKSHOW_TTS_ENGINE`` overrides yaml ``tts.engine``."""
    env = os.environ.get("TALKSHOW_TTS_ENGINE", "").strip().lower()
    if env in ("piper", "cosyvoice"):
        return env
    if cfg is not None and cfg.engine:
        return str(cfg.engine).strip().lower()
    return "piper"


async def synthesize_pcm_for_config(
    cfg: LocaleTTSConfig,
    text: str,
    *,
    emotion: str | None = None,
    locale: str = "en",
    turn_log: TurnJsonlLogger | None = None,
    room: str = "",
    role: str = "",
    step: str = "",
) -> tuple[bytes, int, int]:
    """Return int16 mono PCM for the active TTS engine (piper | cosyvoice)."""
    text = text.strip()
    if not text:
        return b"", cfg.sample_rate, 1

    engine = resolve_tts_engine(cfg)
    t0 = time.monotonic()

    if engine == "cosyvoice":
        from agent.adapters.cosyvoice_client import (
            cosyvoice_enabled,
            synthesize_pcm_via_sidecar,
        )

        if not cosyvoice_enabled():
            logger.warning(
                "engine=cosyvoice but COSYVOICE_SIDECAR_URL unset — falling back to piper"
            )
            engine = "piper"
        else:
            from agent.adapters.cosyvoice_voices import (
                cosyvoice_mode,
                mode_needs_prompt_wav,
                resolve_spk_id,
            )

            mode = cosyvoice_mode()
            instruct = emotion_instruct(emotion=emotion, locale=locale)
            spk_id = resolve_spk_id(
                role=role, locale=locale, model_path=cfg.model_path
            )
            prompt_wav = ""
            if mode_needs_prompt_wav(mode):
                prompt_wav = cfg.model_path
            pcm, sample_rate, num_channels = await synthesize_pcm_via_sidecar(
                text=text,
                instruct=instruct,
                spk_id=spk_id,
                prompt_wav=prompt_wav,
                mode=mode,
                sample_rate_hint=cfg.sample_rate,
            )
            _log_synth(
                turn_log,
                engine="cosyvoice",
                tts_latency_s=time.monotonic() - t0,
                chars=len(text),
                room=room,
                role=role,
                step=step,
                emotion=emotion or "neutral",
            )
            return pcm, sample_rate, num_channels

    from agent.adapters.piper_tts import PiperTTS, _synthesize_pcm
    import asyncio

    stub = PiperTTS(cfg, turn_log=turn_log, room_name=room)
    pcm, sample_rate, num_channels = await asyncio.to_thread(_synthesize_pcm, stub, text)
    _log_synth(
        turn_log,
        engine="piper",
        tts_latency_s=time.monotonic() - t0,
        chars=len(text),
        room=room,
        role=role,
        step=step,
        emotion=emotion or "neutral",
    )
    return pcm, sample_rate, num_channels


def _log_synth(
    turn_log: TurnJsonlLogger | None,
    *,
    engine: str,
    tts_latency_s: float,
    chars: int,
    room: str,
    role: str,
    step: str,
    emotion: str,
) -> None:
    if turn_log is None:
        return
    turn_log.log(
        "tts_synthesize",
        engine=engine,
        tts_latency_s=round(tts_latency_s, 3),
        tts_ms=round(tts_latency_s * 1000),
        chars=chars,
        room=room,
        role=role,
        step=step,
        emotion=emotion,
    )
