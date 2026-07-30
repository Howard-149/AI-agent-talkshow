from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from agent.config import LocaleTTSConfig
from agent.adapters.piper_tts import PiperTTS, _synthesize_pcm

if TYPE_CHECKING:
    from agent.hooks.logging import TurnJsonlLogger


async def synthesize_pcm_for_config(
    cfg: LocaleTTSConfig,
    text: str,
    *,
    turn_log: TurnJsonlLogger | None = None,
    room: str = "",
    role: str = "",
    step: str = "",
) -> tuple[bytes, int, int]:
    """One-shot Piper PCM for sync avatar path (bypasses LiveKit TTS node)."""
    text = text.strip()
    if not text:
        return b"", cfg.sample_rate, 1

    stub = PiperTTS(cfg, turn_log=turn_log, room_name=room)
    t0 = time.monotonic()
    pcm, sample_rate, num_channels = await asyncio.to_thread(_synthesize_pcm, stub, text)
    tts_latency_s = time.monotonic() - t0

    if turn_log is not None:
        turn_log.log(
            "tts_synthesize",
            tts_latency_s=round(tts_latency_s, 3),
            tts_ms=round(tts_latency_s * 1000),
            chars=len(text),
            room=room,
            role=role,
            step=step,
        )
    return pcm, sample_rate, num_channels
