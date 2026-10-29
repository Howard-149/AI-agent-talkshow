"""Fish PCM chunks -> LiveKit audio frames with interruptible HTTP synthesis."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
import time
import uuid
from typing import TYPE_CHECKING

import httpx
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from agent.adapters.fish_client import FishSettings, iter_fish_pcm
from agent.config import LocaleTTSConfig

if TYPE_CHECKING:
    from agent.telemetry.turn_jsonl_logger import TurnJsonlLogger


class FishTTS(tts.TTS):
    def __init__(self, cfg: LocaleTTSConfig, *, role: str = "host", locale: str = "en",
                 turn_log: TurnJsonlLogger | None = None, room_name: str = "",
                 client: httpx.AsyncClient | None = None):
        self._settings = FishSettings.from_env(
            role, locale, reference_id=cfg.model_path if cfg.engine == "fish" else "",
            sample_rate=cfg.sample_rate if cfg.engine == "fish" else None,
        )
        # Full sentences go in; HTTP audio chunks come out progressively. This is
        # not token-by-token text streaming, so LiveKit's sentence adapter is kept.
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=self._settings.sample_rate, num_channels=1)
        self._role, self._locale = role, locale
        self._turn_log, self._room_name = turn_log, room_name
        self._client = client

    @property
    def model(self) -> str:
        return self._settings.model

    @property
    def provider(self) -> str:
        return "fish"

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _FishChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _FishChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        impl: FishTTS = self._tts  # type: ignore[assignment]
        text = self._input_text.strip()
        if not text:
            return
        started, first_pcm, total_bytes, status = time.monotonic(), None, 0, "complete"
        output_emitter.initialize(request_id=str(uuid.uuid4()), sample_rate=impl.sample_rate,
                                  num_channels=1, mime_type="audio/pcm", frame_size_ms=20, stream=False)
        try:
            async with aclosing(iter_fish_pcm(text, impl._settings, client=impl._client)) as stream:
                async for pcm in stream:
                    if first_pcm is None:
                        first_pcm = time.monotonic()
                    total_bytes += len(pcm)
                    output_emitter.push(pcm)
            output_emitter.flush()
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception:
            status = "failed"
            raise
        finally:
            if impl._turn_log is not None:
                impl._turn_log.log(
                    "tts_synthesize", engine="fish", model=impl.model,
                    reference_id=impl._settings.reference_id, latency=impl._settings.latency,
                    tts_latency_s=round(time.monotonic() - started, 3),
                    ttfa_ms=None if first_pcm is None else round((first_pcm - started) * 1000),
                    audio_seconds=round(total_bytes / (impl.sample_rate * 2), 3),
                    status=status, chars=len(text), role=impl._role, locale=impl._locale, room=impl._room_name,
                )
