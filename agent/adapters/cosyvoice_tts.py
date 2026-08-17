"""LiveKit TTS adapter that calls the CosyVoice sidecar (non-streaming)."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from agent.adapters.cosyvoice_client import synthesize_pcm_via_sidecar
from agent.config import LocaleTTSConfig
from agent.adapters.cosyvoice_voices import cosyvoice_mode, mode_needs_prompt_wav, resolve_spk_id
from agent.emotion.tts import emotion_instruct

if TYPE_CHECKING:
    from agent.telemetry.turn_jsonl_logger import TurnJsonlLogger

logger = logging.getLogger(__name__)


class CosyVoiceTTS(tts.TTS):
    def __init__(
        self,
        cfg: LocaleTTSConfig,
        *,
        emotion: str | None = None,
        locale: str = "en",
        role: str = "host",
        turn_log: TurnJsonlLogger | None = None,
        room_name: str = "",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=cfg.sample_rate,
            num_channels=1,
        )
        self._cfg = cfg
        self._role = role
        self._emotion = emotion
        self._locale = locale
        self._turn_log = turn_log
        self._room_name = room_name
        self._mode = cosyvoice_mode()
        self._spk_id = resolve_spk_id(
            role=role, locale=locale, model_path=cfg.model_path
        )
        logger.info(
            "CosyVoiceTTS init mode=%s spk=%s emotion=%s locale=%s",
            self._mode,
            self._spk_id,
            emotion or "neutral",
            locale,
        )

    @property
    def model(self) -> str:
        return self._spk_id or "cosyvoice"

    @property
    def provider(self) -> str:
        return "cosyvoice"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _CosyVoiceChunkedStream(
            tts=self, input_text=text, conn_options=conn_options
        )


class _CosyVoiceChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        impl: CosyVoiceTTS = self._tts  # type: ignore[assignment]
        text = self._input_text.strip()
        if not text:
            return

        instruct = emotion_instruct(emotion=impl._emotion, locale=impl._locale)
        prompt_wav = ""
        if mode_needs_prompt_wav(impl._mode):
            prompt_wav = impl._cfg.model_path
        t0 = time.monotonic()
        pcm_bytes, sample_rate, num_channels = await synthesize_pcm_via_sidecar(
            text=text,
            instruct=instruct,
            spk_id=impl._spk_id,
            prompt_wav=prompt_wav,
            mode=impl._mode,
            sample_rate_hint=impl._cfg.sample_rate,
        )
        tts_latency_s = time.monotonic() - t0

        if impl._turn_log is not None:
            impl._turn_log.log(
                "tts_synthesize",
                engine="cosyvoice",
                mode=impl._mode,
                spk_id=impl._spk_id,
                tts_latency_s=round(tts_latency_s, 3),
                chars=len(text),
                emotion=impl._emotion or "neutral",
                room=impl._room_name,
            )

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=sample_rate,
            num_channels=num_channels,
            mime_type="audio/pcm",
            stream=False,
        )
        output_emitter.push(pcm_bytes)
        output_emitter.flush()
