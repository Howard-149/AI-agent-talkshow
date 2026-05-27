from __future__ import annotations

import logging
import uuid

from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr, is_given
from livekit.agents.utils import AudioBuffer

from agent.adapters.gemma_mm_client import GemmaMMClient
from agent.adapters.turn_store import TurnStore

logger = logging.getLogger(__name__)


def _buffer_to_wav(buffer: AudioBuffer) -> bytes:
    if isinstance(buffer, list):
        frame = rtc.combine_audio_frames(buffer)
    else:
        frame = buffer
    return frame.to_wav_bytes()


class GemmaAudioSTT(stt.STT):
    """
    One vLLM multimodal call per utterance (audio-in).
    Exposes [heard] as STT transcript; stores [reply] for StoredReplyLLM.
    """

    def __init__(self, *, client: GemmaMMClient, turn_store: TurnStore) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            ),
        )
        self._client = client
        self._turn_store = turn_store

    @property
    def model(self) -> str:
        return self._client._cfg.model

    @property
    def provider(self) -> str:
        return "gemma-vllm-multimodal"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        wav = _buffer_to_wav(buffer)
        logger.info("GemmaAudioSTT: wav_bytes=%d", len(wav))

        parsed = await self._client.complete_from_wav(wav)
        self._turn_store.set_turn(parsed.heard, parsed.reply)
        logger.info("heard=%r reply_len=%d", parsed.heard[:80], len(parsed.reply))

        lang = "en"
        if is_given(language):
            lang = str(language)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[
                stt.SpeechData(
                    language=lang,
                    text=parsed.heard,
                    confidence=1.0,
                )
            ],
        )
