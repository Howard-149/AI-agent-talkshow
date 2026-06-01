from __future__ import annotations

import logging
import uuid

from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

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

    def __init__(
        self,
        *,
        client: GemmaMMClient,
        turn_store: TurnStore,
        talkshow_data: object | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            ),
        )
        self._client = client
        self._turn_store = turn_store
        self._talkshow_data = talkshow_data

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

        user_hint: str | None = None
        panel_turn = False
        if self._talkshow_data is not None:
            from agent.supervisor import TurnController

            data = self._talkshow_data
            ctrl = TurnController(data.scenario, data)
            if ctrl.is_panel_mode() and not data.panel_chain_running:
                ctrl.apply_listen_persona()
                from agent.show_context import host_audio_user_hint

                user_hint = host_audio_user_hint()
                panel_turn = True

        hist = None
        if self._talkshow_data is not None:
            hist = self._talkshow_data.show_history.prior_messages()

        parsed = await self._client.complete_from_wav(
            wav, user_text=user_hint, history_messages=hist
        )
        reply = parsed.reply
        if panel_turn:
            from agent.show_context import sanitize_host_panel_reply

            fitted = sanitize_host_panel_reply(reply, parsed.heard)
            if fitted != reply:
                logger.info(
                    "host reply reshaped for panel floor (was %r)",
                    reply[:80],
                )
            reply = fitted
        if self._talkshow_data is not None:
            from agent.show_history import append_human, append_role

            append_human(self._talkshow_data, parsed.heard)
            append_role(self._talkshow_data, "host", reply)
            self._talkshow_data.last_human_heard = parsed.heard
            if panel_turn:
                self._talkshow_data.last_host_panel_tee = reply
        self._turn_store.set_turn(
            parsed.heard, reply, handoff_to=parsed.handoff_to
        )
        logger.info("heard=%r reply_len=%d", parsed.heard[:80], len(reply))

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
