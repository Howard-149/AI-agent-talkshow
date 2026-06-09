from __future__ import annotations

import logging
import os
import uuid

from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from agent.adapters.gemma_mm_client import GemmaMMClient
from agent.adapters.response_parser import is_noise_heard
from agent.adapters.turn_store import TurnStore

logger = logging.getLogger(__name__)

_MIN_WAV_BYTES = int(os.environ.get("TALKSHOW_MIN_UTTERANCE_WAV_BYTES", "12000"))


def _buffer_to_wav(buffer: AudioBuffer) -> bytes:
    if isinstance(buffer, list):
        frame = rtc.combine_audio_frames(buffer)
    else:
        frame = buffer
    return frame.to_wav_bytes()


def _empty_transcript_event(*, language: str = "en") -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=str(uuid.uuid4()),
        alternatives=[stt.SpeechData(language=language, text="", confidence=0.0)],
    )


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
        logger.info("GemmaAudioSTT: wav_bytes=%d active_role=%s", len(wav), getattr(self._talkshow_data, "active_role", "?"))

        if len(wav) < _MIN_WAV_BYTES:
            logger.info(
                "GemmaAudioSTT: skip short utterance (%d < %d bytes)",
                len(wav),
                _MIN_WAV_BYTES,
            )
            return _empty_transcript_event()

        data = self._talkshow_data
        if data is None:
            parsed = await self._client.complete_from_wav(wav)
            self._turn_store.set_turn(parsed.heard, parsed.reply, handoff_to=parsed.handoff_to)
            return self._final_event(parsed.heard, language)

        async with data.human_turn_lock:
            return await self._recognize_human_locked(wav, language=language)

    async def _recognize_human_locked(
        self,
        wav: bytes,
        *,
        language: NotGivenOr[str],
    ) -> stt.SpeechEvent:
        from agent.supervisor import TurnController

        data = self._talkshow_data
        assert data is not None

        ctrl = TurnController(data.scenario, data)
        panel_turn = ctrl.is_panel_mode() and (
            not data.panel_chain_running or data.human_hand_raised
        )

        if panel_turn:
            ctrl.apply_listen_persona()
            if data.agent_session is not None:
                from agent.human_turn import ensure_listen_role_for_human

                await ensure_listen_role_for_human(
                    data.agent_session,
                    data,
                    reason="panel_human_turn",
                    turn_log=data.turn_log,
                    room=data.room_name,
                )
            from agent.show_context import host_audio_user_hint

            user_hint = host_audio_user_hint()
        else:
            user_hint = None

        if data.turn_log is not None:
            data.turn_log.log(
                "gemma_stt_start",
                room=data.room_name,
                active_role=data.active_role,
                panel_turn=panel_turn,
            )

        hist = data.show_history.prior_messages()
        parsed = await self._client.complete_from_wav(
            wav, user_text=user_hint, history_messages=hist
        )

        if is_noise_heard(parsed.heard):
            logger.info("GemmaAudioSTT: skip empty/noise heard=%r", parsed.heard[:60])
            if data.turn_log is not None:
                data.turn_log.log(
                    "gemma_stt_skip",
                    reason="noise",
                    heard=parsed.heard[:80],
                    room=data.room_name,
                )
            return _empty_transcript_event()

        reply = parsed.reply
        resolved_next = parsed.next_speaker
        if panel_turn:
            from agent.floor_control import resolve_floor_after_host_speech
            from agent.show_context import host_used_tee_fallback, sanitize_host_panel_reply

            fitted = sanitize_host_panel_reply(reply, parsed.heard)
            if fitted != reply:
                logger.info(
                    "host reply reshaped for panel floor (was %r)",
                    reply[:80],
                )
            reply = fitted
            from agent.floor_parser import strip_next_tag

            reply = strip_next_tag(reply)
            resolved_next = resolve_floor_after_host_speech(
                data,
                spoken=reply,
                tagged_next=parsed.next_speaker,
                tee_fallback=host_used_tee_fallback(reply, parsed.heard),
                after_human_turn=True,
            )
            logger.info(
                "human turn floor next=%s tag=%r",
                resolved_next,
                parsed.next_speaker,
            )

        from agent.show_history import append_human, append_role
        from agent.ui_events import emit_floor_grant, emit_transcript

        append_human(data, parsed.heard)
        append_role(data, "host", reply)
        await emit_transcript("human", parsed.heard, step="human_turn")
        if data.human_hand_raised:
            await emit_floor_grant("human", reason="human_spoke_hand_up")
        data.queue_transcript("host", reply, step="host_reply")
        data.last_human_heard = parsed.heard
        if panel_turn:
            data.last_host_panel_tee = reply
            data.panel_followup_pending = True
            data.user_turn_pending_panel = True

        self._turn_store.set_turn(parsed.heard, reply, handoff_to=parsed.handoff_to)

        if data.turn_log is not None:
            data.turn_log.log(
                "gemma_stt_done",
                heard=parsed.heard,
                reply=reply,
                floor_next=data.floor_next_speaker,
                tagged_next=parsed.next_speaker,
                resolved_next=resolved_next,
                room=data.room_name,
                active_role=data.active_role,
                panel_followup_pending=data.panel_followup_pending,
            )

        logger.info(
            "heard=%r reply_len=%d floor_next=%s",
            parsed.heard[:80],
            len(reply),
            data.floor_next_speaker,
        )
        return self._final_event(parsed.heard, language)

    def _final_event(self, heard: str, language: NotGivenOr[str]) -> stt.SpeechEvent:
        lang = "en"
        if is_given(language):
            lang = str(language)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[
                stt.SpeechData(language=lang, text=heard, confidence=1.0)
            ],
        )
