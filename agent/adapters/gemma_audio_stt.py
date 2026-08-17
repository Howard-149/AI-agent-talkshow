"""LiveKit STT adapter: human audio → Gemma multimodal heard/reply turn."""

from __future__ import annotations

import logging
import os
import time
import uuid

from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from agent.adapters.gemma_mm_client import GemmaMMClient
from agent.adapters.response_parser import is_noise_heard
from agent.adapters.turn_store import TurnStore

logger = logging.getLogger(__name__)

_MIN_WAV_BYTES = int(os.environ.get("TALKSHOW_MIN_UTTERANCE_WAV_BYTES", "16000"))
_MIN_RMS = float(os.environ.get("TALKSHOW_MIN_UTTERANCE_RMS", "180"))


def _wav_rms(wav_bytes: bytes) -> float:
    """Peak-ish RMS of PCM in a wav container — rejects near-silent VAD blips."""
    import wave
    from io import BytesIO

    import numpy as np

    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
    except wave.Error:
        return 0.0
    if not frames:
        return 0.0
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def _human_has_floor(data: object) -> bool:
    """Panel mode: only run Gemma on human audio when the floor was granted."""
    from agent.floor.floor_control import peek_floor_next
    from agent.floor import TurnController

    if peek_floor_next(data) == "human":  # type: ignore[arg-type]
        return True
    ctrl = TurnController(data.scenario, data)  # type: ignore[arg-type]
    if not ctrl.is_panel_mode():
        return True
    return False


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
    Exposes [heard] as STT transcript; stores [reply] on TalkShowData.pending_host_speak
    for speak_panel_line (TalkShowAgent.on_user_turn_completed).
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

        rms = _wav_rms(wav)
        if rms < _MIN_RMS:
            logger.info(
                "GemmaAudioSTT: skip quiet utterance rms=%.1f < %.1f",
                rms,
                _MIN_RMS,
            )
            return _empty_transcript_event()

        data = self._talkshow_data
        if data is not None and not _human_has_floor(data):
            logger.info(
                "GemmaAudioSTT: skip — human has no floor (floor_next=%s chain=%s hand=%s)",
                getattr(data, "floor_next_speaker", ""),
                getattr(data, "panel_chain_running", False),
                getattr(data, "human_hand_raised", False),
            )
            if data.turn_log is not None:
                data.turn_log.log(
                    "gemma_stt_skip",
                    reason="no_human_floor",
                    room=data.room_name,
                    floor_next=getattr(data, "floor_next_speaker", ""),
                    panel_chain=getattr(data, "panel_chain_running", False),
                )
            return _empty_transcript_event()

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
        from agent.floor import TurnController

        data = self._talkshow_data
        assert data is not None

        ctrl = TurnController(data.scenario, data)
        panel_turn = ctrl.is_panel_mode() and _human_has_floor(data)

        if panel_turn:
            ctrl.apply_listen_persona()
            if data.agent_session is not None:
                from agent.session.human_turn import ensure_listen_role_for_human

                await ensure_listen_role_for_human(
                    data.agent_session,
                    data,
                    reason="panel_human_turn",
                    turn_log=data.turn_log,
                    room=data.room_name,
                )
            from agent.show.show_context import host_audio_user_hint
            from agent.emotion import emotion_prompt_block

            user_hint = (
                f"{host_audio_user_hint()}\n\n{emotion_prompt_block(data, 'host')}"
            )
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
        t0 = time.monotonic()
        parsed = await self._client.complete_from_wav(
            wav, user_text=user_hint, history_messages=hist
        )
        model_latency_s = time.monotonic() - t0

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
        from agent.emotion import apply_emotion_from_parsed, get_role_emotion

        emotion = apply_emotion_from_parsed(data, "host", parsed.emotion)
        if panel_turn:
            from agent.floor.floor_control import resolve_floor_after_host_speech
            from agent.show.show_context import host_used_tee_fallback, sanitize_host_panel_reply

            fitted = sanitize_host_panel_reply(reply, parsed.heard, data.scenario)
            if fitted != reply:
                logger.info(
                    "host reply reshaped for panel floor (was %r)",
                    reply[:80],
                )
            reply = fitted
            from agent.floor.floor_parser import strip_speech_control_tags

            reply = strip_speech_control_tags(reply)
            resolved_next = resolve_floor_after_host_speech(
                data,
                spoken=reply,
                tagged_next=parsed.next_speaker,
                tee_fallback=host_used_tee_fallback(reply, parsed.heard, data.scenario),
                after_human_turn=True,
            )
            logger.info(
                "human turn floor next=%s tag=%r emotion=%s",
                resolved_next,
                parsed.next_speaker,
                emotion,
            )
        else:
            from agent.floor.floor_parser import strip_speech_control_tags

            reply = strip_speech_control_tags(reply)

        from agent.locale.localize import texts_for_needed_locales
        from agent.show.show_history import append_human, append_role
        from agent.ui.ui_events import emit_floor_grant, emit_transcript

        append_human(data, parsed.heard)
        append_role(data, "host", reply)
        human_texts = await texts_for_needed_locales(data, parsed.heard)
        await emit_transcript(
            "human", parsed.heard, step="human_turn", texts=human_texts
        )
        if data.human_hand_raised:
            await emit_floor_grant("human", reason="human_spoke_hand_up")
        # Spoken in TalkShowAgent.on_user_turn_completed via speak_panel_line
        # (same Piper→DyStream path as panel lines). UI queues inside speak_panel_line.
        data.pending_host_speak = reply
        data.last_human_heard = parsed.heard
        if panel_turn:
            data.last_host_panel_tee = reply
            data.panel_followup_pending = True

        # Handoff tag only for non-panel rotation; reply is not spoken by StoredReplyLLM.
        self._turn_store.set_turn(parsed.heard, reply, handoff_to=parsed.handoff_to)

        if data.turn_log is not None:
            data.turn_log.log(
                "gemma_stt_done",
                heard=parsed.heard,
                reply=reply,
                model_latency_s=round(model_latency_s, 3),
                floor_next=data.floor_next_speaker,
                tagged_next=parsed.next_speaker,
                resolved_next=resolved_next,
                emotion=get_role_emotion(data, "host"),
                room=data.room_name,
                active_role=data.active_role,
                panel_followup_pending=data.panel_followup_pending,
            )

        logger.info(
            "heard=%r reply_len=%d floor_next=%s emotion=%s",
            parsed.heard[:80],
            len(reply),
            data.floor_next_speaker,
            get_role_emotion(data, "host"),
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
