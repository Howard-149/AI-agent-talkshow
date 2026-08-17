"""Role switch and speak_panel_line (TTS + optional avatar A/V sync)."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from livekit.agents import AgentSession

from agent.adapters.dystream_bridge import (
    AvatarBridge,
    SyncStreamResult,
    get_avatar_bridge,
)
from agent.adapters.avatar_video import (
    AvatarVideoPublisher,
    clip_duration_sec,
    get_avatar_video_publisher,
    lk_video_enabled,
)
from agent.adapters.tts_synthesize import synthesize_pcm_for_config
from agent.agents.factory import build_agent
from agent.config import LocaleTTSConfig, load_persona_tts
from agent.data import TalkShowData
from agent.ui.participant_display import set_agent_display_name
from agent.session.speak_chunks import avatar_chunk_stream_enabled, split_speak_sentences
from agent.floor import TurnController
from avatar.paths import cleanup_speech_slots
from avatar.pcm_utils import pcm16_to_wav, split_pcm16_by_silence

logger = logging.getLogger(__name__)


async def wait_for_session_agent(session: AgentSession) -> None:
    """Wait until update_agent() has finished swapping AgentActivity."""
    task = getattr(session, "_update_activity_atask", None)
    if task is not None and not task.done():
        await asyncio.shield(task)


async def switch_to_role(
    session: AgentSession,
    data: TalkShowData,
    role: str,
    *,
    reason: str,
) -> None:
    """Hand off to role and wait before speaking (correct Piper + no stray on_enter)."""
    if role == data.active_role:
        return

    ctrl = TurnController(data.scenario, data)
    data.silent_handoff = True
    try:
        ctrl.record_handoff(to_role=role, reason=reason)
        ctrl.apply_persona_for_role(role)
        await set_agent_display_name(role)
        session.update_agent(build_agent(role, data))
        await wait_for_session_agent(session)
        tts_path = load_persona_tts(role, data.runtime.config).model_path
        logger.info("role ready role=%s piper=%s reason=%s", role, tts_path, reason)
    finally:
        data.silent_handoff = False


def queue_speech_ui(
    data: TalkShowData,
    role: str,
    text: str,
    *,
    step: str = "",
    texts: dict[str, str] | None = None,
) -> None:
    """Queue UI — emitted when agent state becomes speaking (playout)."""
    data.queue_speech_ui(role, text, step=step, texts=texts)


async def _say_pcm(
    session: AgentSession,
    text: str,
    pcm: bytes,
    sample_rate: int,
) -> None:
    """Play pre-synthesized PCM via session.say(audio=...) — skips Piper TTS node."""
    from livekit.agents.utils.audio import audio_frames_from_file

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        pcm16_to_wav(pcm, wav_path, sample_rate=sample_rate, num_channels=1)
        handle = session.say(
            text,
            audio=audio_frames_from_file(str(wav_path)),
            allow_interruptions=False,
        )
        await handle.wait_for_playout()
    finally:
        wav_path.unlink(missing_ok=True)


def _pcm_duration_sec(pcm: bytes, sample_rate: int) -> float | None:
    if not pcm or sample_rate <= 0:
        return None
    # Piper PCM is int16 mono
    return (len(pcm) / 2) / float(sample_rate)


@dataclass
class _PreparedChunk:
    text: str
    pcm: bytes
    sample_rate: int
    clip_path: Path | None
    bake_ms: int
    frames: list[bytes] | None
    fps: float
    duration_sec: float | None
    url: str | None
    tts_ms: int
    live_stream: SyncStreamResult | None = None
    preroll_frames: int = 0
    gen_fps: float = 0.0
    expected_frames: int = 0


async def _prepare_chunk(
    *,
    bridge: AvatarBridge,
    tts_cfg: LocaleTTSConfig,
    speak_role: str,
    step: str,
    chunk_text: str,
    chunk_idx: int,
    slot: str,
    turn_log: object | None,
    room: str,
    video_pub: AvatarVideoPublisher | None,
    use_lk_video: bool,
    emotion: str | None = None,
    locale: str = "en",
    pcm: bytes | None = None,
    sample_rate: int | None = None,
    tts_ms: int | None = None,
) -> _PreparedChunk:
    """Bake one sentence chunk. If ``pcm`` is provided, skip TTS (one-shot split)."""
    if pcm is None:
        t0 = time.monotonic()
        pcm, sample_rate, _ = await synthesize_pcm_for_config(
            tts_cfg,
            chunk_text,
            emotion=emotion,
            locale=locale,
            turn_log=turn_log,
            room=room,
            role=speak_role,
            step=f"{step}:c{chunk_idx}",
        )
        tts_ms = round((time.monotonic() - t0) * 1000)
    else:
        sample_rate = int(sample_rate or 22050)
        tts_ms = int(tts_ms or 0)
    if not pcm:
        return _PreparedChunk(
            text=chunk_text,
            pcm=b"",
            sample_rate=sample_rate,
            clip_path=None,
            bake_ms=0,
            frames=None,
            fps=25.0,
            duration_sec=None,
            url=None,
            tts_ms=tts_ms,
            live_stream=None,
        )

    clip = await bridge.bake_sync(
        speak_role,
        f"{step}:c{chunk_idx}",
        pcm,
        sample_rate,
        slot=slot,
    )
    frames: list[bytes] | None = None
    live_stream: SyncStreamResult | None = None
    fps = 25.0
    # Pace video to the same PCM DyStream used for mouth motion.
    duration_sec = _pcm_duration_sec(pcm, sample_rate)
    url: str | None = None
    clip_path: Path | None = None
    bake_ms = 0
    preroll_frames = 0
    gen_fps = 0.0
    expected_frames = 0

    if isinstance(clip, SyncStreamResult):
        live_stream = clip
        fps = clip.fps
        bake_ms = clip.bake_ms
        preroll_frames = clip.preroll_frames
        gen_fps = clip.gen_fps
        expected_frames = clip.expected_frames
    elif clip is not None:
        clip_path = clip.clip_path
        bake_ms = clip.bake_ms
        url = clip.url
        if use_lk_video and video_pub is not None:
            if duration_sec is None:
                duration_sec = await asyncio.to_thread(clip_duration_sec, clip.clip_path)
            prepared = await video_pub.prepare_mp4(clip.clip_path)
            if prepared is not None:
                frames, fps = prepared
                preroll_frames = len(frames)
                expected_frames = len(frames)

    return _PreparedChunk(
        text=chunk_text,
        pcm=pcm,
        sample_rate=sample_rate,
        clip_path=clip_path,
        bake_ms=bake_ms,
        frames=frames,
        fps=fps,
        duration_sec=duration_sec,
        url=url,
        tts_ms=tts_ms,
        live_stream=live_stream,
        preroll_frames=preroll_frames,
        gen_fps=gen_fps,
        expected_frames=expected_frames,
    )


async def _play_prepared_chunk(
    session: AgentSession,
    chunk: _PreparedChunk,
    *,
    video_pub: AvatarVideoPublisher | None,
    use_lk_video: bool,
    audio_via_session: bool = True,
    locale_audio: object | None = None,
) -> None:
    """Video from DyStream live stream / MP4; audio from Piper PCM."""

    async def _play_audio() -> None:
        if not chunk.pcm:
            if audio_via_session:
                handle = session.say(chunk.text, allow_interruptions=False)
                await handle.wait_for_playout()
            return
        if audio_via_session:
            await _say_pcm(session, chunk.text, chunk.pcm, chunk.sample_rate)
            return
        if locale_audio is not None:
            await locale_audio.play_pcm(chunk.pcm, chunk.sample_rate)  # type: ignore[attr-defined]

    if (
        chunk.live_stream is not None
        and use_lk_video
        and video_pub is not None
        and chunk.pcm
    ):
        # Frame 0 on the track first (client already warm-attached), then PCM.
        first_frame = asyncio.Event()

        async def _audio_with_frame0() -> None:
            await first_frame.wait()
            await _play_audio()

        await asyncio.gather(
            video_pub.stream_frames_live(
                chunk.live_stream.frame_queue,
                chunk.live_stream.fps,
                target_duration_sec=chunk.duration_sec,
                on_first_frame=first_frame,
            ),
            _audio_with_frame0(),
        )
        if chunk.live_stream._reader_task is not None:
            try:
                await chunk.live_stream._reader_task
            except Exception:
                pass
        return
    if chunk.frames is not None and use_lk_video and video_pub is not None and chunk.pcm:
        await asyncio.gather(
            video_pub.stream_frames(
                chunk.frames,
                chunk.fps,
                target_duration_sec=chunk.duration_sec,
            ),
            _play_audio(),
        )
        return
    await _play_audio()


async def _speak_avatar_chunked(
    session: AgentSession,
    data: TalkShowData,
    *,
    bridge: AvatarBridge,
    speak_role: str,
    text: str,
    step: str,
    tts_cfg: LocaleTTSConfig,
    turn_log: object | None,
    video_pub: AvatarVideoPublisher | None,
    use_lk_video: bool,
    ui_text: str | None = None,
    texts: dict[str, str] | None = None,
    audio_via_session: bool = True,
    locale_audio: object | None = None,
    emit_ui: bool = True,
    slot_prefix: str = "",
    emotion: str | None = None,
    locale: str = "en",
) -> None:
    sentences = split_speak_sentences(text)
    if not sentences:
        return
    # CosyVoice chunk modes (TALKSHOW_COSYVOICE_CHUNK_MODE):
    #   sentence (default) — per-sentence TTS + DyStream (correct A/V boundaries)
    #   full — one TTS + one DyStream for the whole line
    #   tts_once — one TTS; DyStream chunks via silence heuristic near char-weight
    #              cuts (no real timestamps; opt-in only)
    from agent.adapters.tts_synthesize import resolve_tts_engine

    cosy_mode = (
        os.environ.get("TALKSHOW_COSYVOICE_CHUNK_MODE", "sentence").strip().lower()
        if resolve_tts_engine(tts_cfg) == "cosyvoice"
        else "sentence"
    )
    if cosy_mode in ("full", "whole", "utterance"):
        sentences = [text]
        cosy_mode = "full"
    elif cosy_mode in ("tts_once", "oneshot", "full_split", "split"):
        cosy_mode = "tts_once"
        if len(sentences) == 1 or not avatar_chunk_stream_enabled():
            sentences = [text]
            cosy_mode = "full"
    else:
        cosy_mode = "sentence"

    if cosy_mode == "sentence":
        if len(sentences) > 1 and not audio_via_session:
            # Secondary locale: one bake for the full line (keep A/V in sync with primary).
            sentences = [text]
        elif len(sentences) == 1 or not avatar_chunk_stream_enabled():
            sentences = [text]

    prefix = slot_prefix.strip()
    slots = [f"{prefix}c{i}" if prefix else f"c{i}" for i in range(len(sentences))]
    sync_t0 = time.monotonic()
    total_tts_ms = 0
    total_bake_ms = 0
    first_wait_ms: int | None = None
    # Perceived wait: clock starts when previous chunk finished playing (or line start).
    prev_play_end = sync_t0
    perceived_waits: list[int] = []
    transcript_text = ui_text if ui_text is not None else text

    # Optional: one CosyVoice call, then DyStream per sentence on PCM slices.
    pcm_parts: list[bytes] | None = None
    oneshot_sr = 22050
    oneshot_tts_ms = 0
    if cosy_mode == "tts_once" and len(sentences) > 1:
        t0 = time.monotonic()
        full_pcm, oneshot_sr, _ = await synthesize_pcm_for_config(
            tts_cfg,
            text,
            emotion=emotion,
            locale=locale,
            turn_log=turn_log,
            room=data.room_name,
            role=speak_role,
            step=f"{step}:tts_once",
        )
        oneshot_tts_ms = round((time.monotonic() - t0) * 1000)
        weights = [float(max(1, len(s.strip()))) for s in sentences]
        # CosyVoice returns no sentence timestamps; cut at RMS valleys near
        # char-proportional guesses (still wrong if pauses don't match text).
        pcm_parts = split_pcm16_by_silence(
            full_pcm or b"",
            len(sentences),
            sample_rate=oneshot_sr,
            weights=weights,
        )
        logger.info(
            "cosyvoice tts_once chars=%d sentences=%d tts_ms=%d "
            "(pcm split=silence heuristic, not aligned)",
            len(text),
            len(sentences),
            oneshot_tts_ms,
        )

    from agent.ui.ui_events import emit_avatar_clip

    if use_lk_video and emit_ui:
        await emit_avatar_clip(speak_role, transport="livekit", step=step)

    def _chunk_kwargs(i: int) -> dict:
        kw: dict = dict(
            bridge=bridge,
            tts_cfg=tts_cfg,
            speak_role=speak_role,
            step=step,
            chunk_text=sentences[i],
            chunk_idx=i,
            slot=slots[i],
            turn_log=turn_log,
            room=data.room_name,
            video_pub=video_pub,
            use_lk_video=use_lk_video,
            emotion=emotion,
            locale=locale,
        )
        if pcm_parts is not None:
            kw["pcm"] = pcm_parts[i] if i < len(pcm_parts) else b""
            kw["sample_rate"] = oneshot_sr
            # Attribute the one-shot TTS cost to the first chunk only.
            kw["tts_ms"] = oneshot_tts_ms if i == 0 else 0
        return kw

    next_task: asyncio.Task[_PreparedChunk] | None = asyncio.create_task(
        _prepare_chunk(**_chunk_kwargs(0))
    )

    try:
        for i, sent in enumerate(sentences):
            assert next_task is not None
            chunk = await next_task
            ready_at = time.monotonic()
            # Gap clock: previous playout end (chunk 0: line start).
            # Bake-ahead may finish before prev ends → perceived wait 0.
            perceived_wait_ms = max(0, round((ready_at - prev_play_end) * 1000))
            if first_wait_ms is None:
                first_wait_ms = perceived_wait_ms
            perceived_waits.append(perceived_wait_ms)
            total_tts_ms += chunk.tts_ms
            total_bake_ms += chunk.bake_ms

            if turn_log is not None:
                turn_log.log(
                    "avatar_chunk_play",
                    role=speak_role,
                    step=f"{step}:c{i}",
                    chunk_idx=i,
                    perceived_wait_ms=perceived_wait_ms,
                    bake_wall_ms=chunk.bake_ms,
                    tts_ms=chunk.tts_ms,
                    pcm_duration_sec=chunk.duration_sec,
                    preroll_frames=chunk.preroll_frames,
                    preroll_frames_target=chunk.preroll_frames,
                    expected_frames=chunk.expected_frames,
                    gen_fps=chunk.gen_fps,
                )

            if not use_lk_video and chunk.url and emit_ui:
                await emit_avatar_clip(
                    speak_role, url=chunk.url, transport="http", step=f"{step}:c{i}"
                )

            if i + 1 < len(sentences):
                next_task = asyncio.create_task(
                    _prepare_chunk(**_chunk_kwargs(i + 1))
                )
            else:
                next_task = None

            # Speaking → idle → speaking: re-assert role_active each chunk.
            # Transcript only on first chunk (full line).
            if emit_ui:
                queue_speech_ui(
                    data,
                    speak_role,
                    transcript_text if i == 0 else "",
                    step=step if i == 0 else f"{step}:c{i}",
                    texts=texts if i == 0 else None,
                )
            logger.info(
                "avatar chunk play i=%d/%d chars=%d perceived_wait_ms=%d "
                "bake_wall_ms=%d text=%.60r",
                i + 1,
                len(sentences),
                len(sent),
                perceived_wait_ms,
                chunk.bake_ms,
                sent,
            )
            await _play_prepared_chunk(
                session,
                chunk,
                video_pub=video_pub,
                use_lk_video=use_lk_video,
                audio_via_session=audio_via_session,
                locale_audio=locale_audio,
            )
            prev_play_end = time.monotonic()
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
            try:
                await next_task
            except (asyncio.CancelledError, Exception):
                pass
        cleanup_speech_slots(slots)

    if turn_log is not None:
        turn_log.log(
            "avatar_sync_playout",
            role=speak_role,
            step=step,
            tts_ms=total_tts_ms,
            avatar_bake_ms=total_bake_ms,
            avatar_video_compose_ms=total_bake_ms,
            sync_wait_ms=first_wait_ms or round((time.monotonic() - sync_t0) * 1000),
            first_chunk_sync_wait_ms=first_wait_ms,
            perceived_wait_ms_mean=(
                round(sum(perceived_waits) / len(perceived_waits))
                if perceived_waits
                else None
            ),
            perceived_wait_ms_max=max(perceived_waits) if perceived_waits else None,
            chunk_count=len(sentences),
            video_transport="livekit" if use_lk_video else "http",
            chunk_stream=True,
        )


async def speak_panel_line(
    session: AgentSession,
    data: TalkShowData,
    *,
    speak_role: str,
    text: str,
    step: str,
    texts: dict[str, str] | None = None,
) -> None:
    """Speak any show line via Piper (+ DyStream when avatar sync is on).

    Canonical ``text`` is English (model / history). Delivery localizes to
    ``data.needed_locales`` only — one human language ⇒ one TTS/avatar path.

    Optional ``texts`` skips Gemma translate (canned lines such as session welcome).
    """
    from agent.adapters.locale_audio import get_locale_audio_publisher
    from agent.floor.floor_parser import strip_speech_control_tags
    from agent.locale.localize import texts_for_needed_locales
    from agent.session.session_lifecycle import session_is_active
    from agent.locale.viewer_locales import DEFAULT_LOCALE, primary_delivery_locale

    text = strip_speech_control_tags(text.strip())
    if not text:
        return

    if not session_is_active(session) or data.shutdown_event.is_set():
        logger.info("speak_panel_line skipped — session inactive step=%s", step)
        return

    if speak_role != data.active_role:
        await switch_to_role(session, data, speak_role, reason=f"panel:{step}")

    bridge = get_avatar_bridge()
    turn_log = getattr(data, "turn_log", None)
    use_lk_video = lk_video_enabled()

    needed = frozenset(getattr(data, "needed_locales", None) or {DEFAULT_LOCALE})
    if texts is None:
        texts = await texts_for_needed_locales(data, text)
    else:
        # Keep only locales actually needed; fall back to English canonical.
        texts = {
            loc: (texts.get(loc) or text).strip()
            for loc in needed
            if (texts.get(loc) or text).strip()
        } or {DEFAULT_LOCALE: text}
    primary = primary_delivery_locale(needed)
    primary_text = texts.get(primary) or text
    ui_canonical = texts.get(DEFAULT_LOCALE) or text

    logger.info(
        "PANEL say step=%s role=%s needed=%s primary=%s text=%.80r",
        step,
        speak_role,
        sorted(needed),
        primary,
        primary_text,
    )

    data.speak_line_busy = True
    try:
        if bridge is not None:
            try:
                from agent.emotion import get_role_emotion

                role_emotion = get_role_emotion(data, speak_role)
                primary_tts = load_persona_tts(
                    speak_role, data.runtime.config, locale=primary
                )
                primary_video = get_avatar_video_publisher(primary)

                try:
                    from livekit.agents.job import get_job_context

                    room = get_job_context().room
                    if primary_video is not None:
                        await primary_video.bind_room(room)
                except Exception:
                    logger.exception("primary locale video bind failed locale=%s", primary)

                secondaries = [loc for loc in sorted(needed) if loc != primary]
                secondary_coros = []
                for loc in secondaries:
                    loc_text = texts.get(loc) or text
                    loc_tts = load_persona_tts(
                        speak_role, data.runtime.config, locale=loc
                    )
                    loc_video = get_avatar_video_publisher(loc)
                    loc_audio = get_locale_audio_publisher(loc)
                    if data.agent_session is not None:
                        # Bind secondary tracks once (room from job context inside bind)
                        try:
                            from livekit.agents.job import get_job_context

                            room = get_job_context().room
                            await loc_audio.bind_room(
                                room, sample_rate=loc_tts.sample_rate
                            )
                            if loc_video is not None:
                                await loc_video.bind_room(room)
                        except Exception:
                            logger.exception(
                                "secondary locale bind failed locale=%s", loc
                            )
                    secondary_coros.append(
                        _speak_avatar_chunked(
                            session,
                            data,
                            bridge=bridge,
                            speak_role=speak_role,
                            text=loc_text,
                            step=f"{step}:{loc}",
                            tts_cfg=loc_tts,
                            turn_log=turn_log,
                            video_pub=loc_video,
                            use_lk_video=use_lk_video and loc_video is not None,
                            ui_text=ui_canonical,
                            texts=texts,
                            audio_via_session=False,
                            locale_audio=loc_audio,
                            emit_ui=False,
                            slot_prefix=f"{loc}-",
                            emotion=role_emotion,
                            locale=loc,
                        )
                    )

                primary_coro = _speak_avatar_chunked(
                    session,
                    data,
                    bridge=bridge,
                    speak_role=speak_role,
                    text=primary_text,
                    step=step,
                    tts_cfg=primary_tts,
                    turn_log=turn_log,
                    video_pub=primary_video,
                    use_lk_video=use_lk_video and primary_video is not None,
                    ui_text=ui_canonical,
                    texts=texts,
                    audio_via_session=True,
                    emit_ui=True,
                    emotion=role_emotion,
                    locale=primary,
                )
                if secondary_coros:
                    await asyncio.gather(primary_coro, *secondary_coros)
                else:
                    await primary_coro
            except RuntimeError as exc:
                if "isn't running" in str(exc):
                    logger.info(
                        "speak_panel_line aborted — session stopped step=%s",
                        step,
                    )
                    return
                raise
            return

        queue_speech_ui(
            data, speak_role, ui_canonical, step=step, texts=texts
        )
        try:
            handle = session.say(primary_text, allow_interruptions=False)
            await handle.wait_for_playout()
        except RuntimeError as exc:
            if "isn't running" in str(exc):
                logger.info(
                    "speak_panel_line aborted — session stopped step=%s", step
                )
                return
            raise
    finally:
        data.speak_line_busy = False
        _flush_deferred_panel_followup(data)


def _flush_deferred_panel_followup(data: TalkShowData) -> None:
    """Run panel after speak_panel_line if listening fired between avatar chunks."""
    if not data.panel_followup_deferred:
        return
    data.panel_followup_deferred = False
    if not data.panel_followup_pending or data.panel_chain_running:
        return
    runner = data.panel_followup_runner
    if runner is None:
        return
    data.panel_followup_pending = False
    turn_log = getattr(data, "turn_log", None)
    if turn_log is not None:
        turn_log.log(
            "panel_trigger",
            reason="host_reply_done_deferred",
            floor_next=data.floor_next_speaker or "host",
            room=data.room_name,
        )
    runner()  # type: ignore[operator]
