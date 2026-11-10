"""Speech consumers: render the script the interaction generator produces.

The interaction (generator) runs as fast as its model calls allow: it writes
each line as a ``Cue`` (text, the listeners' backchannels at character offsets,
where the line is cut) and hands it over without waiting for playback. A
consumer only delivers the script. ``ConsoleSpeechConsumer`` prints it
(Text-only); ``AVSpeechConsumer`` queues it for TTS + avatar video and plays
it at speaking pace.

When a human barges in, ``stop`` halts delivery and reports the position the
audience had reached; the generator then discards everything it wrote past it.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from agent.interaction.think_action import Backchannel, insert_backchannels

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

DEFAULT_SPEECH_WORDS_PER_SEC = 2.6
DEFAULT_TURN_GAP_S = 1.0
DEFAULT_TURN_GAP_JITTER_S = 0.5
# Backchannels play over the speaker on their own track; a little louder keeps them audible.
DEFAULT_BACKCHANNEL_GAIN = 1.6
DEFAULT_BACKCHANNEL_MAX_LATENESS_S = 0.25
_WORD_RE = re.compile(r"\S+")


@dataclass(eq=False)
class Cue:
    """One scripted line. ``end`` is set by the generator once the line is
    decided: ``len(text)``, or less when an interrupt cut it there."""

    role: str
    text: str
    step: str
    texts: dict[str, str] | None = None
    whole: bool = False  # canned line, not disclosed interval by interval
    canned: bool = False  # fixed host line: may play from the pre-rendered cache
    # TTS label fixed when the line was queued; None reads the mood at playback.
    emotion: str | None = None
    end: int | None = None
    on_start: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    mark: Any = None  # generator checkpoint for rollback
    interrupted_by: str | None = None  # role whose approved interrupt cut this line
    played_segments: dict[int, tuple[float, float]] = field(default_factory=dict)
    # TTS sentence index -> (audio position at its start, monotonic time the audience
    # started hearing it, duration, word-end timings task or None)
    started_segments: dict[int, tuple[float, float, float, Any]] = field(default_factory=dict)
    playout_done: asyncio.Event = field(default_factory=asyncio.Event)
    playout_started_at: float | None = None
    playout_ended_at: float | None = None  # set only when the line played to its end


# (cue, characters of cue.text the audience heard)
Position = tuple[Cue, int]


class SpeechConsumer(Protocol):
    async def submit(self, data: TalkShowData, cue: Cue) -> None:
        """Queue a line; returns without waiting for it to be delivered."""

    def disclose(
        self, cue: Cue, end: int, chunk: str, backchannels: list[tuple[str, Backchannel, int]]
    ) -> None:
        """``cue.text[:end]`` is decided; (role, backchannel, char offset) fall inside ``chunk``."""

    def close(self, cue: Cue, end: int) -> None:
        """The line is fully decided and ends at ``end``."""

    def stop(self) -> tuple[Position | None, bool]:
        """Barge-in: halt delivery, drop the queue. Returns where the audience is
        and whether anything undelivered was cut."""

    def idle(self) -> bool:
        """Nothing queued or playing."""

    def ahead(self) -> int:
        """Lines queued that have not started playing yet."""

    async def drain(self) -> None:
        """Wait until everything queued has been delivered."""


async def _run_on_start(cue: Cue) -> None:
    for callback in cue.on_start:
        try:
            await callback()
        except Exception:
            logger.exception("cue on_start failed step=%s", cue.step)


class ConsoleSpeechConsumer:
    """Text-only: intervals are printed as soon as the generator decides them."""

    def __init__(self) -> None:
        self._open: Cue | None = None
        self._printed = 0
        self._last: Position | None = None

    async def submit(self, data: TalkShowData, cue: Cue) -> None:
        from agent.config import load_persona_name
        from agent.session.session_handoff import speak_panel_line, switch_to_role

        await _run_on_start(cue)
        if cue.whole:
            await speak_panel_line(
                None, data, speak_role=cue.role, text=cue.text, step=cue.step, texts=cue.texts,
                emotion=cue.emotion,
            )
            self._last = (cue, len(cue.text))
            return
        await switch_to_role(None, data, cue.role, reason=f"panel:{cue.step}")
        print(f"\n{load_persona_name(cue.role)}: ", end="", flush=True)
        self._open = cue
        self._printed = 0

    def disclose(
        self, cue: Cue, end: int, chunk: str, backchannels: list[tuple[str, Backchannel, int]]
    ) -> None:
        from agent.config import load_persona_name

        named = [(load_persona_name(role), event) for role, event, _ in backchannels]
        print(insert_backchannels(chunk, named), end="", flush=True)
        self._printed = end

    def close(self, cue: Cue, end: int) -> None:
        cue.end = end
        if self._open is cue:
            print(flush=True)
            self._open = None
            self._last = (cue, end)

    def stop(self) -> tuple[Position | None, bool]:
        if self._open is None:
            return self._last, False
        cue, self._open = self._open, None
        print(flush=True)
        self._last = (cue, self._printed)
        return self._last, True

    def idle(self) -> bool:
        return True

    def ahead(self) -> int:
        return 0

    async def drain(self) -> None:
        return None


def _char_end_after_words(text: str, words: int) -> int:
    if words <= 0:
        return 0
    spans = list(_WORD_RE.finditer(text))
    if words >= len(spans):
        return len(text)
    return spans[words - 1].end()


def _is_speaking(state: object) -> bool:
    return str(state).lower().rsplit(".", 1)[-1] == "speaking"


def turn_gap_s() -> float:
    """Pause before the next speaker takes over: base + uniform jitter (like a real
    conversation, nobody answers instantly)."""

    def env(name: str, default: float) -> float:
        try:
            return max(0.0, float(os.environ.get(name, default)))
        except ValueError:
            return default

    jitter = env("TALKSHOW_TURN_GAP_JITTER_S", DEFAULT_TURN_GAP_JITTER_S)
    return env("TALKSHOW_TURN_GAP_S", DEFAULT_TURN_GAP_S) + random.uniform(0.0, jitter)


def voice_backchannels_enabled() -> bool:
    return os.environ.get("TALKSHOW_VOICE_BACKCHANNELS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


_ui_tasks: set[asyncio.Task] = set()


def _fire_ui(coro: Awaitable[None]) -> None:
    """Publish a UI event without holding up playback."""
    task = asyncio.ensure_future(coro)
    _ui_tasks.add(task)
    task.add_done_callback(_ui_tasks.discard)


async def _prepare_backchannel(
    data: TalkShowData, role: str, text: str
) -> tuple[Any, bytes, int]:
    """TTS a listener's backchannel for its own audio track (no avatar video)."""
    from livekit.agents.job import get_job_context

    from agent.adapters.locale_audio import get_backchannel_audio_publisher
    from agent.adapters.tts_synthesize import synthesize_pcm_for_config
    from agent.config import load_persona_tts
    from agent.emotion import get_role_emotion
    from agent.locale.localize import translate_text
    from agent.locale.viewer_locales import DEFAULT_LOCALE, primary_delivery_locale

    locale = primary_delivery_locale(getattr(data, "needed_locales", None) or {DEFAULT_LOCALE})
    if locale != DEFAULT_LOCALE:
        text = await translate_text(data, text, target_locale=locale)
    tts_cfg = load_persona_tts(role, data.runtime.config, locale=locale)
    pcm, sample_rate, _ = await synthesize_pcm_for_config(
        tts_cfg,
        text,
        emotion=get_role_emotion(data, role),
        locale=locale,
        turn_log=getattr(data, "turn_log", None),
        room=data.room_name,
        role=role,
        step="backchannel",
    )
    publisher = get_backchannel_audio_publisher(role, locale)
    await publisher.bind_room(get_job_context().room, sample_rate=sample_rate)
    return publisher, _amplify_pcm(pcm, backchannel_gain()), sample_rate


def backchannel_gain() -> float:
    """Volume multiplier for backchannels relative to the main line (``TALKSHOW_BACKCHANNEL_GAIN``)."""
    try:
        gain = float(os.environ.get("TALKSHOW_BACKCHANNEL_GAIN", DEFAULT_BACKCHANNEL_GAIN))
    except ValueError:
        gain = DEFAULT_BACKCHANNEL_GAIN
    return max(0.0, gain)


def _amplify_pcm(pcm: bytes, gain: float) -> bytes:
    """Scale int16 mono PCM by ``gain``, clipping at full scale."""
    if gain == 1.0 or not pcm:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32)
    return np.clip(samples * gain, -32768, 32767).astype(np.int16).tobytes()


class AVSpeechConsumer:
    """LiveKit: cues play one after another through ``speak_panel_line`` (TTS + avatar).

    A cue starts as soon as it is queued, even while the generator is still
    deciding its later intervals; a cut decided later is applied when playback
    reaches it. Playback progress is estimated from the first ``speaking`` agent
    state of the cue and a speaking rate. Backchannels are pre-synthesized and
    voiced over the line when playback reaches their word, timed from the latest
    real playback anchor (line start or a finished TTS sentence).
    """

    _POLL_S = 0.05

    def __init__(self, session: AgentSession, *, words_per_sec: float | None = None) -> None:
        self.session = session
        self._wps = words_per_sec or float(
            os.environ.get("TALKSHOW_SPEECH_WORDS_PER_SEC", DEFAULT_SPEECH_WORDS_PER_SEC)
        )
        self._data: TalkShowData | None = None
        self._queue: collections.deque[Cue] = collections.deque()
        self._worker: asyncio.Task | None = None
        self._speech: asyncio.Task | None = None
        self._current: Cue | None = None
        self._playing_text = ""
        self._started_at: float | None = None
        self._last: Position | None = None
        # When the previous cue finished playing naturally (None: cut, or nothing yet).
        self._last_end: float | None = None
        self._last_role: str | None = None
        self._backchannels: set[asyncio.Task] = set()
        on = getattr(session, "on", None)
        if on is not None:
            on("agent_state_changed", self._on_agent_state)

    def _on_agent_state(self, ev: Any) -> None:
        if self._started_at is None and self._current is not None and _is_speaking(
            getattr(ev, "new_state", "")
        ):
            self._started_at = time.monotonic()
            self._current.playout_started_at = self._started_at

    def _sync_clock(self) -> None:
        # Back-to-back cues may not produce a new speaking transition.
        if self._started_at is None and self._speech is not None and _is_speaking(
            getattr(self.session, "agent_state", "")
        ):
            self._started_at = time.monotonic()
            if self._current is not None:
                self._current.playout_started_at = self._started_at

    def heard_chars(self) -> int:
        """How much of the current cue's text the audience has heard."""
        if self._current is None:
            return 0
        if self._speech is not None and self._speech.done():
            return len(self._playing_text)
        self._sync_clock()
        if self._started_at is None:
            return 0
        words = int((time.monotonic() - self._started_at) * self._wps)
        return _char_end_after_words(self._playing_text, words)

    # --- generator side -----------------------------------------------------

    async def submit(self, data: TalkShowData, cue: Cue) -> None:
        self._data = data
        if cue.whole:
            from agent.session.session_handoff import log_whole_line

            # Panel lines are logged as they are generated; whole lines when queued.
            log_whole_line(data, cue.role, cue.text, cue.step)
        self._queue.append(cue)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="speech:playout")

    def disclose(
        self, cue: Cue, end: int, chunk: str, backchannels: list[tuple[str, Backchannel, int]]
    ) -> None:
        if self._data is None or not voice_backchannels_enabled():
            return
        for role, event, offset in backchannels:
            task = asyncio.create_task(
                self._backchannel(
                    cue, role, event.content, offset, event.target_segment_index
                ),
                name=f"backchannel:{role}",
            )
            self._backchannels.add(task)
            task.add_done_callback(self._backchannel_done)

    def close(self, cue: Cue, end: int) -> None:
        cue.end = end

    def stop(self) -> tuple[Position | None, bool]:
        if self._current is not None:
            position: Position | None = (self._current, self.heard_chars())
        elif self._queue:
            position = (self._queue[0], 0)
        else:
            position = self._last
        cut = self._current is not None or bool(self._queue)
        for task in [self._worker, self._speech, *self._backchannels]:
            if task is not None and not task.done():
                task.cancel()
        for cue in ([self._current] if self._current is not None else []) + list(self._queue):
            cue.playout_done.set()
        if self._current is not None:
            try:
                self.session.interrupt(force=True)
            except Exception:
                logger.debug("session.interrupt failed", exc_info=True)
        self._queue.clear()
        self._worker = self._speech = self._current = None
        self._last_end = None
        if position is not None:
            self._last = position
        return position, cut

    def idle(self) -> bool:
        return self._current is None and not self._queue

    def ahead(self) -> int:
        return len(self._queue)

    async def drain(self) -> None:
        while not self.idle():
            await asyncio.sleep(self._POLL_S)

    # --- playout ------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            if not self._queue:
                if await self._return_to_listen_role():
                    continue
                return
            await self._turn_gap()
            cue = self._queue.popleft()
            self._current = cue
            await self._play(cue)
            self._current = None

    async def _turn_gap(self) -> None:
        """On a normal speaker switch, hold the next cue until a turn gap has passed
        since the last line ended (time the generator already took counts toward it).
        No gap when the same speaker goes on, or for an interrupt (the script cut the
        previous line); backchannels are on their own tracks and never wait."""
        if self._last_end is None or self._queue[0].role == self._last_role:
            return
        wait = turn_gap_s() - (time.monotonic() - self._last_end)
        if wait > 0:
            await asyncio.sleep(wait)

    async def _play(self, cue: Cue) -> None:
        from agent.session.session_handoff import speak_panel_line

        text = cue.text
        if cue.end is not None and cue.end < len(text):
            text = text[: cue.end].rstrip()
        self._playing_text = text
        self._started_at = None
        cue.playout_started_at = None
        self._speech = None
        await _run_on_start(cue)
        await self._announce_transcript(cue, text)

        def segment_played(segment_index: int, audio_position_s: float) -> None:
            played_at = time.monotonic()
            cue.played_segments[segment_index] = (audio_position_s, played_at)

        def segment_started(
            segment_index: int, audio_position_s: float, duration_s: float, word_ends: Any
        ) -> None:
            cue.started_segments[segment_index] = (
                audio_position_s, time.monotonic(), duration_s, word_ends
            )

        speech = self._speech = asyncio.create_task(
            speak_panel_line(
                self.session,
                self._data,
                speak_role=cue.role,
                text=text,
                step=cue.step,
                texts=cue.texts if text == cue.text else None,
                canned=cue.canned and text == cue.text,
                on_segment_played=segment_played,
                on_segment_started=segment_started,
                emotion=cue.emotion,
            ),
            name=f"speech:{cue.role}:{cue.step}",
        )
        heard = len(text)
        while not speech.done():
            self._sync_clock()
            if cue.end is not None and cue.end < len(text) and self.heard_chars() >= cue.end:
                # The script cut this line (approved interrupt) after playback began.
                heard = cue.end
                speech.cancel()
                try:
                    self.session.interrupt(force=True)
                except Exception:
                    logger.debug("session.interrupt failed", exc_info=True)
                break
            await asyncio.wait({speech}, timeout=self._POLL_S)
        try:
            await speech
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise  # stop(): the worker itself is being cancelled
        except Exception:
            logger.exception("speech playback failed step=%s", cue.step)
        cut_by_script = cue.end is not None and cue.end < len(cue.text)
        if not cut_by_script and speech.done() and not speech.cancelled():
            cue.playout_ended_at = time.monotonic()
        cue.playout_done.set()
        self._last = (cue, heard)
        if cut_by_script and cue.interrupted_by:
            from agent.ui.ui_events import emit_interrupt

            _fire_ui(emit_interrupt(cue.role, cue.interrupted_by, heard_text=text[:heard]))
        self._last_end = None if cut_by_script else time.monotonic()
        self._last_role = cue.role

    async def _announce_transcript(self, cue: Cue, text: str) -> None:
        """Open the line's transcript before its audio starts. The synced words on
        ``lk.transcription`` travel the same reliable data channel, so the browser always
        sees this first and attributes the stream to the right speaker."""
        from agent.ui.ui_events import emit_transcript

        announced = getattr(self._data, "announced_transcripts", None)
        if announced is None or not text.strip():
            return
        texts = cue.texts if text == cue.text else None
        await emit_transcript(cue.role, text, step=cue.step, texts=texts, stream=True)
        announced.add((cue.role, cue.step))

    async def _return_to_listen_role(self) -> bool:
        """Drained: hand the session back to the host once the generator is idle too.
        Returns True if a new cue arrived meanwhile."""
        data = self._data
        if data is None or getattr(data, "panel_chain_running", False):
            return False
        from agent.floor import TurnController
        from agent.session.session_handoff import switch_to_role

        listen = TurnController(data.scenario, data).listen_role()
        if data.active_role != listen:
            try:
                await switch_to_role(self.session, data, listen, reason="playout_done")
            except Exception:
                logger.exception("return to %s after playout failed", listen)
        data.touch_activity()
        return bool(self._queue)

    async def _backchannel(
        self,
        cue: Cue,
        role: str,
        text: str,
        offset: int,
        target_segment_index: int | None = None,
    ) -> None:
        """Voice a prepared backchannel when the speaker's audio reaches ``offset``.

        The due time is projected from the latest real playback anchor before the
        target (line start, or a TTS sentence that finished playing) at the line's
        measured speaking rate. A target exactly at a sentence end is due on that
        sentence's playback event. A projection that slips past the lateness window
        waits for a later anchor to confirm where playback is; a target that anchor
        shows was missed, or that is cut or stopped, is dropped."""
        try:
            publisher, pcm, sample_rate = await _prepare_backchannel(self._data, role, text)
            if not pcm:
                self._log_backchannel(cue, role, text, offset, "dropped", "empty_synthesized_audio")
                return
            max_late = _backchannel_max_lateness_s()
            while True:
                if cue.end is not None and cue.end < offset:
                    self._log_backchannel(
                        cue, role, text, offset, "dropped", "cut_before_target",
                        target_segment_index,
                    )
                    return
                timing = self._backchannel_due(cue, offset)
                if timing is not None:
                    due_at, audio_position_s, confirmed = timing
                    late_by_s = time.monotonic() - due_at
                    if late_by_s >= 0 and (confirmed or late_by_s <= max_late):
                        break
                if cue.playout_done.is_set() and (timing is None or not timing[2]):
                    self._log_backchannel(
                        cue, role, text, offset, "dropped", "speaker_stopped",
                        target_segment_index,
                    )
                    return
                wait = -late_by_s if timing is not None and late_by_s < 0 else self._POLL_S
                await asyncio.sleep(min(self._POLL_S, wait))
            if late_by_s > max_late:
                self._log_backchannel(
                    cue, role, text, offset, "dropped", "late_after_target",
                    target_segment_index, audio_position_s, late_by_s,
                )
                return
            interaction = getattr(self._data, "floor_interaction", None)
            controller = getattr(interaction, "backchannel_controller", None)
            if controller is not None:
                allowed, reason = controller.try_start(id(cue), role, text)
                if not allowed:
                    self._log_backchannel(
                        cue, role, text, offset, "dropped", reason,
                        target_segment_index, audio_position_s, late_by_s,
                    )
                    return
            duration_s = len(pcm) / 2 / sample_rate if sample_rate > 0 else 0.0

            def on_started(started_at: float) -> None:
                relative_start_s = (
                    started_at - cue.playout_started_at
                    if cue.playout_started_at is not None
                    else None
                )
                actual_late_by_s = started_at - due_at
                self._log_backchannel(
                    cue,
                    role,
                    text,
                    offset,
                    "started" if actual_late_by_s <= max_late else "started_late",
                    "server_audio_source_capture",
                    target_segment_index,
                    audio_position_s,
                    actual_late_by_s,
                    relative_start_s,
                )
                from agent.ui.ui_events import emit_backchannel

                _fire_ui(
                    emit_backchannel(
                        role,
                        text,
                        speaker_role=cue.role,
                        duration_s=duration_s,
                    )
                )
        except asyncio.CancelledError:
            self._log_backchannel(cue, role, text, offset, "dropped", "playback_cancelled")
            raise
        await publisher.play_pcm(pcm, sample_rate, on_started=on_started)

    def _backchannel_due(self, cue: Cue, offset: int) -> tuple[float, float, bool] | None:
        """(monotonic time, speaker audio position, confirmed) at which playback reaches
        ``offset``, or None while no playback anchor is known yet. ``confirmed``: an
        anchor at or past the target shows playback really got there."""
        spans = _sentence_spans(cue.text)
        # The target's sentence is being heard: its playback start plus where the target
        # word ends in that sentence's audio (forced alignment), or, until alignment is
        # in, the share of the sentence's words before it times its real duration.
        for index, (start, end) in enumerate(spans):
            if start <= offset <= end:
                segment = cue.started_segments.get(index)
                if segment is not None:
                    audio_start, started_at, duration_s, word_ends = segment
                    total = len(_WORD_RE.findall(cue.text[start:end])) or 1
                    words = len(_WORD_RE.findall(cue.text[start:offset]))
                    ends = _word_ends(word_ends)
                    if ends is not None and len(ends) == total and words > 0:
                        into = min(ends[words - 1], duration_s)
                        confirmed = True  # aligned to the audio actually playing
                    else:
                        into = words / total * duration_s
                        confirmed = index in cue.played_segments
                    return started_at + into, audio_start + into, confirmed
                break
        # Anchors: (char offset, monotonic time, audio position) known from real playback.
        anchors: list[tuple[int, float, float]] = []
        if cue.playout_started_at is not None:
            anchors.append((0, cue.playout_started_at, 0.0))
        for index, (audio_position_s, played_at) in sorted(cue.played_segments.items()):
            if index < len(spans):
                anchors.append((spans[index][1], played_at, audio_position_s))
        if cue.playout_ended_at is not None and cue.playout_started_at is not None:
            anchors.append((
                len(cue.text.rstrip()),
                cue.playout_ended_at,
                cue.playout_ended_at - cue.playout_started_at,
            ))
        if not anchors:
            return None
        # Words per second measured on this line so far, else the configured rate.
        wps = self._wps
        last_char, _, last_audio = max(anchors, key=lambda a: a[0])
        if last_char > 0 and last_audio > 0:
            wps = len(_WORD_RE.findall(cue.text[:last_char])) / last_audio
        passed = [a for a in anchors if a[0] >= offset]
        if passed:
            # Playback already reached or passed the target: due at its exact time
            # if this anchor sits on it, else back-projected from that anchor.
            char, at, audio = min(passed, key=lambda a: a[0])
            words = len(_WORD_RE.findall(cue.text[offset:char]))
            return at - words / wps, max(0.0, audio - words / wps), True
        char, at, audio = max(anchors, key=lambda a: a[0])
        words = len(_WORD_RE.findall(cue.text[char:offset]))
        return at + words / wps, audio + words / wps, False

    def _log_backchannel(
        self,
        cue: Cue,
        role: str,
        text: str,
        offset: int,
        decision: str,
        reason: str,
        target_segment_index: int | None = None,
        target_audio_position_s: float | None = None,
        late_by_s: float | None = None,
        actual_relative_start_s: float | None = None,
    ) -> None:
        data = self._data
        client = getattr(getattr(data, "runtime", None), "gemma_client", None)
        log_event = getattr(client, "log_text_only_event", None)
        if log_event is not None:
            log_event(
                role,
                "backchannel_playback",
                candidate=text,
                speaker_role=cue.role,
                speaker_step=cue.step,
                target_char_offset=offset,
                target_segment_index=target_segment_index,
                target_audio_position_s=target_audio_position_s,
                actual_relative_start_s=actual_relative_start_s,
                late_by_s=late_by_s,
                decision=decision,
                reason=reason,
            )

    def _backchannel_done(self, task: asyncio.Task) -> None:
        self._backchannels.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("backchannel playback failed", exc_info=task.exception())


def _word_ends(task: Any) -> list[float] | None:
    """A finished word-alignment result, else None (pending, failed, or absent)."""
    if task is None or not task.done() or task.cancelled() or task.exception() is not None:
        return None
    return task.result()


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each TTS sentence of ``text``; end excludes trailing space."""
    from agent.session.speak_chunks import split_speak_sentences

    spans: list[tuple[int, int]] = []
    search_from = 0
    for segment in split_speak_sentences(text):
        start = text.find(segment, search_from)
        if start < 0:
            break
        spans.append((start, start + len(segment.rstrip())))
        search_from = start + len(segment)
    return spans


def _backchannel_max_lateness_s() -> float:
    try:
        return max(0.0, float(os.environ.get(
            "TALKSHOW_BACKCHANNEL_MAX_LATENESS_S", DEFAULT_BACKCHANNEL_MAX_LATENESS_S
        )))
    except ValueError:
        return DEFAULT_BACKCHANNEL_MAX_LATENESS_S


__all__ = ["AVSpeechConsumer", "ConsoleSpeechConsumer", "Cue", "Position", "SpeechConsumer"]
