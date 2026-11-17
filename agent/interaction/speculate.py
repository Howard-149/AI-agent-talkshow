"""Think while the human is still talking.

Every few seconds of a human utterance, what has been said so far is transcribed (Gemma
audio-in, the same call as the final STT) and every panelist's think call starts on it
(``ThinkActionInteraction.speculate_on_partial``). When the final transcript arrives,
the newest speculation whose text covers it is used instead of thinking from scratch;
otherwise it is discarded. The think call (gpt-5.5 in mode 2: 4.5–9 s) was most of the
silence after a human turn.

TALKSHOW_SPECULATE_HUMAN=0 turns it off.
"""

from __future__ import annotations

import asyncio
import collections
import difflib
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 1.5  # transcribe the utterance so far this often (11/17: 2.5 s left the newest speculation 1.6–3.9 s from done at the end)
DEFAULT_MIN_SPEECH_S = 2.0  # nothing to think about before this much speech
PREROLL_S = 0.8  # VAD reports speech a little after it starts
SAMPLE_RATE = 16000
# The final transcript may run this far past a speculation (its last words, said while
# the speculation was already thinking) and still use it.
DEFAULT_MAX_MISSING_WORDS = 6
DEFAULT_MAX_MISSING_FRACTION = 0.15
MIN_PARTIAL_WORDS = 4

_WORD_RE = re.compile(r"[a-z0-9']+")


def _text_on_main_llm() -> bool:
    """Mode 1: think calls go to the local vLLM, which the speculations share."""
    return not os.environ.get("TALKSHOW_TEXT_LLM_BASE_URL", "").strip()


def interval_s() -> float:
    """How often to transcribe and speculate. On the local LLM the speculations compete
    with each other for the GPU (11/17: 1.5 s kept a think call at ~4.5 s), so less often."""
    return _env_float("TALKSHOW_SPECULATE_INTERVAL_S", 2.5 if _text_on_main_llm() else DEFAULT_INTERVAL_S)


def max_speculations() -> int:
    """Speculations kept at once (older ones are cancelled)."""
    return int(_env_float("TALKSHOW_SPECULATE_MAX", 2 if _text_on_main_llm() else 4))


def one_role() -> bool:
    """TALKSHOW_SPECULATE_ROLES=one (default): think ahead only for the panelist most
    likely to answer; all: every panelist (a full floor arbitration, slower)."""
    return os.environ.get("TALKSHOW_SPECULATE_ROLES", "one").strip().lower() != "all"


def likely_responder(partial: str, roles: list[str], recent_roles: list[str], host: str) -> str:
    """Who most likely answers the human: a panelist they name, else the panelist who has
    gone longest without speaking (the host only if nobody else is there)."""
    from agent.config import load_persona_name

    said = set(words(partial))
    for role in roles:
        try:
            name = load_persona_name(role)
        except Exception:
            name = role
        if name and set(words(name)) & said:
            return role
    panel = [role for role in roles if role != host] or list(roles)
    last_spoke = {role: i for i, role in enumerate(recent_roles)}
    return min(panel, key=lambda role: last_spoke.get(role, -1))


def speculation_enabled() -> bool:
    return os.environ.get("TALKSHOW_SPECULATE_HUMAN", "1").strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def covers(partial: str, final: str) -> tuple[bool, int]:
    """Does a speculation on ``partial`` stand for the turn ``final``? (ok, words of
    ``final`` it has not seen). The two come from separate transcriptions, so they are
    matched word by word, not as a string prefix."""
    p, f = words(partial), words(final)
    if len(p) < MIN_PARTIAL_WORDS or not f:
        return False, len(f)
    matched = sum(block.size for block in difflib.SequenceMatcher(None, p, f, autojunk=False).get_matching_blocks())
    missing = len(f) - matched
    limit = max(
        _env_float("TALKSHOW_SPECULATE_MAX_MISSING_WORDS", DEFAULT_MAX_MISSING_WORDS),
        _env_float("TALKSHOW_SPECULATE_MAX_MISSING_FRACTION", DEFAULT_MAX_MISSING_FRACTION) * len(f),
    )
    # The partial must be (almost all) in the final: a speculation on words the final
    # transcript does not have was thinking about something else.
    return missing <= limit and matched >= 0.8 * len(p), missing


class HumanPartialTranscriber:
    """Follows the human's microphone and feeds partial transcripts to the floor."""

    def __init__(self, data: Any, room: Any, session: Any) -> None:
        self._data = data
        self._room = room
        self._session = session
        self._participant: Any = None
        self._stream_task: asyncio.Task | None = None
        self._ticker: asyncio.Task | None = None
        self._ring: collections.deque = collections.deque()  # recent frames (preroll)
        self._utterance: list[Any] | None = None  # frames since speech start
        self._busy = False

    # --- VAD ---------------------------------------------------------------

    def on_user_state(self, state: str) -> None:
        if not speculation_enabled():
            return
        if state == "speaking":
            self._ensure_stream()
            self._utterance = list(self._ring)
            if self._ticker is None or self._ticker.done():
                self._ticker = asyncio.create_task(self._tick(), name="speculate:tick")
        else:
            self._utterance = None
            if self._ticker is not None:
                self._ticker.cancel()
                self._ticker = None

    # --- audio -------------------------------------------------------------

    def _human_participant(self) -> Any:
        room_io = getattr(self._session, "_room_io", None) or getattr(self._session, "room_io", None)
        linked = getattr(room_io, "linked_participant", None)
        if linked is not None:
            return linked
        for participant in getattr(self._room, "remote_participants", {}).values():
            identity = str(getattr(participant, "identity", ""))
            if "viewer" in identity:
                continue
            for pub in getattr(participant, "track_publications", {}).values():
                if str(getattr(pub, "kind", "")).endswith("AUDIO") or getattr(pub, "kind", None) == 1:
                    return participant
        return None

    def _ensure_stream(self) -> None:
        participant = self._human_participant()
        if participant is None:
            return
        if participant is self._participant and self._stream_task is not None and not self._stream_task.done():
            return
        if self._stream_task is not None:
            self._stream_task.cancel()
        self._participant = participant
        self._stream_task = asyncio.create_task(self._read(participant), name="speculate:audio")

    async def _read(self, participant: Any) -> None:
        from livekit import rtc

        stream = rtc.AudioStream.from_participant(
            participant=participant,
            track_source=rtc.TrackSource.SOURCE_MICROPHONE,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        try:
            async for event in stream:
                frame = event.frame
                self._ring.append(frame)
                held = sum(f.samples_per_channel for f in self._ring) / SAMPLE_RATE
                while held > PREROLL_S and len(self._ring) > 1:
                    held -= self._ring.popleft().samples_per_channel / SAMPLE_RATE
                if self._utterance is not None:
                    self._utterance.append(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("speculation audio stream failed")
        finally:
            await stream.aclose()

    # --- partial transcripts ----------------------------------------------

    async def _tick(self) -> None:
        interval = interval_s()
        min_speech = _env_float("TALKSHOW_SPECULATE_MIN_SPEECH_S", DEFAULT_MIN_SPEECH_S)
        while True:
            await asyncio.sleep(interval)
            frames = self._utterance
            if not frames or self._busy:
                continue
            seconds = sum(f.samples_per_channel for f in frames) / SAMPLE_RATE
            if seconds < min_speech:
                continue
            self._busy = True
            asyncio.create_task(self._transcribe(list(frames), seconds), name="speculate:stt")

    async def _transcribe(self, frames: list[Any], seconds: float) -> None:
        from livekit import rtc

        from agent.adapters.gemma_audio_stt import _human_has_floor
        from agent.show.show_context import host_audio_system_prompt, host_audio_user_hint

        data = self._data
        try:
            if not _human_has_floor(data):
                return
            wav = rtc.combine_audio_frames(frames).to_wav_bytes()
            t0 = time.monotonic()
            parsed = await data.runtime.gemma_client.complete_from_wav(
                wav,
                user_text=host_audio_user_hint(),
                history_messages=[],
                output_role="host",
                system_prompt=host_audio_system_prompt(),
                temperature=0.0,
            )
            heard = (parsed.heard or parsed.reply or "").strip()
            if data.turn_log is not None:
                data.turn_log.log(
                    "speculate_partial",
                    room=data.room_name,
                    speech_s=round(seconds, 2),
                    stt_s=round(time.monotonic() - t0, 3),
                    words=len(words(heard)),
                )
            if self._utterance is None:
                return  # the utterance ended meanwhile: the final STT takes over
            interaction = getattr(data, "floor_interaction", None)
            speculate = getattr(interaction, "speculate_on_partial", None)
            if speculate is not None and heard:
                speculate(data, heard)
        except Exception:
            logger.exception("partial transcript failed")
        finally:
            self._busy = False

    async def aclose(self) -> None:
        for task in (self._ticker, self._stream_task):
            if task is not None:
                task.cancel()
