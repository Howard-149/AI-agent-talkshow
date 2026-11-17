"""Join a LiveKit room as the human guest and speak scripted turns into the show.

Think & Action floor: the human never waits for a floor grant. A scripted turn can say
when to cut in (after how many agent lines, at a line's end or partway into it); a turn
that doesn't waits for the panel to finish a beat (``--beat-panel`` / ``--quiet-sec``).

A turn can also be written live (``generate: true``, live.py): an LLM writes the words
from what the panel has said so far, early enough that the audio is ready when the
script says to cut in.

Turns are spoken as microphone audio — CosyVoice from a recorded reference (voice.py) —
so the agent hears them through VAD, barge-in and STT like a real guest, and a recording
has the guest's voice. Replay scripts (a ``prefix``) and ``--text-turns`` send the text
instead: no audio, no STT, the exact same words every run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from eval.ghost_session.live import TranscriptLine

# Protocol topics — keep in sync with agent.ui.control_events / ui_events.
CONTROL_TOPIC = "talkshow/control"
UI_TOPIC = "talkshow/ui"

logger = logging.getLogger("ghost_session")

_REPO_ROOT = Path(__file__).resolve().parents[2]

WHEN = ("line_end", "mid_line")
# Silence after an agent playout that counts as the end of its line. Sentences of one
# line follow each other closely; a new speaker waits TALKSHOW_TURN_GAP_S (>= 1 s).
LINE_END_QUIET_S = 0.8
MIC_RATE = 48000
FRAME_MS = 10


@dataclass
class ScriptedTurn:
    text: str
    topic: str = ""
    # Cut in once this many agent lines have started since the guest last spoke (since
    # the session started, for the first turn). None: wait for the beat to end instead.
    after_lines: int | None = None
    # line_end: wait for that line to finish. mid_line: cut in after_sec into it.
    when: str = "line_end"
    after_sec: float = 0.0
    emotion: str = ""  # CosyVoice delivery (Anger, Sadness, …); "" = Neutral
    audio: str = ""  # a recorded wav to play instead of synthesizing
    max_wait_sec: float = 180.0  # speak anyway after this long
    # Live turn: no text; an LLM writes it from the show so far (live.py), with ``goal``
    # as a hint for what this turn should do.
    generate: bool = False
    goal: str = ""


@dataclass
class GhostScript:
    locale: str = "en"
    identity: str = "ghost-guest"
    name: str = "Guest"  # shown on the guest's seat and transcript
    turns: list[ScriptedTurn] = field(default_factory=list)
    # Replay mode (worker needs TALKSHOW_GHOST_REPLAY=1): sent with the first turn;
    # the worker swaps in this history / mood state before handling the turn.
    prefix: dict[str, Any] | None = None
    # Live turns: who the guest is and what the show is about (prompt context only).
    persona: str = ""
    topic: str = ""
    # Reactive mode (reactive.py): after the turns above (seeds), the guest decides on
    # every agent line whether to answer / react / cut in, within these limits.
    reactive: bool = False
    max_turns: int = 12  # all of his turns, seeds included (a cap; time ends a show)
    min_lines_between_turns: int = 3  # spontaneous turns only (an answer when asked is always given)
    max_minutes: float = 9.5  # from the first agent line
    nudge_after_lines: int = 5  # quiet this many agent lines → he answers the next eligible one
    # Ending (ending.py): a goodbye at the next gap, then wrap_up, then leave after the
    # host's close. Always on in reactive mode; scripted runs opt in (or --wrap-up).
    wrap_up: bool = False
    goodbye: str = ""  # fixed goodbye text; default: LLM-written


def _turn(item: Any) -> ScriptedTurn | None:
    if isinstance(item, str):
        return ScriptedTurn(text=item.strip()) if item.strip() else None
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or "").strip()
    generate = bool(item.get("generate")) and not text
    if not text and not generate:
        return None
    when = str(item.get("when") or "line_end").strip()
    if when not in WHEN:
        raise ValueError(f"ghost turn: when must be one of {WHEN}, got {when!r}")
    after_lines = item.get("after_lines")
    return ScriptedTurn(
        text=text,
        topic=str(item.get("topic") or "").strip(),
        after_lines=None if after_lines is None else max(0, int(after_lines)),
        when=when,
        after_sec=max(0.0, float(item.get("after_sec", 0.0))),
        emotion=str(item.get("emotion") or "").strip(),
        audio=str(item.get("audio") or "").strip(),
        max_wait_sec=float(item.get("max_wait_sec", 180.0)),
        generate=generate,
        goal=str(item.get("goal") or "").strip(),
    )


def load_script(path: Path) -> GhostScript:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"ghost script must be a mapping: {path}")
    turns = [t for t in (_turn(item) for item in raw.get("turns") or []) if t is not None]
    prefix: dict[str, Any] | None = None
    if raw.get("prefix_file"):
        prefix_path = Path(str(raw["prefix_file"]))
        if not prefix_path.is_absolute():
            prefix_path = _REPO_ROOT / prefix_path
        prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    if raw.get("persona_swap"):
        prefix = dict(prefix or {})
        prefix["persona_swap"] = True
    if raw.get("flags"):
        prefix = dict(prefix or {})
        prefix["flags"] = dict(raw["flags"])
    return GhostScript(
        locale=str(raw.get("locale") or "en").strip() or "en",
        identity=str(raw.get("identity") or "ghost-guest").strip() or "ghost-guest",
        name=str(raw.get("name") or "Guest").strip() or "Guest",
        turns=turns,
        prefix=prefix,
        persona=str(raw.get("persona") or "").strip(),
        topic=str(raw.get("topic") or "").strip(),
        reactive=bool(raw.get("reactive")),
        max_turns=int(raw.get("max_turns", 12)),
        min_lines_between_turns=max(0, int(raw.get("min_lines_between_turns", 3))),
        max_minutes=float(raw.get("max_minutes", 9.5)),
        nudge_after_lines=max(0, int(raw.get("nudge_after_lines", 5))),
        wrap_up=bool(raw.get("wrap_up")),
        goodbye=str(raw.get("goodbye") or "").strip(),
    )


def mint_ghost_token(
    *,
    room: str,
    identity: str,
    locale: str,
    ttl_sec: int = 3600,
    viewer: bool = False,
    can_publish: bool | None = None,
    await_viewer: bool = False,
    name: str = "",
) -> tuple[str, str]:
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    from livekit import api

    if viewer:
        if not identity.startswith("ghost-viewer"):
            identity = "ghost-viewer"
    elif not identity.startswith("ghost-"):
        identity = f"ghost-{identity}"
    meta: dict[str, Any] = {"locale": locale, "ghost": True}
    if viewer:
        meta["ghostViewer"] = True
    if await_viewer:
        # The worker holds the opening until the recording page is up.
        meta["awaitViewer"] = True
    if can_publish is None:
        can_publish = not viewer
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(name or identity)
        .with_metadata(json.dumps(meta, ensure_ascii=False))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=can_publish,
                can_subscribe=True,
            )
        )
        .with_ttl(timedelta(seconds=ttl_sec))
    )
    # Must match the worker's TALKSHOW_AGENT_NAME, or the room gets no agent.
    agent_name = os.environ.get("TALKSHOW_AGENT_NAME", "").strip()
    if agent_name:
        token = token.with_room_config(
            api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=agent_name)])
        )
    return url, token.to_jwt()


def default_room() -> str:
    """A new room per run: a reused room name can keep the previous (agent-less) room
    alive, and a shared name collides with other users' ghosts (DuplicateIdentity)."""
    base = os.environ.get("TALKSHOW_ROOM", "").strip()
    return base or f"talkshow-ghost-{int(time.time())}"


def _decode_ui_event(raw: bytes | str | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        text = raw
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class GhostRoomClient:
    """The guest in the room: follows the show from UI events and speaks on a mic track."""

    def __init__(self, audio_tap: Any | None = None) -> None:
        self.room: Any = None
        self.first_line = asyncio.Event()
        # The worker subscribed to the ghost's mic (it linked this participant).
        self.mic_subscribed = asyncio.Event()
        self._audio_tap = audio_tap
        # Agent lines that started playing (one transcript event per line).
        self.agent_lines = 0
        # Lines on air or queued, by (role, step). A step id is per beat, so the same
        # one comes back in a later beat: a key is a new line again once its line ended.
        self._open: set[tuple[str, str]] = set()
        # Any transcript (the human's own line included) and when the last one came.
        self.transcript_seen = asyncio.Event()
        self.last_transcript_t = 0.0
        self.panel_transcripts = 0  # guest/commentator transcript events seen
        self.speaking = False  # an agent playout (one sentence) is on air
        self.lines_ended = 0  # agent lines the worker reported over (line_end)
        self.idle_since = time.monotonic()
        self._changed = asyncio.Event()
        # Microphone: frames waiting to go out, and when each spoken segment started.
        self._source: Any = None
        self._mic_task: asyncio.Task[None] | None = None
        self._pending: deque[bytes] = deque()
        self._segment_started: asyncio.Future[float] | None = None
        self._segment_done: asyncio.Event | None = None
        self.spoken: list[tuple[Path, float]] = []  # (wav, monotonic start) for the recording
        # What was said on air, in order (None: a line the worker dropped unplayed).
        self._lines: list[TranscriptLine | None] = []
        self._line_at: dict[tuple[str, str], int] = {}  # open agent line → its index
        # Per agent line, by its ordinal (1 = first agent line): where it is in the
        # transcript, when it was announced / ended, and whether it was dropped unplayed.
        self._ordinal_idx: dict[int, int] = {}
        self._ordinal_of: dict[tuple[str, str], int] = {}
        self.line_announced_at: dict[int, float] = {}
        self.line_ended_at: dict[int, float] = {}
        self.line_dropped: set[int] = set()
        self.line_started_at: dict[int, float] = {}  # its audio started (role_active)
        # Agent audio actually on air (role_active … role_idle); ``speaking`` also counts a
        # line announced but not yet playing.
        self.playing = False
        self.playing_since = 0.0
        self.guest_name = ""  # whose floor hand-overs a seed turn must not cut into
        self.close_round_ended_at: float | None = None  # the host's closing line ended
        self.roster_names: set[str] = set()  # panel names from panel_roster

    def known_names(self) -> set[str]:
        """Panel names: the roster plus everyone who has spoken (the guest excluded)."""
        names = set(self.roster_names)
        names.update(line.speaker for line in self.transcript if line.speaker and line.role != "human")
        names.discard(self.guest_name)
        return names

    def agent_line(self, ordinal: int) -> TranscriptLine | None:
        idx = self._ordinal_idx.get(ordinal)
        return None if idx is None else self._lines[idx]

    def transcript_through(self, ordinal: int) -> list[TranscriptLine]:
        """The transcript up to and including agent line ``ordinal``."""
        idx = self._ordinal_idx.get(ordinal)
        if idx is None:
            return self.transcript
        return [line for line in self._lines[: idx + 1] if line is not None]

    @property
    def transcript(self) -> list[TranscriptLine]:
        return [line for line in self._lines if line is not None]

    def record_own_line(self, speaker: str, text: str) -> None:
        """The guest's own turn (the worker's STT of it may differ; keep what was said)."""
        if text.strip():
            self._lines.append(TranscriptLine("human", speaker, text.strip()))

    def _track_transcript(self, ev_type: str, role: str, ev: dict[str, Any]) -> None:
        key = (role, str(ev.get("step") or ""))
        if ev_type == "transcript":
            if not role or role == "human":
                return
            text = str(ev.get("text") or "").strip()
            if not text:
                texts = ev.get("texts")
                if isinstance(texts, dict):
                    text = str(texts.get("en") or next(iter(texts.values()), "")).strip()
            idx = self._line_at.get(key)
            if idx is None:
                ordinal = self.agent_lines + 1  # on_ui_event counts it right after this
                self._ordinal_idx[ordinal] = len(self._lines)
                self._ordinal_of[key] = ordinal
                self.line_announced_at[ordinal] = time.monotonic()
                self._line_at[key] = len(self._lines)
                self._lines.append(TranscriptLine(role, str(ev.get("speaker") or ""), text, key[1]))
            elif text and self._lines[idx] is not None:
                old = self._lines[idx]
                self._lines[idx] = TranscriptLine(old.role, old.speaker, text, old.step)
        elif ev_type == "line_end":
            idx = self._line_at.pop(key, None)
            ordinal = self._ordinal_of.pop(key, None)
            if ordinal is not None:
                self.line_ended_at[ordinal] = time.monotonic()
            if key[1] == "close_round":
                self.close_round_ended_at = time.monotonic()
            if idx is not None and ev.get("played") is False:
                self._lines[idx] = None  # never heard: the UI drops it too
                if ordinal is not None:
                    self.line_dropped.add(ordinal)
        elif ev_type == "interrupt":
            heard = str(ev.get("heard_text") or "").strip()
            if not role:
                return
            for idx in range(len(self._lines) - 1, -1, -1):
                line = self._lines[idx]
                if line is not None and line.role == role:
                    if heard:
                        self._lines[idx] = TranscriptLine(line.role, line.speaker, heard + " —", line.step)
                    break

    # --- following the show -------------------------------------------------

    def on_ui_event(self, ev: dict[str, Any]) -> None:
        ev_type = str(ev.get("type") or "")
        role = str(ev.get("role") or "")
        if ev_type in ("transcript", "line_end", "interrupt"):
            self._track_transcript(ev_type, role, ev)
        if ev_type == "panel_roster":
            for member in ev.get("members") or []:
                if isinstance(member, dict) and str(member.get("name") or "").strip():
                    self.roster_names.add(str(member["name"]).strip())
            return
        if ev_type == "transcript":
            self.transcript_seen.set()
            self.last_transcript_t = time.monotonic()
        if ev_type == "transcript" and role and role != "human":
            key = (role, str(ev.get("step") or ""))
            if key in self._open:
                return
            self._open.add(key)
            self.agent_lines += 1
            # The worker announces a line's transcript before its audio starts: the
            # line is on air from now until its playout goes idle.
            self.speaking = True
            if role in ("guest", "commentator"):
                self.panel_transcripts += 1
            self.first_line.set()
            logger.info(
                "agent line %d %s step=%s %r",
                self.agent_lines, ev.get("speaker") or role, ev.get("step"),
                (ev.get("text") or "")[:80],
            )
        elif ev_type == "line_end":
            key = (role, str(ev.get("step") or ""))
            if key not in self._open:
                return
            self._open.discard(key)
            self.lines_ended += 1
        elif ev_type == "role_active":
            self.speaking = True
            self.playing, self.playing_since = True, time.monotonic()
            # The oldest open line of that role is the one that started playing.
            open_lines = sorted(o for (r, _s), o in self._ordinal_of.items() if r == role)
            for ordinal in open_lines:
                if ordinal not in self.line_started_at:
                    self.line_started_at[ordinal] = self.playing_since
                    break
        elif ev_type == "role_idle":
            self.speaking = False
            self.playing = False
            self.idle_since = time.monotonic()
        else:
            return
        self._changed.set()

    async def _until(self, predicate: Any, deadline: float) -> bool:
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=min(remaining, 0.2))
            except asyncio.TimeoutError:
                pass
        return True

    async def wait_for_turn(
        self,
        turn: ScriptedTurn,
        *,
        lines_before: int,
        on_target: Callable[[], None] | None = None,
    ) -> str:
        """Block until the script says the guest cuts in; returns why it did.

        ``on_target`` runs as soon as the trigger line is announced (its transcript
        comes before its audio): a live turn starts writing and synthesizing then."""
        deadline = time.monotonic() + turn.max_wait_sec
        target = lines_before + (turn.after_lines or 0)
        if not await self._until(lambda: self.agent_lines >= target, deadline):
            return "timeout"
        handed_to = self._floor_handed_to(target) if turn.when == "line_end" else ""
        if handed_to:
            # The trigger line hands the floor to someone else: let them answer first,
            # and cut in after their line instead.
            logger.info("line %d hands the floor to %s — waiting for their line", target, handed_to)
            target = await self._wait_for_line_by(handed_to, after=target, deadline=deadline)
        if on_target is not None:
            on_target()
        if turn.when == "mid_line":
            await asyncio.sleep(min(turn.after_sec, max(0.0, deadline - time.monotonic())))
            return "mid_line"

        def line_over() -> bool:
            # The target line is over (not merely idle between its sentences), and the
            # floor stays quiet for a moment: the next line of the same beat (the host's
            # opening, then its invite) starts right after it, and a guest who waits for
            # the end doesn't talk over that one either.
            if target not in self.line_ended_at and self.lines_ended < target:
                return False
            return not self.speaking and time.monotonic() - self.idle_since >= LINE_END_QUIET_S

        if not await self._until(line_over, deadline):
            return "timeout"
        await asyncio.sleep(turn.after_sec)
        return "line_end"

    def _floor_handed_to(self, ordinal: int) -> str:
        from eval.ghost_session.addressing import addressing

        line = self.agent_line(ordinal)
        if line is None or not self.guest_name:
            return ""
        addr = addressing(line.text, self.guest_name, self.known_names())
        return addr[6:] if addr.startswith("other:") else ""

    async def _wait_for_line_by(
        self, name: str, *, after: int, deadline: float, show_up_sec: float = 20.0
    ) -> int:
        """The ordinal of the first line by ``name`` after line ``after``, once it is
        announced (``after`` itself if they don't speak within ``show_up_sec`` of it)."""

        def theirs() -> int:
            for o in range(after + 1, self.agent_lines + 1):
                line = self.agent_line(o)
                if line is not None and line.speaker.lower() == name.lower():
                    return o
            return 0

        def ready() -> bool:
            if theirs():
                return True
            ended = self.line_ended_at.get(after)
            return ended is not None and time.monotonic() - ended >= show_up_sec

        await self._until(ready, deadline)
        found = theirs()
        if not found:
            logger.info("%s never took the floor — going ahead", name)
        return found or after

    # --- room ---------------------------------------------------------------

    async def connect(self, url: str, token: str) -> None:
        from livekit import rtc

        self.room = rtc.Room()

        def _on_data(*args: Any, **kwargs: Any) -> None:
            packet = args[0] if args else kwargs.get("ev")
            topic = getattr(packet, "topic", None)
            payload = getattr(packet, "data", None)
            if topic is None and len(args) >= 4:
                payload, _participant, _kind, topic = args[:4]
            if topic != UI_TOPIC or payload is None:
                return
            ev = _decode_ui_event(payload)
            if ev is not None:
                self.on_ui_event(ev)

        self.room.on("data_received", _on_data)
        self.room.on("local_track_subscribed", lambda *_: self.mic_subscribed.set())
        if self._audio_tap is not None:
            self._audio_tap.attach(self.room)
        await self.room.connect(url, token)
        if self._audio_tap is not None:
            self._audio_tap.capture_existing(self.room)
        logger.info(
            "ghost joined room=%s identity=%s",
            getattr(self.room, "name", "?"),
            getattr(self.room.local_participant, "identity", "?"),
        )

    async def publish_control(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await self.room.local_participant.publish_data(
            data,
            reliable=True,
            topic=CONTROL_TOPIC,
        )

    async def send_wrap_up(self) -> None:
        """Ask the worker to close the show (host's close_round line next)."""
        await self.publish_control({"type": "wrap_up"})

    async def inject_turn(self, text: str, *, prefix: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"type": "ghost_human_turn", "text": text}
        if prefix:
            payload["prefix"] = prefix
        await self.publish_control(payload)
        logger.info("injected ghost turn chars=%d %r", len(text), text[:80])

    # --- microphone ---------------------------------------------------------

    async def open_mic(self) -> None:
        """Publish a mic track that carries silence between turns, like a real mic
        (the agent's VAD needs the silence after speech to end the turn)."""
        from livekit import rtc

        self._source = rtc.AudioSource(MIC_RATE, 1, queue_size_ms=50)
        track = rtc.LocalAudioTrack.create_audio_track("ghost-mic", self._source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, options)
        self._mic_task = asyncio.create_task(self._mic_loop(), name="ghost:mic")
        logger.info("ghost mic published rate=%d", MIC_RATE)

    async def _mic_loop(self) -> None:
        from livekit import rtc

        samples = MIC_RATE * FRAME_MS // 1000
        silence = bytes(2 * samples)
        while True:
            if self._pending:
                chunk = self._pending.popleft()
                if self._segment_started is not None and not self._segment_started.done():
                    self._segment_started.set_result(time.monotonic())
                if not self._pending and self._segment_done is not None:
                    self._segment_done.set()
            else:
                chunk = silence
            await self._source.capture_frame(rtc.AudioFrame(chunk, MIC_RATE, 1, samples))

    async def say(self, speech: Any, wav: Path, *, on_start: Callable[[], None] | None = None) -> None:
        """Play ``speech`` (voice.Speech) on the mic in real time; ``on_start`` runs when
        its first frame goes out."""
        import numpy as np

        pcm = np.frombuffer(speech.pcm, dtype=np.int16).astype(np.float32)
        if speech.sample_rate != MIC_RATE:
            n = int(round(len(pcm) * MIC_RATE / speech.sample_rate))
            pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm)
        samples = MIC_RATE * FRAME_MS // 1000
        pcm = np.pad(pcm, (0, (-len(pcm)) % samples)).astype(np.int16).tobytes()
        loop = asyncio.get_running_loop()
        self._segment_started, self._segment_done = loop.create_future(), asyncio.Event()
        self._pending.extend(pcm[i : i + 2 * samples] for i in range(0, len(pcm), 2 * samples))
        started = await self._segment_started
        if on_start is not None:
            on_start()
        await self._segment_done.wait()
        self.spoken.append((wav, started))

    async def disconnect(self) -> None:
        if self._mic_task is not None:
            self._mic_task.cancel()
        if self.room is None:
            return
        try:
            await self.room.disconnect()
        except Exception:
            logger.debug("ghost disconnect raised", exc_info=True)


async def _inject_until_heard(
    client: GhostRoomClient,
    text: str,
    *,
    settle_sec: float = 2.0,
    ack_timeout_sec: float = 30.0,
    attempts: int = 8,
    prefix: dict[str, Any] | None = None,
) -> None:
    """Inject, then re-inject until a new transcript shows up (Think & Action: the
    human line itself; hand-raise floor: the host reply).

    Hand-raise floor: the human ``floor_grant`` is emitted while the show beat is still
    finishing; an inject that lands then is dropped by the worker ("panel chain running")
    with no reply on the wire. Extra injects after acceptance are dropped the same way
    (the worker also drops a repeat of its last ghost turn).
    """
    await asyncio.sleep(settle_sec)
    for attempt in range(1, attempts + 1):
        client.transcript_seen.clear()
        await client.inject_turn(text, prefix=prefix)
        try:
            await asyncio.wait_for(client.transcript_seen.wait(), timeout=ack_timeout_sec)
            return
        except asyncio.TimeoutError:
            logger.warning(
                "no transcript %.0fs after inject (attempt %d/%d) — re-injecting",
                ack_timeout_sec,
                attempt,
                attempts,
            )
    raise TimeoutError(f"ghost turn never acknowledged after {attempts} injects")


async def _wait_quiet(
    client: GhostRoomClient,
    *,
    quiet_sec: float,
    max_sec: float,
    stop_after_panel: int = 0,
    panel_base: int = 0,
) -> None:
    """Return once the panel beat is over: ``stop_after_panel`` panelist lines since
    ``panel_base`` (0 = off), or no transcript for ``quiet_sec``, or ``max_sec``.

    Silence alone is a poor end signal: one line can take ~60 s to synthesise and play
    under load, while 60 s after a beat the worker's idle loop starts the next one.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_sec:
        if stop_after_panel and client.panel_transcripts - panel_base >= stop_after_panel:
            await asyncio.sleep(2.0)
            return
        last = max(client.last_transcript_t, t0)
        if time.monotonic() - last >= quiet_sec:
            return
        await asyncio.sleep(1.0)
    logger.warning("panel still talking after %.0fs — moving on", max_sec)


@dataclass
class LiveTurn:
    """A generated turn, ready to say: its words, and audio unless sent as text."""

    text: str
    emotion: str
    speech: tuple[Any, Path] | None
    ready_t: float  # monotonic time it became ready


async def prepare_live_turn(
    client: GhostRoomClient,
    writer: Any,
    script: GhostScript,
    turn: ScriptedTurn,
    *,
    idx: int,
    voice: bool,
) -> LiveTurn:
    """Write the turn from the transcript so far, then synthesize it in the guest's voice."""
    from eval.ghost_session.live import build_messages

    lines = client.transcript
    cut_line = None
    if turn.when == "mid_line" and turn.after_lines and lines and lines[-1].role != "human":
        cut_line, lines = lines[-1], lines[:-1]
    messages = build_messages(
        name=script.name,
        persona=script.persona,
        topic=script.topic,
        transcript=lines,
        goal=turn.goal,
        cut_line=cut_line,
        cut_after_sec=turn.after_sec,
    )
    gen = await writer.write_turn(messages)
    logger.info(
        "turn %d: generated in %.1fs (attempts=%d%s) emotion=%s words=%d %r",
        idx, gen.seconds, gen.attempts, ", FALLBACK" if gen.fallback else "",
        gen.emotion, len(gen.text.split()), gen.text,
    )
    speech = None
    if voice:
        from eval.ghost_session.voice import speech_for_turn

        t0 = time.monotonic()
        try:
            speech = await speech_for_turn(gen.text, emotion=gen.emotion, locale=script.locale)
        except Exception:
            # Better a text turn than no turn: the show goes on without the voice.
            logger.exception("turn %d: TTS failed — sending the text instead", idx)
        else:
            logger.info(
                "turn %d: TTS %.1fs for %.1fs of audio", idx, time.monotonic() - t0, speech[0].duration_s
            )
    return LiveTurn(gen.text, gen.emotion, speech, time.monotonic())


async def prepare_reaction(
    writer: Any, script: GhostScript, messages: list[dict[str, str]], must_answer: bool, *, voice: bool
) -> Any:
    """Reactive mode: decide + write (one LLM call), then synthesize unless silent."""
    from eval.ghost_session.reactive import PreparedReply

    try:
        r = await writer.react(messages, must_answer=must_answer)
    except Exception:
        logger.exception("reactive: LLM failed")
        return PreparedReply("silent", "", "", None, 0.0, 0.0)
    speech = None
    t0 = time.monotonic()
    if r.action != "silent" and voice:
        from eval.ghost_session.voice import speech_for_turn

        try:
            speech = await speech_for_turn(r.text, emotion=r.emotion, locale=script.locale)
        except Exception:
            logger.exception("reactive: TTS failed — sending the text instead")
    return PreparedReply(r.action, r.emotion, r.text, speech, r.seconds, time.monotonic() - t0)


async def prepare_goodbye(
    client: GhostRoomClient, writer: Any, script: GhostScript, *, voice: bool
) -> tuple[str, Any]:
    """The goodbye's words (script ``goodbye:``, else LLM-written) and its audio."""
    from eval.ghost_session.live import (
        GOODBYE_FALLBACK,
        GOODBYE_MAX_WORDS,
        build_goodbye_messages,
        trim_words,
    )

    text, emotion = script.goodbye, "Happiness"
    if not text:
        text = GOODBYE_FALLBACK
        if writer is not None:
            gen = await writer.write_turn(
                build_goodbye_messages(
                    name=script.name, persona=script.persona, topic=script.topic,
                    transcript=client.transcript,
                )
            )
            if not gen.fallback:
                text, emotion = trim_words(gen.text, GOODBYE_MAX_WORDS), gen.emotion
    speech = None
    if voice:
        from eval.ghost_session.voice import speech_for_turn

        try:
            speech = await speech_for_turn(text, emotion=emotion, locale=script.locale)
        except Exception:
            logger.exception("goodbye: TTS failed — sending the text instead")
    return text, speech


def speaks_turns(script: GhostScript, *, text_turns: bool) -> bool:
    """Voice unless asked for text. A replay (prefix) compares runs that differ in one
    word, so it always sends the exact text: no TTS, no STT."""
    return not text_turns and script.prefix is None


async def run_ghost_session(
    *,
    room: str,
    script: GhostScript,
    opening_only: bool = False,
    text_turns: bool = False,
    welcome_timeout_sec: float = 90.0,
    floor_timeout_sec: float = 180.0,
    hold_sec: float = 8.0,
    quiet_sec: float = 90.0,
    stop_after_panel: int = 0,
    beat_panel: int = 0,
    wrap_up: bool = False,
    record: bool = False,
    frontend_url: str = "http://localhost:3000",
    start_frontend: bool = False,
    record_dir: Path | None = None,
    headed: bool = False,
) -> Path | None:
    """Turns with ``after_lines`` cut in when the script says; the others wait for the
    beat to end (``beat_panel`` panel lines before the first turn, then quiet or
    ``beat_panel`` / ``stop_after_panel`` panel lines after each turn).

    Ending: reactive mode (or ``wrap_up``) ends with a goodbye, wrap_up and the host's
    close (ending.py) instead of leaving after ``stop_after_panel`` lines."""
    from eval.ghost_session.record_frontend import (
        FrontendRecorder,
        RoomAudioTap,
        default_record_dir,
        mux_recording,
        viewer_page_url,
    )

    turns = [] if opening_only else script.turns
    voice = speaks_turns(script, text_turns=text_turns)
    speeches: list[tuple[Any, Path] | None] = [None] * len(turns)
    if voice:
        from eval.ghost_session.voice import speech_for_turn

        # Synthesize before joining, so a turn is never late because of TTS (live
        # turns are synthesized during the show, as soon as their words are written).
        for i, turn in enumerate(turns):
            if turn.generate:
                continue
            speeches[i] = await speech_for_turn(
                turn.text, emotion=turn.emotion, audio=turn.audio, locale=script.locale
            )

    reactive = script.reactive and not opening_only
    ending = reactive or ((wrap_up or script.wrap_up) and bool(turns))
    if reactive and stop_after_panel:
        logger.info("reactive mode ends with goodbye + wrap_up: --stop-after-panel ignored")
    writer: Any = None
    if reactive or any(turn.generate for turn in turns) or (ending and not script.goodbye):
        from eval.ghost_session.live import LiveWriter

        writer = LiveWriter()
        logger.info("live turns: LLM at %s", writer.config.base_url)

    url, token = mint_ghost_token(
        room=room,
        identity=script.identity,
        name=script.name,
        locale=script.locale,
        await_viewer=record,
    )
    stamp = int(time.time())
    out_dir = record_dir or default_record_dir()
    recorder: FrontendRecorder | None = None
    audio_tap: RoomAudioTap | None = None
    recorded: Path | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_tap = RoomAudioTap(out_dir / f"ghost-{room}-{stamp}.wav")
        recorder = FrontendRecorder(
            frontend_url=frontend_url,
            record_dir=out_dir,
            start_frontend=start_frontend,
            headed=headed,
        )
        _viewer_url, viewer_token = mint_ghost_token(
            room=room,
            identity="ghost-viewer",
            locale=script.locale,
            viewer=True,
        )
        page_url = viewer_page_url(
            frontend_url=frontend_url,
            livekit_url=url,
            token=viewer_token,
            locale=script.locale,
        )
    else:
        page_url = ""

    client = GhostRoomClient(audio_tap=audio_tap)
    client.guest_name = script.name
    await client.connect(url, token)
    try:
        if voice and (turns or reactive):
            await client.open_mic()
        if recorder is not None:
            if voice and (turns or reactive):
                # The worker listens to the first participant it finds; the viewer
                # joining first would leave the guest's mic unheard all session.
                try:
                    await asyncio.wait_for(client.mic_subscribed.wait(), timeout=welcome_timeout_sec)
                except asyncio.TimeoutError:
                    logger.warning("worker never subscribed to the ghost mic — recording anyway")
            await recorder.start(page_url)
        try:
            await asyncio.wait_for(client.first_line.wait(), timeout=welcome_timeout_sec)
        except asyncio.TimeoutError:
            logger.warning("no agent line after %.0fs — continuing", welcome_timeout_sec)
        show_started_t = time.monotonic()

        async def speak(
            text: str, speech: Any, *, idx: int = 0, on_start: Callable[[], Any] | None = None
        ) -> float | None:
            """Say a turn (mic audio, or text when there is none); returns when the audio
            started (monotonic), None for a text turn."""
            client.record_own_line(script.name, text)
            if speech is not None:
                await client.say(*speech, on_start=on_start)
                return client.spoken[-1][1]
            if on_start is not None:
                started = on_start()
                if asyncio.isfuture(started):
                    await started  # text turns: let it go out before the text does
            await _inject_until_heard(client, text, prefix=script.prefix if idx == 1 else None)
            return None

        lines_before = 0
        last_turn_ordinal = 0
        for idx, (turn, speech) in enumerate(zip(turns, speeches), start=1):
            live_task: asyncio.Task[LiveTurn] | None = None

            def start_live(turn: ScriptedTurn = turn, idx: int = idx) -> None:
                nonlocal live_task
                if turn.generate and live_task is None:
                    logger.info("turn %d: trigger line announced — writing the turn", idx)
                    live_task = asyncio.create_task(
                        prepare_live_turn(client, writer, script, turn, idx=idx, voice=voice),
                        name=f"ghost:live-{idx}",
                    )

            if turn.after_lines is not None:
                why = await client.wait_for_turn(turn, lines_before=lines_before, on_target=start_live)
                logger.info(
                    "turn %d: cutting in (%s, %d agent lines since the last turn)",
                    idx, why, client.agent_lines - lines_before,
                )
            elif idx == 1 and beat_panel and not script.prefix:
                # Let the opening run to a few panel lines before the first turn
                # (a replay prefix replaces the opening anyway).
                await _wait_quiet(
                    client,
                    quiet_sec=quiet_sec,
                    max_sec=floor_timeout_sec,
                    stop_after_panel=beat_panel,
                    panel_base=0,
                )
            text = turn.text
            if turn.generate:
                trigger_t = time.monotonic()
                start_live()  # no-op when already started at the trigger line
                assert live_task is not None
                live = await live_task
                late = live.ready_t - trigger_t
                if late > 0.05:
                    logger.warning("turn %d: ready %.1fs after the cut-in point", idx, late)
                else:
                    logger.info("turn %d: ready %.1fs before the cut-in point", idx, -late)
                text, speech = live.text, live.speech
            panel_base = client.panel_transcripts
            last_turn_ordinal = client.agent_lines
            await speak(text, speech, idx=idx)
            lines_before = client.agent_lines
            last = idx == len(turns)
            if last and reactive:
                pass  # the reactive guest takes over right away
            elif last or turns[idx].after_lines is None:
                await _wait_quiet(
                    client,
                    quiet_sec=quiet_sec,
                    max_sec=floor_timeout_sec * 3,
                    # Last turn: leave after N panel lines. Earlier turns: let the beat run
                    # to N panel lines (its full length) before the next turn.
                    stop_after_panel=stop_after_panel if last else beat_panel,
                    panel_base=panel_base,
                )
        if reactive:
            from eval.ghost_session.live import build_reaction_messages
            from eval.ghost_session.reactive import ReactiveGuest, ReactiveLimits

            guest = ReactiveGuest(
                client,
                name=script.name,
                limits=ReactiveLimits(
                    max_turns=script.max_turns,
                    min_lines_between_turns=script.min_lines_between_turns,
                    max_minutes=script.max_minutes,
                    nudge_after_lines=script.nudge_after_lines,
                ),
                prepare=lambda messages, must: prepare_reaction(writer, script, messages, must, voice=voice),
                speak=speak,
                build_messages=lambda **kw: build_reaction_messages(
                    name=script.name, persona=script.persona, topic=script.topic, **kw
                ),
                turns_done=len(turns),
                show_started_t=show_started_t,
            )
            logger.info(
                "reactive: on (max_turns=%d incl. %d seeds, min_lines_between_turns=%d, max_minutes=%.1f)",
                script.max_turns, len(turns), script.min_lines_between_turns, script.max_minutes,
            )
            await guest.run(last_turn_ordinal=last_turn_ordinal)
        if ending:
            from eval.ghost_session.ending import goodbye_and_wrap_up

            await goodbye_and_wrap_up(
                client,
                prepare=lambda: prepare_goodbye(client, writer, script, voice=voice),
                speak=speak,
            )
        else:
            await asyncio.sleep(max(0.0, hold_sec))
    finally:
        if writer is not None:
            await writer.aclose()
        wav: Path | None = None
        video: Path | None = None
        if recorder is not None:
            # Close the page while the room is still up so wav/video stop together.
            video = await recorder.stop()
        await client.disconnect()
        if audio_tap is not None:
            wav = await audio_tap.finish()
        if recorder is not None and video is not None:
            dest = out_dir / f"ghost-{room}-{stamp}.mp4"
            recorded = mux_recording(
                video=video,
                audio=wav,
                dest=dest,
                audio_t0=getattr(audio_tap, "started_at", None),
                video_t0=recorder.started_at,
                guest_audio=client.spoken,
            )
            logger.info("ghost recording → %s", recorded)
    return recorded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join LiveKit as the human guest and speak scripted turns into the show "
            "(CosyVoice audio on a mic track, or --text-turns)."
        )
    )
    parser.add_argument(
        "--room",
        default=None,
        help="LiveKit room (default: TALKSHOW_ROOM, else a new talkshow-ghost-<epoch> room)",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=_REPO_ROOT / "eval" / "ghost_session" / "scripts" / "smoke.yaml",
        help="YAML with locale + turns",
    )
    parser.add_argument(
        "--opening-only",
        action="store_true",
        help="Join, wait for the first agent line, hold; speak no turns",
    )
    parser.add_argument(
        "--text-turns",
        action="store_true",
        help="Send the turns as text (no audio / STT): the exact words every run. "
        "Replay scripts (prefix_file) always do",
    )
    parser.add_argument("--locale", default="", help="Override script locale")
    parser.add_argument("--name", default="", help="Override the guest's display name (script name:)")
    parser.add_argument(
        "--welcome-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for the first agent line",
    )
    parser.add_argument(
        "--floor-timeout",
        type=float,
        default=180.0,
        help="Longest wait for a beat to end (a turn without after_lines)",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=8.0,
        help="Stay connected after the last beat so TTS can finish",
    )
    parser.add_argument(
        "--quiet-sec",
        type=float,
        default=90.0,
        help="A beat counts as over after this long without a transcript (the room "
        "closes when the ghost leaves — keep it above the slowest line under load)",
    )
    parser.add_argument(
        "--stop-after-panel",
        type=int,
        default=0,
        help="After the LAST scripted turn, leave once this many panelist lines were heard (0 = off). "
        "Ignored in reactive mode, which ends with a goodbye + wrap_up instead",
    )
    parser.add_argument(
        "--beat-panel",
        type=int,
        default=0,
        help="Before a NON-final scripted turn, wait for this many panelist lines (a full beat) — "
        "with --quiet-sec as fallback for beats that end early (0 = silence only)",
    )
    parser.add_argument(
        "--wrap-up",
        action="store_true",
        help="Scripted runs: after the last turn's beat, say goodbye, send wrap_up and leave after "
        "the host's closing line (reactive scripts always do; there --stop-after-panel is ignored)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="If set (on Babel), summarize the newest matching session JSONL",
    )
    parser.add_argument(
        "--summarize",
        type=Path,
        default=None,
        help="Only print a recap of this JSONL (no LiveKit join)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Headless-record talkshow-web (panel + transcript) and mux show audio",
    )
    parser.add_argument(
        "--frontend-url",
        default=os.environ.get("TALKSHOW_FRONTEND_URL", "http://localhost:3000"),
        help="talkshow-web origin for --record",
    )
    parser.add_argument(
        "--start-frontend",
        action="store_true",
        help="If --record and the origin is down, run `pnpm dev` in talkshow-web/",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Where to write ghost-*.mp4 (default logs/ghost-recordings)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the Chromium window while recording (debug)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(_REPO_ROOT / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    if args.summarize is not None:
        from eval.ghost_session.summarize import summarize_session

        print(summarize_session(args.summarize))
        return 0

    if args.room is None:
        args.room = default_room()
    script = load_script(args.script)
    if args.locale:
        script.locale = args.locale
    if args.name:
        script.name = args.name
    logger.info(
        "ghost session room=%s locale=%s turns=%d opening_only=%s text_turns=%s record=%s",
        args.room,
        script.locale,
        0 if args.opening_only else len(script.turns),
        args.opening_only,
        args.text_turns,
        args.record,
    )
    started_at = time.time()
    recorded = asyncio.run(
        run_ghost_session(
            room=args.room,
            script=script,
            opening_only=args.opening_only,
            text_turns=args.text_turns,
            welcome_timeout_sec=args.welcome_timeout,
            floor_timeout_sec=args.floor_timeout,
            hold_sec=args.hold_sec,
            quiet_sec=args.quiet_sec,
            stop_after_panel=args.stop_after_panel,
            beat_panel=args.beat_panel,
            wrap_up=args.wrap_up,
            record=args.record,
            frontend_url=args.frontend_url,
            start_frontend=args.start_frontend,
            record_dir=args.record_dir,
            headed=args.headed,
        )
    )
    if recorded is not None:
        print(f"RECORDING={recorded}")
    log_dir = args.log_dir
    if log_dir is None:
        default_dir = Path(os.environ.get("LOG_DIR", "logs"))
        if default_dir.is_dir():
            log_dir = default_dir
    if log_dir is not None:
        from eval.ghost_session.summarize import find_session_log, summarize_session

        path = find_session_log(log_dir, args.room, newer_than=started_at)
        if path is None:
            logger.warning(
                "no session JSONL for room=%s written during this run in %s — the live "
                "log is on the agent worker; on Babel: python eval/pad_report.py",
                args.room,
                log_dir,
            )
        else:
            print(summarize_session(path))
            from eval.ghost_session.judge_pad import format_pad_judgment, judge_pad_events
            from eval.log_parse import load_events

            print(format_pad_judgment(judge_pad_events(load_events(path))))
    else:
        logger.info(
            "Session JSONL is on the agent worker (LOG_DIR, default logs/). "
            "On Babel: python -m eval.ghost_session --summarize logs/session-<id>.jsonl"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
