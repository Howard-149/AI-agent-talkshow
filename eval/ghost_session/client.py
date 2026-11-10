"""Join a LiveKit room as the human guest and speak scripted turns into the show.

Think & Action floor: the human never waits for a floor grant. A scripted turn can say
when to cut in (after how many agent lines, at a line's end or partway into it); a turn
that doesn't waits for the panel to finish a beat (``--beat-panel`` / ``--quiet-sec``).

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
from typing import Any

from dotenv import load_dotenv

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


@dataclass
class GhostScript:
    locale: str = "en"
    identity: str = "ghost-guest"
    turns: list[ScriptedTurn] = field(default_factory=list)
    # Replay mode (worker needs TALKSHOW_GHOST_REPLAY=1): sent with the first turn;
    # the worker swaps in this history / mood state before handling the turn.
    prefix: dict[str, Any] | None = None


def _turn(item: Any) -> ScriptedTurn | None:
    if isinstance(item, str):
        return ScriptedTurn(text=item.strip()) if item.strip() else None
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or "").strip()
    if not text:
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
        turns=turns,
        prefix=prefix,
    )


def mint_ghost_token(
    *,
    room: str,
    identity: str,
    locale: str,
    ttl_sec: int = 3600,
    viewer: bool = False,
    can_publish: bool | None = None,
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
    if can_publish is None:
        can_publish = not viewer
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
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
        self._audio_tap = audio_tap
        # Agent lines that started playing (one transcript event per line).
        self.agent_lines = 0
        self._seen: set[tuple[str, str]] = set()
        # Any transcript (the human's own line included) and when the last one came.
        self.transcript_seen = asyncio.Event()
        self.last_transcript_t = 0.0
        self.panel_transcripts = 0  # guest/commentator transcript events seen
        self.speaking = False  # an agent playout (one sentence) is on air
        self.idle_since = time.monotonic()
        self._changed = asyncio.Event()
        # Microphone: frames waiting to go out, and when each spoken segment started.
        self._source: Any = None
        self._mic_task: asyncio.Task[None] | None = None
        self._pending: deque[bytes] = deque()
        self._segment_started: asyncio.Future[float] | None = None
        self._segment_done: asyncio.Event | None = None
        self.spoken: list[tuple[Path, float]] = []  # (wav, monotonic start) for the recording

    # --- following the show -------------------------------------------------

    def on_ui_event(self, ev: dict[str, Any]) -> None:
        ev_type = str(ev.get("type") or "")
        role = str(ev.get("role") or "")
        if ev_type == "transcript":
            self.transcript_seen.set()
            self.last_transcript_t = time.monotonic()
        if ev_type == "transcript" and role and role != "human":
            key = (role, str(ev.get("step") or ""))
            if key in self._seen:
                return
            self._seen.add(key)
            self.agent_lines += 1
            if role in ("guest", "commentator"):
                self.panel_transcripts += 1
            self.first_line.set()
            logger.info(
                "agent line %d %s step=%s %r",
                self.agent_lines, ev.get("speaker") or role, ev.get("step"),
                (ev.get("text") or "")[:80],
            )
        elif ev_type == "role_active":
            self.speaking = True
        elif ev_type == "role_idle":
            self.speaking = False
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

    async def wait_for_turn(self, turn: ScriptedTurn, *, lines_before: int) -> str:
        """Block until the script says the guest cuts in; returns why it did."""
        deadline = time.monotonic() + turn.max_wait_sec
        target = lines_before + (turn.after_lines or 0)
        if not await self._until(lambda: self.agent_lines >= target, deadline):
            return "timeout"
        if turn.when == "mid_line":
            await asyncio.sleep(min(turn.after_sec, max(0.0, deadline - time.monotonic())))
            return "mid_line"

        def line_over() -> bool:
            # A later line starting also means the target line is over.
            return self.agent_lines > target or (
                not self.speaking and time.monotonic() - self.idle_since >= LINE_END_QUIET_S
            )

        if not await self._until(line_over, deadline):
            return "timeout"
        await asyncio.sleep(turn.after_sec)
        return "line_end"

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

    async def say(self, speech: Any, wav: Path) -> None:
        """Play ``speech`` (voice.Speech) on the mic in real time."""
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
    record: bool = False,
    frontend_url: str = "http://localhost:3000",
    start_frontend: bool = False,
    record_dir: Path | None = None,
    headed: bool = False,
) -> Path | None:
    """Turns with ``after_lines`` cut in when the script says; the others wait for the
    beat to end (``beat_panel`` panel lines before the first turn, then quiet or
    ``beat_panel`` / ``stop_after_panel`` panel lines after each turn)."""
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

        # Synthesize before joining, so a turn is never late because of TTS.
        for i, turn in enumerate(turns):
            speeches[i] = await speech_for_turn(
                turn.text, emotion=turn.emotion, audio=turn.audio, locale=script.locale
            )

    url, token = mint_ghost_token(
        room=room,
        identity=script.identity,
        locale=script.locale,
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
    await client.connect(url, token)
    try:
        if voice and turns:
            await client.open_mic()
        if recorder is not None:
            await recorder.start(page_url)
        try:
            await asyncio.wait_for(client.first_line.wait(), timeout=welcome_timeout_sec)
        except asyncio.TimeoutError:
            logger.warning("no agent line after %.0fs — continuing", welcome_timeout_sec)

        lines_before = 0
        for idx, (turn, speech) in enumerate(zip(turns, speeches), start=1):
            if turn.after_lines is not None:
                why = await client.wait_for_turn(turn, lines_before=lines_before)
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
            panel_base = client.panel_transcripts
            if speech is not None:
                await client.say(*speech)
            else:
                await _inject_until_heard(
                    client,
                    turn.text,
                    prefix=script.prefix if idx == 1 else None,
                )
            lines_before = client.agent_lines
            last = idx == len(turns)
            if last or turns[idx].after_lines is None:
                await _wait_quiet(
                    client,
                    quiet_sec=quiet_sec,
                    max_sec=floor_timeout_sec * 3,
                    # Last turn: leave after N panel lines. Earlier turns: let the beat run
                    # to N panel lines (its full length) before the next turn.
                    stop_after_panel=stop_after_panel if last else beat_panel,
                    panel_base=panel_base,
                )
        await asyncio.sleep(max(0.0, hold_sec))
    finally:
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
        help="After the LAST scripted turn, leave once this many panelist lines were heard (0 = off)",
    )
    parser.add_argument(
        "--beat-panel",
        type=int,
        default=0,
        help="Before a NON-final scripted turn, wait for this many panelist lines (a full beat) — "
        "with --quiet-sec as fallback for beats that end early (0 = silence only)",
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
