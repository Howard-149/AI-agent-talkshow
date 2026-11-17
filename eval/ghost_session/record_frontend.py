"""Headless talkshow-web capture + mux with LiveKit audio."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

logger = logging.getLogger("ghost_session.record")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def viewer_page_url(
    *,
    frontend_url: str,
    livekit_url: str,
    token: str,
    locale: str,
) -> str:
    base = frontend_url.rstrip("/") + "/custom/"
    query = urlencode(
        {
            "liveKitUrl": livekit_url,
            "token": token,
            "locale": locale,
            "record": "1",
        }
    )
    return f"{base}?{query}"


def frontend_reachable(frontend_url: str, *, timeout_sec: float = 2.0) -> bool:
    try:
        with urlopen(frontend_url, timeout=timeout_sec) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except Exception:
        return False


async def wait_frontend(frontend_url: str, *, timeout_sec: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if frontend_reachable(frontend_url):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(f"talkshow-web not reachable at {frontend_url}")


async def start_frontend(frontend_url: str) -> asyncio.subprocess.Process:
    web = _REPO_ROOT / "talkshow-web"
    if not (web / "package.json").is_file():
        raise FileNotFoundError(f"talkshow-web missing at {web}")
    logger.info("starting talkshow-web via pnpm dev")
    proc = await asyncio.create_subprocess_exec(
        "pnpm",
        "dev",
        cwd=str(web),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await wait_frontend(frontend_url)
    except Exception:
        proc.terminate()
        raise
    return proc


def audio_mux_shift_s(*, audio_t0: float | None, video_t0: float | None) -> float:
    """Seconds to shift wav onto Playwright's clock (positive = delay audio)."""
    if audio_t0 is None or video_t0 is None:
        return 0.0
    return float(audio_t0) - float(video_t0)


def mux_recording(
    *,
    video: Path,
    audio: Path | None,
    dest: Path,
    audio_t0: float | None = None,
    video_t0: float | None = None,
    guest_audio: list[tuple[Path, float]] | None = None,
) -> Path:
    """``guest_audio``: the ghost's own spoken wavs with their monotonic start times,
    mixed over the show audio (the tap only hears the room, not the ghost's mic)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        fallback = dest.with_suffix(".webm")
        shutil.copy2(video, fallback)
        logger.warning("ffmpeg not found — saved video-only %s", fallback)
        if audio and audio.is_file():
            logger.warning("audio sidecar %s (install ffmpeg to mux)", audio)
        return fallback
    cmd = [ffmpeg, "-y"]
    if audio is not None and audio.is_file() and audio.stat().st_size > 44:
        shift = audio_mux_shift_s(audio_t0=audio_t0, video_t0=video_t0)
        cmd.extend(["-i", str(video)])
        if shift > 0.02:
            # Wav t=0 is later than video t=0 — delay audio.
            cmd.extend(["-itsoffset", f"{shift:.3f}", "-i", str(audio)])
        elif shift < -0.02:
            # Wav started first (ghost joined before Chromium) — drop the lead.
            cmd.extend(["-ss", f"{-shift:.3f}", "-i", str(audio)])
        else:
            cmd.extend(["-i", str(audio)])
        if abs(shift) > 0.02:
            logger.info("mux clock shift audio-video=%.3fs", shift)
        segments = [(p, t) for p, t in (guest_audio or []) if p.is_file()]
        for path, _ in segments:
            cmd.extend(["-i", str(path)])
        audio_map = "1:a:0"
        if segments:
            cmd.extend(["-filter_complex", guest_mix_filter(segments, video_t0=video_t0)])
            audio_map = "[aout]"
        cmd.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                audio_map,
                "-shortest",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ]
        )
    else:
        logger.warning(
            "no LiveKit wav to mux (missing or empty %s) — writing video-only mp4",
            audio,
        )
        cmd.extend(
            ["-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"]
        )
    cmd.extend(["-movflags", "+faststart", str(dest)])
    logger.info("mux %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return dest


def guest_mix_filter(segments: list[tuple[Path, float]], *, video_t0: float | None) -> str:
    """ffmpeg filter: show audio (input 1) plus each guest wav (inputs 2…) delayed to
    when it was spoken, on the video's clock."""
    t0 = video_t0 if video_t0 is not None else min(t for _, t in segments)
    parts, labels = [], ["[1:a]"]
    for i, (_, started) in enumerate(segments):
        ms = max(0, int(round((started - t0) * 1000)))
        parts.append(f"[{i + 2}:a]adelay={ms}:all=1[g{i}]")
        labels.append(f"[g{i}]")
    parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=first[aout]")
    return ";".join(parts)


@dataclass
class FrontendRecording:
    video_path: Path | None
    _browser: Any
    _context: Any
    _page: Any
    _frontend_proc: asyncio.subprocess.Process | None
    started_at: float | None = None


class FrontendRecorder:
    def __init__(
        self,
        *,
        frontend_url: str,
        record_dir: Path,
        start_frontend: bool = False,
        headed: bool = False,
        width: int = 1600,
        height: int = 900,
    ) -> None:
        self.frontend_url = frontend_url.rstrip("/")
        self.record_dir = record_dir
        self.start_frontend = start_frontend
        self.headed = headed
        self.width = width
        self.height = height
        self._playwright: Any = None
        self._recording: FrontendRecording | None = None
        self.started_at: float | None = None

    async def start(self, page_url: str) -> FrontendRecording:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for --record. "
                "pip install playwright && python -m playwright install chromium"
            ) from exc

        frontend_proc = None
        if not frontend_reachable(self.frontend_url):
            if not self.start_frontend:
                raise RuntimeError(
                    f"talkshow-web not running at {self.frontend_url}. "
                    "Start `pnpm dev` in talkshow-web/ or pass --start-frontend."
                )
            frontend_proc = await start_frontend(self.frontend_url)
        else:
            logger.info("talkshow-web reachable at %s", self.frontend_url)

        self.record_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = self.record_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        self._playwright = await async_playwright().start()
        try:
            browser = await self._playwright.chromium.launch(
                headless=not self.headed,
                args=["--autoplay-policy=no-user-gesture-required"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Playwright Chromium is missing for this Python package. "
                "From the same venv: python -m playwright install chromium"
            ) from exc
        context = await browser.new_context(
            viewport={"width": self.width, "height": self.height},
            record_video_dir=str(raw_dir),
            record_video_size={"width": self.width, "height": self.height},
        )
        page = await context.new_page()
        video_t0 = time.monotonic()
        logger.info("opening viewer %s", page_url.split("token=")[0] + "token=…")
        await page.goto(page_url, wait_until="domcontentloaded")
        try:
            await page.locator('[data-talkshow-ready="true"]').wait_for(timeout=120_000)
            logger.info("talkshow stage ready")
        except Exception:
            logger.warning("stage ready selector timed out — recording anyway")
        rec = FrontendRecording(
            video_path=None,
            _browser=browser,
            _context=context,
            _page=page,
            _frontend_proc=frontend_proc,
            started_at=video_t0,
        )
        self._recording = rec
        self.started_at = video_t0
        return rec

    async def stop(self) -> Path | None:
        rec = self._recording
        self._recording = None
        if rec is None:
            return None
        video_path: Path | None = None
        try:
            if rec._page is not None:
                handle = rec._page.video
                await rec._context.close()
                if handle is not None:
                    raw = await handle.path()
                    video_path = Path(raw)
            else:
                await rec._context.close()
        except Exception:
            logger.exception("closing playwright context failed")
        try:
            await rec._browser.close()
        except Exception:
            logger.debug("browser close skipped", exc_info=True)
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        if rec._frontend_proc is not None and rec._frontend_proc.returncode is None:
            rec._frontend_proc.terminate()
        rec.video_path = video_path
        return video_path


def track_is_audio(track: Any) -> bool:
    """LiveKit protobuf TrackKind.KIND_AUDIO is int 1; str(kind) is '1', not 'audio'."""
    kind = getattr(track, "kind", None)
    if kind is None:
        return False
    try:
        from livekit import rtc

        return int(kind) == int(rtc.TrackKind.KIND_AUDIO)
    except Exception:
        name = str(getattr(kind, "name", kind)).lower()
        return "audio" in name or name == "1"


class RoomAudioTap:
    """Record the show's audio: every remote audio track (the speaker's line, and the
    listeners' backchannels and a cut speaker's trailing words, which play on their own
    tracks at the same time), mixed into one WAV. Recording only the first track left
    every overlap out of the video."""

    RATE = 48000
    # A gap this long in a track's frames is silence the track did not send: keep the
    # track on the wall clock by padding it.
    _GAP_S = 0.25

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tracks: dict[str, tuple[asyncio.Task[None], Path]] = {}
        self._starts: dict[str, float] = {}
        self.started_at: float | None = None

    @property
    def _started(self) -> bool:
        return bool(self._tracks)

    def _maybe_start(self, track: Any) -> None:
        if track is None or not track_is_audio(track):
            if track is not None:
                logger.debug("audio tap skip kind=%r", getattr(track, "kind", None))
            return
        key = str(getattr(track, "sid", "") or id(track))
        if key in self._tracks:
            return
        part = self.path.with_name(f"{self.path.stem}.track{len(self._tracks)}.wav")
        logger.info("audio tap subscribed kind=%r track=%s", getattr(track, "kind", None), key)
        self._tracks[key] = (asyncio.create_task(self._consume(track, key, part)), part)

    def attach(self, room: Any) -> None:
        def _on_track(*args: Any, **kwargs: Any) -> None:
            track = args[0] if args else kwargs.get("track")
            self._maybe_start(track)

        room.on("track_subscribed", _on_track)
        self.capture_existing(room)

    def capture_existing(self, room: Any) -> None:
        remotes = getattr(room, "remote_participants", None)
        if remotes is None:
            return
        values = remotes.values() if hasattr(remotes, "values") else remotes
        for participant in values:
            pubs = getattr(participant, "track_publications", None) or {}
            items = pubs.values() if hasattr(pubs, "values") else pubs
            for pub in items:
                self._maybe_start(getattr(pub, "track", None))

    async def _consume(self, track: Any, key: str, part: Path) -> None:
        try:
            from livekit import rtc
        except ImportError:
            logger.warning("livekit rtc missing — cannot tap room audio")
            return
        import wave

        stream = rtc.AudioStream(track, sample_rate=self.RATE, num_channels=1)
        part.parent.mkdir(parents=True, exist_ok=True)
        wf: wave.Wave_write | None = None
        written = 0  # samples
        try:
            async for event in stream:
                data = bytes(getattr(event.frame, "data", b"") or b"")
                now = time.monotonic()
                if wf is None:
                    self._starts[key] = now
                    if self.started_at is None or now < self.started_at:
                        self.started_at = now
                    wf = wave.open(str(part), "wb")
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.RATE)
                    logger.info("audio tap → %s", part)
                behind = int((now - self._starts[key]) * self.RATE) - written
                if behind > self._GAP_S * self.RATE:
                    wf.writeframes(b"\x00\x00" * behind)
                    written += behind
                if data:
                    wf.writeframes(data)
                    written += len(data) // 2
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("audio tap failed track=%s", key)
        finally:
            if wf is not None:
                wf.close()

    def _mix(self, parts: list[tuple[float, Path]]) -> None:
        import wave

        import numpy as np

        t0 = min(start for start, _ in parts)
        tracks = []
        for start, part in parts:
            with wave.open(str(part)) as wf:
                samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            tracks.append((int(round((start - t0) * self.RATE)), samples))
        total = max(offset + len(samples) for offset, samples in tracks)
        mix = np.zeros(total, dtype=np.int32)
        for offset, samples in tracks:
            mix[offset : offset + len(samples)] += samples
        with wave.open(str(self.path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(np.clip(mix, -32768, 32767).astype(np.int16).tobytes())
        self.started_at = t0

    async def finish(self) -> Path | None:
        for task, _ in self._tracks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        parts = [
            (self._starts[key], part)
            for key, (_, part) in self._tracks.items()
            if key in self._starts and part.is_file() and part.stat().st_size > 44
        ]
        if not parts:
            logger.warning("audio tap produced no wav (tracks=%d path=%s)", len(self._tracks), self.path)
            return None
        self._mix(parts)
        for _, part in parts:
            part.unlink(missing_ok=True)
        logger.info("audio tap mixed %d track(s) → %s", len(parts), self.path)
        return self.path


def default_record_dir() -> Path:
    raw = os.environ.get("GHOST_RECORD_DIR", "").strip()
    if raw:
        return Path(raw)
    return _REPO_ROOT / "logs" / "ghost-recordings"
