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
) -> Path:
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
        cmd.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
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
    """Write the first subscribed remote audio track to a WAV (show TTS)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._wave: Any = None
        self._started = False
        self._task: asyncio.Task[None] | None = None
        self.started_at: float | None = None

    def _maybe_start(self, track: Any) -> None:
        if self._started or track is None:
            return
        if not track_is_audio(track):
            logger.debug("audio tap skip kind=%r", getattr(track, "kind", None))
            return
        self._started = True
        logger.info("audio tap subscribed kind=%r", getattr(track, "kind", None))
        self._task = asyncio.create_task(self._consume(track))

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

    async def _consume(self, track: Any) -> None:
        try:
            from livekit import rtc
        except ImportError:
            logger.warning("livekit rtc missing — cannot tap room audio")
            return
        import wave

        stream = rtc.AudioStream(track)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        wf: wave.Wave_write | None = None
        try:
            async for event in stream:
                frame = event.frame
                rate = int(getattr(frame, "sample_rate", 48000) or 48000)
                ch = int(getattr(frame, "num_channels", 1) or 1)
                data = getattr(frame, "data", b"")
                if wf is None:
                    self.started_at = time.monotonic()
                    wf = wave.open(str(self.path), "wb")
                    wf.setnchannels(ch)
                    wf.setsampwidth(2)
                    wf.setframerate(rate)
                    logger.info("audio tap → %s rate=%d ch=%d", self.path, rate, ch)
                if data:
                    wf.writeframes(bytes(data))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("audio tap failed")
        finally:
            if wf is not None:
                wf.close()

    async def finish(self) -> Path | None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self.path.is_file() and self.path.stat().st_size > 44:
            return self.path
        logger.warning("audio tap produced no wav (started=%s path=%s)", self._started, self.path)
        return None


def default_record_dir() -> Path:
    raw = os.environ.get("GHOST_RECORD_DIR", "").strip()
    if raw:
        return Path(raw)
    return _REPO_ROOT / "logs" / "ghost-recordings"
