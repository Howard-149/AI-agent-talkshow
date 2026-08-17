"""Publish DyStream RGBA/MP4 frames as LiveKit avatar video tracks."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import av
from livekit import rtc

logger = logging.getLogger(__name__)

TRACK_NAME = "talkshow-avatar"

# Sentinel placed on live frame queues when the producer finishes.
FRAME_EOS: object = object()


def track_name_for_locale(locale: str | None) -> str:
    from agent.locale.viewer_locales import avatar_track_name

    return avatar_track_name(locale)


def lk_video_enabled() -> bool:
    if os.environ.get("TALKSHOW_AVATAR_ENABLED", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    raw = os.environ.get("TALKSHOW_AVATAR_LK_VIDEO", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def clip_duration_sec(path: Path) -> float | None:
    """Duration from speech MP4 (prefers audio stream)."""
    if not path.is_file():
        return None
    try:
        with av.open(str(path)) as container:
            for stream in container.streams.audio:
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
            for stream in container.streams.video:
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
            if container.duration:
                return float(container.duration) / 1_000_000
    except Exception as exc:
        logger.warning("clip_duration_sec failed path=%s: %s", path, exc)
    return None


def _decode_mp4_frames(path: Path, width: int, height: int) -> tuple[list[bytes], float]:
    frames: list[bytes] = []
    fps = 25.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate:
            fps = max(float(stream.average_rate), 1.0)
        for frame in container.decode(video=0):
            rgba = frame.reformat(width=width, height=height, format="rgba").to_ndarray()
            frames.append(rgba.tobytes())
    return frames, fps


class AvatarVideoPublisher:
    """Publish DyStream frames on a muted LiveKit video track (audio stays Piper)."""

    def __init__(self, *, track_name: str = TRACK_NAME) -> None:
        self._room: rtc.Room | None = None
        self._source: rtc.VideoSource | None = None
        self._width = int(os.environ.get("TALKSHOW_AVATAR_VIDEO_WIDTH", "512"))
        self._height = int(os.environ.get("TALKSHOW_AVATAR_VIDEO_HEIGHT", "512"))
        self._published = False
        self._lock = asyncio.Lock()
        self._play_task: asyncio.Task[None] | None = None
        # Must stay monotonic across clips — resetting to 0 each line can drop early frames.
        self._ts_us = 0
        self.track_name = track_name

    @property
    def enabled(self) -> bool:
        return lk_video_enabled()

    async def bind_room(self, room: rtc.Room) -> None:
        self._room = room
        if self.enabled:
            await self._ensure_published()

    async def _ensure_published(self) -> None:
        if self._published or self._room is None:
            return
        lp = self._room.local_participant
        if lp is None:
            return
        self._source = rtc.VideoSource(self._width, self._height)
        track = rtc.LocalVideoTrack.create_video_track(self.track_name, self._source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,
            video_encoding=rtc.VideoEncoding(
                max_framerate=30,
                max_bitrate=int(os.environ.get("TALKSHOW_AVATAR_VIDEO_BITRATE", "2500000")),
            ),
        )
        await lp.publish_track(track, options)
        self._published = True
        logger.info(
            "avatar video track published name=%s %dx%d",
            self.track_name,
            self._width,
            self._height,
        )

    def _next_ts_us(self, fps: float) -> int:
        step = max(1, int(1_000_000 / max(fps, 1.0)))
        self._ts_us += step
        return self._ts_us

    def _capture_rgba(self, data: bytes, *, fps: float) -> None:
        if self._source is None:
            return
        w, h = self._width, self._height
        rgba = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, data)
        try:
            frame = rgba.convert(rtc.VideoBufferType.I420)
        except Exception:
            frame = rgba
        self._source.capture_frame(frame, timestamp_us=self._next_ts_us(fps))

    async def prepare_mp4(self, path: Path) -> tuple[list[bytes], float] | None:
        """Decode MP4 off the hot path so audio/video can start together."""
        if not path.is_file():
            return None
        try:
            frame_bytes, fps = await asyncio.to_thread(
                _decode_mp4_frames, path, self._width, self._height
            )
        except Exception as exc:
            logger.warning("avatar lk video decode failed path=%s: %s", path, exc)
            return None
        if not frame_bytes:
            return None
        return frame_bytes, fps

    async def stream_frames(
        self,
        frame_bytes: list[bytes],
        fps: float,
        *,
        target_duration_sec: float | None = None,
    ) -> None:
        """Push pre-decoded frames using wall-clock pacing."""
        await self._ensure_published()
        if self._source is None or not frame_bytes:
            return

        if target_duration_sec and target_duration_sec > 0:
            fps = max(len(frame_bytes) / target_duration_sec, 1.0)

        frame_interval = 1.0 / fps
        logger.info(
            "avatar lk video stream start frames=%d fps=%.2f target_s=%s",
            len(frame_bytes),
            fps,
            f"{target_duration_sec:.3f}" if target_duration_sec else "—",
        )

        start = asyncio.get_running_loop().time()
        for i, data in enumerate(frame_bytes):
            self._capture_rgba(data, fps=fps)
            if i == 0:
                logger.info("avatar lk video first frame sent ts_us=%d", self._ts_us)
            target = start + (i + 1) * frame_interval
            delay = target - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)

        logger.info("avatar lk video stream done frames=%d", len(frame_bytes))

    async def stream_frames_live(
        self,
        frame_queue: asyncio.Queue,
        fps: float,
        *,
        target_duration_sec: float | None = None,
        eos_sentinel: object | None = None,
        on_first_frame: asyncio.Event | None = None,
    ) -> None:
        """Play prerolled + live RGBA at fps. Signals on_first_frame after frame 0 capture."""
        await self._ensure_published()
        if self._source is None:
            if on_first_frame is not None:
                on_first_frame.set()
            return

        sentinel = eos_sentinel if eos_sentinel is not None else FRAME_EOS
        frame_interval = 1.0 / max(fps, 1.0)
        logger.info(
            "avatar lk video live start fps=%.2f target_s=%s",
            fps,
            f"{target_duration_sec:.3f}" if target_duration_sec else "—",
        )

        start = asyncio.get_running_loop().time()
        deadline = (
            start + target_duration_sec
            if target_duration_sec and target_duration_sec > 0
            else None
        )
        i = 0
        last: bytes | None = None
        underruns = 0
        eos = False

        while True:
            now = asyncio.get_running_loop().time()
            if deadline is not None and now >= deadline and i > 0:
                break

            data: bytes | None = None
            if not eos:
                try:
                    if i == 0:
                        item = await frame_queue.get()
                    else:
                        timeout = frame_interval
                        if deadline is not None:
                            timeout = max(0.0, min(frame_interval, deadline - now))
                        item = await asyncio.wait_for(
                            frame_queue.get(), timeout=timeout
                        )
                except asyncio.TimeoutError:
                    underruns += 1
                    data = last
                else:
                    if item is sentinel:
                        eos = True
                        data = last
                        if data is None:
                            break
                    else:
                        last = bytes(item)
                        data = last
            else:
                data = last
                if data is None:
                    break
                if deadline is None:
                    break

            if data is None:
                continue

            self._capture_rgba(data, fps=fps)
            if i == 0:
                logger.info("avatar lk video live first frame ts_us=%d", self._ts_us)
                if on_first_frame is not None:
                    on_first_frame.set()
            i += 1
            target = start + i * frame_interval
            delay = target - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)

        if on_first_frame is not None and not on_first_frame.is_set():
            on_first_frame.set()

        logger.info(
            "avatar lk video live done frames=%d underruns=%d eos=%s",
            i,
            underruns,
            eos,
        )


_publishers: dict[str, AvatarVideoPublisher] = {}


def get_avatar_video_publisher(
    locale: str | None = None,
) -> AvatarVideoPublisher | None:
    if not lk_video_enabled():
        return None
    name = track_name_for_locale(locale)
    pub = _publishers.get(name)
    if pub is None:
        pub = AvatarVideoPublisher(track_name=name)
        _publishers[name] = pub
    return pub


def init_avatar_video_publisher(*, room: rtc.Room) -> AvatarVideoPublisher | None:
    pub = get_avatar_video_publisher()
    if pub is None:
        return None

    async def _bind() -> None:
        try:
            await pub.bind_room(room)
        except Exception as exc:
            logger.warning("avatar video bind failed: %s", exc)

    asyncio.create_task(_bind())
    return pub
