"""Secondary locale audio track (named talkshow-audio-<locale>)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from livekit import rtc

from agent.locale.viewer_locales import audio_track_name

logger = logging.getLogger(__name__)

# ~20 ms frames at 22050 Hz
_FRAME_MS = 20


class LocaleAudioPublisher:
    """Publish PCM for a non-primary locale on a dedicated LiveKit audio track."""

    def __init__(self, locale: str, *, track_name: str | None = None) -> None:
        self.locale = locale
        self.track_name = track_name or audio_track_name(locale)
        self._room: rtc.Room | None = None
        self._source: rtc.AudioSource | None = None
        self._published = False
        self._sample_rate = 22050
        self._lock = asyncio.Lock()

    async def bind_room(self, room: rtc.Room, *, sample_rate: int = 22050) -> None:
        self._room = room
        self._sample_rate = sample_rate
        await self._ensure_published()

    async def _ensure_published(self) -> None:
        if self._published or self._room is None:
            return
        lp = self._room.local_participant
        if lp is None:
            return
        self._source = rtc.AudioSource(self._sample_rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track(self.track_name, self._source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE,
        )
        await lp.publish_track(track, options)
        self._published = True
        logger.info(
            "locale audio track published name=%s rate=%d",
            self.track_name,
            self._sample_rate,
        )

    async def play_pcm(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        on_started: Callable[[float], None] | None = None,
    ) -> None:
        """Stream int16 mono PCM at roughly real-time pace."""
        async with self._lock:
            if sample_rate > 0:
                self._sample_rate = sample_rate
            await self._ensure_published()
            if self._source is None or not pcm:
                return

            samples_per_frame = max(1, int(self._sample_rate * _FRAME_MS / 1000))
            bytes_per_frame = samples_per_frame * 2
            frame_interval = samples_per_frame / float(self._sample_rate)
            loop = asyncio.get_running_loop()
            start = loop.time()
            i = 0
            offset = 0
            while offset < len(pcm):
                chunk = pcm[offset : offset + bytes_per_frame]
                offset += bytes_per_frame
                if len(chunk) < bytes_per_frame:
                    chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                    samples_per_channel=samples_per_frame,
                )
                await self._source.capture_frame(frame)
                if i == 0 and on_started is not None:
                    on_started(time.monotonic())
                i += 1
                target = start + i * frame_interval
                delay = target - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)


_secondary: dict[str, LocaleAudioPublisher] = {}


def get_locale_audio_publisher(locale: str) -> LocaleAudioPublisher:
    if locale not in _secondary:
        _secondary[locale] = LocaleAudioPublisher(locale)
    return _secondary[locale]


_backchannel: dict[tuple[str, str], LocaleAudioPublisher] = {}


def get_backchannel_audio_publisher(role: str, locale: str) -> LocaleAudioPublisher:
    """One independent track per listener role and locale.

    Backchannels overlap the speaker and are subscribed by matching-locale viewers.
    """
    key = (role, locale)
    if key not in _backchannel:
        _backchannel[key] = LocaleAudioPublisher(
            locale, track_name=f"talkshow-backchannel-{locale}-{role}"
        )
    return _backchannel[key]
