from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from livekit import rtc
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from agent.config import LocaleTTSConfig

logger = logging.getLogger(__name__)


class PiperTTS(tts.TTS):
    def __init__(self, cfg: LocaleTTSConfig) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=cfg.sample_rate,
            num_channels=1,
        )
        self._model_path = Path(cfg.model_path)
        self._piper_bin = shutil.which("piper")

    @property
    def model(self) -> str:
        return self._model_path.name

    @property
    def provider(self) -> str:
        return "piper"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _PiperChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _PiperChunkedStream(tts.ChunkedStream):
    async def _run(self) -> None:
        tts_impl: PiperTTS = self._tts  # type: ignore[assignment]
        text = self._input_text.strip()
        if not text:
            return

        wav_bytes = await asyncio.to_thread(_synthesize_wav, tts_impl, text)
        frame = _wav_to_frame(wav_bytes, tts_impl.sample_rate)

        await self._event_ch.send(
            tts.SynthesizedAudio(
                frame=frame,
                request_id=str(uuid.uuid4()),
                is_final=True,
            )
        )


def _synthesize_wav(pipert: PiperTTS, text: str) -> bytes:
    if not pipert._model_path.is_file():
        raise FileNotFoundError(f"Piper model not found: {pipert._model_path}")

    # Prefer Python piper-tts if installed
    try:
        from piper import PiperVoice  # type: ignore[import-untyped]

        voice = PiperVoice.load(str(pipert._model_path))
        chunks = list(voice.synthesize(text))
        return b"".join(c.audio_int16_bytes for c in chunks)
    except ImportError:
        pass

    if not pipert._piper_bin:
        raise RuntimeError(
            "Install piper-tts (`pip install piper-tts`) or put `piper` CLI on PATH"
        )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        subprocess.run(
            [
                pipert._piper_bin,
                "--model",
                str(pipert._model_path),
                "--output_file",
                str(out),
            ],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
        )
        return out.read_bytes()


def _wav_to_frame(wav_bytes: bytes, sample_rate: int) -> rtc.AudioFrame:
    import wave
    from io import BytesIO

    with BytesIO(wav_bytes) as bio, wave.open(bio, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if rate != sample_rate:
        logger.warning("Piper wav rate %s != configured %s", rate, sample_rate)

    samples = len(frames) // 2 // channels
    return rtc.AudioFrame(
        data=frames,
        sample_rate=rate,
        num_channels=channels,
        samples_per_channel=samples,
    )
