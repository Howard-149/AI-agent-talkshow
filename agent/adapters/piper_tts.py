from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from agent.config import LocaleTTSConfig

logger = logging.getLogger(__name__)

# Reuse loaded ONNX voices across role handoffs (avoids gaps between panel lines).
_voice_cache: dict[str, object] = {}


class PiperTTS(tts.TTS):
    def __init__(self, cfg: LocaleTTSConfig) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=cfg.sample_rate,
            num_channels=1,
        )
        self._model_path = Path(cfg.model_path)
        self._piper_bin = shutil.which("piper")
        logger.info("PiperTTS init model_path=%s", self._model_path)

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
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        tts_impl: PiperTTS = self._tts  # type: ignore[assignment]
        text = self._input_text.strip()
        if not text:
            return

        pcm_bytes, sample_rate, num_channels = await asyncio.to_thread(
            _synthesize_pcm, tts_impl, text
        )

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=sample_rate,
            num_channels=num_channels,
            mime_type="audio/pcm",
            stream=False,
        )
        output_emitter.push(pcm_bytes)
        output_emitter.flush()


def _piper_config_path(model_path: Path) -> Path:
    """Piper 1.x expects `<name>.onnx.json` beside the ONNX file."""
    legacy = model_path.with_suffix(".json")  # rhasspy: en_US-foo-medium.json
    modern = Path(f"{model_path}.json")  # piper-tts: en_US-foo-medium.onnx.json
    if modern.is_file():
        return modern
    if legacy.is_file():
        return legacy
    return modern


def _load_voice(model_path: Path, config_path: Path):
    from piper import PiperVoice  # type: ignore[import-untyped]

    key = str(model_path.resolve())
    cached = _voice_cache.get(key)
    if cached is not None:
        return cached
    voice = PiperVoice.load(str(model_path), config_path=str(config_path))
    _voice_cache[key] = voice
    logger.info("PiperVoice cached model=%s", model_path.name)
    return voice


def _synthesize_pcm(pipert: PiperTTS, text: str) -> tuple[bytes, int, int]:
    if not pipert._model_path.is_file():
        raise FileNotFoundError(f"Piper model not found: {pipert._model_path}")

    config_path = _piper_config_path(pipert._model_path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Piper config missing for {pipert._model_path.name}: "
            f"expected {config_path}. "
            f"Re-run: bash deploy/download-piper-voices.sh --dest $(dirname {pipert._model_path}) ryan amy lessac"
        )

    # Prefer Python piper-tts if installed
    try:
        logger.debug(
            "PiperVoice.load model=%s config=%s",
            pipert._model_path.name,
            config_path.name,
        )
        voice = _load_voice(pipert._model_path, config_path)
        pcm, rate = _synthesize_piper_voice(voice, text)
        return pcm, rate, 1
    except ImportError:
        pass

    if not pipert._piper_bin:
        raise RuntimeError(
            "Install piper-tts (`pip install piper-tts`) or put `piper` CLI on PATH"
        )

    sentence_silence = _env_float("TALKSHOW_PIPER_SENTENCE_SILENCE", 0.0)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        cmd = [
            pipert._piper_bin,
            "--model",
            str(pipert._model_path),
            "--output_file",
            str(out),
        ]
        if sentence_silence >= 0:
            cmd.extend(["--sentence-silence", str(sentence_silence)])
        subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
        )
        return _read_wav_pcm(out.read_bytes(), pipert.sample_rate)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _synthesize_piper_voice(voice: object, text: str) -> tuple[bytes, int]:
    """
    One continuous PCM buffer for the full line — avoids long gaps between sentences.
    """
    import numpy as np

    rate = voice.config.sample_rate  # type: ignore[attr-defined]

    if hasattr(voice, "phonemize") and hasattr(voice, "phoneme_ids_to_audio"):
        from piper.config import SynthesisConfig

        syn = SynthesisConfig()
        phoneme_ids: list[int] = []
        for phonemes in voice.phonemize(text):  # type: ignore[attr-defined]
            if phonemes:
                phoneme_ids.extend(voice.phonemes_to_ids(phonemes))  # type: ignore[attr-defined]
        if phoneme_ids:
            audio = voice.phoneme_ids_to_audio(phoneme_ids, syn_config=syn)  # type: ignore[attr-defined]
            if isinstance(audio, tuple):
                audio = audio[0]
            pcm = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16).tobytes()
            return pcm, rate

    sentence_silence = _env_float("TALKSHOW_PIPER_SENTENCE_SILENCE", 0.0)
    try:
        chunks = list(voice.synthesize(text, sentence_silence=sentence_silence))  # type: ignore[attr-defined]
    except TypeError:
        try:
            from piper.config import SynthesisConfig

            chunks = list(
                voice.synthesize(text, syn_config=SynthesisConfig())  # type: ignore[attr-defined]
            )
        except TypeError:
            chunks = list(voice.synthesize(text))  # type: ignore[attr-defined]

    pcm = b"".join(c.audio_int16_bytes for c in chunks)
    return pcm, rate


def _read_wav_pcm(wav_bytes: bytes, expected_rate: int) -> tuple[bytes, int, int]:
    import wave
    from io import BytesIO

    with BytesIO(wav_bytes) as bio, wave.open(bio, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if rate != expected_rate:
        logger.warning("Piper wav rate %s != configured %s", rate, expected_rate)

    return frames, rate, channels
