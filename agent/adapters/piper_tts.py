"""LiveKit TTS adapter wrapping local Piper ONNX voices."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.telemetry.turn_jsonl_logger import TurnJsonlLogger

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from agent.config import LocaleTTSConfig

logger = logging.getLogger(__name__)

# Reuse loaded ONNX voices across role handoffs (avoids gaps between panel lines).
_voice_cache: dict[str, object] = {}


class PiperTTS(tts.TTS):
    def __init__(
        self,
        cfg: LocaleTTSConfig,
        *,
        turn_log: TurnJsonlLogger | None = None,
        room_name: str = "",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=cfg.sample_rate,
            num_channels=1,
        )
        self._model_path = Path(cfg.model_path)
        self._piper_bin = shutil.which("piper")
        self._turn_log = turn_log
        self._room_name = room_name
        logger.info(
            "PiperTTS init model_path=%s cuda=%s device=%s",
            self._model_path,
            _piper_use_cuda(),
            _piper_cuda_device() if _piper_use_cuda() else "cpu",
        )

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

        t0 = time.monotonic()
        pcm_bytes, sample_rate, num_channels = await asyncio.to_thread(
            _synthesize_pcm, tts_impl, text
        )
        tts_latency_s = time.monotonic() - t0

        if tts_impl._turn_log is not None:
            tts_impl._turn_log.log(
                "tts_synthesize",
                tts_latency_s=round(tts_latency_s, 3),
                chars=len(text),
                room=tts_impl._room_name,
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


def _piper_use_cuda() -> bool:
    raw = os.environ.get("TALKSHOW_PIPER_CUDA", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _piper_cuda_device() -> int:
    """Physical CUDA device for Piper (default: same as DyStream, GPU 1)."""
    raw = os.environ.get("TALKSHOW_PIPER_CUDA_DEVICE", "").strip()
    if not raw:
        raw = os.environ.get("DYSTREAM_CUDA_DEVICE", "1").strip() or "1"
    try:
        return max(int(raw), 0)
    except ValueError:
        logger.warning("Invalid TALKSHOW_PIPER_CUDA_DEVICE=%r; using 1", raw)
        return 1


def _invalidate_cuda_voices() -> None:
    for key in [k for k in _voice_cache if "#cuda_dev=" in k]:
        _voice_cache.pop(key, None)


def _is_ort_cuda_fail(exc: BaseException) -> bool:
    name = type(exc).__module__ + "." + type(exc).__name__
    msg = str(exc).lower()
    return (
        "onnxruntime" in name.lower()
        or "cudnn" in msg
        or "cuda" in msg
        or "cublas" in msg
        or "gpu=" in msg
    )


def _load_voice_cuda(model_path: Path, config_path: Path, device_id: int):
    """Load Piper ONNX on a specific CUDA device (vLLM stays on GPU 0)."""
    import json

    import onnxruntime as ort
    from piper import PiperVoice  # type: ignore[import-untyped]
    from piper.config import PiperConfig  # type: ignore[import-untyped]

    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    providers: list = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": device_id,
                "cudnn_conv_algo_search": "HEURISTIC",
            },
        ),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(
        str(model_path),
        sess_options=ort.SessionOptions(),
        providers=providers,
    )
    return PiperVoice(
        config=PiperConfig.from_dict(config_dict),
        session=session,
    )


def _load_voice(model_path: Path, config_path: Path, *, force_cpu: bool = False):
    from piper import PiperVoice  # type: ignore[import-untyped]

    want_cuda = _piper_use_cuda() and not force_cpu
    device_id = _piper_cuda_device() if want_cuda else -1
    key = (
        f"{model_path.resolve()}#cuda_dev={device_id}"
        if want_cuda
        else f"{model_path.resolve()}#cuda_dev=-1"
    )
    cached = _voice_cache.get(key)
    if cached is not None:
        return cached

    if want_cuda:
        try:
            voice = _load_voice_cuda(model_path, config_path, device_id)
            providers = voice.session.get_providers()
            if "CUDAExecutionProvider" in providers:
                _voice_cache[key] = voice
                logger.info(
                    "PiperVoice cached model=%s cuda_device=%d providers=%s",
                    model_path.name,
                    device_id,
                    providers,
                )
                return voice
            logger.warning(
                "TALKSHOW_PIPER_CUDA=1 device=%d but ORT providers=%s; using CPU",
                device_id,
                providers,
            )
        except Exception as exc:
            logger.warning(
                "Piper CUDA load failed device=%d (%s); using CPU",
                device_id,
                exc,
            )

    voice = PiperVoice.load(
        str(model_path), config_path=str(config_path), use_cuda=False
    )
    cpu_key = f"{model_path.resolve()}#cuda_dev=-1"
    _voice_cache[cpu_key] = voice
    logger.info(
        "PiperVoice cached model=%s providers=%s",
        model_path.name,
        voice.session.get_providers(),
    )
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
            "PiperVoice.load model=%s config=%s cuda_device=%s",
            pipert._model_path.name,
            config_path.name,
            _piper_cuda_device() if _piper_use_cuda() else "cpu",
        )
        voice = _load_voice(pipert._model_path, config_path)
        try:
            pcm, rate = _synthesize_piper_voice(voice, text)
            return pcm, rate, 1
        except Exception as exc:
            # CUDA / cuDNN often blows up when sharing a GPU with another process.
            if _piper_use_cuda() and _is_ort_cuda_fail(exc):
                logger.warning(
                    "Piper CUDA synth failed (%s); invalidating CUDA sessions → CPU",
                    exc,
                )
                _invalidate_cuda_voices()
                voice = _load_voice(
                    pipert._model_path, config_path, force_cpu=True
                )
                pcm, rate = _synthesize_piper_voice(voice, text)
                return pcm, rate, 1
            raise
    except ImportError:
        pass

    if not pipert._piper_bin:
        raise RuntimeError(
            "Install piper-tts (`pip install piper-tts`) or put `piper` CLI on PATH"
        )

    sentence_silence = _env_float("TALKSHOW_PIPER_SENTENCE_SILENCE", 0.0)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        # CLI piper has no --device; pin the process to Piper's CUDA GPU.
        env = os.environ.copy()
        if _piper_use_cuda():
            env["CUDA_VISIBLE_DEVICES"] = str(_piper_cuda_device())
        cmd = [
            pipert._piper_bin,
            "--model",
            str(pipert._model_path),
            "--output_file",
            str(out),
        ]
        if _piper_use_cuda():
            cmd.append("--cuda")
        if sentence_silence >= 0:
            cmd.extend(["--sentence-silence", str(sentence_silence)])
        subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
            env=env,
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
    sentence_silence = _env_float("TALKSHOW_PIPER_SENTENCE_SILENCE", 0.0)
    if sentence_silence > 0:
        logger.warning(
            "TALKSHOW_PIPER_SENTENCE_SILENCE=%.2f adds gaps between sentences; use 0 for panel lines",
            sentence_silence,
        )

    if hasattr(voice, "phonemize") and hasattr(voice, "phoneme_ids_to_audio"):
        from piper.config import SynthesisConfig

        syn = SynthesisConfig()
        phoneme_ids: list[int] = []
        for phonemes in voice.phonemize(text):  # type: ignore[attr-defined]
            if phonemes:
                phoneme_ids.extend(voice.phonemes_to_ids(phonemes))  # type: ignore[attr-defined]
        if phoneme_ids:
            logger.info(
                "Piper phonemize path chars=%d ids=%d",
                len(text),
                len(phoneme_ids),
            )
            audio = voice.phoneme_ids_to_audio(phoneme_ids, syn_config=syn)  # type: ignore[attr-defined]
            if isinstance(audio, tuple):
                audio = audio[0]
            pcm = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16).tobytes()
            return pcm, rate
        logger.warning(
            "Piper phonemize returned no ids; falling back to synthesize() chunks"
        )

    chunk_silence = 0.0  # never add inter-sentence silence when chunking
    logger.info(
        "Piper chunk path chars=%d sentence_silence=%.2f",
        len(text),
        chunk_silence,
    )
    try:
        chunks = list(
            voice.synthesize(text, sentence_silence=chunk_silence)  # type: ignore[attr-defined]
        )
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


def prewarm_piper_voices(app_cfg: object | None = None) -> None:
    """Load all panel Piper ONNX voices (and run a tiny CUDA synth) at process start."""
    from agent.config import AppConfig, load_config, load_persona_tts

    cfg: AppConfig = app_cfg if isinstance(app_cfg, AppConfig) else load_config()
    roles = ("host", "guest", "commentator")
    t_all = time.monotonic()
    loaded = 0
    for role in roles:
        tts_cfg = load_persona_tts(role, cfg)
        model_path = Path(tts_cfg.model_path)
        config_path = _piper_config_path(model_path)
        if not model_path.is_file():
            logger.warning("Piper prewarm skip role=%s missing %s", role, model_path)
            continue
        if not config_path.is_file():
            logger.warning("Piper prewarm skip role=%s missing %s", role, config_path)
            continue
        t0 = time.monotonic()
        try:
            voice = _load_voice(model_path, config_path)
            # Touch CUDA/cuDNN kernels so first real line is not the cold start.
            _synthesize_piper_voice(voice, "Hi.")
            loaded += 1
            logger.info(
                "Piper prewarm ready role=%s model=%s ms=%d",
                role,
                model_path.name,
                round((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:
            logger.warning("Piper prewarm failed role=%s: %s", role, exc)
    logger.info(
        "Piper prewarm done voices=%d/%d total_ms=%d cuda=%s device=%s",
        loaded,
        len(roles),
        round((time.monotonic() - t_all) * 1000),
        _piper_use_cuda(),
        _piper_cuda_device() if _piper_use_cuda() else "cpu",
    )
