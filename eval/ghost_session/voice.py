"""The ghost human's voice: scripted turns synthesized with CosyVoice from a reference.

References live in ``GHOST_VOICE_DIR`` (default ``eval/ghost_session/voice/``, not in
git — it is someone's voice): ``Neutral.wav`` is required, ``<Emotion>.wav`` (Anger,
Sadness, Happiness, …) is used for a turn with that emotion when present. A
``<name>.txt`` next to a wav holds what is said in it.

Synthesis goes through the CosyVoice sidecar (``COSYVOICE_SIDECAR_URL``) in instruct2
mode: the reference sets the timbre, the emotion instruct sets the delivery. The
sidecar reads the reference by path, so it must see the same file system (on Babel,
point ``GHOST_VOICE_DIR`` at a shared copy). When the sidecar runs elsewhere (a laptop
reaching it through a tunnel), set ``GHOST_VOICE_REMOTE_DIR`` to the folder holding the
same references on the sidecar's machine: the reference is still chosen and checked in
``GHOST_VOICE_DIR``, but the sidecar is sent ``GHOST_VOICE_REMOTE_DIR/<same file name>``. Results are cached per text, emotion and
reference in ``GHOST_VOICE_CACHE`` (default ``logs/ghost-voice-cache/``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("ghost_session")

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Speech:
    """Mono int16 PCM."""

    pcm: bytes
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


def voice_dir() -> Path:
    raw = os.environ.get("GHOST_VOICE_DIR", "").strip()
    return Path(raw) if raw else _REPO_ROOT / "eval" / "ghost_session" / "voice"


def cache_dir() -> Path:
    raw = os.environ.get("GHOST_VOICE_CACHE", "").strip()
    return Path(raw) if raw else _REPO_ROOT / "logs" / "ghost-voice-cache"


def reference_wav(emotion: str = "") -> Path:
    """``<Emotion>.wav`` when recorded, else ``Neutral.wav``."""
    folder = voice_dir()
    if emotion:
        own = folder / f"{emotion.strip().capitalize()}.wav"
        if own.is_file():
            return own
    neutral = folder / "Neutral.wav"
    if not neutral.is_file():
        raise FileNotFoundError(
            f"no ghost voice reference at {neutral} (set GHOST_VOICE_DIR, or use --text-turns)"
        )
    return neutral


def sidecar_reference_path(reference: Path) -> str:
    """The path the sidecar opens: ``GHOST_VOICE_REMOTE_DIR/<name>`` when set (the
    sidecar is on another machine), else the local file."""
    remote = os.environ.get("GHOST_VOICE_REMOTE_DIR", "").strip().rstrip("/")
    if remote:
        return f"{remote}/{reference.name}"
    return str(reference.resolve())


def _sidecar_synth() -> Any:
    """``agent.adapters.cosyvoice_client.synthesize_pcm_via_sidecar``. On a laptop without
    the agents SDK the ``agent.adapters`` package ``__init__`` (LiveKit TTS adapters) fails
    to import; the HTTP client itself only needs httpx, so load it without that init."""
    try:
        from agent.adapters import cosyvoice_client
    except ImportError:
        import importlib
        import sys
        import types

        pkg = types.ModuleType("agent.adapters")
        pkg.__path__ = [str(_REPO_ROOT / "agent" / "adapters")]
        sys.modules["agent.adapters"] = pkg
        cosyvoice_client = importlib.import_module("agent.adapters.cosyvoice_client")
    return cosyvoice_client.synthesize_pcm_via_sidecar


def read_wav(path: Path) -> Speech:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM")
        rate, channels = wf.getframerate(), wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    if channels > 1:
        import numpy as np

        pcm = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels).mean(axis=1)
        pcm = pcm.astype(np.int16).tobytes()
    return Speech(pcm, rate)


def write_wav(path: Path, speech: Speech) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(speech.sample_rate)
        wf.writeframes(speech.pcm)


def _cache_key(text: str, emotion: str, reference: Path, instruct: str) -> str:
    stat = reference.stat()
    raw = "\n".join([text, emotion, str(reference.resolve()), str(stat.st_mtime_ns), instruct])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


async def speech_for_turn(text: str, *, emotion: str = "", audio: str = "", locale: str = "en") -> tuple[Speech, Path]:
    """The turn's audio and the wav it came from: ``audio`` if the script names one,
    else the cached or freshly synthesized line."""
    if audio:
        path = Path(audio)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return read_wav(path), path

    from agent.emotion.tts import emotion_instruct

    synthesize_pcm_via_sidecar = _sidecar_synth()

    reference = reference_wav(emotion)
    instruct = emotion_instruct(emotion=emotion or "Neutral", locale=locale)
    cached = cache_dir() / f"{_cache_key(text, emotion, reference, instruct)}.wav"
    if cached.is_file():
        return read_wav(cached), cached
    t0 = time.monotonic()
    pcm, rate, _ = await synthesize_pcm_via_sidecar(
        text=text, instruct=instruct, prompt_wav=sidecar_reference_path(reference), mode="instruct2"
    )
    speech = Speech(pcm, rate)
    write_wav(cached, speech)
    logger.info(
        "ghost voice synthesized %.1fs of audio in %.1fs emotion=%s ref=%s → %s",
        speech.duration_s, time.monotonic() - t0, emotion or "Neutral", reference.name, cached,
    )
    return speech, cached
