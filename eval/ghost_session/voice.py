"""The ghost human's voice: scripted turns synthesized with CosyVoice from a reference.

References live in ``GHOST_VOICE_DIR`` (default ``eval/ghost_session/voice/``, not in
git — it is someone's voice): ``Neutral.wav`` is required, ``<Emotion>.wav`` (Anger,
Sadness, Happiness, …) is used for a turn with that emotion when present. A
``<name>.txt`` next to a wav holds what is said in it.

Synthesis goes through the CosyVoice sidecar (``COSYVOICE_SIDECAR_URL``) in instruct2
mode: the reference sets the timbre, the emotion instruct sets the delivery. The
sidecar reads the reference by path, so it must see the same file system (on Babel,
point ``GHOST_VOICE_DIR`` at a shared copy). Results are cached per text, emotion and
reference in ``GHOST_VOICE_CACHE`` (default ``logs/ghost-voice-cache/``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path

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

    from agent.adapters.cosyvoice_client import synthesize_pcm_via_sidecar
    from agent.emotion.tts import emotion_instruct

    reference = reference_wav(emotion)
    instruct = emotion_instruct(emotion=emotion or "Neutral", locale=locale)
    cached = cache_dir() / f"{_cache_key(text, emotion, reference, instruct)}.wav"
    if cached.is_file():
        return read_wav(cached), cached
    pcm, rate, _ = await synthesize_pcm_via_sidecar(
        text=text, instruct=instruct, prompt_wav=str(reference.resolve()), mode="instruct2"
    )
    speech = Speech(pcm, rate)
    write_wav(cached, speech)
    logger.info(
        "ghost voice synthesized %.1fs emotion=%s ref=%s → %s",
        speech.duration_s, emotion or "Neutral", reference.name, cached,
    )
    return speech, cached
