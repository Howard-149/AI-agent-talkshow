from __future__ import annotations

import struct
import wave
from pathlib import Path


def pcm16_to_wav(
    pcm: bytes,
    dest: Path,
    *,
    sample_rate: int = 22050,
    num_channels: int = 1,
) -> Path:
    """Write mono/stereo PCM16 LE to a WAV file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return dest


def silent_wav(
    dest: Path,
    *,
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
) -> Path:
    """Generate silent audio for idle-loop DyStream bakes."""
    n_samples = max(1, int(duration_sec * sample_rate))
    pcm = b"\x00\x00" * n_samples
    return pcm16_to_wav(pcm, dest, sample_rate=sample_rate, num_channels=1)


def resample_pcm16_mono(
    pcm: bytes,
    src_rate: int,
    dst_rate: int,
) -> bytes:
    """Lightweight PCM resample for DyStream (expects 16 kHz)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    import audioop

    converted, _ = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
    return converted


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def pcm_rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    if not samples:
        return 0.0
    mean_sq = sum(s * s for s in samples) / len(samples)
    return mean_sq**0.5
