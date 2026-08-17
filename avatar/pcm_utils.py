"""PCM16 ↔ WAV helpers and resampling for avatar bake input audio."""

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


def split_pcm16_by_weights(pcm: bytes, weights: list[float]) -> list[bytes]:
    """Split mono PCM16 by relative weights (e.g. sentence char counts).

    Approximate only — CosyVoice does not return per-sentence timestamps.
    Prefer per-sentence TTS when mouth sync at boundaries matters.
    """
    if not weights:
        return [pcm] if pcm else []
    if len(weights) == 1:
        return [pcm]
    total_w = sum(max(0.0, float(w)) for w in weights)
    if total_w <= 0 or not pcm:
        # Equal split fallback.
        n = len(weights)
        frame = (len(pcm) // 2) // n
        out: list[bytes] = []
        for i in range(n):
            start = i * frame * 2
            end = len(pcm) if i == n - 1 else (i + 1) * frame * 2
            out.append(pcm[start:end])
        return out
    n_samples = len(pcm) // 2
    out: list[bytes] = []
    cursor = 0
    for i, w in enumerate(weights):
        if i == len(weights) - 1:
            out.append(pcm[cursor * 2 :])
            break
        take = int(round(n_samples * (max(0.0, float(w)) / total_w)))
        take = max(0, min(take, n_samples - cursor))
        # Keep at least 1 sample for non-empty weight when audio remains.
        if take == 0 and float(w) > 0 and cursor < n_samples:
            take = 1
        end = cursor + take
        out.append(pcm[cursor * 2 : end * 2])
        cursor = end
    return out


def split_pcm16_by_silence(
    pcm: bytes,
    n_parts: int,
    *,
    sample_rate: int,
    weights: list[float] | None = None,
    window_ms: float = 30.0,
    search_radius_ms: float = 400.0,
) -> list[bytes]:
    """Split PCM at local RMS minima near expected cut points.

    Still heuristic (no forced alignment). If silence valleys are weak,
    falls back to ``split_pcm16_by_weights``.
    """
    if n_parts <= 1 or not pcm:
        return [pcm] if pcm else []
    if weights is None or len(weights) != n_parts:
        weights = [1.0] * n_parts

    n_samples = len(pcm) // 2
    win = max(1, int(sample_rate * window_ms / 1000.0))
    # Frame RMS envelope.
    n_frames = max(1, n_samples // win)
    rms: list[float] = []
    for fi in range(n_frames):
        start = fi * win
        end = min(n_samples, start + win)
        chunk = pcm[start * 2 : end * 2]
        rms.append(pcm_rms(chunk))

    total_w = sum(max(0.0, float(w)) for w in weights) or float(n_parts)
    expected: list[int] = []
    acc = 0.0
    for i in range(n_parts - 1):
        acc += max(0.0, float(weights[i])) / total_w
        expected.append(int(round(acc * n_samples)))

    radius = max(win, int(sample_rate * search_radius_ms / 1000.0))
    cuts: list[int] = []
    prev = 0
    for exp in expected:
        lo = max(prev + win, exp - radius)
        hi = min(n_samples - win, exp + radius)
        if hi <= lo:
            cuts.append(max(prev + 1, min(exp, n_samples - 1)))
            prev = cuts[-1]
            continue
        # Search lowest RMS frame in [lo, hi).
        best_s = exp
        best_r = float("inf")
        for s in range(lo, hi, win):
            fi = min(n_frames - 1, s // win)
            r = rms[fi]
            if r < best_r:
                best_r = r
                best_s = s
        cuts.append(best_s)
        prev = best_s

    out: list[bytes] = []
    cursor = 0
    for cut in cuts:
        cut = max(cursor + 1, min(cut, n_samples - (n_parts - len(out) - 1)))
        out.append(pcm[cursor * 2 : cut * 2])
        cursor = cut
    out.append(pcm[cursor * 2 :])
    if len(out) != n_parts:
        return split_pcm16_by_weights(pcm, weights)
    return out
