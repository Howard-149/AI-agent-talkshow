"""HTTP client for the CosyVoice TTS sidecar (emotion-aware PCM)."""

from __future__ import annotations

import array
import base64
import io
import logging
import os
import wave
from typing import Any

import httpx

from agent.adapters.cosyvoice_voices import cosyvoice_mode, mode_needs_prompt_wav

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def cosyvoice_sidecar_url() -> str:
    return os.environ.get("COSYVOICE_SIDECAR_URL", "").strip().rstrip("/")


def cosyvoice_enabled() -> bool:
    return bool(cosyvoice_sidecar_url())


def _read_wav_pcm(data: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(
            f"CosyVoice sidecar WAV must be 16-bit PCM, got sampwidth={sampwidth}"
        )
    if channels != 1:
        samples = array.array("h")
        samples.frombytes(frames)
        mono = array.array(
            "h", (samples[i] for i in range(0, len(samples), channels))
        )
        frames = mono.tobytes()
        channels = 1
    return frames, sample_rate, channels


async def _http_client(timeout_sec: float) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=timeout_sec)
    return _client


async def synthesize_pcm_via_sidecar(
    *,
    text: str,
    instruct: str,
    spk_id: str = "",
    prompt_wav: str = "",
    mode: str | None = None,
    sample_rate_hint: int = 22050,
    timeout_sec: float | None = None,
) -> tuple[bytes, int, int]:
    """POST /synthesize → int16 mono PCM + rate + channels."""
    base = cosyvoice_sidecar_url()
    if not base:
        raise RuntimeError("COSYVOICE_SIDECAR_URL is not set")

    use_mode = (mode or cosyvoice_mode()).strip()
    if "#" in use_mode:
        use_mode = use_mode.split("#", 1)[0].strip()
    use_mode = (use_mode.split() or ["instruct"])[0].lower()
    if mode_needs_prompt_wav(use_mode) and not prompt_wav:
        raise RuntimeError(f"mode={use_mode} requires prompt_wav")
    if use_mode in ("instruct", "sft") and not spk_id:
        raise RuntimeError(f"mode={use_mode} requires spk_id")

    if timeout_sec is None:
        raw = os.environ.get("COSYVOICE_TIMEOUT_SEC", "120").strip()
        timeout_sec = float(raw) if raw else 120.0

    payload: dict[str, Any] = {
        "text": text,
        "instruct": instruct,
        "mode": use_mode,
        "stream": False,
        "format": "json",
    }
    speed_raw = os.environ.get("COSYVOICE_SPEED", "").strip()
    if speed_raw:
        try:
            payload["speed"] = max(0.5, min(2.0, float(speed_raw)))
        except ValueError:
            pass
    if spk_id:
        payload["spk_id"] = spk_id
    if prompt_wav:
        payload["prompt_wav"] = prompt_wav

    url = f"{base}/synthesize"
    client = await _http_client(timeout_sec)
    resp = await client.post(
        url,
        json=payload,
        headers={"Accept": "application/json"},
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"CosyVoice sidecar HTTP {resp.status_code}: {resp.text[:300]}"
        )
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in ctype:
        data = resp.json()
        b64 = data.get("pcm_s16le_b64") or data.get("pcm_b64")
        if not b64:
            raise RuntimeError("CosyVoice JSON response missing pcm_s16le_b64")
        pcm = base64.b64decode(b64)
        rate = int(data.get("sample_rate") or sample_rate_hint)
        ch = int(data.get("num_channels") or 1)
        sidecar_lat = data.get("latency_s")
        if sidecar_lat is not None:
            logger.debug(
                "cosyvoice sidecar latency_s=%s chars=%d mode=%s",
                sidecar_lat,
                len(text),
                use_mode,
            )
        return pcm, rate, ch
    return _read_wav_pcm(resp.content)


async def sidecar_health() -> dict[str, Any] | None:
    base = cosyvoice_sidecar_url()
    if not base:
        return None
    try:
        client = await _http_client(5.0)
        resp = await client.get(f"{base}/health")
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("cosyvoice health failed: %s", exc)
        return None
