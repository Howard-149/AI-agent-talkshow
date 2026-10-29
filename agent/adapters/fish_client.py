"""Fish Audio HTTP PCM streaming; credentials stay in the backend environment."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field
import os

import httpx
from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError

FISH_TTS_URL = "https://api.fish.audio/v1/tts"
# Same voice IDs as Gefei's existing Fish text-show demo.
DEFAULT_FISH_VOICES = {
    "host": "bf322df2096a46f18c579d0baa36f41d",
    "commentator": "536d3a5e000945adb7038665781a4aca",
    "guest": "933563129e564b19a115bedd57b7406a",
}
FISH_MODELS = {"s1", "s2-pro", "s2.1-pro", "s2.1-pro-free", "drama-3-preview"}
FISH_PCM_RATES = {8000, 16000, 24000, 32000, 44100}


def fish_voice_for_role(role: str, locale: str = "en") -> str:
    if role not in DEFAULT_FISH_VOICES:
        raise ValueError(f"No Fish voice configured for role {role!r}")
    return (
        os.environ.get(f"FISH_REFERENCE_ID_{locale.upper()}_{role.upper()}", "").strip()
        or os.environ.get(f"FISH_REFERENCE_ID_{role.upper()}", "").strip()
        or DEFAULT_FISH_VOICES[role]
    )


def fish_sample_rate() -> int:
    rate = int(os.environ.get("FISH_SAMPLE_RATE", "44100"))
    if rate not in FISH_PCM_RATES:
        raise ValueError("Unsupported FISH_SAMPLE_RATE for Fish PCM")
    return rate


@dataclass(frozen=True)
class FishSettings:
    api_key: str = field(repr=False)
    reference_id: str
    model: str = "s2.1-pro-free"
    latency: str = "balanced"
    sample_rate: int = 44100
    timeout_s: float = 45.0

    def __post_init__(self):
        if not self.api_key.strip():
            raise ValueError("TALKSHOW_TTS_ENGINE=fish requires FISH_API_KEY in the backend .env")
        # Fish silently falls back to a paid model on an unknown header value.
        # Reject typos locally; never change tiers or fall back to Piper silently.
        if self.model not in FISH_MODELS:
            raise ValueError("Unsupported FISH_MODEL (model header must be explicit)")
        if self.latency not in {"normal", "balanced", "low"}:
            raise ValueError("Unsupported FISH_LATENCY")
        if self.sample_rate not in FISH_PCM_RATES:
            raise ValueError("Unsupported Fish PCM sample rate")
        if self.timeout_s <= 0 or not self.reference_id:
            raise ValueError("Fish timeout must be positive and reference_id must be configured")

    @classmethod
    def from_env(cls, role: str, locale: str = "en", *, reference_id: str = "", sample_rate: int | None = None):
        return cls(
            api_key=os.environ.get("FISH_API_KEY", "").strip(),
            reference_id=reference_id or fish_voice_for_role(role, locale),
            model=os.environ.get("FISH_MODEL", "s2.1-pro-free").strip(),
            latency=os.environ.get("FISH_LATENCY", "balanced").strip(),
            sample_rate=sample_rate if sample_rate is not None else fish_sample_rate(),
            timeout_s=float(os.environ.get("FISH_TIMEOUT_S", "45")),
        )


async def iter_fish_pcm(text: str, settings: FishSettings, *, client: httpx.AsyncClient | None = None) -> AsyncIterator[bytes]:
    """Yield aligned int16 mono chunks; cancellation closes the HTTP stream."""
    text = text.strip()
    if not text:
        return
    owned_client = client is None
    if client is None:
        client = httpx.AsyncClient()
    received = 0
    try:
        async with client.stream(
            "POST", FISH_TTS_URL,
            headers={"Authorization": f"Bearer {settings.api_key}", "model": settings.model},
            json={"text": text, "reference_id": settings.reference_id, "format": "pcm",
                  "sample_rate": settings.sample_rate, "latency": settings.latency},
            timeout=httpx.Timeout(settings.timeout_s, connect=min(settings.timeout_s, 10.0)),
        ) as response:
            if response.status_code != 200:
                raise APIStatusError(f"Fish Audio TTS returned HTTP {response.status_code}",
                                     status_code=response.status_code, retryable=response.status_code >= 500 or response.status_code == 429)
            content_type = response.headers.get("content-type", "").split(";")[0].lower()
            if content_type not in {"audio/pcm", "audio/raw", "application/octet-stream", "binary/octet-stream"}:
                raise APIError("Fish Audio did not return raw PCM", retryable=False)
            pending = b""
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                received += len(chunk)
                if received > 20 * 1024 * 1024:
                    raise APIError("Fish Audio PCM response exceeded the demo limit", retryable=False)
                data = pending + chunk
                aligned = len(data) & ~1
                pending = data[aligned:]
                if aligned:
                    yield data[:aligned]
            if pending:
                raise APIError("Fish Audio returned a truncated PCM sample", retryable=False)
            if not received:
                raise APIError("Fish Audio returned empty PCM", retryable=False)
    except httpx.TimeoutException:
        raise APITimeoutError("Fish Audio TTS timed out", retryable=received == 0) from None
    except httpx.HTTPError:
        # Do not repeat already-audible speech after a mid-stream failure.
        # Never expose the upstream response body, headers or credentials in logs.
        raise APIConnectionError("Fish Audio TTS connection failed", retryable=received == 0) from None
    finally:
        if owned_client:
            await client.aclose()


async def synthesize_fish_pcm(text: str, settings: FishSettings) -> tuple[bytes, int, int]:
    chunks = []
    async with aclosing(iter_fish_pcm(text, settings)) as stream:
        async for chunk in stream:
            chunks.append(chunk)
    return b"".join(chunks), settings.sample_rate, 1
