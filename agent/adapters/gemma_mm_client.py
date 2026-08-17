"""HTTP client for vLLM Gemma multimodal chat (audio-in and text complete)."""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import wave
from typing import Any

import httpx

from agent.adapters.response_parser import ParsedTurn, parse_heard_reply
from agent.config import LocaleLLMConfig

logger = logging.getLogger(__name__)


def _silent_wav_bytes(*, duration_sec: float = 0.35, sample_rate: int = 16000) -> bytes:
    """Minimal mono PCM16 WAV for multimodal path warm-up."""
    n = max(1, int(duration_sec * sample_rate))
    pcm = b"\x00\x00" * n
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class GemmaMMClient:
    """vLLM OpenAI-compatible chat with audio_url multimodal input."""

    def __init__(self, llm_cfg: LocaleLLMConfig, *, system_prompt: str) -> None:
        self._cfg = llm_cfg
        self._system_prompt = system_prompt
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self._mm_warmed = False

    @property
    def active_persona_id(self) -> str:
        return getattr(self, "_persona_id", "host")

    def set_persona(self, persona_id: str, instructions: str) -> None:
        self._persona_id = persona_id
        self._system_prompt = instructions
        logger.info("Gemma persona → %s", persona_id)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def warm_multimodal(self) -> None:
        """Hit vLLM with a tiny audio_url chat once so first human turn is not cold.

        Opt out: TALKSHOW_GEMMA_MM_WARM=0
        """
        raw = os.environ.get("TALKSHOW_GEMMA_MM_WARM", "1").strip().lower()
        if raw in ("0", "false", "no", "off"):
            logger.info("Gemma MM warm skipped (TALKSHOW_GEMMA_MM_WARM=%s)", raw)
            return
        if self._mm_warmed:
            return
        t0 = time.monotonic()
        try:
            await self.complete_from_wav(
                _silent_wav_bytes(),
                user_text=(
                    "Warm-up only. Reply with exactly:\n"
                    "[heard]: (silence)\n"
                    "[reply]: OK\n"
                    "[next]: host\n"
                    "[emotion]: neutral"
                ),
                history_messages=None,
            )
            self._mm_warmed = True
            logger.info(
                "Gemma MM warm done model=%s ms=%d",
                self._cfg.model,
                round((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gemma MM warm failed after %.1fs: %s",
                time.monotonic() - t0,
                exc,
            )

    def _build_messages(
        self,
        system: str,
        *,
        history_messages: list[dict[str, Any]] | None,
        tail_user: str | list[Any],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if history_messages:
            messages.extend(history_messages)
        if isinstance(tail_user, str):
            messages.append({"role": "user", "content": tail_user})
        else:
            messages.append({"role": "user", "content": tail_user})
        return messages

    async def complete_from_wav(
        self,
        wav_bytes: bytes,
        *,
        user_text: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
    ) -> ParsedTurn:
        b64 = base64.standard_b64encode(wav_bytes).decode("ascii")
        audio_url = f"data:audio/wav;base64,{b64}"

        if user_text is None:
            user_text = (
                "Listen to the user's audio turn and respond using the required "
                "[heard]: / [reply]: format."
            )

        tail: list[Any] = [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {"type": "text", "text": user_text},
        ]
        messages = self._build_messages(
            self._system_prompt,
            history_messages=history_messages,
            tail_user=tail,
        )

        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "max_tokens": self._cfg.max_tokens,
            "temperature": 0.5,
        }

        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}

        hist_n = len(history_messages) if history_messages else 0
        logger.info(
            "Gemma MM request → %s model=%s history_msgs=%d",
            url,
            self._cfg.model,
            hist_n,
        )
        resp = await self._http.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error(
                "Gemma MM HTTP %s model=%s body=%.500s",
                resp.status_code,
                self._cfg.model,
                resp.text,
            )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            content = str(content)

        return parse_heard_reply(content)

    async def complete_text(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Text-only turn with optional prior conversation as chat messages."""
        system = system_prompt if system_prompt is not None else self._system_prompt
        messages = self._build_messages(
            system,
            history_messages=history_messages,
            tail_user=user_text,
        )
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "max_tokens": self._cfg.max_tokens,
            "temperature": 0.45,
        }
        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}
        hist_n = len(history_messages) if history_messages else 0
        logger.info(
            "Gemma text request → %s persona=%s history_msgs=%d",
            url,
            self.active_persona_id,
            hist_n,
        )
        resp = await self._http.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error(
                "Gemma text HTTP %s model=%s body=%.500s",
                resp.status_code,
                self._cfg.model,
                resp.text,
            )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            content = str(content)
        text = content.strip()
        for marker in ("[reply]:", "[heard]:"):
            if marker.lower() in text.lower():
                parsed = parse_heard_reply(text)
                return parsed.reply or text
        return text
