from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from agent.adapters.response_parser import ParsedTurn, parse_heard_reply
from agent.config import LocaleLLMConfig

logger = logging.getLogger(__name__)


class GemmaMMClient:
    """vLLM OpenAI-compatible chat with audio_url multimodal input."""

    def __init__(self, llm_cfg: LocaleLLMConfig, *, system_prompt: str) -> None:
        self._cfg = llm_cfg
        self._system_prompt = system_prompt
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete_from_wav(self, wav_bytes: bytes) -> ParsedTurn:
        b64 = base64.standard_b64encode(wav_bytes).decode("ascii")
        audio_url = f"data:audio/wav;base64,{b64}"

        user_text = (
            "Listen to the user's audio turn and respond using the required "
            "[heard]: / [reply]: format."
        )

        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "max_tokens": self._cfg.max_tokens,
            "temperature": 0.7,
        }

        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}

        logger.info("Gemma MM request → %s model=%s", url, self._cfg.model)
        resp = await self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            content = str(content)

        return parse_heard_reply(content)
