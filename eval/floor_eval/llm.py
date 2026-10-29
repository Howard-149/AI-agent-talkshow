"""Minimal OpenAI-compatible chat call for stimulus generation and validation.

Listener decisions do NOT use this; they go through the production GemmaMMClient.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from agent.adapters.gemma_mm_client import _sampling_params


async def chat_json(
    http: httpx.AsyncClient, model: str, system: str, user: str, *,
    max_tokens: int = 1200, temperature: float = 0.7, retries: int = 3,
) -> dict[str, Any]:
    url = f"{os.environ['VLLM_BASE_URL'].rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        **_sampling_params(model, max_tokens=max_tokens, temperature=temperature),
    }
    for attempt in range(retries):
        try:
            resp = await http.post(url, json=body, headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"})
            resp.raise_for_status()
            return json.loads(resp.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            if attempt + 1 == retries:
                raise
            await asyncio.sleep(2 ** attempt)
    raise AssertionError("unreachable")
