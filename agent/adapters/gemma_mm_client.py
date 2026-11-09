"""HTTP client for vLLM Gemma multimodal chat (audio-in and text complete)."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import wave
from typing import Any, Callable

import httpx

from agent.adapters.response_parser import ParsedTurn, parse_heard_reply
from agent.config import LocaleLLMConfig, show_seed
from agent.telemetry.model_output_logger import ModelOutputLogger

logger = logging.getLogger(__name__)


def _add_seed(payload: dict[str, Any]) -> None:
    """Sample with TALKSHOW_SEED when it's set, so a repeated prompt gets the same reply."""
    seed = show_seed()
    if seed is not None:
        payload["seed"] = seed


def sse_content_delta(line: str) -> str | None:
    """Text delta from one server-sent-events line of a streamed chat completion.

    "" for lines that carry no text (blank lines, role-only or empty deltas), None at
    the ``[DONE]`` sentinel.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return ""
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        logger.warning("Gemma stream: unparsable event %.200s", data)
        return ""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("delta") or {}).get("content")
    return content if isinstance(content, str) else ""


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
        self._output_logger: ModelOutputLogger | None = None

    def set_output_logger(self, output_logger: ModelOutputLogger) -> None:
        self._output_logger = output_logger

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
            messages.extend(dict(m) for m in history_messages)
        last = messages[-1]
        if last["role"] == "user" and isinstance(last.get("content"), str):
            # History ended with other speakers' lines: fold the instruction into that
            # user turn so roles keep alternating.
            if isinstance(tail_user, str):
                last["content"] = f"{last['content']}\n\n{tail_user}"
            else:
                last["content"] = [{"type": "text", "text": last["content"]}, *tail_user]
            return messages
        messages.append({"role": "user", "content": tail_user})
        return messages

    async def complete_from_wav(
        self,
        wav_bytes: bytes,
        *,
        user_text: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        output_role: str | None = None,
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
        _add_seed(payload)

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
        if self._output_logger is not None and output_role is not None:
            self._output_logger.log(output_role, content)

        return parse_heard_reply(content)

    async def complete_text(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        raw: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        output_role: str | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> str:
        """Text-only turn with optional prior conversation as chat messages.

        ``raw=True`` returns the model output untouched so callers can parse
        control tags ([next]/[pad]/…). Otherwise a [reply]-formatted answer is
        reduced to its spoken reply, which strips those tags.

        ``on_text`` streams the request and is called with the raw output so far after
        each delta, so a caller can start on the first sentence while the rest is
        written. The return value is the same either way.
        """
        system = system_prompt if system_prompt is not None else self._system_prompt
        messages = self._build_messages(
            system,
            history_messages=history_messages,
            tail_user=user_text,
        )
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "max_tokens": max_tokens or self._cfg.max_tokens,
            "temperature": 0.45 if temperature is None else temperature,
        }
        _add_seed(payload)
        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}
        hist_n = len(history_messages) if history_messages else 0
        logger.info(
            "Gemma text request → %s persona=%s history_msgs=%d stream=%s",
            url,
            self.active_persona_id,
            hist_n,
            on_text is not None,
        )
        if on_text is not None:
            content = await self._stream_text(url, payload, headers, on_text)
        else:
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
        if self._output_logger is not None and output_role is not None:
            self._output_logger.log(output_role, content)
        text = content.strip()
        if raw:
            return text
        for marker in ("[reply]:", "[heard]:"):
            if marker.lower() in text.lower():
                parsed = parse_heard_reply(text)
                return parsed.reply or text
        return text

    async def _stream_text(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        on_text: Callable[[str], None],
    ) -> str:
        """POST with ``stream: true``; call ``on_text`` with the output so far per delta."""
        parts: list[str] = []
        async with self._http.stream(
            "POST", url, json={**payload, "stream": True}, headers=headers
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                logger.error(
                    "Gemma text HTTP %s model=%s body=%.500s",
                    resp.status_code,
                    self._cfg.model,
                    body,
                )
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                delta = sse_content_delta(line)
                if delta is None:
                    break
                if not delta:
                    continue
                parts.append(delta)
                try:
                    on_text("".join(parts))
                except Exception:
                    # A failing listener must not lose the reply itself.
                    logger.exception("Gemma stream listener failed")
        return "".join(parts)

    async def complete_tools(
        self,
        user_text: str,
        *,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "required",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
        """Text turn with OpenAI-style tools → (content, [(tool name, arguments)]).

        ``tool_choice="required"`` makes vLLM constrain the output to a tool call.
        Unparsable arguments come back as ``{}``.
        """
        system = system_prompt if system_prompt is not None else self._system_prompt
        messages = self._build_messages(
            system,
            history_messages=history_messages,
            tail_user=user_text,
        )
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens or self._cfg.max_tokens,
            "temperature": 0.45 if temperature is None else temperature,
        }
        _add_seed(payload)
        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}
        resp = await self._http.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error(
                "Gemma tools HTTP %s model=%s body=%.500s",
                resp.status_code,
                self._cfg.model,
                resp.text,
            )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        calls: list[tuple[str, dict[str, Any]]] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append((str(fn.get("name") or ""), args if isinstance(args, dict) else {}))
        return str(message.get("content") or "").strip(), calls
