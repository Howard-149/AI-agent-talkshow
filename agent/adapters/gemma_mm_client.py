"""HTTP client for vLLM Gemma multimodal chat (audio-in and text complete)."""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import wave
from pathlib import Path
from typing import Any

import httpx

from agent.adapters.response_parser import ParsedTurn, parse_heard_reply
from agent.config import LocaleLLMConfig, show_seed
from agent.telemetry.model_output_logger import ModelOutputLogger

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


def _add_seed(payload: dict[str, Any]) -> None:
    """Sample with TALKSHOW_SEED when it's set, so a repeated prompt gets the same reply."""
    seed = show_seed()
    if seed is not None:
        payload["seed"] = seed


def _openai_flag(name: str, base_url: str) -> bool:
    """OPENAI_USE_* flag: "1"/"0" when set, otherwise on for the OpenAI API itself."""
    raw = os.environ.get(name, "").strip()
    if raw:
        return raw == "1"
    return "api.openai.com" in base_url


def decisions_on_main_llm() -> bool:
    """Short structured calls (floor control, uptake plan) stay on the main (local) endpoint
    when a separate text LLM is configured: each one is a serial step before the next line,
    and a remote round trip costs 1-2 s apiece. TALKSHOW_DECISIONS_ON_MAIN_LLM=0 sends them
    to the text endpoint too."""
    return os.environ.get("TALKSHOW_DECISIONS_ON_MAIN_LLM", "1").strip() != "0"


def _takes_reasoning_effort(model: str) -> bool:
    return model.strip().lower().startswith("gpt-6-luna")


class GemmaMMClient:
    """vLLM OpenAI-compatible chat with audio_url multimodal input."""

    def __init__(self, llm_cfg: LocaleLLMConfig, *, system_prompt: str) -> None:
        self._cfg = llm_cfg
        self._system_prompt = system_prompt
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self._mm_warmed = False
        self._output_logger: ModelOutputLogger | None = None

    def _text_endpoint(self, decision: bool = False) -> tuple[str, str, str]:
        """(base_url, model, api_key) for text calls; plain configs without text_* use the main endpoint.
        ``decision`` calls use the main endpoint unless TALKSHOW_DECISIONS_ON_MAIN_LLM=0."""
        if decision and decisions_on_main_llm():
            return self._cfg.base_url, self._cfg.model, self._cfg.api_key
        endpoint = getattr(self._cfg, "text_endpoint", None)
        if endpoint is not None:
            return endpoint
        return self._cfg.base_url, self._cfg.model, self._cfg.api_key

    def set_output_logger(self, output_logger: ModelOutputLogger) -> None:
        self._output_logger = output_logger

    @property
    def output_log_path(self) -> Path | None:
        return self._output_logger.path if self._output_logger is not None else None

    def log_text_only_event(self, role: str, kind: str, **fields: Any) -> None:
        if self._output_logger is not None:
            self._output_logger.event(role, kind, **fields)

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
        system_prompt: str | None = None,
        temperature: float = 0.5,
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
            system_prompt if system_prompt is not None else self._system_prompt,
            history_messages=history_messages,
            tail_user=tail,
        )

        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "max_tokens": self._cfg.max_tokens,
            "temperature": temperature,
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
        resp = await self._post(url, payload, headers)
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

    async def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        """POST with one retry on a dropped connection (e.g. a pooled keep-alive socket
        the vLLM server already closed — seen with several concurrent rooms)."""
        try:
            return await self._http.post(url, json=payload, headers=headers)
        except httpx.TransportError as exc:
            logger.warning("Gemma request transport error (%r) — retrying once", exc)
            return await self._http.post(url, json=payload, headers=headers)

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
        reasoning_effort: str | None = None,
        decision: bool = False,
    ) -> str:
        """Text-only turn with optional prior conversation as chat messages.

        ``raw=True`` returns the model output untouched so callers can parse
        control tags ([next]/[pad]/…). Otherwise a [reply]-formatted answer is
        reduced to its spoken reply, which strips those tags.

        ``reasoning_effort`` is sent only to models that take it: callers pass "low"
        for decisions (floor control, reactions, appraisals) and "none" for plain
        spoken output. ``OPENAI_REASONING_EFFORT`` overrides it for every call.

        ``decision=True`` marks a short structured call (floor control, uptake plan) that
        stays on the main endpoint; see ``decisions_on_main_llm``.
        """
        system = system_prompt if system_prompt is not None else self._system_prompt
        messages = self._build_messages(
            system,
            history_messages=history_messages,
            tail_user=user_text,
        )
        base_url, model, api_key = self._text_endpoint(decision)
        token_limit = max_tokens or self._cfg.max_tokens
        openai_reasoning = _openai_flag("OPENAI_USE_MAX_COMPLETION_TOKENS", base_url)
        if openai_reasoning:
            # Reasoning models spend completion tokens on hidden reasoning first; the
            # small budgets sized for Gemma's visible output make them fail with 400.
            token_limit = max(
                token_limit, int(os.environ.get("OPENAI_MIN_COMPLETION_TOKENS", "4096"))
            )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            ("max_completion_tokens" if openai_reasoning else "max_tokens"): token_limit,
        }
        effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip() or (
            reasoning_effort if _takes_reasoning_effort(model) else None
        )
        if openai_reasoning and effort:
            payload["reasoning_effort"] = effort
        if not _openai_flag("OPENAI_USE_DEFAULT_SAMPLING", base_url):
            payload["temperature"] = 0.45 if temperature is None else temperature
            _add_seed(payload)
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        hist_n = len(history_messages) if history_messages else 0
        logger.info(
            "Gemma text request → %s persona=%s history_msgs=%d",
            url,
            self.active_persona_id,
            hist_n,
        )
        resp = await self._post(url, payload, headers)
        if resp.status_code >= 400:
            logger.error(
                "Gemma text HTTP %s model=%s body=%.500s",
                resp.status_code,
                model,
                resp.text,
            )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            content = str(content)
        text = content.strip()
        if raw:
            return text
        for marker in ("[reply]:", "[heard]:"):
            if marker.lower() in text.lower():
                parsed = parse_heard_reply(text)
                return parsed.reply or text
        return text

    async def complete_text_n(
        self,
        user_text: str,
        n: int,
        *,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        output_role: str | None = None,
        reasoning_effort: str | None = None,
    ) -> list[str]:
        """``n`` raw samples of the same text turn in ONE request (vLLM ``n``: shared prefill, parallel decode).
        Used by best-of-N panel drafting (TALKSHOW_PANEL_BEST_OF); same payload as complete_text otherwise."""
        system = system_prompt if system_prompt is not None else self._system_prompt
        messages = self._build_messages(system, history_messages=history_messages, tail_user=user_text)
        base_url, model, api_key = self._text_endpoint()
        token_limit = max_tokens or self._cfg.max_tokens
        openai_reasoning = _openai_flag("OPENAI_USE_MAX_COMPLETION_TOKENS", base_url)
        if openai_reasoning:
            token_limit = max(
                token_limit, int(os.environ.get("OPENAI_MIN_COMPLETION_TOKENS", "4096"))
            )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            ("max_completion_tokens" if openai_reasoning else "max_tokens"): token_limit,
            "n": int(n),
        }
        effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip() or (
            reasoning_effort if _takes_reasoning_effort(model) else None
        )
        if openai_reasoning and effort:
            payload["reasoning_effort"] = effort
        if not _openai_flag("OPENAI_USE_DEFAULT_SAMPLING", base_url):
            payload["temperature"] = 0.45 if temperature is None else temperature
            _add_seed(payload)
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        logger.info("Gemma text request (n=%d) → %s persona=%s", n, url, self.active_persona_id)
        resp = await self._post(url, payload, headers)
        if resp.status_code >= 400:
            logger.error("Gemma text HTTP %s model=%s body=%.500s", resp.status_code, model, resp.text)
        resp.raise_for_status()
        out = []
        for ch in resp.json()["choices"]:
            content = ch["message"]["content"]
            content = content if isinstance(content, str) else str(content)
            out.append(content.strip())
        return out
