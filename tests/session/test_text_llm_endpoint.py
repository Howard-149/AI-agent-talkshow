"""Mode 2: text turns on a separate endpoint (e.g. the OpenAI API), audio stays on vLLM."""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.adapters.gemma_mm_client import GemmaMMClient
from agent.config import LocaleLLMConfig

VLLM = "http://127.0.0.1:8000/v1"
OAI = "https://api.openai.com/v1"


def _cfg(**text: str) -> LocaleLLMConfig:
    return LocaleLLMConfig(
        model="google/gemma-4-E4B-it",
        base_url=VLLM,
        api_key="EMPTY",
        max_tokens=256,
        reply_language="English",
        include_heard_summary=True,
        **text,
    )


def _client(cfg: LocaleLLMConfig, content: str = "[heard]: hi\n[reply]: hello") -> tuple[GemmaMMClient, AsyncMock]:
    response = SimpleNamespace(
        status_code=200,
        raise_for_status=Mock(),
        json=Mock(return_value={"choices": [{"message": {"content": content}}]}),
        text="",
    )
    client = GemmaMMClient(cfg, system_prompt="test")
    post = AsyncMock(return_value=response)
    client._http = SimpleNamespace(post=post)
    return client, post


CLEAN_ENV = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith(("OPENAI_USE_", "OPENAI_REASONING", "TALKSHOW_TEXT_LLM_", "TALKSHOW_DECISIONS_"))
}


@patch.dict("os.environ", CLEAN_ENV, clear=True)
class TextEndpointTests(unittest.TestCase):
    def test_mode1_text_uses_vllm(self) -> None:
        client, post = _client(_cfg())
        asyncio.run(client.complete_text("go"))
        url, payload, headers = post.call_args.args[0], post.call_args.kwargs["json"], post.call_args.kwargs["headers"]
        self.assertEqual(url, f"{VLLM}/chat/completions")
        self.assertEqual(payload["model"], "google/gemma-4-E4B-it")
        self.assertIn("max_tokens", payload)
        self.assertIn("temperature", payload)
        self.assertEqual(headers["Authorization"], "Bearer EMPTY")

    def test_mode2_text_goes_to_openai_with_its_flags(self) -> None:
        client, post = _client(_cfg(text_base_url=OAI, text_model="gpt-5.5", text_api_key="sk-test"))
        asyncio.run(client.complete_text("go"))
        url, payload, headers = post.call_args.args[0], post.call_args.kwargs["json"], post.call_args.kwargs["headers"]
        self.assertEqual(url, f"{OAI}/chat/completions")
        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertIn("max_completion_tokens", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertEqual(headers["Authorization"], "Bearer sk-test")

    def test_mode2_decisions_stay_on_vllm(self) -> None:
        client, post = _client(_cfg(text_base_url=OAI, text_model="gpt-5.5", text_api_key="sk-test"))
        asyncio.run(client.complete_text("who speaks?", raw=True, decision=True, reasoning_effort="low"))
        url, payload, headers = post.call_args.args[0], post.call_args.kwargs["json"], post.call_args.kwargs["headers"]
        self.assertEqual(url, f"{VLLM}/chat/completions")
        self.assertEqual(payload["model"], "google/gemma-4-E4B-it")
        self.assertIn("max_tokens", payload)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(headers["Authorization"], "Bearer EMPTY")

    def test_mode2_decisions_on_openai_when_opted_out(self) -> None:
        client, post = _client(_cfg(text_base_url=OAI, text_model="gpt-5.5", text_api_key="sk-test"))
        with patch.dict("os.environ", {"TALKSHOW_DECISIONS_ON_MAIN_LLM": "0"}):
            asyncio.run(client.complete_text("who speaks?", raw=True, decision=True))
        self.assertEqual(post.call_args.args[0], f"{OAI}/chat/completions")

    def test_mode2_n_samples_go_to_openai(self) -> None:
        client, post = _client(_cfg(text_base_url=OAI, text_model="gpt-5.5", text_api_key="sk-test"))
        asyncio.run(client.complete_text_n("go", 3))
        self.assertEqual(post.call_args.args[0], f"{OAI}/chat/completions")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "gpt-5.5")

    def test_mode2_audio_stays_on_vllm(self) -> None:
        client, post = _client(_cfg(text_base_url=OAI, text_model="gpt-5.5", text_api_key="sk-test"))
        asyncio.run(client.complete_from_wav(b"RIFF"))
        url, payload, headers = post.call_args.args[0], post.call_args.kwargs["json"], post.call_args.kwargs["headers"]
        self.assertEqual(url, f"{VLLM}/chat/completions")
        self.assertEqual(payload["model"], "google/gemma-4-E4B-it")
        self.assertEqual(headers["Authorization"], "Bearer EMPTY")

    def test_transcription_uses_its_own_system_prompt(self) -> None:
        from agent.show.show_context import host_audio_system_prompt

        client, post = _client(_cfg(), content="What's your favorite sport?")
        client.set_persona("host", "You are Claire, the host.")
        parsed = asyncio.run(
            client.complete_from_wav(b"RIFF", system_prompt=host_audio_system_prompt(), temperature=0.0)
        )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["content"], host_audio_system_prompt())
        self.assertNotIn("Claire", payload["messages"][0]["content"])
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(parsed.heard, "")  # no tag: the STT adapter falls back to the reply
        self.assertEqual(parsed.reply, "What's your favorite sport?")

    def test_default_audio_call_keeps_the_persona(self) -> None:
        client, post = _client(_cfg())
        client.set_persona("host", "You are Claire, the host.")
        asyncio.run(client.complete_from_wav(b"RIFF"))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["content"], "You are Claire, the host.")
        self.assertEqual(payload["temperature"], 0.5)

    def test_explicit_flag_overrides_openai_default(self) -> None:
        client, post = _client(_cfg(text_base_url=OAI, text_model="gpt-4.1", text_api_key="sk-test"))
        with patch.dict("os.environ", {"OPENAI_USE_DEFAULT_SAMPLING": "0"}):
            asyncio.run(client.complete_text("go"))
        self.assertIn("temperature", post.call_args.kwargs["json"])

    def test_load_config_reads_text_env(self) -> None:
        from agent.config import load_config

        env = {
            "TALKSHOW_TEXT_LLM_BASE_URL": OAI,
            "TALKSHOW_TEXT_LLM_MODEL": "gpt-5.5",
            "TALKSHOW_TEXT_LLM_API_KEY": "sk-test",
        }
        with patch.dict("os.environ", env):
            llm = load_config().locales["en"].llm
        self.assertEqual(llm.text_endpoint, (OAI, "gpt-5.5", "sk-test"))
        self.assertNotEqual(llm.base_url, OAI)


if __name__ == "__main__":
    unittest.main()
