"""Speaker-perspective chat history (fixes models answering as another panelist)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from agent.show.show_history import ShowHistory

LINES = [
    ("host", "Welcome."),
    ("human", "Should we freeze hiring?"),
    ("host", "Great question. Ryan?"),
    ("commentator", "Bad idea."),
    ("host", "Amy, you're up."),
    ("guest", "I disagree with Ryan."),
]


def _history() -> ShowHistory:
    h = ShowHistory()
    for role, text in LINES:
        h.append(role, text)
    return h


class PerspectiveTests(unittest.TestCase):
    def test_own_lines_are_unlabelled_assistant_turns(self) -> None:
        msgs = _history().prior_messages(perspective="commentator")
        self.assertEqual(
            msgs,
            [
                {"role": "user", "content": "Lessac (host): Welcome.\nHuman guest: Should we freeze hiring?\nLessac (host): Great question. Ryan?"},
                {"role": "assistant", "content": "Bad idea."},
                {"role": "user", "content": "Lessac (host): Amy, you're up.\nAmy (Guest): I disagree with Ryan."},
            ],
        )

    def test_other_panelists_never_appear_as_assistant(self) -> None:
        for role in ("guest", "commentator"):
            msgs = _history().prior_messages(perspective=role)
            for m in msgs:
                if m["role"] == "assistant":
                    self.assertNotIn("(", m["content"])  # no "Name (Role):" label
        guest_view = _history().prior_messages(perspective="guest")
        self.assertIn("Ryan (Commentator): Bad idea.", guest_view[0]["content"])

    def test_roles_alternate(self) -> None:
        for role in ("host", "guest", "commentator"):
            msgs = _history().prior_messages(perspective=role)
            for a, b in zip(msgs, msgs[1:]):
                self.assertNotEqual(a["role"], b["role"])

    def test_legacy_format_available(self) -> None:
        legacy = _history().prior_messages()
        self.assertEqual(legacy[3], {"role": "assistant", "content": "Ryan (Commentator): Bad idea."})
        with mock.patch.dict(os.environ, {"TALKSHOW_HISTORY_FORMAT": "legacy"}):
            self.assertEqual(_history().prior_messages(perspective="guest"), legacy)


class ClientMergeTests(unittest.TestCase):
    def _client(self):
        try:
            from agent.adapters.gemma_mm_client import GemmaMMClient
        except ImportError as exc:  # livekit.agents missing locally
            self.skipTest(f"gemma client import: {exc}")
        return GemmaMMClient.__new__(GemmaMMClient)

    def test_instruction_folds_into_trailing_user_turn(self) -> None:
        client = self._client()
        hist = _history().prior_messages(perspective="commentator")
        msgs = client._build_messages("SYS", history_messages=hist, tail_user="Your turn.")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertTrue(msgs[-1]["content"].endswith("\n\nYour turn."))
        self.assertEqual(hist[-1]["content"].count("Your turn."), 0)  # caller's list untouched

    def test_audio_tail_becomes_multipart(self) -> None:
        client = self._client()
        hist = _history().prior_messages(perspective="commentator")
        audio = [{"type": "audio_url", "audio_url": {"url": "data:"}}, {"type": "text", "text": "hint"}]
        msgs = client._build_messages("SYS", history_messages=hist, tail_user=audio)
        self.assertEqual(msgs[-1]["content"][0]["type"], "text")
        self.assertEqual(msgs[-1]["content"][1:], audio)

    def test_appends_when_history_ends_with_own_line(self) -> None:
        client = self._client()
        hist = _history().prior_messages(perspective="guest")  # ends with Amy's own line
        msgs = client._build_messages("SYS", history_messages=hist, tail_user="Go.")
        self.assertEqual(msgs[-1], {"role": "user", "content": "Go."})


if __name__ == "__main__":
    unittest.main()


class ReasoningEffortTests(unittest.TestCase):
    def payload(self, model: str, effort: str | None, env: dict[str, str]) -> dict:
        import asyncio
        from types import SimpleNamespace

        try:
            from agent.adapters.gemma_mm_client import GemmaMMClient
        except ImportError as exc:  # livekit.agents missing locally
            self.skipTest(f"gemma client import: {exc}")
        client = GemmaMMClient.__new__(GemmaMMClient)
        client._cfg = SimpleNamespace(
            model=model, base_url="https://api.example/v1", api_key="k", max_tokens=256
        )
        client._system_prompt = "SYS"
        client._output_logger = None
        response = mock.Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "Hi."}}]}
        client._http = mock.Mock(post=mock.AsyncMock(return_value=response))
        with mock.patch.dict(os.environ, env, clear=False):
            asyncio.run(client.complete_text("Go.", raw=True, reasoning_effort=effort))
        return client._http.post.await_args.kwargs["json"]

    OPENAI = {"OPENAI_USE_MAX_COMPLETION_TOKENS": "1", "OPENAI_REASONING_EFFORT": ""}

    def test_luna_gets_the_per_call_effort(self) -> None:
        self.assertEqual(self.payload("gpt-6-luna", "low", self.OPENAI)["reasoning_effort"], "low")
        self.assertEqual(self.payload("gpt-6-luna", "none", self.OPENAI)["reasoning_effort"], "none")

    def test_other_models_get_no_effort(self) -> None:
        self.assertNotIn("reasoning_effort", self.payload("gpt-5o", "low", self.OPENAI))
        local = {"OPENAI_USE_MAX_COMPLETION_TOKENS": "", "OPENAI_REASONING_EFFORT": ""}
        self.assertNotIn("reasoning_effort", self.payload("gpt-6-luna", "low", local))

    def test_env_overrides_every_call(self) -> None:
        env = {**self.OPENAI, "OPENAI_REASONING_EFFORT": "medium"}
        self.assertEqual(self.payload("gpt-6-luna", "none", env)["reasoning_effort"], "medium")

