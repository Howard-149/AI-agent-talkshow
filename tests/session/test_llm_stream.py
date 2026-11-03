"""LLM streaming: a live-drafted line's first sentence starts TTS and the avatar early."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

import httpx

# agent.agents first: session_handoff and agents.host import each other, and the
# cycle only resolves in this order (as in agent.main).
import agent.agents  # noqa: F401
from agent.adapters import gemma_mm_client as gmc
from agent.adapters import response_parser as rp
from agent.multi_agent.types import AgentDraft
from agent.panel import panel_speech
from agent.session import session_handoff as sh
from agent.show_graph.actuators import LiveActuators

LINE = "The freeze is a mistake. Senior engineers leave first. Then the juniors follow."
FIRST = "The freeze is a mistake."


def _sse(*deltas: str) -> bytes:
    events = [{"choices": [{"delta": {"role": "assistant"}}]}]
    events += [{"choices": [{"delta": {"content": d}}]} for d in deltas]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return body.encode()


class SseTests(unittest.TestCase):
    def test_content_delta(self) -> None:
        line = 'data: {"choices": [{"delta": {"content": "Hi"}}]}'
        self.assertEqual(gmc.sse_content_delta(line), "Hi")

    def test_no_text(self) -> None:
        self.assertEqual(gmc.sse_content_delta(""), "")
        self.assertEqual(gmc.sse_content_delta(": keep-alive"), "")
        self.assertEqual(gmc.sse_content_delta('data: {"choices": [{"delta": {"role": "assistant"}}]}'), "")
        self.assertEqual(gmc.sse_content_delta('data: {"choices": []}'), "")
        self.assertEqual(gmc.sse_content_delta("data: not json"), "")

    def test_done(self) -> None:
        self.assertIsNone(gmc.sse_content_delta("data: [DONE]"))


class StreamTextTests(unittest.TestCase):
    """GemmaMMClient._stream_text against a mocked vLLM server."""

    def _client(self, handler: Callable[[httpx.Request], httpx.Response]) -> gmc.GemmaMMClient:
        client = gmc.GemmaMMClient.__new__(gmc.GemmaMMClient)
        client._cfg = SimpleNamespace(model="m")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client

    def test_streams_deltas_and_returns_the_whole_text(self) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, content=_sse("The freeze ", "is a mistake. ", "Senior"))

        partials: list[str] = []
        client = self._client(handler)
        out = asyncio.run(client._stream_text("http://x/v1/chat/completions", {"model": "m"}, {}, partials.append))
        self.assertEqual(out, "The freeze is a mistake. Senior")
        self.assertEqual(partials, ["The freeze ", "The freeze is a mistake. ", "The freeze is a mistake. Senior"])
        self.assertTrue(seen[0]["stream"])

    def test_a_failing_listener_keeps_the_reply(self) -> None:
        client = self._client(lambda r: httpx.Response(200, content=_sse("A. ", "B.")))

        def boom(_: str) -> None:
            raise RuntimeError("listener")

        out = asyncio.run(client._stream_text("http://x", {}, {}, boom))
        self.assertEqual(out, "A. B.")

    def test_http_error_raises(self) -> None:
        client = self._client(lambda r: httpx.Response(500, content=b"bad"))
        with self.assertRaises(httpx.HTTPStatusError):
            asyncio.run(client._stream_text("http://x", {}, {}, lambda _: None))


class SettledChunkTests(unittest.TestCase):
    def test_first_sentence_settles_once_the_next_starts(self) -> None:
        self.assertIsNone(sh.settled_first_chunk("The freeze is a mis"))
        self.assertIsNone(sh.settled_first_chunk("The freeze is a mistake."))
        self.assertEqual(sh.settled_first_chunk("The freeze is a mistake. Senior eng"), FIRST)

    def test_matches_the_finished_line(self) -> None:
        self.assertEqual(sh.settled_first_chunk(LINE[:30]), sh.split_speak_sentences(LINE)[0])


ONE = "That's a massive leap, and I'm skeptical of the hidden costs."


def _prefixes(text: str) -> list[str]:
    return [text[:i] for i in range(1, len(text) + 1)]


def _settled_while_streaming(raw: str) -> list[tuple[str, str]]:
    """(prefix, first chunk) for each prefix of ``raw`` whose streamed view settles one."""
    out = []
    for prefix in _prefixes(raw):
        first = sh.settled_first_chunk(rp.streamed_reply_so_far(prefix))
        if first is not None:
            out.append((prefix, first))
    return out


class StreamedReplyTests(unittest.TestCase):
    """The spoken reply as a panelist's output is streamed, one character at a time."""

    def test_a_one_sentence_reply_never_settles_early(self) -> None:
        for raw in (
            f"[reply]: {ONE}\n[next]: host",
            f"[reply]: {ONE}\n[next:host]",
            f"[reply]: {ONE}\n[next]: host\n[emotion]: happy",
            f"[reply]: {ONE}\n[emotion:happy]\n[next]: host",
            f"[reply]: {ONE}\n[next]: host\n[pad]: P=0.10 A=0.20 D=0.00",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(_settled_while_streaming(raw), [])

    def test_a_longer_reply_settles_on_its_first_sentence(self) -> None:
        for raw in (
            f"[reply]: {LINE}\n[next]: host",
            f"[reply]: {LINE}\n[next:host]",
            f"[emotion]: calm\n[reply]: {LINE}\n[next]: host",
        ):
            with self.subTest(raw=raw):
                prefix, first = _settled_while_streaming(raw)[0]
                self.assertEqual(first, FIRST)
                final = rp.parse_host_speech(raw).reply
                start = _ahead(rp.streamed_reply_so_far(prefix))
                self.assertEqual(sh.adopt_streamed_start(start, final)[1], "")

    def test_half_written_tags_are_cut(self) -> None:
        for tail in ("[", "[ne", "[next:ho", "[next]", "[next]:", "[next]: ho", "[emotion]: hap"):
            with self.subTest(tail=tail):
                got = rp.streamed_reply_so_far(f"[reply]: {LINE}\n{tail}")
                self.assertNotIn("[", got)
                self.assertNotIn(":", got)
                self.assertEqual(len(sh.split_speak_sentences(got)), 3)


def _ahead(text: str, *, texts: dict[str, str] | None = None) -> sh.PreparedLineStart:
    return sh.PreparedLineStart(
        role="guest", text=text, locale="en", texts=texts or {"en": text}, emotion="Neutral",
        first_chunk=sh.split_speak_sentences(text)[0], slot="ahead-s",
        task=mock.Mock(done=lambda: True),
    )


class AdoptTests(unittest.TestCase):
    def test_same_first_sentence_binds_the_final_line(self) -> None:
        ahead, why = sh.adopt_streamed_start(_ahead(LINE[:30]), LINE)
        self.assertEqual(why, "")
        assert ahead is not None
        self.assertEqual((ahead.text, ahead.texts), (LINE, {"en": LINE}))

    def test_changed_first_sentence_is_dropped(self) -> None:
        start = _ahead(LINE[:30])
        start.discard = mock.Mock()  # type: ignore[method-assign]
        ahead, why = sh.adopt_streamed_start(start, "A whole other opening. And more.")
        self.assertEqual((ahead, why), (None, "changed"))
        start.discard.assert_called_once()

    def test_translated_start_is_dropped(self) -> None:
        start = _ahead(LINE[:30], texts={"en": "Übersetzt. Mehr"})
        start.discard = mock.Mock()  # type: ignore[method-assign]
        self.assertEqual(sh.adopt_streamed_start(start, LINE), (None, "translated"))

    def test_nothing_started(self) -> None:
        self.assertEqual(sh.adopt_streamed_start(None, LINE), (None, "not_started"))


class ActuatorStreamTests(unittest.TestCase):
    """LiveActuators with a fake streamed draft, line start and speech."""

    def setUp(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.data = SimpleNamespace(
            show_history=SimpleNamespace(lines=[]),
            turn_log=SimpleNamespace(log=lambda e, **f: self.events.append((e, f))),
            room_name="r", role_emotion={}, dialogue_example_index=None,
        )
        self.final = LINE
        self.started: list[str] = []
        self.spoken_with: list[Any] = []
        self.streamed: list[bool] = []

        async def draft(data: Any, *, speak_role: str, step: str,
                        on_reply: Callable[[str], None] | None = None) -> AgentDraft:
            self.streamed.append(on_reply is not None)
            if on_reply is not None:
                for n in (10, 30, 50):
                    on_reply(LINE[:n])
                    await asyncio.sleep(0)
            return AgentDraft(role=speak_role, speech=self.final, next_role="host", emotion=None,
                              dialogue_example_index=None, history_version=0)

        async def prepare(data: Any, *, speak_role: str, text: str, step: str,
                          emotion_tag: str | None = None) -> sh.PreparedLineStart:
            self.started.append(text)
            return sh.PreparedLineStart(
                role=speak_role, text=text, locale="en", texts={"en": text}, emotion="Neutral",
                first_chunk=sh.split_speak_sentences(text)[0], slot=f"ahead-{step}",
                task=asyncio.ensure_future(asyncio.sleep(0)),
            )

        async def speak(session: Any, data: Any, **kw: Any) -> None:
            self.spoken_with.append(kw.get("ahead"))

        for patch in (
            mock.patch.object(panel_speech, "draft_panelist", draft),
            mock.patch.object(sh, "prepare_line_start", prepare),
            mock.patch.object(sh, "speak_panel_line", speak),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _show(self, env: dict[str, str]) -> None:
        act = LiveActuators(None, self.data, None)  # type: ignore[arg-type]

        async def go() -> None:
            line = await act.generate_panelist_line("guest", "speech_guest_0")
            await act.speak({"role": "guest", "text": line.text, "step": "speech_guest_0", "kind": "panelist"})

        with mock.patch.dict(os.environ, env):
            asyncio.run(go())

    def _names(self) -> list[str]:
        return [e for e, _ in self.events]

    def test_the_first_sentence_starts_once_written_and_plays_with_the_line(self) -> None:
        self._show({"TALKSHOW_LLM_STREAM": "1"})
        self.assertEqual(self.streamed, [True])
        self.assertEqual(self.started, [LINE[:30]])  # once, as soon as it settled
        self.assertEqual(len(self.spoken_with), 1)
        self.assertIsNotNone(self.spoken_with[0])
        self.assertEqual(self.spoken_with[0].text, LINE)
        self.assertIn("stream_ahead_started", self._names())

    def test_a_changed_opening_speaks_live(self) -> None:
        self.final = "Actually, a different opening here. And more after it."
        self._show({"TALKSHOW_LLM_STREAM": "1"})
        self.assertEqual(self.spoken_with, [None])
        reasons = [f["reason"] for e, f in self.events if e == "stream_ahead_dropped"]
        self.assertEqual(reasons, ["changed"])

    def test_off_by_default(self) -> None:
        self._show({"TALKSHOW_LLM_STREAM": ""})
        self.assertEqual(self.streamed, [False])
        self.assertEqual(self.started, [])
        self.assertEqual(self.spoken_with, [None])


if __name__ == "__main__":
    unittest.main()
