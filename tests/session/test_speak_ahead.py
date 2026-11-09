"""Speak-ahead: a drafted line's first chunk starts during the intro and plays when due."""

from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

# agent.agents first: session_handoff and agents.host import each other, and the
# cycle only resolves in this order (as in agent.main).
import agent.agents  # noqa: F401
from agent.config import LocaleTTSConfig
from agent.emotion import appraisal
from agent.emotion.appraisal import Utterance
from agent.floor import host_floor
from agent.floor.floor_parser import HandRaiseResult
from agent.multi_agent.types import AgentDraft
from agent.panel import panel_speech
from agent.session import session_handoff as sh
from agent.show_graph import actuators as act_mod
from agent.show_graph.actuators import LiveActuators
from agent.show_graph.state import FloorPick

LINE = "The freeze is a mistake. Senior engineers leave first. Then the juniors follow."
TTS = LocaleTTSConfig(engine="piper", model_path="voice.onnx", sample_rate=22050)
FLAGS = {"TALKSHOW_DRAFT_AHEAD": "1", "TALKSHOW_SPEAK_AHEAD": "1", "TALKSHOW_PRERENDER_CANNED": "1"}


class Log:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []

    def log(self, event: str, **fields: Any) -> None:
        self.rows.append((event, fields))

    def events(self, event: str) -> list[dict[str, Any]]:
        return [f for e, f in self.rows if e == event]


def _chunk(text: str) -> sh._PreparedChunk:
    return sh._PreparedChunk(
        text=text, pcm=b"", sample_rate=22050, clip_path=None, bake_ms=0,
        frames=None, fps=25.0, duration_sec=None, url=None, tts_ms=0,
    )


def _first_chunk(text: str = LINE) -> str:
    return sh._plan_chunks(text, TTS, audio_via_session=True)[0][0]


class SpeakPathTests(unittest.TestCase):
    """_speak_avatar_chunked with fake chunk prepare / play (no TTS, DyStream or LiveKit)."""

    def setUp(self) -> None:
        self.prepared: list[tuple[int, str]] = []
        self.played: list[sh._PreparedChunk] = []
        self.cleaned: list[str] = []

        async def prepare(**kw: Any) -> sh._PreparedChunk:
            self.prepared.append((kw["chunk_idx"], kw["slot"]))
            return _chunk(kw["chunk_text"])

        async def play(session: Any, chunk: sh._PreparedChunk, **kw: Any) -> None:
            self.played.append(chunk)

        env = {k: v for k, v in os.environ.items() if k != "TALKSHOW_AVATAR_CHUNK_STREAM"}
        for patch in (
            mock.patch.object(sh, "_prepare_chunk", prepare),
            mock.patch.object(sh, "_play_prepared_chunk", play),
            mock.patch.object(sh, "cleanup_speech_slots", self.cleaned.extend),
            mock.patch.dict(os.environ, env, clear=True),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _speak(self, make_ahead: Any, *, emotion: str = "Neutral") -> tuple[Log, sh.PreparedLineStart]:
        log = Log()

        async def go() -> sh.PreparedLineStart:
            ahead = make_ahead()
            await sh._speak_avatar_chunked(
                None, SimpleNamespace(room_name="r"), bridge=object(), speak_role="guest",
                text=LINE, step="speech_guest_1", tts_cfg=TTS, turn_log=log, video_pub=None,
                use_lk_video=False, emit_ui=False, emotion=emotion, locale="en", ahead=ahead,
            )
            return ahead

        return log, asyncio.run(go())

    @staticmethod
    def _ahead(work: Any, *, emotion: str = "Neutral") -> Any:
        def make() -> sh.PreparedLineStart:
            return sh.PreparedLineStart(
                role="guest", text=LINE, locale="en", texts={"en": LINE}, emotion=emotion,
                first_chunk=_first_chunk(), slot="ahead-speech_guest_1",
                task=asyncio.ensure_future(work()),
            )

        return make

    def test_line_plays_the_chunk_started_ahead(self) -> None:
        started = _chunk(_first_chunk())

        async def ready() -> sh._PreparedChunk:
            return started

        log, _ = self._speak(self._ahead(ready))
        self.assertIs(self.played[0], started)
        self.assertEqual([i for i, _ in self.prepared], [1, 2])  # only the later chunks render live
        self.assertEqual(len(log.events("speak_ahead_used")), 1)
        self.assertIn("ahead-speech_guest_1", self.cleaned)

    def test_changed_emotion_drops_the_ahead_chunk_and_renders_it_live(self) -> None:
        async def never() -> sh._PreparedChunk:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        log, ahead = self._speak(self._ahead(never, emotion="Happy"))
        self.assertTrue(ahead.task.cancelled())
        self.assertEqual(self.prepared, [(0, "c0"), (1, "c1"), (2, "c2")])
        self.assertEqual([f["reason"] for f in log.events("speak_ahead_dropped")], ["changed"])
        self.assertEqual(log.events("speak_ahead_used"), [])

    def test_failed_ahead_renders_the_first_chunk_live(self) -> None:
        async def fail() -> sh._PreparedChunk:
            raise RuntimeError("cosyvoice down")

        self._speak(self._ahead(fail))
        self.assertEqual(self.prepared[0], (0, "c0"))
        self.assertEqual([c.text for c in self.played], sh._plan_chunks(LINE, TTS, audio_via_session=True)[0])

    def test_ahead_cancelled_elsewhere_does_not_cancel_the_line(self) -> None:
        def make() -> sh.PreparedLineStart:
            async def never() -> sh._PreparedChunk:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            ahead = self._ahead(never)()
            ahead.discard()
            return ahead

        self._speak(make)
        self.assertEqual(self.prepared[0], (0, "c0"))
        self.assertEqual(len(self.played), 3)


class PrepareLineStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[dict[str, Any]] = []

        async def prepare(**kw: Any) -> sh._PreparedChunk:
            self.calls.append(kw)
            return _chunk(kw["chunk_text"])

        for patch in (
            mock.patch.object(sh, "_prepare_chunk", prepare),
            mock.patch.object(sh, "get_avatar_bridge", lambda: object()),
            mock.patch.object(sh, "load_persona_tts", lambda role, config, locale="en": TTS),
            mock.patch.object(sh, "get_avatar_video_publisher", lambda locale: None),
            mock.patch.object(sh, "lk_video_enabled", lambda: False),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _start(self, **data: Any) -> sh.PreparedLineStart | None:
        fields = dict(needed_locales=frozenset({"en"}), runtime=SimpleNamespace(config=None),
                      room_name="r", turn_log=None, role_emotion={})
        fields.update(data)

        async def go() -> sh.PreparedLineStart | None:
            ahead = await sh.prepare_line_start(
                SimpleNamespace(**fields), speak_role="guest", text=LINE, step="speech_guest_1"
            )
            if ahead is not None:
                await ahead.task
            return ahead

        return asyncio.run(go())

    def test_starts_the_first_chunk_in_its_own_slot(self) -> None:
        ahead = self._start()
        assert ahead is not None
        self.assertEqual(ahead.first_chunk, _first_chunk())
        self.assertEqual(ahead.texts, {"en": LINE})
        self.assertEqual([(c["chunk_idx"], c["slot"]) for c in self.calls], [(0, "ahead-speech_guest_1")])

    def test_nothing_to_start_with_two_viewer_locales(self) -> None:
        self.assertIsNone(self._start(needed_locales=frozenset({"en", "zh"})))
        self.assertEqual(self.calls, [])


class _ActuatorCase(unittest.TestCase):
    """LiveActuators with a fake draft, speak-ahead start and speech."""

    def setUp(self) -> None:
        self.log = Log()
        self.data = SimpleNamespace(
            show_history=SimpleNamespace(lines=[]), turn_log=self.log, room_name="r",
            role_emotion={}, dialogue_example_index=None,
        )
        self.started: list[str] = []
        self.spoken_with: list[Any] = []
        self.drafts = 0

        async def draft(data: Any, *, speak_role: str, step: str) -> AgentDraft:
            self.drafts += 1
            return AgentDraft(role=speak_role, speech=LINE, next_role="host", emotion=None,
                              dialogue_example_index=None, history_version=0)

        async def prepare(data: Any, *, speak_role: str, text: str, step: str,
                          emotion_tag: str | None = None) -> sh.PreparedLineStart:
            self.started.append(text)
            return sh.PreparedLineStart(
                role=speak_role, text=text, locale="en", texts={"en": text}, emotion="Neutral",
                first_chunk=_first_chunk(text), slot=f"ahead-{step}",
                task=asyncio.ensure_future(asyncio.sleep(0, _chunk(text))),
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


class ActuatorTests(_ActuatorCase):
    """LiveActuators hands the speak-ahead start to the line it was started for."""

    def _show(self, *, said_meanwhile: bool = False) -> LiveActuators:
        act = LiveActuators(None, self.data, None)  # type: ignore[arg-type]

        async def go() -> None:
            act.start_panelist_draft("guest", "speech_guest_0")
            await asyncio.sleep(0.01)  # the intro plays
            if said_meanwhile:
                self.data.show_history.lines.append(SimpleNamespace(text="Hold on, one thing."))
            line = await act.generate_panelist_line("guest", "speech_guest_0")
            await act.speak({"role": "guest", "text": line.text, "step": "speech_guest_0", "kind": "panelist"})

        asyncio.run(go())
        return act

    def test_the_drafted_line_speaks_with_its_ahead_start(self) -> None:
        with mock.patch.dict(os.environ, FLAGS):
            self._show()
        self.assertEqual(self.started, [LINE])
        self.assertEqual(len(self.spoken_with), 1)
        self.assertIsNotNone(self.spoken_with[0])
        self.assertEqual(self.spoken_with[0].text, LINE)

    def test_a_stale_draft_drops_its_ahead_start(self) -> None:
        with mock.patch.dict(os.environ, FLAGS):
            self._show(said_meanwhile=True)
        self.assertEqual(self.spoken_with, [None])
        self.assertEqual(self.drafts, 2)  # the line was drafted again, live
        self.assertEqual([f["reason"] for f in self.log.events("speak_ahead_dropped")], ["stale"])

    def test_off_without_pre_rendered_host_lines(self) -> None:
        env = {**FLAGS, "TALKSHOW_PRERENDER_CANNED": "0"}
        with mock.patch.dict(os.environ, env):
            self._show()
        self.assertEqual(self.started, [])
        self.assertEqual(self.spoken_with, [None])


class EarlyDraftTests(_ActuatorCase):
    """With the early poll on, the poll's winner starts drafting during the open-floor line."""

    def setUp(self) -> None:
        super().setUp()
        self.data.hand_raise_queue = SimpleNamespace(roles=lambda: [])
        self.data.panel_priority = ["commentator", "guest"]
        self.act = LiveActuators(  # type: ignore[arg-type]
            None, self.data, SimpleNamespace(panel_speaker_roles=lambda: ["guest", "commentator"])
        )

        async def poll(data: Any, roles: list[str]) -> dict[str, HandRaiseResult]:
            return {"guest": HandRaiseResult("guest", True), "commentator": HandRaiseResult("commentator", False)}

        for patch in (
            mock.patch.object(host_floor, "poll_panel_hand_raises", poll),
            mock.patch.dict(os.environ, {**FLAGS, "TALKSHOW_POLL_DURING_OPEN_FLOOR": "1"}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _open_floor(self, pick: str) -> None:
        async def run_round(*args: Any, **kw: Any) -> FloorPick:
            await kw["prefetched_poll"]
            return FloorPick(pick, "queue_fifo")

        async def go() -> None:
            self.act.prefetch_hand_raise_poll(0)
            await asyncio.sleep(0.01)  # the open-floor line plays; the poll lands
            with mock.patch.object(host_floor, "run_hand_raise_round", run_round):
                await self.act.hand_raise_round(0, [])
            if pick != "guest":
                return
            # The intro: a canned line, then the draft-ahead start the graph makes for it.
            self.data.show_history.lines.append(SimpleNamespace(text="Amy, you're up.", procedural=True))
            self.act.start_panelist_draft("guest", "speech_guest_0")
            line = await self.act.generate_panelist_line("guest", "speech_guest_0")
            await self.act.speak({"role": "guest", "text": line.text, "step": "speech_guest_0", "kind": "panelist"})

        asyncio.run(go())

    def test_the_poll_winner_starts_before_the_intro_and_is_spoken(self) -> None:
        self._open_floor("guest")
        self.assertEqual(self.drafts, 1)  # the intro kept the early draft
        self.assertEqual(self.started, [LINE])
        self.assertIsNotNone(self.spoken_with[0])
        self.assertEqual([f["step"] for f in self.log.events("draft_ahead_early")], ["speech_guest_0"])
        self.assertEqual(self.log.events("draft_ahead_dropped"), [])

    def test_floor_going_elsewhere_drops_the_early_draft(self) -> None:
        self._open_floor("human")
        self.assertEqual(self.drafts, 1)
        self.assertEqual([f["reason"] for f in self.log.events("draft_ahead_dropped")], ["not_picked"])
        self.assertEqual([f["reason"] for f in self.log.events("speak_ahead_dropped")], ["not_picked"])
        self.assertIsNone(self.act._draft_ahead)


if __name__ == "__main__":
    unittest.main()


class HostCallOutTests(_ActuatorCase):
    """The LLM host's call-out starts the named panelist's draft once that panelist
    has appraised it, so the line is written and rendered in the mood it's spoken in."""

    CALL_OUT = Utterance("host", "Amy, does that match what you saw?", seq=7)

    def setUp(self) -> None:
        super().setUp()
        self.order: list[str] = []
        self.listening = ["guest", "commentator"]
        self.act = LiveActuators(None, self.data, None)  # type: ignore[arg-type]
        draft = panel_speech.draft_panelist

        async def logged_draft(data: Any, **kw: Any) -> AgentDraft:
            self.order.append("draft")
            return await draft(data, **kw)

        async def run_appraisal(data: Any, listener: str, utt: Utterance) -> Any:
            return SimpleNamespace(delta=None, word=None)

        def apply_appraisal(data: Any, utt: Utterance, result: Any) -> None:
            self.order.append("appraised")

        for patch in (
            mock.patch.object(panel_speech, "draft_panelist", logged_draft),
            mock.patch.object(appraisal, "run_appraisal", run_appraisal),
            mock.patch.object(appraisal, "apply_appraisal", apply_appraisal),
            mock.patch.object(LiveActuators, "listeners", lambda _self, speaker: list(self.listening)),
            mock.patch.dict(os.environ, FLAGS),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.data.show_history.lines.append(SimpleNamespace(text=self.CALL_OUT.text, procedural=False))

    def _call_out(self, *, appraise: bool = True) -> None:
        async def go() -> None:
            self.act.start_panelist_draft("guest", "speech_guest_0", after_appraisal_of=self.CALL_OUT)
            await asyncio.sleep(0.01)
            # A panelist who appraises the call-out hasn't started drafting yet.
            self.assertEqual(self.drafts, 0 if "guest" in self.listening else 1)
            if appraise:
                await self.act.appraise(self.CALL_OUT, "guest")
            line = await self.act.generate_panelist_line("guest", "speech_guest_0")
            await self.act.speak({"role": "guest", "text": line.text, "step": "speech_guest_0", "kind": "panelist"})

        asyncio.run(go())

    def test_the_draft_waits_for_the_appraisal_then_speaks_ahead(self) -> None:
        self._call_out()
        self.assertEqual(self.order, ["appraised", "draft"])
        self.assertEqual(self.started, [LINE])
        self.assertIsNotNone(self.spoken_with[0])
        (logged,) = self.log.events("draft_ahead_from_host_line")
        self.assertFalse(logged["appraisal_timed_out"])
        self.assertEqual(self.log.events("draft_ahead_dropped"), [])

    def test_a_missing_appraisal_only_delays_the_draft(self) -> None:
        with mock.patch.object(act_mod, "DRAFT_AFTER_APPRAISAL_WAIT_S", 0.05):
            self._call_out(appraise=False)
        self.assertEqual(self.drafts, 1)
        self.assertIsNotNone(self.spoken_with[0])
        self.assertTrue(self.log.events("draft_ahead_from_host_line")[0]["appraisal_timed_out"])

    def test_a_panelist_who_doesnt_appraise_drafts_at_once(self) -> None:
        self.listening = ["commentator"]
        self._call_out(appraise=False)
        self.assertEqual(self.drafts, 1)
        self.assertEqual(self.log.events("draft_ahead_from_host_line"), [])


class HostPickAheadTests(_ActuatorCase):
    """Under the LLM host, the next pick starts while a panelist's line plays: the poll
    at once, the decision once every listener has appraised the line, then the
    call-out's first sentence. The pick is used when the request made when it's due
    matches the one made ahead."""

    LINE_UTT = Utterance("commentator", "Hiring freezes backfire.", seq=3)
    CALL_OUT = "Amy, does that match what you saw?"

    def setUp(self) -> None:
        super().setUp()
        from agent.floor import host_decide
        from agent.floor.hand_raise_queue import HandRaiseQueue

        self.data.hand_raise_queue = HandRaiseQueue()
        self.data.panel_priority = ["guest", "commentator"]
        self.data.show_history.lines.append(SimpleNamespace(text=self.LINE_UTT.text, procedural=False))
        self.act = LiveActuators(  # type: ignore[arg-type]
            None,
            self.data,
            SimpleNamespace(panel_speaker_roles=lambda: ["guest", "commentator"], listen_role=lambda: "host"),
        )
        self.decided: list[bool] = []  # ahead=True / False, per decision request
        self.prompt = "Who speaks next?"

        async def poll(data: Any, roles: list[str]) -> dict[str, HandRaiseResult]:
            return {"guest": HandRaiseResult("guest", True), "commentator": HandRaiseResult("commentator", False)}

        def build(data: Any, *, queue: Any = None, **kw: Any) -> host_decide.DecideRequest:
            raised = (queue or data.hand_raise_queue).snapshot(["guest", "commentator"])["guest"].raised
            return host_decide.DecideRequest(
                prompt=f"{self.prompt} Amy raised: {raised}", system="host", history=[], tools=[],
                names={"guest": "Amy"}, raises=(), human=host_decide.Raise("Human guest", False),
                turns={}, human_must_go=kw["human_must_go"],
            )

        async def run_decide(data: Any, req: Any, *, ahead: bool = False) -> host_decide.HostDecision:
            self.decided.append(ahead)
            return host_decide.HostDecision("guest", self.CALL_OUT, host_decide.CALL_ON)

        async def run_appraisal(data: Any, listener: str, utt: Utterance) -> Any:
            return SimpleNamespace(delta=None, word=None)

        for patch in (
            mock.patch.object(host_floor, "poll_panel_hand_raises", poll),
            mock.patch.object(host_decide, "build_decide_request", build),
            mock.patch.object(host_decide, "run_decide", run_decide),
            mock.patch.object(appraisal, "run_appraisal", run_appraisal),
            mock.patch.object(appraisal, "apply_appraisal", lambda *a: None),
            mock.patch.object(LiveActuators, "listeners", lambda _self, speaker: ["guest", "host"]),
            mock.patch.dict(os.environ, {**FLAGS, "TALKSHOW_POLL_DURING_OPEN_FLOOR": "1", "TALKSHOW_HOST_DECIDE": "llm"}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _round(self, *, said_meanwhile: bool = False, prompt_now: str | None = None) -> None:
        from agent.floor.host_decide import build_decide_request, run_decide

        async def round_(*args: Any, **kw: Any) -> FloorPick:
            # As _run_llm_host_round: the poll lands in the queue, then the pick request.
            poll = await kw["prefetched_poll"] if kw["prefetched_poll"] is not None else None
            if poll:
                self.data.hand_raise_queue.apply_poll_batch(poll, self.data.panel_priority)
            if prompt_now is not None:
                self.prompt = prompt_now
            req = build_decide_request(self.data, human_must_go=False)
            decision = await host_floor._decision_made_ahead(self.data, kw["decision_ahead"], req)
            if decision is host_floor._NOT_AHEAD:
                decision = await run_decide(self.data, req)
            return FloorPick(decision.role, "host_llm", decision.line)

        async def go() -> None:
            self.act.prefetch_host_pick(1, after_appraisal_of=self.LINE_UTT)
            await asyncio.sleep(0.01)
            self.assertEqual(self.decided, [])  # waits for the listeners' appraisals
            for listener in ("guest", "host"):
                await self.act.appraise(self.LINE_UTT, listener)
            await asyncio.sleep(0.01)
            if said_meanwhile:
                self.data.show_history.lines.append(SimpleNamespace(text="Wait, one more thing.", procedural=False))
            with mock.patch.object(host_floor, "run_hand_raise_round", round_):
                pick = await self.act.hand_raise_round(1, ["commentator"])
            await self.act.speak({"role": "host", "text": pick.host_line, "step": "direct_call_host_llm_1",
                                  "kind": "direct_call", "target": pick.role})

        asyncio.run(go())

    def test_the_pick_made_ahead_is_used_and_its_call_out_spoken_ahead(self) -> None:
        self._round()
        self.assertEqual(self.decided, [True])  # one request, made ahead
        self.assertEqual(self.started, [self.CALL_OUT])
        self.assertIsNotNone(self.spoken_with[0])
        self.assertEqual(len(self.log.events("host_decide_ahead_used")), 1)

    def test_a_changed_request_is_decided_again(self) -> None:
        self._round(prompt_now="The human raised a hand.")
        self.assertEqual(self.decided, [True, False])
        self.assertEqual([f["reason"] for f in self.log.events("host_decide_ahead_dropped")], ["changed"])

    def test_anything_said_meanwhile_drops_the_pick(self) -> None:
        self._round(said_meanwhile=True)
        self.assertEqual(self.decided, [True, False])
        self.assertEqual([f["reason"] for f in self.log.events("host_decide_ahead_dropped")], ["stale"])
        self.assertEqual(self.spoken_with, [None])  # its call-out start went with it
