"""Policy, cancellation races, memory and actual graph routing, without API cost."""
import asyncio
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agent.config import load_scenario
from agent.data import TalkShowData
from agent.floor.reactive_policy import FloorRequest, ReactivePolicy, parse_request
from agent.multi_agent.reactive_session import ActiveTurn, ReactiveSession
from agent.multi_agent.reactive_demo_plan import DEMO_FOLLOWUPS, generate_demo_line
from agent.session.ghost_flags import is_ai_only_participant
from agent.ui.control_events import parse_control_payload

CLAIM = "AI answers teach students automatically."
TAIL = "Students can submit them without checking. An unplayed secret is purple aardvark."
RAW = json.dumps({"action": "interrupt", "kind": "rebuttal", "reason": "Understanding requires practice",
                  "evidence": CLAIM, "speech": "I disagree, Amy. Students need practice to understand an answer."})


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ReactivePolicy(min_speech_s=3, cooldown_s=15)
        self.req = parse_request(RAW, role="commentator", epoch=8)
        self.context = dict(epoch=8, speaker="guest", heard=CLAIM, elapsed_s=5, now=100,
                            eligible_roles=["guest", "commentator"])

    def decide(self, req=None, **kwargs):
        return self.policy.decide(req or self.req, **{**self.context, **kwargs})

    def test_grounded_rebuttal_granted(self):
        self.assertEqual(self.decide().status, "granted")

    def test_minimum_time(self):
        self.assertEqual(self.decide(elapsed_s=1).rule, "minimum_speaking_time")

    def test_cooldown(self):
        self.policy.commit_cut(self.req, now=99)
        self.assertEqual(self.decide().rule, "cooldown")
        self.assertEqual(self.decide(now=115).status, "granted")

    def test_human_overrides_cooldown_and_minimum(self):
        self.policy.commit_cut(self.req, now=99)
        human = FloorRequest("human", 8, "interrupt", source="human")
        self.assertEqual(self.decide(human, elapsed_s=0).rule, "human_priority")

    def test_stale_human_also_rejected(self):
        self.assertEqual(self.decide(FloorRequest("human", 7, "interrupt", source="human")).rule, "stale_epoch")

    def test_unheard_evidence_rejected(self):
        self.assertEqual(self.decide(heard="A different statement.").rule, "evidence_not_heard")

    def test_self_request_rejected(self):
        self.assertEqual(self.decide(speaker="commentator").rule, "ineligible_role")

    def test_addition_queues(self):
        req = FloorRequest("commentator", 8, "interrupt", kind="addition")
        self.assertEqual(self.decide(req).rule, "wait_for_turn_end")

    def test_queue_mode_and_human_pending(self):
        self.assertEqual(self.decide(allow_ai_cut=False).status, "queued")
        self.assertEqual(self.decide(human_pending=True).rule, "human_priority")

    def test_malformed_json_fails_closed(self):
        for raw in ["not json", "[]", '{"action":"steal"}', '{"action":"interrupt","speech":123}']:
            with self.assertRaises((ValueError, TypeError)):
                parse_request(raw, role="commentator", epoch=8)

    def test_model_cannot_choose_identity_or_epoch(self):
        obj = json.loads(RAW); obj.update(role="human", epoch=999, source="human")
        req = parse_request(json.dumps(obj), role="commentator", epoch=8)
        self.assertEqual((req.role, req.epoch, req.source), ("commentator", 8, "llm"))

    def test_control_invalid_modes_and_nonobjects(self):
        for raw in [b'[]', b'null', b'{"type":"reactive_demo","mode":"evil"}']:
            self.assertIsNone(parse_control_payload(raw))
        self.assertEqual(parse_control_payload(b'{"type":"human_interrupt"}').type, "human_interrupt")
        self.assertEqual(parse_control_payload(b'{"type":"reactive_stop"}').type, "reactive_stop")

    def test_ai_only_identity_requires_true_token_metadata(self):
        self.assertTrue(is_ai_only_participant(SimpleNamespace(metadata='{"aiOnly":true}')))
        for raw in ['{"aiOnly":"true"}', '[]', 'invalid', '{}']:
            self.assertFalse(is_ai_only_participant(SimpleNamespace(metadata=raw)))
        self.assertFalse(is_ai_only_participant(None))


class FakeHandle:
    def __init__(self, owner, text):
        self.owner = owner; self.text = text; self.interrupted = False
        self.event = asyncio.Event()
        self.task = asyncio.create_task(self._play())

    async def _play(self):
        self.owner.concurrent += 1
        self.owner.max_concurrent = max(self.owner.max_concurrent, self.owner.concurrent)
        try:
            await asyncio.wait_for(self.event.wait(), timeout=0.07)
        except asyncio.TimeoutError:
            pass
        finally:
            self.owner.concurrent -= 1

    def interrupt(self, **_):
        self.interrupted = True; self.event.set()
        return self

    async def wait_for_playout(self):
        await self.task

    def done(self):
        return self.task.done()

    def exception(self):
        return None


class FakeSession:
    def __init__(self):
        self.spoken = []; self.concurrent = 0; self.max_concurrent = 0

    def say(self, text, **_):
        self.spoken.append(text)
        return FakeHandle(self, text)


class FakeGemma:
    def __init__(self, *, invalid=False, delayed=False):
        self.prompts = []; self.started = asyncio.Event()
        self.invalid = invalid; self.delayed = delayed

    async def complete_text(self, prompt, **kwargs):
        self.prompts.append(prompt); self.started.set()
        if self.delayed:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                # Deliberately badly behaved transport delivers after cancellation.
                return RAW
        await asyncio.sleep(0.012)
        return "invalid" if self.invalid else RAW


class FakeAct:
    def __init__(self): self.grants = []; self.generated = []
    def listeners(self, speaker): return []
    def relax(self, role): pass
    async def appraise(self, utt, role): return {"ok": True}
    async def grant_panelist(self, role, reason): self.grants.append((role, reason))
    async def generate_panelist_line(self, role, step):
        self.generated.append(role)
        return SimpleNamespace(text="A fresh reply considers the entire completed speech. I prefer guided practice.")
    async def generate_demo_line(self, beat):
        self.generated.append(beat.role)
        return {"role": beat.role, "text": "A fresh reply considers the played discussion. I prefer guided practice.",
                "step": beat.step, "kind": beat.kind, "procedural": False}


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = FakeGemma()
        self.data = TalkShowData(scenario=load_scenario(), runtime=SimpleNamespace(gemma_client=self.client))
        self.session = FakeSession()
        ctrl = SimpleNamespace(panel_speaker_roles=lambda: ["guest", "commentator"])
        self.broker = ReactiveSession(self.session, self.data, ctrl)
        self.broker.policy.min_speech_s = 0
        self.events = []
        self.broker.log = lambda event, **fields: self.events.append({"event": event, **fields})
        self.broker.emit = AsyncMock()
        self.act = FakeAct()
        self.line = {"role": "guest", "text": CLAIM + " " + TAIL, "kind": "panelist", "step": "test"}
        self.patches = [patch("agent.session.session_handoff.switch_to_role", new_callable=AsyncMock),
                        patch("agent.ui.ui_events.emit_role_active", new_callable=AsyncMock),
                        patch("agent.ui.ui_events.emit_transcript", new_callable=AsyncMock),
                        patch("agent.ui.ui_events.emit_queue_state", new_callable=AsyncMock),
                        patch("agent.ui.ui_events.emit_floor_grant", new_callable=AsyncMock)]
        for p in self.patches: p.start()

    async def asyncTearDown(self):
        for p in self.patches: p.stop()

    async def test_ai_cut_and_no_unplayed_memory(self):
        result = await self.broker.speak(self.act, self.line, extra_utterances=[])
        self.assertIn(("commentator", "ai_interrupt"), self.act.grants)
        self.assertEqual(self.data.show_history.lines[0].text, CLAIM)
        self.assertNotIn("purple aardvark", " ".join(x.text for x in self.data.show_history.lines))
        self.assertNotIn("purple aardvark", self.client.prompts[0])
        self.assertEqual(self.session.max_concurrent, 1)
        self.assertEqual(result["extra_turns"], 1)
        self.assertFalse(self.data.show_history.lines[0].procedural)

    async def test_ai_only_ignores_human_interrupt(self):
        self.broker.ai_only = True
        self.broker.active = ActiveTurn(epoch=8, role="guest", step="test")
        await self.broker.request_human()
        self.assertIsNone(self.broker.active.winner)
        self.assertFalse(self.broker.pending_human)
        self.broker.emit.assert_not_awaited()

    async def test_operator_stop_is_not_human_takeover(self):
        self.broker.ai_only = self.broker.demo_active = True
        run = asyncio.create_task(self.broker.speak(self.act, self.line, extra_utterances=[]))
        while self.broker.active is None:
            await asyncio.sleep(0.005)
        await self.broker.stop_demo()
        result = await run
        self.assertTrue(result["stopped"])
        self.assertFalse(result["human"])
        self.assertEqual(self.data.floor_next_speaker, "host")
        self.assertFalse(any(e.get("role") == "human" for e in self.events))
        self.assertEqual(self.session.concurrent, 0)

    async def test_operator_stop_cancels_live_draft(self):
        self.broker.ai_only = self.broker.demo_active = True
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def delayed(beat):
            started.set()
            try:
                await asyncio.sleep(5)
            finally:
                cancelled.set()
        self.act.generate_demo_line = delayed
        run = asyncio.create_task(self.broker.speak(self.act, self.line, extra_utterances=[]))
        await started.wait()
        await self.broker.stop_demo()
        result = await run
        self.assertTrue(result["stopped"])
        self.assertTrue(cancelled.is_set())
        self.assertFalse(result["human"])
        self.assertIsNone(self.broker.demo_generation)

    async def test_ai_only_round_has_no_human_history_or_final_grant(self):
        from agent.multi_agent.reactive_demo import run_reactive_demo
        from agent.show_graph.actuators import LiveActuators

        self.broker.ai_only = True
        self.data.reactive_session = self.broker
        self.data.floor_next_speaker = "host"
        live_act = LiveActuators(self.session, self.data, self.broker.controller)
        async def post(*args, **kwargs):
            result = await self.broker.speak(self.act, self.line, extra_utterances=[])
            await live_act.after_line(result["line"])
            return {"last_outcome": "close"}
        self.data.show_session = SimpleNamespace(running=True, post=post)
        with patch("agent.ui.ui_events.emit_floor_grant", new_callable=AsyncMock) as grant:
            ok = await run_reactive_demo(self.session, self.data, self.broker.controller, mode="live")
        self.assertTrue(ok)
        self.assertFalse(any(x.role_id == "human" for x in self.data.show_history.lines))
        self.assertFalse(any(c.args[0] == "human" for c in grant.await_args_list))
        self.assertEqual(self.data.floor_next_speaker, "host")
        self.assertFalse(self.data.panel_chain_running)
        self.assertFalse(self.broker.demo_active)

    async def test_ai_only_idle_never_opens_legacy_floor(self):
        from agent.ui.control_handler import idle_topic_loop

        class TwoTicks:
            def __init__(self): self.ticks = 0
            def is_set(self): return self.ticks >= 2
            async def wait(self):
                self.ticks += 1
                if self.ticks == 1: raise asyncio.TimeoutError
        data = SimpleNamespace(shutdown_event=TwoTicks(),
                               reactive_session=SimpleNamespace(ai_only=True))
        controller = SimpleNamespace(is_host_moderated_mode=lambda: True)
        with patch("agent.ui.control_handler.session_is_active", return_value=True), \
             patch("agent.ui.control_handler.room_has_humans", return_value=True), \
             patch("agent.ui.control_handler.run_host_moderation_from_queue", new_callable=AsyncMock) as moderate:
            await idle_topic_loop(None, data, controller, room=SimpleNamespace(name="ai-only-test"))
        moderate.assert_not_awaited()

    async def test_queue_waits_and_regenerates(self):
        self.broker.mode = "queue"; self.broker.demo_active = True
        await self.broker.speak(self.act, self.line, extra_utterances=[])
        self.assertEqual(self.data.show_history.lines[0].text, self.line["text"])
        self.assertEqual(self.act.generated, ["commentator", "guest", "host", "commentator", "guest", "host"])
        self.assertIn(("commentator", "queued_after_turn"), self.act.grants)
        self.assertFalse(any(reason == "ai_interrupt" for _, reason in self.act.grants))

    async def test_baseline_has_no_intent_requests(self):
        self.broker.mode = "baseline"; self.broker.demo_active = True
        await self.broker.speak(self.act, self.line, extra_utterances=[])
        self.assertFalse(self.client.prompts)
        self.assertEqual(self.act.generated, ["commentator", "guest", "host", "commentator", "guest", "host"])

    async def test_natural_demo_is_bounded_and_closes_with_host(self):
        self.broker.demo_active = True
        result = await self.broker.speak(self.act, self.line, extra_utterances=[])
        self.assertEqual(result["roles"], ["guest", "commentator", "guest", "host", "commentator", "guest", "host"])
        self.assertTrue(result["closed"])
        self.assertEqual(result["line"]["kind"], "close")
        self.assertEqual(result["line"]["step"], "demo_host_wrap")
        self.assertEqual(sum(reason == "ai_interrupt" for _, reason in self.act.grants), 1)
        self.assertEqual(self.session.max_concurrent, 1)
        self.assertFalse(self.data.hand_raise_queue.roles())
        self.assertNotIn("purple aardvark", " ".join(x.text for x in self.data.show_history.lines))
        self.assertEqual(self.act.generated, [b.role for b in DEMO_FOLLOWUPS])

    async def test_no_ai_intent_still_gets_normal_reply_not_a_forced_cut(self):
        self.broker.demo_active = True
        self.client.invalid = True
        result = await self.broker.speak(self.act, self.line, extra_utterances=[])
        self.assertTrue(result["closed"])
        self.assertEqual(len(result["roles"]), 7)
        self.assertFalse(any(reason == "ai_interrupt" for _, reason in self.act.grants))
        self.assertEqual(self.data.show_history.lines[0].text, self.line["text"])

    async def test_human_can_stop_amys_later_response(self):
        self.broker.demo_active = True
        run = asyncio.create_task(self.broker.speak(self.act, self.line, extra_utterances=[]))
        while self.broker.active is None or self.broker.active.step != "demo_amy_response":
            await asyncio.sleep(0.005)
        await self.broker.request_human()
        result = await run
        self.assertTrue(result["human"])
        self.assertEqual(result["roles"], ["guest", "commentator", "guest"])
        self.assertNotIn("host", self.act.generated)
        self.assertFalse(self.data.hand_raise_queue.roles())

    async def test_human_cancels_generation_gap_without_playing_next_voice(self):
        self.broker.demo_active = True
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def delayed(beat):
            started.set()
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        self.act.generate_demo_line = delayed
        run = asyncio.create_task(self.broker.speak(self.act, self.line, extra_utterances=[]))
        await started.wait()
        await self.broker.request_human()
        result = await run
        self.assertTrue(result["human"])
        self.assertTrue(cancelled.is_set())
        self.assertEqual(result["roles"], ["guest", "commentator"])
        self.assertIsNone(self.broker.demo_generation)

    async def test_parent_cancellation_cleans_up_generation(self):
        self.broker.demo_active = True
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def delayed(beat):
            started.set()
            try:
                await asyncio.sleep(5)
            finally:
                cancelled.set()
        self.act.generate_demo_line = delayed
        run = asyncio.create_task(self.broker.speak(self.act, self.line, extra_utterances=[]))
        await started.wait()
        run.cancel()
        with self.assertRaises(asyncio.CancelledError): await run
        self.assertTrue(cancelled.is_set())
        self.assertIsNone(self.broker.demo_generation)
        self.assertIsNone(self.broker.active)

    async def test_generated_reply_uses_only_committed_history(self):
        self.data.show_history.append("commentator", "Learning needs practice.")
        self.client.complete_text = AsyncMock(return_value="I learn from worked examples, too.")
        line = await generate_demo_line(self.data, DEMO_FOLLOWUPS[0])
        self.assertEqual(line["role"], "guest")
        kwargs = self.client.complete_text.call_args.kwargs
        self.assertIn("Learning needs practice.", str(kwargs["history_messages"]))
        self.assertNotIn("purple aardvark", str(kwargs))
        self.assertIn("Respond directly to Ryan", kwargs["system_prompt"])

    async def test_empty_continuation_fails_not_canned_dialogue(self):
        self.client.complete_text = AsyncMock(return_value=" ")
        with self.assertRaises(ValueError):
            await generate_demo_line(self.data, DEMO_FOLLOWUPS[0])

    async def test_human_preempts_delayed_ai_and_late_result_is_stale(self):
        self.client.delayed = True
        run = asyncio.create_task(self.broker.speak(self.act, self.line, extra_utterances=[]))
        await self.client.started.wait()
        await self.broker.request_human()
        result = await run
        self.assertTrue(result["human"])
        self.assertEqual(self.data.floor_next_speaker, "human")
        self.assertFalse(self.act.grants)
        self.assertEqual(self.session.max_concurrent, 1)

    async def test_late_poll_after_turn_end_rejected_by_epoch(self):
        turn = ActiveTurn(epoch=7, role="guest", step="old", started=0, heard=[CLAIM])
        self.broker.epoch = 7
        self.broker.active = None
        await self.broker._poll(turn, allow_ai_cut=True)
        self.assertTrue(any(e.get("rule") == "stale_epoch" for e in self.events))
        self.assertIsNone(turn.winner)

    async def test_human_supersedes_reserved_ai_winner(self):
        winner = parse_request(RAW, role="commentator", epoch=7)
        turn = ActiveTurn(epoch=7, role="guest", step="old", winner=winner)
        self.broker.active = turn
        await self.broker.request_human()
        self.assertEqual(turn.winner.role, "human")
        self.assertTrue(any(e.get("rule") == "superseded_by_human" for e in self.events))

    async def test_invalid_model_response_never_cuts(self):
        self.client.invalid = True
        await self.broker.speak(self.act, self.line, extra_utterances=[])
        self.assertEqual(self.data.show_history.lines[0].text, self.line["text"])
        self.assertFalse(self.act.grants)

    async def test_cancel_clears_audio_and_poll(self):
        run = asyncio.create_task(self.broker.speak(self.act, self.line, extra_utterances=[]))
        await asyncio.sleep(0.02)
        run.cancel()
        with self.assertRaises(asyncio.CancelledError): await run
        self.assertIsNone(self.broker.active)
        self.assertEqual(self.session.concurrent, 0)
        self.assertFalse(self.data.speak_line_busy)

    async def test_epoch_is_monotonic_across_demos(self):
        self.broker.mode = "baseline"
        await self.broker.speak(self.act, self.line, extra_utterances=[])
        first = self.broker.epoch
        await self.broker.speak(self.act, self.line, extra_utterances=[])
        self.assertGreater(self.broker.epoch, first)


class GraphIntegrationTests(unittest.TestCase):
    def test_contextual_wrap_finishes_without_canned_human_handoff(self):
        from test_show_graph import FakeActuators, run

        class ClosedAct(FakeActuators):
            def uses_reactive(self, line): return line.get("kind") == "panelist"
            async def speak_reactive(self, line, extra):
                return {"line": {"role": "host", "text": "Thanks, Amy and Ryan.",
                                 "kind": "close", "step": "demo_host_wrap"},
                        "human": False, "closed": True, "roles": ["guest", "commentator", "host"],
                        "steps": ["demo_host_wrap"], "appraisals": [], "extra_turns": 2}

        act = ClosedAct(floor=["guest"])
        result = run(act, entry="panel")
        self.assertEqual(result["outcome"], "close")
        self.assertEqual(act.kinds("after_line"), ["close"])
        self.assertFalse(act.kinds("speak"))
        self.assertFalse(act.kinds("hand_raise"))

    def test_graph_never_prerecords_full_reactive_draft(self):
        from test_show_graph import FakeActuators, run
        from agent.show_graph.state import FloorPick

        class ReactiveAct(FakeActuators):
            def uses_reactive(self, line): return line.get("kind") == "panelist"
            def record_line(self, line):
                if line.get("kind") == "panelist":
                    raise AssertionError("graph committed a full unplayed draft")
                return super().record_line(line)
            async def speak_reactive(self, line, extra):
                self.history.append((line["role"], "Only this sentence played.", False))
                return {"line": {**line, "text": "Only this sentence played.", "next_tag": "human"},
                        "human": False, "roles": [line["role"]], "steps": ["reactive"],
                        "appraisals": [], "extra_turns": 0}

        act = ReactiveAct(picks=[FloorPick("guest", "test")])
        result = run(act, human_event={"heard": "AI homework?", "parsed": None})
        self.assertEqual(result["outcome"], "human")
        self.assertEqual([x[1] for x in act.history if x[0] == "guest"], ["Only this sentence played."])


if __name__ == "__main__": unittest.main()
