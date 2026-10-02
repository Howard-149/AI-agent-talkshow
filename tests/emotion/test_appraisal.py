"""Separate PAD appraisal: prompt, parse, scheduling, and prompt-mode switches."""

from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

from agent.config import load_scenario
from agent.emotion import appraisal as A
from agent.emotion.state import (
    affect_descriptor,
    mood_output_lines,
    pad_prompt_block,
    parse_pad_appraisal,
)
from agent.show.show_history import ShowHistory


class _ListLog:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log(self, event: str, **fields) -> None:
        self.rows.append({"event": event, **fields})


class _FakeClient:
    def __init__(self, replies: dict[str, str], delay_s: float = 0.0) -> None:
        self.replies = replies
        self.delay_s = delay_s
        self.calls: list[dict] = []

    async def complete_text(self, user_text, **kw):
        self.calls.append({"user": user_text, **kw})
        await asyncio.sleep(self.delay_s)
        for name, reply in self.replies.items():
            if f"inner emotional reactions of {name}" in kw.get("system_prompt", ""):
                return reply
        return "[pad]: unmoved 0 0 0"


@dataclass
class _FakeData:
    scenario: object = field(default_factory=load_scenario)
    show_history: ShowHistory = field(default_factory=ShowHistory)
    role_emotion: dict = field(default_factory=dict)
    role_pad: dict = field(default_factory=dict)
    role_pad_ts: dict = field(default_factory=dict)
    role_pad_neighbors: dict = field(default_factory=dict)
    pending_appraisals: dict = field(default_factory=dict)
    turn_log: _ListLog = field(default_factory=_ListLog)
    room_name: str = "test"
    runtime: object = None


_SEPARATE = {"TALKSHOW_PAD_DELTA_SCALE": "1", "TALKSHOW_EMOTION_SOURCE": "pad"}


class ParseTests(unittest.TestCase):
    def test_word_and_delta(self) -> None:
        self.assertEqual(
            parse_pad_appraisal("[pad]: stung -0.30 0.25 -0.20"),
            ("stung", (-0.3, 0.25, -0.2)),
        )
        self.assertEqual(parse_pad_appraisal("[pad]: 0.1 0 0"), (None, (0.1, 0.0, 0.0)))
        # Axis labels are not mistaken for the feeling word.
        self.assertEqual(
            parse_pad_appraisal("[pad]: P 0.2 A 0.1 D 0"), (None, (0.2, 0.1, 0.0))
        )

    def test_descriptor_tie_keeps_all_labels(self) -> None:
        s = affect_descriptor((0.4, 0.3, 0.1), ["Anger", "Surprise", "Contempt"])
        self.assertIn("a mix of anger, surprise and contempt", s)

    def test_descriptor_has_no_numbers(self) -> None:
        self.assertEqual(affect_descriptor((0.0, 0.02, 0.0), ["Neutral"]), "calm and neutral")
        s = affect_descriptor((-0.5, 0.5, -0.4), ["Anger", "Anger", "Contempt"])
        self.assertIn("anger with a touch of contempt", s)
        self.assertIn("on the back foot", s)
        self.assertFalse(any(ch.isdigit() for ch in s))


class PromptModeTests(unittest.TestCase):
    def test_separate_mode_drops_pad_contract(self) -> None:
        data = _FakeData()
        data.role_pad["guest"] = (0.3, 0.3, 0.0)
        data.role_pad_neighbors["guest"] = ["Happiness", "Surprise", "Happiness"]
        with mock.patch.dict(os.environ, _SEPARATE):
            block = pad_prompt_block(data, "guest")
            self.assertNotIn("[pad]", block)
            self.assertNotIn("P=", block)
            self.assertIn("happiness", block)
            self.assertEqual(mood_output_lines(), "")


    def test_llm_rollback_disables_appraisal(self) -> None:
        with mock.patch.dict(os.environ, {"TALKSHOW_EMOTION_SOURCE": "llm"}):
            self.assertFalse(A.appraisal_enabled())
            self.assertIn("[emotion]", mood_output_lines())


class AppraisalPromptTests(unittest.TestCase):
    def test_prompt_contents(self) -> None:
        data = _FakeData()
        data.show_history.append("guest", "I think we should keep the mid-level hires.")
        data.show_history.append("commentator", "Amy, that's sentimental, not strategy.")
        system, user = A.build_appraisal_prompt(
            data,
            "guest",
            A.Utterance("commentator", "Amy, that's sentimental, not strategy."),
        )
        self.assertIn("Amy", system)
        self.assertIn("a reply to what Amy said above", user)  # Ryan answers Amy
        self.assertIn("keep the mid-level hires", user)  # context line
        self.assertEqual(user.count("that's sentimental"), 1)  # new line not duplicated
        self.assertTrue(user.rstrip().endswith("<ΔD>"))

    def test_host_uses_appraisal_profile(self) -> None:
        data = _FakeData()
        system, _ = A.build_appraisal_prompt(data, "host", A.Utterance("guest", "Hi all."))
        self.assertIn("professionally neutral", system)

    def test_render_exchange_lists_every_line(self) -> None:
        _, user = A.render_appraisal_prompt(
            name="Tom",
            persona="A shopkeeper.",
            mood="calm and neutral",
            pad=(0.0, 0.0, 0.0),
            context_lines=[],
            new_lines=[("Tom", "Morning."), ("John", "Hi Tom.")],
        )
        self.assertIn("New exchange:", user)
        self.assertIn('John: "Hi Tom."', user)
        self.assertNotIn("Use 0 on an axis only if", user)

    def test_reply_to_listener_is_appraised_as_exchange(self) -> None:
        data = _FakeData()
        data.show_history.append("guest", "We should keep the mid-level hires.")
        data.show_history.append("commentator", "That's sentimental, not strategy.")
        utt = A.Utterance(
            "commentator",
            "That's sentimental, not strategy.",
            context=A.snapshot_context(data, "That's sentimental, not strategy."),
        )
        # Default protocol "reply": only the reply is appraised; own line stays context.
        _, guest_user = A.build_appraisal_prompt(data, "guest", utt)
        self.assertIn("a reply to what Amy said above", guest_user)
        self.assertIn("Amy: We should keep the mid-level hires.", guest_user)  # context
        self.assertNotIn('Amy: "We should keep', guest_user)  # not appraised
        _, host_user = A.build_appraisal_prompt(data, "host", utt)
        self.assertIn("New line:", host_user)  # host was not the one being answered
        env = "TALKSHOW_PAD_APPRAISAL_PROTOCOL"
        with mock.patch.dict(os.environ, {env: "exchange"}):
            _, ex = A.build_appraisal_prompt(data, "guest", utt)
            self.assertIn("Amy's own line and the reply to it", ex)
            self.assertIn('Amy: "We should keep the mid-level hires."', ex)
        with mock.patch.dict(os.environ, {env: "line"}):
            _, off = A.build_appraisal_prompt(data, "guest", utt)
            self.assertNotIn("reply to", off)

    def test_procedural_host_lines_do_not_steal_the_reply(self) -> None:
        data = _FakeData()
        data.show_history.append("guest", "Institutional knowledge matters.")
        data.show_history.append("host", "Ryan, you're up.", procedural=True)
        text = "Amy's point is valid, but velocity matters more."
        data.show_history.append("commentator", text)
        utt = A.Utterance("commentator", text, context=A.snapshot_context(data, text))
        self.assertEqual(A.replied_line("guest", utt, utt.context)[0], "guest")
        self.assertIsNone(A.replied_line("host", utt, utt.context))
        _, guest_user = A.build_appraisal_prompt(data, "guest", utt)
        self.assertIn("a reply to what Amy said above", guest_user)
        self.assertIn("Institutional knowledge matters.   ← the line being answered", guest_user)
        _, host_user = A.build_appraisal_prompt(data, "host", utt)
        self.assertNotIn("reply to", host_user)

    def test_substantive_host_line_is_still_a_reply_target(self) -> None:
        data = _FakeData()
        data.show_history.append("host", "Ryan, is the AI plan realistic?")
        data.show_history.append("commentator", "Not at that scale.")
        utt = A.Utterance(
            "commentator", "Not at that scale.",
            context=A.snapshot_context(data, "Not at that scale."),
        )
        self.assertIsNotNone(A.replied_line("host", utt, utt.context))

    def test_context_is_snapshotted_at_schedule_time(self) -> None:
        data = _FakeData()
        data.show_history.append("guest", "first")
        data.show_history.append("commentator", "second")
        utt = A.Utterance("commentator", "second", context=A.snapshot_context(data, "second"))
        data.show_history.append("host", "a later line")  # arrives before the task runs
        _, user = A.build_appraisal_prompt(data, "guest", utt)
        self.assertNotIn("a later line", user)

    def test_prompt_asks_for_valence_and_empathy(self) -> None:
        _, user = A.render_appraisal_prompt(
            name="Amy", persona="x", mood="calm and neutral", pad=(0, 0, 0),
            context_lines=[], new_lines=[("Human guest", "My mentor was fired.")],
        )
        self.assertIn("<pleasant|unpleasant|neutral>", user)
        self.assertIn("suffering, loss, or unfair treatment lowers P", user)
        self.assertIn("saddened on their behalf | unpleasant |", user)

    def test_listeners_exclude_speaker(self) -> None:
        data = _FakeData()
        self.assertNotIn("guest", A.appraisal_listeners(data, "guest"))
        self.assertIn("host", A.appraisal_listeners(data, "guest"))
        self.assertNotIn("human", A.appraisal_listeners(data, "human"))
        self.assertIn("guest", A.appraisal_listeners(data, "human"))


class SchedulingTests(unittest.TestCase):
    def test_schedule_apply_and_wait(self) -> None:
        async def run() -> _FakeData:
            data = _FakeData()
            client = _FakeClient(
                {"Amy": "[pad]: stung -0.30 0.25 -0.20", "Lessac": "[pad]: alert 0 0.1 0"},
                delay_s=0.01,
            )
            data.runtime = SimpleNamespace(gemma_client=client)
            text = "Amy, that's sentimental, not strategy."
            data.show_history.append("commentator", text)
            with mock.patch.dict(os.environ, _SEPARATE), mock.patch(
                "agent.emotion.pad_pipeline.get_pad_knn", return_value=None
            ):
                listeners = A.schedule_appraisals(data, "commentator", text)
                self.assertNotIn("commentator", listeners)
                ok = await A.await_pending_appraisal(data, "guest", timeout_s=2.0)
                self.assertTrue(ok)
                await A.await_pending_appraisal(data, "host", timeout_s=2.0)
            self.assertTrue(all(c["raw"] for c in client.calls))
            self.assertTrue(all(c["temperature"] == 0.9 for c in client.calls))
            return data

        data = asyncio.run(run())
        self.assertEqual(
            tuple(round(x, 4) for x in data.role_pad["guest"]), (-0.3, 0.25, -0.2)
        )
        rows = [r for r in data.turn_log.rows if r["event"] == "pad_appraisal"]
        guest = next(r for r in rows if r["role"] == "guest")
        self.assertEqual(guest["word"], "stung")
        self.assertTrue(guest["ok"])
        maps = [r for r in data.turn_log.rows if r["event"] == "pad_map"]
        self.assertTrue(all(r["reason"] == "appraisal" for r in maps))

    def test_per_role_order_is_preserved(self) -> None:
        async def run() -> _FakeData:
            data = _FakeData()
            data.runtime = SimpleNamespace(
                gemma_client=_FakeClient({"Amy": "[pad]: x 0.1 0 0"})
            )
            with mock.patch.dict(os.environ, _SEPARATE), mock.patch(
                "agent.emotion.pad_pipeline.get_pad_knn", return_value=None
            ):
                for line in ("one", "two", "three"):
                    data.show_history.append("commentator", line)
                    A.schedule_appraisals(data, "commentator", line)
                await A.await_pending_appraisal(data, "guest", timeout_s=2.0)
            return data

        data = asyncio.run(run())
        seqs = [
            r["seq"]
            for r in data.turn_log.rows
            if r["event"] == "pad_appraisal" and r["role"] == "guest"
        ]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), 3)

    def test_expression_relaxes_speaker_arousal_only(self) -> None:
        data = _FakeData()
        data.role_pad["guest"] = (0.4, 0.5, -0.2)
        with mock.patch.dict(os.environ, _SEPARATE):
            self.assertAlmostEqual(A.relax_after_expression(data, "guest"), 0.44)
            self.assertEqual(data.role_pad["guest"][0], 0.4)
            self.assertEqual(data.role_pad["guest"][2], -0.2)
            self.assertIsNone(A.relax_after_expression(data, "human"))
        with mock.patch.dict(os.environ, {**_SEPARATE, "TALKSHOW_PAD_EXPRESSION_AROUSAL_DECAY": "0"}):
            self.assertIsNone(A.relax_after_expression(data, "guest"))
        row = next(r for r in data.turn_log.rows if r["event"] == "pad_expression")
        self.assertEqual(row["arousal_before"], 0.5)

    def test_disabled_without_loop_or_in_llm_mode(self) -> None:
        data = _FakeData()
        data.runtime = SimpleNamespace(gemma_client=_FakeClient({}))
        with mock.patch.dict(os.environ, _SEPARATE):
            self.assertEqual(A.schedule_appraisals(data, "guest", "hi"), [])  # no loop
        with mock.patch.dict(os.environ, {"TALKSHOW_EMOTION_SOURCE": "llm"}):
            self.assertEqual(A.schedule_appraisals(data, "guest", "hi"), [])


if __name__ == "__main__":
    unittest.main()
