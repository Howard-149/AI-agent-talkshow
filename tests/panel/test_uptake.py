"""Uptake planner + scaffold fix (agent/panel/uptake.py) — no LiveKit, no model."""

from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent.panel import uptake
from agent.show.show_history import ShowHistory

ROOT = Path(__file__).resolve().parents[2]


class FakeGemma:
    def __init__(self, reply: str = "- $170 billion over budget\n- Vogtle\n- Addressed: Ryan", delay: float = 0.0):
        self.reply, self.delay, self.calls = reply, delay, 0

    async def complete_text(self, *_a, **_kw) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.reply


def fake_data(lines: list[tuple[str, str]], gemma: FakeGemma | None = None):
    h = ShowHistory()
    for role, text in lines:
        h.append(role, text, speaker=role)
    return SimpleNamespace(show_history=h, runtime=SimpleNamespace(gemma_client=gemma or FakeGemma()),
                           turn_log=None, room_name="t")


class FlagTests(unittest.TestCase):
    def test_default_off(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("TALKSHOW_UPTAKE_PLANNER", "TALKSHOW_PANEL_SCAFFOLD_FIX"):
                os.environ.pop(k, None)
            self.assertFalse(uptake.uptake_planner_enabled())
            self.assertFalse(uptake.scaffold_fix_enabled())


class LiteModeTests(unittest.TestCase):
    def test_modes(self) -> None:
        with mock.patch.dict(os.environ, {"TALKSHOW_UPTAKE_PLANNER": "lite"}):
            self.assertTrue(uptake.uptake_planner_enabled())
            self.assertTrue(uptake.uptake_planner_lite())
            self.assertNotIn("[engages]", uptake.plan_block("- $170 billion"))
            self.assertIn("$170 billion", uptake.plan_block("- $170 billion"))
        with mock.patch.dict(os.environ, {"TALKSHOW_UPTAKE_PLANNER": "1"}):
            self.assertTrue(uptake.uptake_planner_enabled())
            self.assertFalse(uptake.uptake_planner_lite())
            self.assertIn("[engages]", uptake.plan_block("- x"))


class ScaffoldFixTests(unittest.TestCase):
    def test_fix_on_the_real_panel_prompt(self) -> None:
        from agent.multi_agent.worker import PersonaWorker
        from tests.panel.test_concrete import FakeGemma as CaptureGemma, live_data

        g = CaptureGemma(["[reply]: Vogtle is one plant, not the rule."])
        with mock.patch.dict(os.environ, {"TALKSHOW_PANEL_SCAFFOLD_FIX": "1"}):
            os.environ.pop("TALKSHOW_UPTAKE_PLANNER", None)
            asyncio.run(PersonaWorker("commentator").draft(live_data(g), step="test"))
        _kind, user, kw = g.calls[0]
        text = kw["system_prompt"] + "\n" + user
        for gone in ("NOT your conversation partner", "address whoever last spoke", "respond to the latest speaker"):
            self.assertNotIn(gone, text)

    def test_off_or_not_first_answer_is_untouched(self) -> None:
        data = fake_data([("human", "x"), ("host", "y"), ("guest", "z")])  # a panelist already answered
        with mock.patch.dict(os.environ, {"TALKSHOW_PANEL_SCAFFOLD_FIX": "1"}):
            s, p, info = uptake.adapt_panel_prompt(data, "SYS FROZEN", "USR")
        self.assertEqual((s, p, info), ("SYS FROZEN", "USR", {}))


class EngagesTests(unittest.TestCase):
    def test_strip_before_and_after_reply(self) -> None:
        from agent.adapters.response_parser import parse_host_speech

        for raw in ("[engages]: $170bn — systemic\n[reply]: It is the business model.\n[next]: host",
                    "[reply]: It is the business model.\n[engages]: x — y\n[next]: host"):
            reply = parse_host_speech(uptake.strip_engages(raw)).reply
            self.assertNotIn("engages", reply.lower())
            self.assertIn("business model", reply)


class FirstAnswerTests(unittest.TestCase):
    def test_human_then_host(self) -> None:
        self.assertEqual(uptake.first_answer_to_human(fake_data([("human", "H"), ("host", "ack")])), "H")

    def test_after_panelist(self) -> None:
        self.assertIsNone(uptake.first_answer_to_human(fake_data([("human", "H"), ("commentator", "C")])))


class PlannerTests(unittest.TestCase):
    def test_start_once_then_await(self) -> None:
        async def run():
            g = FakeGemma(delay=0.05)
            data = fake_data([("human", "H")], g)
            with mock.patch.dict(os.environ, {"TALKSHOW_UPTAKE_PLANNER": "1"}):
                uptake.start_uptake_plan(data, "H")
                uptake.start_uptake_plan(data, "H")  # dedup by text
                plan, waited = await uptake.await_uptake_plan(data, "H")
            return g.calls, plan, waited
        calls, plan, waited = asyncio.run(run())
        self.assertEqual(calls, 1)
        self.assertIn("$170 billion", plan)
        self.assertLess(waited, 1.0)

    def test_late_plan_times_out_empty(self) -> None:
        async def run():
            data = fake_data([("human", "H")], FakeGemma(delay=5))
            with mock.patch.dict(os.environ, {"TALKSHOW_UPTAKE_PLANNER": "1"}), \
                    mock.patch.object(uptake, "PLAN_WAIT_TIMEOUT_S", 0.05):
                uptake.start_uptake_plan(data, "H")
                plan, _ = await uptake.await_uptake_plan(data, "H")
                data.uptake_plan_task.cancel()
            return plan
        self.assertEqual(asyncio.run(run()), "")

    def test_flag_off_never_calls_model(self) -> None:
        async def run():
            g = FakeGemma()
            data = fake_data([("human", "H")], g)
            os.environ.pop("TALKSHOW_UPTAKE_PLANNER", None)
            uptake.start_uptake_plan(data, "H")
            return g.calls
        self.assertEqual(asyncio.run(run()), 0)


if __name__ == "__main__":
    unittest.main()
