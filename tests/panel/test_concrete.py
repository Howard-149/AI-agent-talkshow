"""Concrete panel flags (agent/panel/concrete.py) — no LiveKit, no model."""

from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent.panel import concrete

ROOT = Path(__file__).resolve().parents[2]
FLAGS = ("TALKSHOW_DIALOGUE_LIBRARY", "TALKSHOW_PANEL_CONCRETE", "TALKSHOW_PANEL_BEST_OF")


class FakeGemma:
    """complete_text → one raw reply; complete_text_n → the given candidates (records the prompts)."""

    def __init__(self, cands: list[str]):
        self.cands, self.calls = cands, []

    async def complete_text(self, user_text, **kw) -> str:
        self.calls.append(("one", user_text, kw))
        return self.cands[0]

    async def complete_text_n(self, user_text, n, **kw) -> list[str]:
        self.calls.append(("n", user_text, n, kw))
        return self.cands[:n]


def live_data(gemma):
    """A real TalkShowData (production scenario) after a short exchange and a human turn."""
    from agent.config import load_scenario
    from agent.data import TalkShowData

    data = TalkShowData(scenario=load_scenario(), runtime=SimpleNamespace(gemma_client=gemma))  # type: ignore[arg-type]
    data.room_name, data.turn_log = "test", None
    for role, text in [
        ("host", "Tonight: should we build more nuclear power plants?"),
        ("guest", "Nuclear is the only firm low-carbon power we can scale."),
        ("commentator", "Firm, sure. Affordable, no. Look at the cost overruns."),
        ("host", "Let us bring in our guest from the audience."),
        ("human", "Vogtle ran 17 billion dollars over budget. Why would the next plant be different?"),
    ]:
        data.show_history.append(role, text, speaker=role)
    return data


def run_draft(gemma: FakeGemma, env: dict):
    from agent.multi_agent.worker import PersonaWorker

    with mock.patch.dict(os.environ, env):
        for k in FLAGS:
            if k not in env:
                os.environ.pop(k, None)
        return asyncio.run(PersonaWorker("guest").draft(live_data(gemma), step="test"))


class FlagTests(unittest.TestCase):
    def test_default_off(self) -> None:
        with mock.patch.dict(os.environ, {}):
            for k in FLAGS:
                os.environ.pop(k, None)
            self.assertIsNone(concrete.dialogue_library_override())
            self.assertFalse(concrete.concrete_instruction_enabled())
            self.assertEqual(concrete.best_of(), 1)
        with mock.patch.dict(os.environ, {"TALKSHOW_PANEL_BEST_OF": "0", "TALKSHOW_DIALOGUE_LIBRARY": ""}):
            self.assertEqual(concrete.best_of(), 1)
            self.assertIsNone(concrete.dialogue_library_override())


class WorkerTests(unittest.TestCase):
    GENERIC = "[reply]: I hear Ryan's point, and I think this is a really complex issue with good points on both sides."
    CONCRETE_LINE = "[reply]: Ontario shut its last coal plant in 2014 by running Bruce and Pickering flat out."

    def test_off_is_production(self) -> None:
        g = FakeGemma([self.GENERIC])
        d = run_draft(g, {})
        self.assertEqual([c[0] for c in g.calls], ["one"])
        self.assertNotIn(concrete.CONCRETE, g.calls[0][1])
        self.assertIn("complex issue", d.speech)

    def test_instruction_and_library(self) -> None:
        g = FakeGemma([self.GENERIC])
        run_draft(g, {"TALKSHOW_PANEL_CONCRETE": "1", "TALKSHOW_DIALOGUE_LIBRARY": "panel_concrete"})
        _kind, user, kw = g.calls[0]
        self.assertTrue(user.rstrip().endswith(concrete.CONCRETE))
        self.assertIn("Rue de Rivoli", kw["system_prompt"])  # a panel_concrete example, absent from panel.yaml

    def test_best_of_picks_the_concrete_candidate(self) -> None:
        g = FakeGemma([self.GENERIC, self.CONCRETE_LINE, self.GENERIC, self.GENERIC])
        d = run_draft(g, {"TALKSHOW_PANEL_BEST_OF": "4"})
        self.assertEqual(g.calls[0][0], "n")
        self.assertEqual(g.calls[0][2], 4)
        self.assertIn("Ontario", d.speech)


if __name__ == "__main__":
    unittest.main()
