"""Context variants of the Think & Action panelist think prompt (agent/panel/think_prompt.py). No model."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest import mock

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent.interaction.think_action import parse_think_action
from agent.panel import think_prompt as tp
from eval.think_action_prompt import think_request

FLAGS = ("TALKSHOW_GHOST_REPLAY", "TALKSHOW_ADDRESSEE_FLOOR", "TALKSHOW_UPTAKE_PLANNER", "TALKSHOW_PANEL_SCAFFOLD_FIX",
         "TALKSHOW_DIALOGUE_LIBRARY", "TALKSHOW_PANEL_CONCRETE", "TALKSHOW_PANEL_BEST_OF", "TALKSHOW_FACT_PACK",
         "TALKSHOW_THINK_LEAN", "TALKSHOW_THINK_PERSONA_NEAR", "TALKSHOW_THINK_FOCUS")
FAST = {"TALKSHOW_UPTAKE_PLANNER": "lite", "TALKSHOW_PANEL_SCAFFOLD_FIX": "1",
        "TALKSHOW_DIALOGUE_LIBRARY": "panel_concrete", "TALKSHOW_PANEL_CONCRETE": "1"}
OPEN_FLOOR = "All right, open floor — who'd like to speak?"
RYAN_UP = "Ryan, you're up."
PREFIX = {"lines": [
    {"role_id": "host", "speaker": "Lessac", "text": "You wanted to talk about rent control — the floor is yours.",
     "procedural": True},
    {"role_id": "human", "speaker": "Human guest", "text": "Rent control keeps families in their homes.",
     "procedural": False},
    {"role_id": "host", "speaker": "Lessac", "text": OPEN_FLOOR, "procedural": True},
    {"role_id": "host", "speaker": "Lessac", "text": RYAN_UP, "procedural": True},
    {"role_id": "commentator", "speaker": "Ryan", "text": "And it freezes the supply.", "procedural": False},
    {"role_id": "host", "speaker": "Lessac", "text": "That's a sharp point about supply.", "procedural": False},
]}
HUMAN = "Ryan, rents here rose 40 percent in two years. Doesn't that change your view?"


class _Env(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"TALKSHOW_TEXT_TOKENIZER": "word"})
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in FLAGS:
            os.environ.pop(k, None)

    def req(self, role="commentator", plan="", human=HUMAN, **flags):
        with mock.patch.dict(os.environ, flags):
            return think_request(PREFIX, role, human_text=human, plan=plan)

    @staticmethod
    def text(req) -> str:
        return req["messages"][-1]["content"]

    @staticmethod
    def history_text(req) -> str:
        return "\n".join(m["content"] for m in req["messages"][1:-1]) + "\n" + req["messages"][-1]["content"]


class FlagsOffTests(_Env):
    def test_off_is_unchanged(self) -> None:
        base = self.req()
        off = self.req(TALKSHOW_THINK_LEAN="0", TALKSHOW_THINK_PERSONA_NEAR="0", TALKSHOW_THINK_FOCUS="0")
        self.assertEqual(base["messages"], off["messages"])
        t = self.text(base)
        self.assertIn(f"Latest public interval: {HUMAN}\n", t)
        self.assertIn("Backchannels are OPTIONAL", t)
        self.assertIn(OPEN_FLOOR, self.history_text(base))
        self.assertNotIn(tp.FOCUS_HEADER, t)
        self.assertNotIn("Write the content as", t)


class LeanTests(_Env):
    def test_free_floor(self) -> None:
        t_req = self.req(TALKSHOW_THINK_LEAN="1")
        t = self.text(t_req)
        self.assertNotIn("Latest public interval", t)
        self.assertNotIn("Complete visible speech prefix", t)
        self.assertIn(tp.LEAN_SAME_LINE, t)
        self.assertEqual(t.count(HUMAN), 1)  # once, as the transcript's last line
        self.assertNotIn("Backchannels are OPTIONAL", t)
        self.assertNotIn("You MAY REQUEST interrupt", t)
        self.assertIn('"backchannels": []', t)
        self.assertIn("JSON schema:", t)
        hist = self.history_text(t_req)
        for canned in (OPEN_FLOOR, RYAN_UP, "the floor is yours"):
            self.assertNotIn(canned, hist)
        for kept in ("Rent control keeps families", "And it freezes the supply.", "sharp point about supply"):
            self.assertIn(kept, hist)

    def test_occupied_floor_unchanged(self) -> None:
        from types import SimpleNamespace

        from agent.interaction import ConsoleSpeechConsumer, ThinkActionInteraction
        from agent.show.show_history import ShowHistory
        from eval.think_action_prompt import _data_for

        def occupied(**flags):
            with mock.patch.dict(os.environ, flags):
                data = _data_for(PREFIX, HUMAN, "")
                data.runtime = SimpleNamespace(gemma_client=data.runtime.gemma_client)
                asyncio.run(ThinkActionInteraction(ConsoleSpeechConsumer())._think_and_act(
                    data, "guest", history=ShowHistory(lines=list(data.show_history.lines)), speaker="human",
                    interval_text="rents here rose", visible_text="Ryan, rents here rose", checkpoint=1,
                    floor_state="occupied"))
                (r,) = data.runtime.gemma_client.requests
                return r

        a, b = occupied(), occupied(TALKSHOW_THINK_LEAN="1", TALKSHOW_THINK_FOCUS="1")
        self.assertEqual(a["prompt"], b["prompt"])
        self.assertEqual(a["history_messages"], b["history_messages"])
        self.assertIn("Backchannels are OPTIONAL", a["prompt"])

    def test_parser_accepts_lean_outputs(self) -> None:
        raw = json.dumps({"think": "x", "action": {"decision": "speak", "intent": {
            "type": "answer", "goal": "g", "target": None}, "content": "Forty percent is a crisis.",
            "backchannels": []}})
        action = parse_think_action(raw, floor_state="free", interval_text=HUMAN, allow_backchannels=True).action
        self.assertEqual((action.decision, action.content, action.backchannels), ("speak", "Forty percent is a crisis.", ()))
        silent = json.dumps({"think": "", "action": {"decision": "silent", "intent": None, "content": None,
                                                    "backchannels": []}})
        self.assertEqual(parse_think_action(silent, floor_state="free", interval_text=HUMAN).action.decision, "silent")


class PersonaNearTests(_Env):
    def test_texts(self) -> None:
        self.assertEqual(tp.persona_reminder("commentator"),
                         "Write the content as Ryan: Keep lines tight (1–2 sentences). Sound like live TV commentary, "
                         "not a polite essay.")
        self.assertEqual(tp.persona_reminder("guest"),
                         "Write the content as Amy: Keep answers concise (2–3 sentences). Sound like a thoughtful "
                         "friend, not a debater.")

    def test_placement(self) -> None:
        near = tp.persona_reminder("commentator")
        t = self.text(self.req(TALKSHOW_THINK_PERSONA_NEAR="1"))  # no additions: before the JSON schema
        self.assertIn(f"{near}\n\nJSON schema:", t)
        t = self.text(self.req(plan="- 40 percent", TALKSHOW_THINK_PERSONA_NEAR="1", **FAST))
        self.assertIn(f"\n\n{near}\n\n{tp.HEADER}", t)
        self.assertEqual(t.count("Write the content as"), 1)
        self.assertNotIn("Write the content as", self.text(self.req(role="host", TALKSHOW_THINK_PERSONA_NEAR="1")))


class FocusTests(_Env):
    def test_block(self) -> None:
        t = self.text(self.req(TALKSHOW_THINK_FOCUS="1"))
        self.assertIn(f"{tp.FOCUS_HEADER}\n- Last speaker: Human guest (human).\n- They addressed: you (Ryan).\n"
                      f"{tp.FOCUS_RULE}", t)
        amy = self.text(self.req(role="guest", TALKSHOW_THINK_FOCUS="1"))
        self.assertIn("- They addressed: Ryan (commentator).", amy)
        nobody = self.text(self.req(human="Rents rose 40 percent here.", TALKSHOW_THINK_FOCUS="1"))
        self.assertIn("- They addressed: nobody by name.", nobody)
        self.assertNotIn(tp.FOCUS_HEADER, self.text(self.req(role="host", TALKSHOW_THINK_FOCUS="1")))

    def test_planner_points(self) -> None:
        t = self.text(self.req(plan="- 40 percent in two years\n- addressed Ryan", TALKSHOW_THINK_FOCUS="1", **FAST))
        self.assertIn("- Their points:\n  - 40 percent in two years\n  - addressed Ryan\n" + tp.FOCUS_RULE, t)


class ArmTests(unittest.TestCase):
    def test_context_arms(self) -> None:
        from eval.bench.systems import SHOW_ARMS
        from eval.quality_ruler import arms

        for name, keys in (("fast_lean", ["TALKSHOW_THINK_LEAN"]), ("fast_persona", ["TALKSHOW_THINK_PERSONA_NEAR"]),
                           ("fast_focus", ["TALKSHOW_THINK_FOCUS"]),
                           ("fast_context_all", ["TALKSHOW_THINK_LEAN", "TALKSHOW_THINK_PERSONA_NEAR",
                                                 "TALKSHOW_THINK_FOCUS"])):
            flags = arms.ARMS[name]["live_flags"]
            self.assertEqual({k: v for k, v in flags.items() if k not in keys}, arms.FAST_FLAGS)
            self.assertTrue(all(flags[k] == "1" for k in keys))
            self.assertIn(name, SHOW_ARMS)
        with mock.patch.dict(os.environ, {"TALKSHOW_THINK_LEAN": "1"}):
            arms.apply_env(arms.ARMS["fast"])  # the fast arm clears a context flag left in the environment
            self.assertNotIn("TALKSHOW_THINK_LEAN", os.environ)


if __name__ == "__main__":
    unittest.main()
