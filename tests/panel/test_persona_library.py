"""Persona library (agent/persona_library.py): schema, loader, rendering, seat selection. No model."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent import persona_library as pl
from agent.config import load_persona_name, load_persona_personality, set_persona_swap

LIB_FLAGS = ("TALKSHOW_PERSONA_LIBRARY", "TALKSHOW_PERSONA_LIBRARY_NAMES", "TALKSHOW_PERSONA_COMMENTATOR",
             "TALKSHOW_PERSONA_GUEST", "TALKSHOW_PERSONA_HOST", "TALKSHOW_THINK_PERSONA_NEAR", "TALKSHOW_THINK_LEAN",
             "TALKSHOW_THINK_FOCUS", "TALKSHOW_UPTAKE_PLANNER", "TALKSHOW_PANEL_SCAFFOLD_FIX", "TALKSHOW_DIALOGUE_LIBRARY",
             "TALKSHOW_PANEL_CONCRETE", "TALKSHOW_FACT_PACK", "TALKSHOW_ADDRESSEE_FLOOR")

RYAN_CARD = ("You are Ryan. On this show: color commentator.\n"
             "Background: former sports commentator turned political pundit.\n"
             "Leaning: skeptical of big promises, market-minded.\n"
             "With Amy: friendly sparring; pushes Amy for a real position.")
RYAN_NEAR = ("Write the content as Ryan (1-2 sentences, punchy, ironic, blunt but not cruel, live TV commentary).\n"
             "Use one of your moves if it fits: seizes a number the other side used and turns it back on them; calls "
             "out a dodge or a hand-wave by name; agrees in one clause, then moves the thread forward; asks a pointed "
             "one-line question.\n"
             'Never: long explanations or lists; "it\'s a complex issue" style filler; opening with "I hear your point".')
AMY_CARD = ("You are Amy. On this show: guest, community advocate.\n"
            "Background: runs a neighbourhood nonprofit.\n"
            "Leaning: starts from how people are affected.")
AMY_NEAR = ("Write the content as Amy (2-3 sentences, warm, thoughtful, concrete, a thoughtful friend, not a debater).\n"
            "Use one of your moves if it fits: tells one short real-life story or case; names what is at stake for "
            "ordinary people; concedes the other side's best point, then reframes; asks the audience member a "
            "follow-up question.\n"
            "Never: abstract policy jargon; attacking a person instead of a point.")


class _Env(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"TALKSHOW_TEXT_TOKENIZER": "word"})
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in LIB_FLAGS:
            os.environ.pop(k, None)
        self.addCleanup(set_persona_swap, False)


class SchemaTests(_Env):
    def test_entries_load(self) -> None:
        self.assertEqual(pl.available(), ["amy_warm", "dana_economist", "lessac_host", "ryan_skeptic", "theo_comedian"])
        for pid in pl.available():
            p = pl.load(pid)
            self.assertEqual(p.id, pid)
            self.assertTrue(p.moves and all(p.style.values()))
        self.assertEqual(pl.load("dana_economist").style, {"length": "2 sentences", "tone": "dry, precise",
                                                            "register": "expert on a news panel"})
        self.assertEqual(pl.load("theo_comedian").avoid, ("punching down", "jokes with no point"))

    def test_validation(self) -> None:
        good = {"id": "x", "name": "X", "role": "r", "style": {"length": "1", "tone": "t", "register": "g"},
                "moves": ["m"]}
        pl.parse(good)
        for bad in ({k: v for k, v in good.items() if k != "moves"}, {**good, "style": {"length": "1"}},
                    {**good, "moves": ["m"] * 6}, {**good, "colour": "red"}):
            with self.assertRaises(ValueError):
                pl.parse(bad)
        d = tempfile.mkdtemp()
        Path(d, "y.yaml").write_text("id: z\nname: Z\nrole: r\nstyle: {length: a, tone: b, register: c}\nmoves: [m]\n")
        with self.assertRaises(ValueError):
            pl.load("y", d)  # id must match the file name
        with self.assertRaises(ValueError):
            pl.load("nobody")


class RenderTests(_Env):
    def test_ryan_and_amy(self) -> None:
        names = {"host": "Lessac", "commentator": "Ryan", "guest": "Amy"}
        self.assertEqual(pl.render_card(pl.load("ryan_skeptic"), "Ryan", names), RYAN_CARD)
        self.assertEqual(pl.render_near(pl.load("ryan_skeptic"), "Ryan"), RYAN_NEAR)
        self.assertEqual(pl.render_card(pl.load("amy_warm"), "Amy", names), AMY_CARD)
        self.assertEqual(pl.render_near(pl.load("amy_warm"), "Amy"), AMY_NEAR)


class SeatTests(_Env):
    def test_flag_off_keeps_the_prose_cards(self) -> None:
        from agent.config import _raw_personality

        self.assertEqual(load_persona_personality("commentator"), _raw_personality("commentator"))
        self.assertEqual(load_persona_name("commentator"), "Ryan")

    def test_library_cards_and_names(self) -> None:
        with mock.patch.dict(os.environ, {"TALKSHOW_PERSONA_LIBRARY": "1"}):
            self.assertEqual(load_persona_personality("commentator"), RYAN_CARD)
            self.assertEqual(load_persona_personality("guest"), AMY_CARD)
            self.assertEqual(load_persona_personality("host"), "")  # the host is not rendered from the library
            with mock.patch.dict(os.environ, {"TALKSHOW_PERSONA_COMMENTATOR": "dana_economist"}):
                self.assertTrue(load_persona_personality("commentator").startswith("You are Ryan. On this show: "
                                                                                   "economist."))
                self.assertEqual(load_persona_name("commentator"), "Ryan")  # the seat keeps its name by default
                with mock.patch.dict(os.environ, {"TALKSHOW_PERSONA_LIBRARY_NAMES": "1"}):
                    self.assertEqual(load_persona_name("commentator"), "Dana")
                    from agent.floor.addressee import addressed_role

                    names = {r: load_persona_name(r) for r in ("commentator", "guest")}
                    self.assertEqual(addressed_role("Dana, what does it cost?", names), "commentator")
                    self.assertIsNone(addressed_role("Ryan, what does it cost?", names))

    def test_swap(self) -> None:
        with mock.patch.dict(os.environ, {"TALKSHOW_PERSONA_LIBRARY": "1"}):
            set_persona_swap(True)
            self.assertTrue(load_persona_personality("commentator").startswith(
                "You are Ryan. On this show: guest, community advocate."))
            from agent.panel.think_prompt import near_block

            self.assertTrue(near_block("commentator").startswith("Write the content as Ryan (2-3 sentences, warm"))


class PromptTests(_Env):
    def think(self, **flags):
        from eval.think_action_prompt import think_request

        prefix = {"lines": [{"role_id": "host", "speaker": "Lessac", "text": "Tonight: rent control.",
                             "procedural": False}]}
        with mock.patch.dict(os.environ, flags):
            return think_request(prefix, "commentator", human_text="Ryan, rents rose 40 percent.")

    def test_think_prompt(self) -> None:
        req = self.think(TALKSHOW_PERSONA_LIBRARY="1")
        self.assertEqual(req["messages"][0]["content"], RYAN_CARD)
        self.assertIn(RYAN_NEAR + "\n\nJSON schema:", req["messages"][-1]["content"])
        both = self.think(TALKSHOW_PERSONA_LIBRARY="1", TALKSHOW_THINK_PERSONA_NEAR="1")
        text = both["messages"][-1]["content"]
        self.assertEqual(text.count("Write the content as"), 1)  # the library block supersedes the reminder
        self.assertIn(RYAN_NEAR, text)
        off = self.think()
        self.assertNotIn("Write the content as", off["messages"][-1]["content"])
        self.assertNotEqual(off["messages"][0]["content"], RYAN_CARD)

    def test_bench_arm(self) -> None:
        from eval.bench import items as bitems
        from eval.bench import systems
        from eval.bench.build import persona_cards
        from eval.quality_ruler.arms import ARMS

        self.assertIn("fast_v2_lib", systems.SHOW_ARMS)
        self.assertEqual(ARMS["fast_v2_lib"]["live_flags"]["TALKSHOW_PERSONA_COMMENTATOR"], "ryan_skeptic")
        sysm = systems.TalkshowThinkAction(systems.DryRunBackend(), arm="fast_v2_lib")
        with mock.patch.dict(os.environ, {"TALKSHOW_PERSONA_LIBRARY": "1"}):
            cards = persona_cards(False)  # items keep the prose cards even with the library on
        self.assertNotEqual(cards["commentator"], RYAN_CARD)
        obs = {"item_id": "x", "task": "T1", "topic_id": "t", "topic": "Rent control", "role": "commentator",
               "name": "Ryan", "names": {"host": "Lessac", "commentator": "Ryan", "guest": "Amy"},
               "personas": cards, "history": [{"speaker": "Lessac", "role": "host", "text": "Tonight: rent."}],
               "audience_turn": "Ryan, rents rose.", "fact_pack": None, "show_state": {}, "floor_choice": False,
               "floor_item": False, "fact_instruction": None}
        sysm.respond(obs)
        think = [r for r in sysm.backend.records if r.kind == "think"][0]
        self.assertEqual(think.messages[0]["content"], RYAN_CARD)
        self.assertIn(RYAN_NEAR, think.messages[-1]["content"])
        self.assertTrue(callable(bitems.observation))


if __name__ == "__main__":
    unittest.main()
