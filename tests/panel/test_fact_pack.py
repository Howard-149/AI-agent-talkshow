"""Fact pack (agent/panel/fact_pack.py): flag off, topic matching, fact selection, prompt block, live/offline parity.

No LiveKit, no model: the worker runs against a fake Gemma client that records the prompts.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent.panel import concrete, fact_pack
from tests.panel.test_concrete import FakeGemma, live_data

PANEL_FLAGS = ("TALKSHOW_DIALOGUE_LIBRARY", "TALKSHOW_PANEL_CONCRETE", "TALKSHOW_PANEL_BEST_OF",
               "TALKSHOW_UPTAKE_PLANNER", "TALKSHOW_PANEL_SCAFFOLD_FIX", "TALKSHOW_ADDRESSEE_FLOOR",
               "TALKSHOW_FACT_PACK", "TALKSHOW_FACT_PACK_K", "TALKSHOW_FACT_PACK_TOPIC")
FAST = {"TALKSHOW_ADDRESSEE_FLOOR": "1", "TALKSHOW_UPTAKE_PLANNER": "lite", "TALKSHOW_PANEL_SCAFFOLD_FIX": "1",
        "TALKSHOW_DIALOGUE_LIBRARY": "panel_concrete", "TALKSHOW_PANEL_CONCRETE": "1"}

NUCLEAR_FACTS = [
    "Germany shut down its last three nuclear plants in April 2023.",
    "France gets about 70% of its electricity from nuclear power.",
    "Vogtle Units 3 and 4 in Georgia cost more than $30 billion.",
    "The Chernobyl disaster in 1986 released radioactive material across Europe.",
    "Ontario closed its last coal plant in 2014.",
    "Nuclear plants in the United States have a capacity factor above 90%.",
    "Finland's Olkiluoto 3 reactor started regular production in 2023 after years of delay.",
    "The Fukushima accident in 2011 followed a tsunami.",
]
PACK = {
    "nuclear-power": {"topic": "Should we build more nuclear power plants?", "pages": [],
                      "facts": [{"fact": f, "side": "for", "source_sentence": f, "page": "p", "url": "u", "overlap": 1.0}
                                for f in NUCLEAR_FACTS]},
    "tipping": {"topic": "Has tipping culture gone too far?", "pages": [],
                "facts": [{"fact": "Federal law lets employers pay tipped workers $2.13 an hour.", "side": "against",
                           "source_sentence": "s", "page": "p", "url": "u", "overlap": 1.0}]},
    "iq2-1": {"topic": "Embrace The Common Core", "pages": [],
              "facts": [{"fact": "The Common Core was launched in 2010.", "source_sentence": "s"}]},
}


def write_pack(tmp: str) -> str:
    path = os.path.join(tmp, "pack.json")
    Path(path).write_text(json.dumps(PACK))
    return path


def clean_env(env: dict) -> mock._patch_dict:
    """Patch os.environ to exactly the given panel flags (all others removed)."""
    patcher = mock.patch.dict(os.environ, {})
    patcher.start()
    for k in PANEL_FLAGS:
        os.environ.pop(k, None)
    os.environ.update(env)
    return patcher


def draft(env: dict, gemma: FakeGemma | None = None, data=None):
    import asyncio

    from agent.multi_agent.worker import PersonaWorker

    gemma = gemma or FakeGemma(["[reply]: Germany burned more coal after 2023."])
    data = data or live_data(gemma)
    data.runtime = SimpleNamespace(gemma_client=gemma)
    p = clean_env(env)
    try:
        d = asyncio.run(PersonaWorker("guest").draft(data, step="test"))
    finally:
        p.stop()
    one = [c for c in gemma.calls if c[0] == "one"]
    return d, one[-1], gemma


class FlagOffTests(unittest.TestCase):
    def setUp(self) -> None:
        fact_pack.load_pack.cache_clear()
        fact_pack._topic_index.cache_clear()

    def test_off_by_default(self) -> None:
        p = clean_env({})
        try:
            self.assertFalse(fact_pack.fact_pack_enabled())
            self.assertEqual(fact_pack.pack_block(live_data(FakeGemma(["x"]))), ("", {}))
        finally:
            p.stop()

    def test_off_prompt_is_byte_identical(self) -> None:
        """Flag off: the worker sends exactly panel_speech_prompt's text (no flags) or that + CONCRETE (concrete on)."""
        from agent.config import load_persona_name
        from agent.floor import TurnController
        from agent.panel.panel_prompts import panel_speech_prompt

        data = live_data(FakeGemma(["x"]))
        expected = panel_speech_prompt(role="guest", name=load_persona_name("guest"),
                                       host_name=load_persona_name("host"), scenario=data.scenario,
                                       panel_roles=TurnController(data.scenario, data).panel_speaker_roles(),
                                       history=data.show_history, latest_human=data.show_history.latest_human_text(),
                                       data=data)
        _, (_k, user, _kw), _ = draft({})
        self.assertEqual(user, expected)
        _, (_k, user2, _kw), _ = draft({"TALKSHOW_FACT_PACK": ""})
        self.assertEqual(user2, expected)
        _, (_k, user3, _kw), _ = draft({"TALKSHOW_PANEL_CONCRETE": "1"})
        self.assertEqual(user3, expected + "\n\n" + concrete.CONCRETE)

    def test_off_offline_builder_unchanged(self) -> None:
        from eval.model_only import build_request

        prefix = json.loads((Path(__file__).resolve().parents[2] / "eval/data/minimal_pairs/prefix/nuclear-power.json")
                            .read_text())
        p = clean_env({})
        try:
            a = build_request(prefix, "Ryan, is nuclear too slow?", "commentator", False)["messages"]
            b = build_request(prefix, "Ryan, is nuclear too slow?", "commentator", False, live_fixes=True,
                              plan="- a plan", topic_id="nuclear-power")["messages"]
        finally:
            p.stop()
        self.assertEqual(a, b)

    def test_unreadable_pack_is_a_no_op(self) -> None:
        p = clean_env({"TALKSHOW_FACT_PACK": "/nonexistent/pack.json"})
        try:
            block, info = fact_pack.pack_block(live_data(FakeGemma(["x"])))
        finally:
            p.stop()
        self.assertEqual(block, "")
        self.assertEqual(info, {"fact_pack_topic": None})


class TopicTests(unittest.TestCase):
    def setUp(self) -> None:
        fact_pack.load_pack.cache_clear()
        fact_pack._topic_index.cache_clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_pack(self.tmp.name)
        self.p = clean_env({"TALKSHOW_FACT_PACK": self.path})

    def tearDown(self) -> None:
        self.p.stop()
        self.tmp.cleanup()

    def resolve(self, data, topic_id=None):
        return fact_pack.resolve_topic(data, fact_pack.load_pack(self.path), self.path, topic_id)

    def bare(self, lines, room="live-room"):
        data = live_data(FakeGemma(["x"]))
        data.show_history.lines = []
        data.room_name = room
        for role, text in lines:
            data.show_history.append(role, text, speaker=role)
        return data

    def test_text_in_host_line(self) -> None:
        d = self.bare([("host", "About Should we build more nuclear power plants?: the mic is yours whenever you're ready."),
                       ("human", "Germany shut its plants.")])
        self.assertEqual(self.resolve(d), ("nuclear-power", "text"))

    def test_text_case_and_punctuation(self) -> None:
        d = self.bare([("human", 'I want to put a motion to the panel: "embrace the common core." Where do you land?')])
        self.assertEqual(self.resolve(d), ("iq2-1", "text"))

    def test_newest_topic_wins(self) -> None:
        d = self.bare([("host", "You wanted to talk about Has tipping culture gone too far — go ahead."),
                       ("host", "You flagged Should we build more nuclear power plants? — over to you.")])
        self.assertEqual(self.resolve(d)[0], "nuclear-power")

    def test_hand_raise_topic(self) -> None:
        d = self.bare([("human", "hello")])
        d.set_human_hand(raised=True, topic="Has tipping culture gone too far?")
        self.assertEqual(self.resolve(d), ("tipping", "text"))

    def test_room_name(self) -> None:
        self.assertEqual(self.resolve(self.bare([], room="ghost-tipping__orig__k0__vB")), ("tipping", "room"))
        self.assertEqual(self.resolve(self.bare([], room="ghost-pfx-nuclear-power-3")), ("nuclear-power", "room"))

    def test_topic_id_and_env_pin_first(self) -> None:
        d = self.bare([("host", "About Should we build more nuclear power plants?: go.")])
        self.assertEqual(self.resolve(d, "tipping"), ("tipping", "topic_id"))
        os.environ["TALKSHOW_FACT_PACK_TOPIC"] = "iq2-1"
        self.assertEqual(self.resolve(d), ("iq2-1", "env"))
        self.assertEqual(self.resolve(d, "tipping"), ("tipping", "topic_id"))

    def test_no_match_is_logged_once(self) -> None:
        d = self.bare([("human", "Should we ban homework?")], room="room-no-match")
        with self.assertLogs("agent.panel.fact_pack", level="INFO") as cm:
            self.assertEqual(fact_pack.pack_block(d), ("", {"fact_pack_topic": None}))
            self.assertEqual(fact_pack.pack_block(d), ("", {"fact_pack_topic": None}))
        self.assertEqual(sum("no topic match" in m for m in cm.output), 1)


class SelectionTests(unittest.TestCase):
    FACTS = [{"fact": f} for f in NUCLEAR_FACTS]

    def test_relevance(self) -> None:
        idx = fact_pack.select_facts(self.FACTS, "What about France and its electricity?", [], 2)
        self.assertEqual(idx[0], 1)
        self.assertEqual(len(idx), 2)

    def test_ties_keep_pack_order(self) -> None:
        self.assertEqual(fact_pack.select_facts(self.FACTS, "", [], 3), [0, 1, 2])

    def test_used_facts_go_last(self) -> None:
        used = ["France already runs on nuclear: 70% of its power."]  # shares the figure 70
        idx = fact_pack.select_facts(self.FACTS, "France electricity nuclear", used, len(self.FACTS))
        self.assertEqual(idx[-1], 1)
        used = ["Ontario closed its last coal plant, remember?"]  # most content words, no figure
        self.assertTrue(fact_pack.fact_used(NUCLEAR_FACTS[4], used))
        self.assertFalse(fact_pack.fact_used(NUCLEAR_FACTS[0], used))

    def test_k_from_env(self) -> None:
        p = clean_env({"TALKSHOW_FACT_PACK_K": "3"})
        try:
            self.assertEqual(fact_pack.k_facts(), 3)
            os.environ["TALKSHOW_FACT_PACK_K"] = "x"
            self.assertEqual(fact_pack.k_facts(), fact_pack.DEFAULT_K)
        finally:
            p.stop()

    def test_query_uses_latest_human_and_last_two_lines(self) -> None:
        data = live_data(FakeGemma(["x"]))
        q = fact_pack.query_text(data)
        self.assertIn("Vogtle", q)
        self.assertIn("bring in our guest", q)
        self.assertNotIn("Affordable", q)


class BlockTests(unittest.TestCase):
    def setUp(self) -> None:
        fact_pack.load_pack.cache_clear()
        fact_pack._topic_index.cache_clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_pack(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rule_text(self) -> None:
        self.assertEqual(fact_pack.RULE,
                         "Use specific numbers, names, studies or cases ONLY from these facts or from what was said in "
                         "the show. Otherwise speak in general terms, or say 'for example' and keep it clearly "
                         "hypothetical.")

    def test_block_format(self) -> None:
        b = fact_pack.fact_block([{"fact": " A. "}, {"fact": "B."}, {"fact": "C."}], [2, 0])
        self.assertEqual(b, "\n\n" + fact_pack.HEADER + "\n- C.\n- A.\n" + fact_pack.RULE)

    def test_worker_pack_only(self) -> None:
        d, (_k, user, _kw), g = draft({"TALKSHOW_FACT_PACK": self.path})
        self.assertIn(fact_pack.HEADER, user)
        self.assertTrue(user.endswith(fact_pack.RULE))
        self.assertEqual(user.split(fact_pack.HEADER)[1].count("\n- "), fact_pack.DEFAULT_K)
        self.assertIn("Vogtle Units 3 and 4", user)  # the human asked about Vogtle
        self.assertNotIn(concrete.CONCRETE, user)

    def test_worker_pack_and_concrete_do_not_conflict(self) -> None:
        _d, (_k, user, _kw), _g = draft({"TALKSHOW_FACT_PACK": self.path, "TALKSHOW_PANEL_CONCRETE": "1"})
        self.assertNotIn("from what you know", user)
        self.assertNotIn(concrete.CONCRETE, user)
        self.assertTrue(user.endswith(fact_pack.RULE + "\n\n" + fact_pack.CONCRETE_WITH_PACK))

    def test_worker_concrete_without_match_keeps_old_text(self) -> None:
        gemma = FakeGemma(["[reply]: ok"])
        data = live_data(gemma)
        data.show_history.lines = [ln for ln in data.show_history.lines if "nuclear" not in ln.text.lower()]
        _d, (_k, user, _kw), _g = draft({"TALKSHOW_FACT_PACK": self.path, "TALKSHOW_PANEL_CONCRETE": "1"},
                                         gemma, data)
        self.assertTrue(user.endswith("\n\n" + concrete.CONCRETE))

    def test_turn_log_fields(self) -> None:
        logged = []
        gemma = FakeGemma(["[reply]: ok"])
        data = live_data(gemma)
        data.turn_log = SimpleNamespace(log=lambda ev, **kw: logged.append((ev, kw)))
        draft({"TALKSHOW_FACT_PACK": self.path}, gemma, data)
        done = [kw for ev, kw in logged if ev == "panel_model_done"][0]
        self.assertEqual(done["fact_pack_topic"], "nuclear-power")
        self.assertEqual(done["fact_pack_match"], "text")
        self.assertEqual(len(done["fact_pack_offered"]), fact_pack.DEFAULT_K)

    def test_live_and_offline_prompts_are_identical(self) -> None:
        """FAST flags + pack: the worker and eval.model_only.build_request(live_fixes=True) build the same messages."""
        from eval.model_only import build_request

        gemma = FakeGemma(["- Vogtle: 17 billion dollars over budget"])  # planner output, then the panel line
        data = live_data(gemma)
        prefix = {"lines": [{"role_id": ln.role_id, "speaker": ln.speaker, "text": ln.text,
                             "procedural": ln.procedural} for ln in data.show_history.lines]}
        env = {**FAST, "TALKSHOW_FACT_PACK": self.path}
        _d, (_k, user, kw), _g = draft(env, gemma, data)
        self.assertIn("concrete points", user)  # the plan block is there
        live = [{"role": "system", "content": kw["system_prompt"]}, *[dict(m) for m in kw["history_messages"]]]
        if live[-1]["role"] == "user":
            live[-1]["content"] = f"{live[-1]['content']}\n\n{user}"
        else:
            live.append({"role": "user", "content": user})
        p = clean_env(env)
        try:
            req = build_request(prefix, "", "guest", False, include_human=False, live_fixes=True,
                                plan="- Vogtle: 17 billion dollars over budget")
        finally:
            p.stop()
        self.assertEqual(req["messages"], live)
        self.assertEqual(req["fact_pack"]["fact_pack_topic"], "nuclear-power")


if __name__ == "__main__":
    unittest.main()
