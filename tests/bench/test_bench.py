"""Benchmark foundation (eval/bench): item io, builders, the fact-edit rule, adapters, dry run, CPU scoring. No model."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from eval.bench import build, fact_edit, items, judge, proc, run, score, systems

ROOT = Path(__file__).resolve().parents[2]
FLAGS = ("TALKSHOW_DIALOGUE_LIBRARY", "TALKSHOW_PANEL_CONCRETE", "TALKSHOW_PANEL_BEST_OF", "TALKSHOW_UPTAKE_PLANNER",
         "TALKSHOW_PANEL_SCAFFOLD_FIX", "TALKSHOW_ADDRESSEE_FLOOR", "TALKSHOW_FACT_PACK", "TALKSHOW_FACT_PACK_K",
         "TALKSHOW_FACT_PACK_TOPIC", "TALKSHOW_GHOST_REPLAY")
TOPICS = ("tipping", "nuclear-power")
FACTS = {
    "nuclear-power": [
        "Germany shut down its last three nuclear plants in April 2023.",
        "France gets about 70% of its electricity from nuclear power.",
        "Vogtle Units 3 and 4 in Georgia cost more than $30 billion.",
        "The Chernobyl disaster in 1986 released radioactive material across Europe.",
        "Ontario closed its last coal plant in 2014.",
        "Nuclear plants in the United States have a capacity factor above 90%.",
        "Nuclear power produces about 10% of the world's electricity.",
        "Spent fuel is stored on site at most US plants.",
    ],
    "tipping": [
        "Federal law lets employers pay tipped workers a cash wage of $2.13 an hour.",
        "Under the Fair Labor Standards Act, employers must pay minimum wage unless employees receive more than $30 a "
        "month in tips.",
        "Tipping in restaurants is customary in the United States.",
        "Many countries in Europe include service in the menu price.",
        "Tip pooling shares tips among front and back of house staff.",
        "Some restaurants have replaced tipping with a service charge.",
        "Card readers now suggest tip amounts at the counter.",
    ],
}
# One page per fact, except two nuclear facts that share a page (only one of them may be offered with the other).
PAGES = {
    "nuclear-power": ["Energy in Germany", "Nuclear power in France", "Vogtle Electric Generating Plant",
                      "Chernobyl disaster", "Energy in Ontario", "Capacity factor", "Nuclear power",
                      "Nuclear power"],
    "tipping": ["Tipped wage (United States)", "Fair Labor Standards Act of 1938", "Gratuity", "Service charge",
                "Tip pooling", "Restaurant", "Point of sale"],
}
PACK = {k: {"topic": k, "pages": [], "facts": [{"fact": f, "page": PAGES[k][i], "url": "u"} for i, f in enumerate(v)]}
        for k, v in FACTS.items()}


class _Env(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"TALKSHOW_TEXT_TOKENIZER": "word"})
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in FLAGS:
            os.environ.pop(k, None)
        self.tmp = Path(tempfile.mkdtemp())
        self.pack = self.tmp / "pack.json"
        self.pack.write_text(json.dumps(PACK))

    def pairs_file(self, test_like: bool = False) -> Path:
        """Two development topics; test_like adds a cause edit (as the public pairs have)."""
        topics = [t for t in yaml.safe_load(build.DEV_PAIRS.read_text())["topics"] if t["slug"] in TOPICS]
        if test_like:
            for t in topics:
                t["turn2"]["causal_flip"] = t["turn2"]["orig"].replace("because", "even though")
        path = self.tmp / ("pairs_test.yaml" if test_like else "pairs.yaml")
        path.write_text(yaml.safe_dump({"topics": topics}))
        return path

    def dev_items(self) -> list[items.Item]:
        return build.build("dev", pack=str(self.pack), pairs=self.pairs_file())


# ------------------------------------------------------------------------------------------------ items

class ItemIOTests(_Env):
    def test_round_trip(self) -> None:
        its = self.dev_items()
        path = self.tmp / "items.jsonl"
        self.assertEqual(items.write_items(path, its), len(its))
        back = items.read_items(path)
        self.assertEqual([i.to_dict() for i in back], [i.to_dict() for i in its])
        self.assertEqual([items.item_hash(i) for i in back], [items.item_hash(i) for i in its])

    def test_rejects_bad_items(self) -> None:
        d = self.dev_items()[0].to_dict()
        with self.assertRaises(ValueError):
            items.Item.from_dict({**d, "extra": 1})
        with self.assertRaises(ValueError):
            items.Item.from_dict({**d, "task": "T9"})
        with self.assertRaises(ValueError):
            items.Item.from_dict({**d, "history": [{"role": "nobody", "speaker": "x", "text": "y"}]})

    def test_future_tasks_fit_the_format(self) -> None:
        d = self.dev_items()[0].to_dict()
        host = items.Item.from_dict({**d, "task": "T5", "role_under_test": "host", "audience_turn": None})
        self.assertIsNone(items.observation(host)["audience_turn"])

    def test_observation(self) -> None:
        it = next(i for i in self.dev_items() if i.task == "T3a")
        obs = items.observation(it)
        self.assertEqual(obs["name"], "Ryan")
        self.assertEqual(obs["fact_pack"], it.fact_pack)
        self.assertEqual(obs["history"], it.history)


# ------------------------------------------------------------------------------------------------ builders

class BuilderTests(_Env):
    def by(self, its, task):
        return [i for i in its if i.task == task]

    def assert_pairs_differ_only_in(self, its, field):
        groups = {}
        for it in its:
            groups.setdefault(it.pair_group, []).append(it)
        for group in groups.values():
            ref = next(i for i in group if i.version == i.meta["reference_version"])
            for it in group:
                for f in ("topic_id", "role_under_test", "history", "topic", "personas", "audience_turn", "fact_pack"):
                    if f != field:
                        self.assertEqual(getattr(it, f), getattr(ref, f), (it.item_id, f))
                if it is not ref:
                    self.assertNotEqual(getattr(it, field), getattr(ref, field), it.item_id)

    def test_dev_counts_pairing_controls(self) -> None:
        its = self.dev_items()
        t1, t2, t3, t4 = (self.by(its, t) for t in ("T1", "T2", "T3a", "T4"))
        self.assertEqual(len(t1), 2 * 5)  # orig, paraphrase, number, stance, entity (no cause edit in dev)
        self.assertEqual(len(t2), 2 * 2 * 2)  # 2 addressees x 2 panelists
        self.assertEqual(len(t4), 2 * 2 * 2)
        self.assertEqual({i.version for i in t1}, {"orig", "paraphrase", "number", "stance", "entity"})
        self.assertEqual([i.version for i in its if i.is_control], ["paraphrase", "paraphrase"])
        self.assertEqual(len({i.pair_group for i in t1}), 2)
        self.assertEqual(len({i.pair_group for i in t2}), 4)
        self.assertEqual({i.role_under_test for i in t1}, {"commentator"})
        self.assert_pairs_differ_only_in(t1, "audience_turn")
        self.assert_pairs_differ_only_in(t2, "audience_turn")
        self.assert_pairs_differ_only_in(t3, "fact_pack")
        self.assert_pairs_differ_only_in(self.by(its, "T3b"), "fact_pack")
        self.assert_pairs_differ_only_in(t4, "personas")
        self.assertEqual(len({i.item_id for i in its}), len(its))

    def test_t2_template_and_addressee(self) -> None:
        from agent.floor.addressee import addressed_role

        t2 = self.by(self.dev_items(), "T2")
        for it in t2:
            name = it.meta["names"][it.expected["addressee"]]
            self.assertTrue(it.audience_turn.startswith(f"{name}, "))
            self.assertEqual(addressed_role(it.audience_turn, {"commentator": "Ryan", "guest": "Amy"}),
                             it.expected["addressee"])
            self.assertEqual(it.expected["should_answer"], it.role_under_test == it.expected["addressee"])
        pairs = {t["slug"]: t for t in yaml.safe_load(build.DEV_PAIRS.read_text())["topics"]}
        amy = next(i for i in t2 if i.version == "addr_guest" and i.topic_id == "tipping")
        self.assertEqual(amy.audience_turn, pairs["tipping"]["turn2"]["addressee"])  # the hand-written pair agrees

    def test_t3_pairs_target_the_edited_fact(self) -> None:
        from agent.floor.addressee import addressed_role

        t3 = self.by(self.dev_items(), "T3a")
        self.assertEqual({i.topic_id for i in t3}, set(TOPICS))
        groups: dict = {}
        for it in t3:
            groups.setdefault(it.pair_group, {})[it.version.rsplit("_", 1)[0]] = it
        self.assertGreaterEqual(len(groups), 2)
        positions = set()
        for g in groups.values():
            orig, edit = g["fact_orig"], g["fact_edit"]
            self.assertEqual(orig.meta["reference_version"], orig.version)
            j = edit.expected["fact_index"]
            positions.add(j)
            # same question, no digit in it, put to the panelist under test
            self.assertEqual(orig.audience_turn, edit.audience_turn)
            self.assertIsNone(re.search(r"\d", orig.audience_turn))
            self.assertEqual(addressed_role(orig.audience_turn, {"commentator": "Ryan", "guest": "Amy"}),
                             orig.role_under_test)
            page = orig.meta["fact_sources"][j]["page"]
            self.assertIn(build.page_subject(page), orig.audience_turn)
            # 6 facts, only the edited one from its page, the lists differ in that fact only
            self.assertEqual(len(orig.fact_pack), 6)
            self.assertEqual([s["page"] for s in orig.meta["fact_sources"]].count(page), 1)
            self.assertEqual(sum(a != b for a, b in zip(orig.fact_pack, edit.fact_pack)), 1)
            self.assertEqual(orig.fact_pack[j], edit.expected["old_fact"])
            self.assertEqual(edit.fact_pack[j], edit.expected["new_fact"])
            self.assertIn(edit.expected["new_value"], edit.expected["new_fact"])
            self.assertEqual(orig.expected["edits"][0]["fact_index"], j)
        self.assertEqual(build.page_subject("Fair Labor Standards Act of 1938"), "Fair Labor Standards Act")
        self.assertEqual(build.page_subject("Tipped wage (United States)"), "Tipped wage")
        offered = build.offered_facts(str(self.pack), build.load_topics("dev", self.pairs_file(), build.DEV_PREFIX,
                                                                         "prefix")[1],
                                      yaml.safe_load(build.DEV_PAIRS.read_text())["topics"][1]["turn2"]["orig"])
        self.assertEqual(len(offered), 6)  # the show's default number of offered facts

    def test_t4_cards_swapped(self) -> None:
        t4 = self.by(self.dev_items(), "T4")
        orig = next(i for i in t4 if i.version == "persona_orig" and i.role_under_test == "commentator")
        swap = next(i for i in t4 if i.version == "persona_swap" and i.role_under_test == "commentator")
        self.assertIn("Ryan", swap.personas["commentator"])  # the name stays with the seat
        self.assertNotEqual(orig.personas["commentator"], swap.personas["commentator"])
        self.assertEqual(swap.expected["persona_of"], "guest")

    def test_test_like_split(self) -> None:
        its = build.build("test", tasks=("T1",), pairs=self.pairs_file(test_like=True))
        self.assertEqual(len(its), 2 * 6)
        self.assertIn("cause", {i.version for i in its})
        it = its[0]
        self.assertEqual(it.meta["history_source"], "opening")
        self.assertEqual([h["role"] for h in it.history], ["host", "host"])
        self.assertNotIn("show_state", it.meta)
        rows = [build.manifest_row(i) for i in its]
        self.assertTrue(all(set(r) >= {"item_id", "sha", "source"} for r in rows))
        self.assertNotIn("audience_turn", rows[0])  # no source text in the manifest
        path = self.tmp / "manifest.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertEqual(build.check_manifest(its, path), [])
        its[0].audience_turn += " Changed."
        self.assertEqual(build.check_manifest(its, path), [f"differs {its[0].item_id}"])

    def test_deterministic(self) -> None:
        a = [items.item_hash(i) for i in self.dev_items()]
        b = [items.item_hash(i) for i in self.dev_items()]
        self.assertEqual(a, b)


# ------------------------------------------------------------------------------------------------ fact edit rule

class FactEditTests(unittest.TestCase):
    def edit(self, text: str) -> str:
        got = fact_edit.edit_fact(text)
        return got[0] if got else ""

    def test_percent(self) -> None:
        self.assertEqual(self.edit("Debt was 38 percent of GDP."), "Debt was 76 percent of GDP.")
        self.assertEqual(self.edit("Turnout reached 75% that year."), "Turnout reached 37.5% that year.")
        self.assertEqual(self.edit("It rose by more than 100 percent."), "It rose by more than 50 percent.")
        self.assertEqual(self.edit("Sales fell by 9.6% after the tax."), "Sales fell by 19.2% after the tax.")

    def test_money(self) -> None:
        self.assertEqual(self.edit("Tips above $30 a month count."), "Tips above $60 a month count.")
        self.assertEqual(self.edit("The federal minimum is $7.25 an hour."), "The federal minimum is $14.50 an hour.")
        self.assertEqual(self.edit("The Games cost $1.5 billion."), "The Games cost $3 billion.")
        self.assertEqual(self.edit("It could raise 14.9 billion dollars."), "It could raise 29.8 billion dollars.")

    def test_year(self) -> None:
        self.assertEqual(self.edit("Nixon closed the gold window in 1971."), "Nixon closed the gold window in 1961.")
        self.assertEqual(self.edit("In October 2018, a law limited them."), "In October 2008, a law limited them.")

    def test_counts_and_separators(self) -> None:
        self.assertEqual(self.edit("Each plant saved 800,000 life years."), "Each plant saved 1,600,000 life years.")
        self.assertEqual(self.edit("The trial had 61 companies."), "The trial had 122 companies.")

    def test_skipped(self) -> None:
        for text in ("Section 230 protects platforms.", "COVID-19 closed schools.", "Prices rose in the 1970s.",
                     "It takes 10-20 years.", "Adopted with 62 of 84 deputies in favor.", "In 33 of its 93 interventions.",
                     "Ranked 30th out of 197 countries.", "The P5+1 talks.", "From around 500,000 to nearly 2 million.", "Retirement age is 67.",
                     "The 4 Day Week Global trial ran.", "It happened on 24 April.", "Zero is 0."):
            self.assertIsNone(fact_edit.edit_fact(text), text)

    def test_skips_to_next_number_and_context(self) -> None:
        # 2010 also appears in the context, so the next number is edited
        new, e = fact_edit.edit_fact("Launched in 2010, it reached 40 states.", context=["Since 2010 things changed."])
        self.assertEqual(new, "Launched in 2010, it reached 80 states.")
        self.assertEqual((e.old, e.new, e.kind, e.old_key, e.new_key), ("40", "80", "number", "n:40", "n:80"))

    def test_implausible_dropped(self) -> None:
        cases = {
            "Incorporating online content increased pass rates to 91% from 55%.": "comparison",
            "Wages rose to $9 an hour from $7.25.": "comparison",
            "Sales fell by 9.6% while prices were up 3%.": "comparison",
            "In 2016 it reversed the 2010 rule.": "year_context",
            "Founded in 1971, at the height of the 1960s boom.": "year_context",
            "It was banned in 1923 after a 19th century campaign.": "year_context",
        }
        for text, why in cases.items():
            dropped: list = []
            self.assertIsNone(fact_edit.edit_fact(text, dropped=dropped), text)
            self.assertIn(why, {d[0] for d in dropped}, text)
        c = fact_edit.NumberEdit(0, 2, "60", "120", "percent")
        self.assertEqual(fact_edit.implausible("60% agreed.", c), "bound")

    def test_implausible_falls_through_to_next_number(self) -> None:
        # the year has another year in its sentence; the count does not compare with anything
        new, e = fact_edit.edit_fact("Between 1990 and 2000 the club signed 40 players.")
        self.assertEqual((e.kind, e.old, e.new), ("number", "40", "80"))
        dropped: list = []
        new, e = fact_edit.edit_fact("In 2016 it reversed the 2010 rule on 40 farms.", dropped=dropped)
        self.assertEqual((e.old, e.new, new), ("40", "80", "In 2016 it reversed the 2010 rule on 80 farms."))
        self.assertEqual([d[0] for d in dropped], ["year_context", "year_context"])

    def test_plausible_kept(self) -> None:
        for text, want in (("Up to 25% of U.S. jobs are at risk of automation.", "Up to 50% of U.S. jobs"),
                           ("Florida allowed autonomous car testing on public roads in 2012.", "in 2002."),
                           ("Each plant saved 800,000 life years, a study estimated.", "1,600,000"),
                           ("The trial involved 61 companies and almost 3,000 workers.", "122 companies")):
            got = fact_edit.edit_fact(text)
            self.assertIsNotNone(got, text)
            self.assertIn(want, got[0])

    def test_follows(self) -> None:
        _new, e = fact_edit.edit_fact("Debt was 38 percent of GDP.")
        self.assertEqual(fact_edit.follows("Debt at seventy-six percent is a crisis.", e.old_key, e.new_key), "new")
        self.assertEqual(fact_edit.follows("38% is nothing.", e.old_key, e.new_key), "old")
        self.assertEqual(fact_edit.follows("Not 38 but 76 percent.", e.old_key, e.new_key), "both")
        self.assertEqual(fact_edit.follows("Debt is high.", e.old_key, e.new_key), "neither")


# ------------------------------------------------------------------------------------------------ adapters

class AdapterTests(_Env):
    def item(self, task, version, role="commentator"):
        return next(i for i in self.dev_items() if i.task == task and i.version == version
                    and i.role_under_test == role and i.topic_id == "nuclear-power")

    def prompt(self, system, it):
        backend = systems.DryRunBackend()
        sys_ = system(backend) if system is systems.PlainLLM else system(backend, arm="production")
        sys_.respond(items.observation(it), seed=1)
        return [r for r in backend.records if r.kind in ("reply", "think")][0].messages

    def prefix(self):
        return json.loads((build.DEV_PREFIX / "nuclear-power.json").read_text())

    def test_worker_matches_model_only(self) -> None:
        from eval.model_only import build_request

        for task, version, swap in (("T1", "number", False), ("T4", "persona_swap", True)):
            it = self.item(task, version)
            want = build_request(self.prefix(), it.audience_turn, "commentator", swap)["messages"]
            self.assertEqual(self.prompt(systems.TalkshowWorker, it), want, version)

    def test_think_action_matches_think_request(self) -> None:
        from eval.think_action_prompt import think_request

        it = self.item("T1", "entity")
        want = think_request(self.prefix(), "commentator", human_text=it.audience_turn)["messages"]
        self.assertEqual(self.prompt(systems.TalkshowThinkAction, it), want)
        swap = self.item("T4", "persona_swap", "guest")
        want = think_request({**self.prefix(), "persona_swap": True}, "guest", human_text=swap.audience_turn)["messages"]
        self.assertEqual(self.prompt(systems.TalkshowThinkAction, swap), want)

    def test_fact_items_reach_the_show_as_given(self) -> None:
        """The show offers exactly the item's facts (in its own relevance order), with its usual fact block."""
        from agent.panel.fact_pack import HEADER

        for system in (systems.TalkshowWorker, systems.TalkshowThinkAction):
            for version in ("fact_orig_1", "fact_edit_1"):
                it = self.item("T3a", version)
                text = self.prompt(system, it)[-1]["content"]
                block = text[text.index(HEADER):].split("\n")[1:]
                shown = [line[2:] for line in block if line.startswith("- ")][:len(it.fact_pack)]
                self.assertEqual(sorted(shown), sorted(it.fact_pack), (system.name, version))

    def test_plain_prompt(self) -> None:
        it = self.item("T3a", "fact_edit_1")
        msgs = self.prompt(systems.PlainLLM, it)
        self.assertEqual(msgs[0]["content"], it.personas["commentator"])
        user = msgs[1]["content"]
        self.assertIn(f"Audience member: {it.audience_turn}\n", user)
        self.assertIn(f"- {it.expected['new_fact']}", user)
        self.assertNotIn(it.expected["old_fact"], user)
        self.assertTrue(user.endswith("You are Ryan. Reply with your next spoken line only: 1-3 sentences, plain text, "
                                      "no speaker label."))
        self.assertNotIn("Checked facts", self.prompt(systems.PlainLLM, self.item("T1", "orig"))[1]["content"])

    def test_show_refuses_foreign_cards(self) -> None:
        it = self.item("T1", "orig")
        obs = items.observation(it)
        obs["personas"]["commentator"] = "You are a pirate."
        with self.assertRaises(ValueError):
            systems.TalkshowWorker(systems.FakeBackend()).respond(obs)

    def test_floor_action_and_fallback(self) -> None:
        silent = json.dumps({"think": "", "action": {"decision": "silent", "intent": None, "content": None,
                                                    "backchannels": []}})

        def responder(req):
            return silent if req.kind == "think" else "[reply]: Thirty billion is the real number."

        ta = systems.TalkshowThinkAction(systems.FakeBackend(responder))
        res = ta.respond(items.observation(self.item("T2", "addr_guest")))
        self.assertEqual((res["floor_action"], res["text_source"]), ("wait", "worker_fallback"))
        self.assertTrue(res["text"].startswith("Thirty billion is the real number"))  # the worker's post-processing
        self.assertEqual([c["kind"] for c in res["calls"]], ["think", "fallback"])
        nofb = systems.TalkshowThinkAction(systems.FakeBackend(responder), fallback=False)
        self.assertEqual(nofb.respond(items.observation(self.item("T2", "addr_guest")))["text"], "")
        self.assertEqual(systems.floor_action("silent", [object()]), "backchannel")

    def test_pass_parsing(self) -> None:
        for text in ("PASS", "pass", " Pass. ", "PASS!", "pass..."):
            self.assertTrue(systems.is_pass(text), text)
        for text in ("PASS on that one", "I pass", "Passing the mic", "", "Pass, Amy should answer."):
            self.assertFalse(systems.is_pass(text), text)

    def test_yield_per_adapter(self) -> None:
        addr_other = self.item("T2", "addr_guest")  # Ryan tested, Amy addressed
        obs = items.observation(addr_other)
        # plain: the PASS line only on T2; exact PASS -> yield, no text; anything else -> speak
        plain_pass = systems.PlainLLM(systems.FakeBackend(lambda r: "Pass."))
        self.assertEqual({k: plain_pass.respond(obs)[k] for k in ("floor_action", "text")},
                         {"floor_action": "yield", "text": ""})
        plain_talk = systems.PlainLLM(systems.FakeBackend(lambda r: "Amy should take this, but I will add one thing."))
        self.assertEqual(plain_talk.respond(obs)["floor_action"], "speak")
        self.assertTrue(self.prompt(systems.PlainLLM, addr_other)[1]["content"].endswith("\n" + systems.PLAIN_PASS))
        for task, version in (("T1", "orig"), ("T3a", "fact_edit_1"), ("T4", "persona_swap")):
            it = self.item(task, version)
            self.assertNotIn("PASS", self.prompt(systems.PlainLLM, it)[1]["content"])
            self.assertIsNone(plain_pass.respond(items.observation(it))["floor_action"])  # no floor choice
        # think & action: silent -> wait (a yield for T2); worker: no floor decision
        silent = json.dumps({"think": "", "action": {"decision": "silent", "intent": None, "content": None,
                                                    "backchannels": []}})
        ta = systems.TalkshowThinkAction(systems.FakeBackend(lambda r: silent if r.kind == "think" else "[reply]: x"))
        self.assertIn(ta.respond(obs)["floor_action"], systems.YIELDS)
        self.assertIsNone(systems.TalkshowWorker(systems.FakeBackend()).respond(obs)["floor_action"])

    def test_t3b_items(self) -> None:
        its = self.dev_items()
        a = {i.item_id.split("/", 1)[1]: i for i in its if i.task == "T3a"}
        b = {i.item_id.split("/", 1)[1]: i for i in its if i.task == "T3b"}
        self.assertEqual(set(a), set(b))
        for k, ia in a.items():
            ib = b[k]
            self.assertEqual(ib.pair_group, "T3b:" + ia.pair_group.split(":", 1)[1])
            self.assertEqual(ib.meta["fact_instruction"], items.FACT_INSTRUCTION)
            self.assertNotIn("fact_instruction", ia.meta)
            for f in ("audience_turn", "fact_pack", "history", "personas", "expected", "version", "role_under_test"):
                self.assertEqual(getattr(ia, f), getattr(ib, f), f)
        self.assertEqual(items.expand_tasks("T1,T3"), ["T1", "T3a", "T3b"])
        self.assertEqual({i.task for i in build.build("dev", tasks=("T3b",), pack=str(self.pack),
                                                      pairs=self.pairs_file())}, {"T3b"})

    def test_t3b_prompts(self) -> None:
        from agent.panel.fact_pack import CONCRETE_WITH_PACK, RULE

        line = items.FACT_INSTRUCTION
        ia, ib = self.item("T3a", "fact_edit_1"), self.item("T3b", "fact_edit_1")
        pa, pb = self.prompt(systems.PlainLLM, ia), self.prompt(systems.PlainLLM, ib)
        last_fact = f"- {ia.fact_pack[-1]}\n"
        self.assertEqual(pb[1]["content"], pa[1]["content"].replace(last_fact, last_fact + line + "\n", 1))
        self.assertNotIn(line, pa[1]["content"])
        for system in (systems.TalkshowWorker, systems.TalkshowThinkAction):
            # production: no grounding instruction of its own, so the line follows the fact block's rule
            wa, wb = self.prompt(system, ia), self.prompt(system, ib)
            self.assertEqual(wb[-1]["content"], wa[-1]["content"].replace(RULE, RULE + "\n" + line, 1), system.name)
            res = system(systems.FakeBackend(), arm="production").respond(items.observation(ib))
            self.assertEqual(res["fact_instruction"], "adapter_line")
            # fast: the show's concrete instruction already asks for the facts, so T3b = T3a
            backend_a, backend_b = systems.DryRunBackend(), systems.DryRunBackend()
            system(backend_a, arm="fast").respond(items.observation(ia))
            res = system(backend_b, arm="fast").respond(items.observation(ib))
            msgs = [[r.messages for r in be.records if r.kind in ("reply", "think")][0] for be in (backend_a, backend_b)]
            self.assertIn(CONCRETE_WITH_PACK, msgs[0][-1]["content"])
            self.assertEqual(msgs[0], msgs[1])
            self.assertEqual(res["fact_instruction"], "show_rule")
        msgs, how = systems.show_fact_instruction([{"role": "user", "content": "no facts here"}],
                                                  {"fact_instruction": line})
        self.assertEqual(how, "not_applicable")

    def test_fast_arm_runs_the_planner(self) -> None:
        backend = systems.DryRunBackend()
        systems.TalkshowWorker(backend, arm="fast").respond(items.observation(self.item("T1", "orig")))
        self.assertEqual([r.kind for r in backend.records], ["planner", "reply"])
        self.assertIn("(dry run: the planner's extracted points go here)", backend.records[1].messages[-1]["content"])


# ------------------------------------------------------------------------------------------------ run + score

class RunScoreTests(_Env):
    def items_file(self) -> Path:
        path = self.tmp / "dev.jsonl"
        items.write_items(path, self.dev_items())
        return path

    def test_dry_run_end_to_end(self) -> None:
        path = self.items_file()
        n = len(items.read_items(path))
        for system in ("plain", "worker", "think_action"):
            out = self.tmp / f"dry_{system}"
            with contextlib.redirect_stdout(io.StringIO()):
                run.main(["--system", system, "--items", str(path), "--split", "dev", "--dry-run", "--out",
                          str(out)])
            rows = [json.loads(l) for l in (out / "prompts.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), n, system)
            self.assertEqual(len(list((out / "prompts").glob("*.txt"))), n)
            self.assertFalse((out / "outputs.jsonl").exists())

    def test_fake_run_and_score(self) -> None:
        path = self.items_file()
        out = self.tmp / "fake"
        with contextlib.redirect_stdout(io.StringIO()):
            run.main(["--system", "think_action", "--items", str(path), "--split", "dev", "--fake", "--reruns", "2",
                      "--out", str(out)])
        rows = [json.loads(l) for l in (out / "outputs.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2 * len(items.read_items(path)))
        self.assertTrue(all({"text", "floor_action", "latency_s", "prompt_hash", "seed"} <= set(r) for r in rows))
        # paired versions share seeds; reruns differ
        t1 = [r for r in rows if r["task"] == "T1" and r["topic_id"] == "tipping"]
        self.assertEqual(len({r["seed"] for r in t1 if r["rerun"] == 0}), 1)
        self.assertEqual(len({r["seed"] for r in t1}), 2)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = score.score(out, str(path))
        self.assertEqual(set(rep["tasks"]), {"T1", "T2", "T3a", "T3b", "T4"})
        rf = rep["tasks"]["T2"]["right_floor"]  # the fake speaks every time
        self.assertEqual((rf["acc_addressed"], rf["acc_not_addressed"], rf["balanced"]), (1.0, 0.0, 0.5))
        self.assertTrue((out / "score" / "R" / "T1" / "lines.jsonl").is_file())
        ident = [json.loads(l) for l in (out / "score" / "judge" / "identification.jsonl").read_text().splitlines()]
        self.assertEqual({r["task"] for r in ident}, {"T1", "T2", "T3a", "T3b", "T4"})
        t2 = [r for r in ident if r["task"] == "T2"]  # only replies given when addressed are judged
        self.assertTrue(t2 and all(r["item_id"].split("/")[3] == f"addr_{r['item_id'].split('/')[4]}" for r in t2))
        self.assertTrue(all(r["true"] in ("A", "B") and r["messages"][-1]["role"] == "user" for r in ident))
        self.assertTrue((out / "score" / "score.json").is_file())

    def test_rules(self) -> None:
        its = {i.item_id: i for i in self.dev_items()}
        rows = []
        for it in its.values():
            if it.task == "T2":
                act = "speak" if it.role_under_test == it.expected["addressee"] else "wait"
                rows.append({"item_id": it.item_id, "task": "T2", "rerun": 0, "floor_action": act, "text": "x"})
        rf = score.right_floor(rows, its)
        self.assertEqual((rf["n"], rf["balanced"], rf["acc_not_addressed"]), (8, 1.0, 1.0))
        rows = []
        for it in its.values():
            if it.task == "T3a" and it.version.startswith("fact_edit"):
                rows.append({"item_id": it.item_id, "task": "T3a", "rerun": 0,
                             "text": f"The number is {it.expected['new_value']}, believe it."})
        res = score.follows_fact(rows, its)
        self.assertEqual(res["edited"]["n"], len(rows))
        self.assertGreater(res["edited"]["new"] + res["edited"]["both"], 0.0)


class JudgeTests(_Env):
    def fake_run(self, name: str = "fake", system: str = "think_action") -> Path:
        path = self.tmp / "dev.jsonl"
        if not path.exists():
            items.write_items(path, self.dev_items())
        out = self.tmp / name
        with contextlib.redirect_stdout(io.StringIO()):
            run.main(["--system", system, "--items", str(path), "--split", "dev", "--fake", "--reruns", "2",
                      "--out", str(out)])
            score.score(out)
        return out

    @staticmethod
    def oracle(reqs):
        """A judge that always answers right: A/B = the true version, yes for every rubric item."""
        out = []
        for r in reqs:
            tok = ("Yes" if r["request"] == "rubric" else r["true"])
            other = ("No" if tok == "Yes" else "B" if tok == "A" else "A")
            lp = {0: mock.Mock(decoded_token=tok, logprob=-0.01), 1: mock.Mock(decoded_token=other, logprob=-5.0)}
            out.append(mock.Mock(outputs=[mock.Mock(logprobs=[lp])]))
        return out

    def test_requests_rubric_rerun_0_only(self) -> None:
        out = self.fake_run()
        rub = [json.loads(l) for l in (out / "score/judge/rubric.jsonl").read_text().splitlines()]
        ident = [json.loads(l) for l in (out / "score/judge/identification.jsonl").read_text().splitlines()]
        self.assertEqual({r["rerun"] for r in rub}, {0})
        self.assertEqual({r["rerun"] for r in ident}, {0, 1})
        with contextlib.redirect_stdout(io.StringIO()):
            score.score(out, rubric_reruns=None)
        rub_all = (out / "score/judge/rubric.jsonl").read_text().splitlines()
        self.assertEqual(len(rub_all), 2 * len(rub))

    def test_judge_resumes_and_scores_merge(self) -> None:
        out = self.fake_run()
        d = out / "score"
        n_req = sum(len((d / f"judge/{n}.jsonl").read_text().splitlines()) for n in ("identification", "rubric"))
        calls = []

        def flaky(reqs):
            calls.append(len(reqs))
            if len(calls) == 2:
                raise RuntimeError("preempted")
            return self.oracle(reqs)

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                judge.run([d], "oracle", flaky, {0}, chunk=50)
            seen = []
            judge.run([d], "oracle", lambda reqs: seen.append(len(reqs)) or self.oracle(reqs), {0}, chunk=50)
        self.assertEqual(sum(seen), n_req - 50)  # the first chunk was not asked again
        answers = [json.loads(l) for l in (d / "judge/answers_oracle.jsonl").read_text().splitlines()]
        self.assertEqual(len(answers), n_req)
        with contextlib.redirect_stdout(io.StringIO()):
            judge.main(["--requests", str(d), "--tag", "fake", "--fake"])
            rep = score.score(out)
        judged = rep["judged"]
        self.assertEqual(judged["judges"], ["fake", "oracle"])
        t1 = judged["tasks"]["T1"]
        self.assertEqual(t1["oracle"]["J"]["J"], 1.0)
        self.assertEqual(t1["oracle"]["J_control"]["J"], 1.0)
        self.assertIn("orig vs paraphrase", t1["oracle"]["J_by_pair"])
        self.assertTrue(0.0 <= t1["fake"]["J"]["J"] <= 1.0)
        lo, hi = t1["all"]["J"]["ci95"]
        self.assertLessEqual(lo, t1["all"]["J"]["J"])
        self.assertLessEqual(t1["all"]["J"]["J"], hi)
        self.assertEqual(judged["tasks"]["T4"]["oracle"]["rubric_pass"]["fidelity"], 1.0)
        self.assertEqual(set(rep["task_scores"]), {"T1", "T2", "T3a", "T3b", "T4", "overall"})
        self.assertIn("right_floor", rep["task_scores"]["T2"]["components"])
        self.assertIn("# think_action (production)", (d / "report.md").read_text())

    def test_balanced_floor_accuracy(self) -> None:
        its = {i.item_id: i for i in self.dev_items()}
        t2 = [i for i in its.values() if i.task == "T2"]

        def rows(policy):
            return [{"item_id": i.item_id, "task": "T2", "rerun": k, "floor_action": policy(i), "text": "x"}
                    for i in t2 for k in range(3)]

        addressed = lambda i: i.role_under_test == i.expected["addressee"]  # noqa: E731
        cases = {"always speak": (lambda i: "speak", 1.0, 0.0, 0.5, 0.0),
                 "always yield": (lambda i: "yield", 0.0, 1.0, 0.5, 0.0),
                 "perfect": (lambda i: "speak" if addressed(i) else "yield", 1.0, 1.0, 1.0, 100.0),
                 "perfect, T&A words": (lambda i: "speak" if addressed(i) else "backchannel", 1.0, 1.0, 1.0, 100.0)}
        for name, (policy, a, na, bal, sc) in cases.items():
            rf = score.right_floor(rows(policy), its)
            self.assertEqual((rf["acc_addressed"], rf["acc_not_addressed"], rf["balanced"]), (a, na, bal), name)
            self.assertEqual(score.task_scores({"T2": {"right_floor": rf}}, {})["T2"]["components"]["right_floor"],
                             sc, name)
        none = score.right_floor(rows(lambda i: None), its)  # the worker: no floor decision
        self.assertFalse(none["applicable"])
        self.assertNotIn("T2", score.task_scores({"T2": {"right_floor": none}}, {}))

    def test_judge_exits_before_loading_a_model(self) -> None:
        missing = self.tmp / "no-such-run" / "score"
        empty = self.tmp / "empty" / "score"
        (empty / "judge").mkdir(parents=True)
        for n in ("identification", "rubric"):
            (empty / "judge" / f"{n}.jsonl").write_text("")
        with mock.patch.object(judge, "vllm_generate", side_effect=AssertionError("model loaded")):
            for dirs in (str(missing), str(empty), f"{missing},{empty}"):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    judge.main(["--requests", dirs, "--tag", "q", "--model", "org/judge"])
                self.assertIn("nothing to do", out.getvalue(), dirs)
        self.assertFalse(missing.exists())  # a skipped dir is not created
        self.assertEqual(list((empty / "judge").glob("answers_*")), [])

    def test_judge_skips_a_missing_dir_and_answers_the_rest(self) -> None:
        out = self.fake_run()
        missing = self.tmp / "gone" / "score"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            judge.main(["--requests", f"{missing},{out / 'score'}", "--tag", "fake", "--fake"])
        self.assertIn(f"skip {missing}", buf.getvalue())
        self.assertTrue((out / "score/judge/answers_fake.jsonl").is_file())

    def test_run_judges_creates_the_judge_dir(self) -> None:
        d = self.tmp / "made" / "score"
        (d / "judge").mkdir(parents=True)
        (d / "judge" / "rubric.jsonl").write_text(json.dumps(
            {"id": "x#0|specific|S2", "task": "T1", "item_id": "x", "rerun": 0, "dim": "specific",
             "rubric_item": "S2", "reverse": False, "messages": [{"role": "user", "content": "q"}]}) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            judge.main(["--requests", str(d), "--tag", "fake", "--fake"])
        self.assertEqual(len((d / "judge/answers_fake.jsonl").read_text().splitlines()), 1)

    def test_exit_codes(self) -> None:
        self.assertEqual(proc.exit_code(lambda: None), 0)
        self.assertEqual(proc.exit_code(lambda: sys.exit(3)), 3)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(proc.exit_code(lambda: sys.exit("bad model")), 1)
            self.assertEqual(proc.exit_code(lambda: 1 / 0), 1)
        self.assertIn("ZeroDivisionError", err.getvalue())

    def entry(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        t0 = time.time()
        res = subprocess.run([sys.executable, "-m", *args], cwd=str(ROOT), capture_output=True, text=True,
                             timeout=120, env={**os.environ, **(env or {})})
        self.assertLess(time.time() - t0, 60, args)
        return res

    def test_entry_points_exit_promptly(self) -> None:
        # judge: missing dir -> 0; a broken request file -> non-zero, never a hang
        res = self.entry("eval.bench.judge", "--requests", str(self.tmp / "nope"), "--tag", "q", "--model", "org/j")
        self.assertEqual(res.returncode, 0, res.stderr)
        bad = self.tmp / "bad" / "score" / "judge"
        bad.mkdir(parents=True)
        (bad / "identification.jsonl").write_text("{not json\n")
        res = self.entry("eval.bench.judge", "--requests", str(bad.parent), "--tag", "q", "--model", "org/j")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("JSONDecodeError", res.stderr)
        # run: a model with no local snapshot while offline -> a clear message and a non-zero code, before vLLM
        path = self.tmp / "dev.jsonl"
        items.write_items(path, self.dev_items())
        res = self.entry("eval.bench.run", "--system", "plain", "--model", "org/not-cached", "--items", str(path),
                         "--split", "dev", "--out", str(self.tmp / "r"),
                         env={"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(self.tmp / "empty-cache")})
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("no local snapshot", res.stderr)
        self.assertNotIn("vllm", res.stderr.lower())

    def test_task_score_formula(self) -> None:
        per_task = {"T1": {}, "T3a": {"follows_fact": {"edited": {"new": 0.4}},
                                     "unsupported_specifics": {"share_lines_unsupported": 0.25}}}
        judged = {"tasks": {"T1": {"all": {"J": {"J": 0.8}}}, "T3a": {"all": {"J": {"J": 0.45}}}}}
        sc = score.task_scores(per_task, judged)
        self.assertEqual(sc["T1"]["score"], 60.0)  # 200 * (0.8 - 0.5)
        self.assertEqual(sc["T3a"]["components"], {"identification": 0.0, "follows_fact": 40.0,
                                                  "supported_specifics": 75.0})
        self.assertEqual(sc["T3a"]["score"], 38.3)
        self.assertEqual(sc["overall"], round((60.0 + 38.3) / 2, 1))

    def test_leaderboard(self) -> None:
        a, b = self.fake_run("a"), self.fake_run("b", system="plain")
        with contextlib.redirect_stdout(io.StringIO()):
            for d in (a, b):
                judge.main(["--requests", str(d / "score"), "--tag", "fake", "--fake"])
                score.score(d)
            table = score.leaderboard([a / "score", b / "score"])
        self.assertEqual(len(table.strip().splitlines()), 4)
        self.assertIn("plain", table)
        self.assertIn("think_action (production)", table)


if __name__ == "__main__":
    unittest.main()
