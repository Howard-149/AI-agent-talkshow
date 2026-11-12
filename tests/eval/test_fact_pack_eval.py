"""Fact pack, offline side: arm presets, the episode generator's dry run, the unsupported-specifics metric. CPU only."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import agent.agents  # noqa: F401  (import order the app uses)
from eval.quality_ruler import arms, specifics
from tests.panel.test_fact_pack import PACK, PANEL_FLAGS, clean_env, write_pack

ROOT = Path(__file__).resolve().parents[2]


def ep(lines, topic_id="nuclear-power", topic="Should we build more nuclear power plants?"):
    L = [{"role": r, "speaker": r, "text": t} for r, t in lines]
    return {"id": "e1", "topic_id": topic_id, "topic": topic, "lines": L,
            "scored": [i for i, l in enumerate(L) if l["role"] in ("commentator", "guest")]}


class ArmTests(unittest.TestCase):
    def test_fast_matches_the_live_arms_file(self) -> None:
        line = next(l for l in (ROOT / "deploy/arms/dialogue-quality.arms").read_text().splitlines()
                    if l.startswith("FAST "))
        flags = dict(kv.split("=", 1) for kv in line.split()[1:])
        self.assertEqual({k: v for k, v in arms.FAST_FLAGS.items() if k != "TALKSHOW_PANEL_BEST_OF"}, flags)
        self.assertEqual(arms.ARMS["fast"]["live_flags"]["TALKSHOW_PANEL_BEST_OF"], "1")
        self.assertEqual(arms.ARMS["concrete_fact_pack"], {**arms.ARMS["fast"], "fact_pack": True})

    def test_apply_env(self) -> None:
        p = clean_env({"TALKSHOW_PANEL_CONCRETE": "1", "TALKSHOW_FACT_PACK": "/x.json"})
        try:
            arms.apply_env(arms.ARMS["production"])
            self.assertFalse(any(k in os.environ for k in PANEL_FLAGS if k != "TALKSHOW_FACT_PACK_K"))
            os.environ["TALKSHOW_FACT_PACK"] = "/x.json"
            arms.apply_env(arms.ARMS["concrete_fact_pack"])
            self.assertEqual(os.environ["TALKSHOW_FACT_PACK"], "/x.json")
            self.assertEqual(os.environ["TALKSHOW_UPTAKE_PLANNER"], "lite")
            arms.apply_env(arms.ARMS["fast"])
            self.assertNotIn("TALKSHOW_FACT_PACK", os.environ)
            with self.assertRaises(SystemExit):
                arms.apply_env(arms.ARMS["concrete_fact_pack"])
        finally:
            p.stop()


class DryRunTests(unittest.TestCase):
    def test_generate_dry_run(self) -> None:
        from agent.panel import fact_pack
        from eval.quality_ruler import episodes as E

        fact_pack.load_pack.cache_clear()
        fact_pack._topic_index.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            pairs = Path(tmp) / "pairs.yaml"
            pairs.write_text(json.dumps({"topics": [{
                "slug": "nuclear-power", "topic": "Should we build more nuclear power plants?",
                "turn1": "Germany shut its last plants in 2023.", "turn2": {"orig": "Ryan, is France the model?"}}]}))
            p = clean_env({"TALKSHOW_FACT_PACK": write_pack(tmp)})
            try:
                E.generate(Namespace(arm="concrete_fact_pack", arm_spec="", pairs=str(pairs), topics="", k=1,
                                     lines_per_beat=2, seed=0, model="m", tp=1, out=str(Path(tmp) / "x.jsonl"),
                                     dry_run=str(Path(tmp) / "dry")))
            finally:
                p.stop()
            rows = [json.loads(l) for l in (Path(tmp) / "dry/prompts.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)  # 2 beats x 2 lines
            users = [r["messages"][-1]["content"] for r in rows]
            self.assertTrue(all(fact_pack.HEADER in u and fact_pack.CONCRETE_WITH_PACK in u for u in users))
            plan = [E.DryRunLLM.PLAN in u for u in users]
            self.assertEqual(plan, [True, False, True, False])  # only the first line after a human turn
            self.assertTrue(any((Path(tmp) / "dry").glob("*.txt")))
            self.assertFalse((Path(tmp) / "x.jsonl").exists())


class SpecificsTests(unittest.TestCase):
    def kinds(self, text):
        return [(s["kind"], s["key"]) for s in specifics.extract(text)]

    def test_numbers(self) -> None:
        self.assertEqual(self.kinds("about 40% of them"), [("percent", "n:40")])
        self.assertEqual(self.kinds("forty percent"), [("percent", "n:40")])
        self.assertEqual(self.kinds("six hundred workers"), [("number", "n:600")])
        self.assertEqual(self.kinds("twenty-five plants"), [("number", "n:25")])
        self.assertEqual(self.kinds("two hundred and fifty thousand"), [("number", "n:250000")])
        self.assertEqual(self.kinds("$1.5 million"), [("money", "n:1.5e+06")])
        self.assertEqual(self.kinds("17 billion dollars"), [("money", "n:1.7e+10")])
        self.assertEqual(self.kinds("since 2011, and the 1970s"), [("year", "n:2011"), ("year", "n:1970")])
        self.assertEqual(self.kinds("1,500,000 people"), [("number", "n:1.5e+06")])

    def test_small_spelled_numbers_are_not_specifics(self) -> None:
        self.assertEqual(self.kinds("There are two sides and one reason."), [])
        self.assertEqual(self.kinds("three percent"), [("percent", "n:3")])

    def test_names(self) -> None:
        self.assertEqual([s["text"] for s in specifics.extract(
            "The Fair Labor Standards Act and the Bank of England. But Germany said so. Ryan, I hear Amy.")],
            ["Fair Labor Standards Act", "Bank of England"])

    def test_support_from_pack(self) -> None:
        rows = specifics.episode_lines(ep([("human", "hi"), ("guest", "France gets 70 percent from nuclear.")]), PACK)
        self.assertEqual([s["supported"] for s in rows[0]["specifics"]], [True])
        rows = specifics.episode_lines(ep([("guest", "France gets 85 percent from nuclear.")]), PACK)
        self.assertEqual([s["supported"] for s in rows[0]["specifics"]], [False])

    def test_support_from_earlier_lines_of_others_only(self) -> None:
        e = ep([("human", "My bill went up 300 dollars."), ("commentator", "Hinkley Point C is late."),
                ("guest", "300 dollars, and Hinkley Point C, and the Sizewell Plant, and 300 again."),
                ("commentator", "Sizewell Plant again.")])
        rows = specifics.episode_lines(e, PACK)
        sup = {r["index"]: [(s["text"], s["supported"]) for s in r["specifics"]] for r in rows}
        self.assertEqual(sup[1], [("Hinkley Point C", False)])
        self.assertEqual(dict(sup[2]), {"300 dollars": True, "Hinkley Point C": True, "Sizewell Plant": False,
                                        "300": True})
        self.assertEqual(sup[3], [("Sizewell Plant", True)])  # said earlier by the other panelist
        e2 = ep([("guest", "Sizewell Plant."), ("human", "ok"), ("guest", "Sizewell Plant again.")])
        self.assertFalse(specifics.episode_lines(e2, PACK)[1]["specifics"][0]["supported"])  # own line is no source

    def test_summary(self) -> None:
        e = ep([("human", "Germany shut its plants."),
                ("guest", "France gets 70 percent of its power from nuclear, and Sizewell Plant costs 40 billion."),
                ("commentator", "No figures here at all.")])
        s = specifics.score_episodes([e], PACK)
        self.assertEqual((s["lines"], s["specifics"]), (2, 3))
        self.assertAlmostEqual(s["share_supported"], 1 / 3)
        self.assertAlmostEqual(s["share_lines_unsupported"], 0.5)
        self.assertAlmostEqual(s["specifics_per100w"], 100 * 3 / s["words"])
        self.assertEqual(s["lines_without_pack"], 0)

    def test_no_pack_for_topic_still_scores(self) -> None:
        s = specifics.score_episodes([ep([("guest", "In 1999 the Bank of England moved.")], "x", "Unknown motion")],
                                     PACK)
        self.assertEqual((s["specifics"], s["share_supported"], s["lines_without_pack"]), (2, 0.0, 1))

    def test_topic_text_fallback(self) -> None:
        s = specifics.score_episodes([ep([("guest", "France gets 70 percent.")], "live-slug",
                                          "should we build more nuclear power plants")], PACK)
        self.assertEqual(s["share_supported"], 1.0)


if __name__ == "__main__":
    unittest.main()
