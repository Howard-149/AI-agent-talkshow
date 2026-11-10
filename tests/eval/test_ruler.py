"""Quality ruler: automatic metrics, perturbations, statistics, jury answer parsing. CPU only, no model."""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval.quality_ruler import auto, jury, perturb, ruler, stats
from eval.quality_ruler.items import GENERIC_POOL


def _ep(lines: list[tuple[str, str]], eid: str = "e1", **kw) -> dict:
    names = {"host": "Lessac", "human": "Human guest", "commentator": "Ryan", "guest": "Amy"}
    L = [{"role": r, "speaker": names[r], "text": t} for r, t in lines]
    return {"id": eid, "topic_id": "t1", "topic": "Tipping", "source": "system", "lines": L,
            "scored": [i for i, l in enumerate(L) if l["role"] in ("commentator", "guest")], **kw}


BEAT = _ep([
    ("human", "Tipping in Chicago pays the rent for waiters."),
    ("commentator", "Chicago waiters earn 2 dollars an hour, so tips are the wage."),
    ("guest", "Ryan is right that Chicago waiters depend on tips, and my cousin Maria says so too."),
    ("commentator", "Then abolish the tipped wage in Illinois."),
    ("guest", "Then abolish the tipped wage in Illinois."),
])


class AutoMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.m = auto.metrics([BEAT], Path("/nonexistent"))["e1"]

    def test_figures_and_names_per_100_words(self) -> None:
        # line 1: NUM "2" + figure "2" (sentence-initial "Chicago" is not a name); line 2: "chicago", "maria"
        # (initial "Ryan" skipped); lines 3 and 4: "illinois" -> 6
        txt = [BEAT["lines"][i]["text"] for i in BEAT["scored"]]
        nw = sum(len(auto.words(t)) for t in txt)
        hits = sum(len(auto.NUM.findall(t)) + len(auto.names_figs(t)) for t in txt)
        self.assertEqual(hits, 6)
        self.assertAlmostEqual(self.m["specific.figs_names_per100w"], 100 * 6 / nw)

    def test_names_figs(self) -> None:
        # capitalised words count unless they start a sentence ("Then" follows a full stop)
        self.assertEqual(auto.names_figs("We paid 12.5% to Boeing. Then Airbus"), {"12.5%", "boeing", "airbus"})

    def test_lexical_link(self) -> None:
        # panel lines 2 and 4 share a content bigram with an earlier line by another speaker ("chicago waiters",
        # "abolish tipped ..."); line 3 shares nothing with the human or Amy -> 2 of 3
        self.assertAlmostEqual(self.m["interaction.lexical_link"], 2 / 3)

    def test_repetition(self) -> None:
        self.assertLess(self.m["repetition.distinct2"], 1.0)
        # four openers, "then abolish the" twice -> one repeat -> 1 - 1/4
        self.assertAlmostEqual(self.m["repetition.1-repeated_opener"], 0.75)

    def test_flow(self) -> None:
        self.assertEqual(self.m["flow.alternation"], 1.0)
        self.assertAlmostEqual(self.m["flow.turn_entropy"], 1.0)
        self.assertEqual(self.m["flow.human_answered"], 1.0)  # "chicago" is shared

    def test_words_per_line(self) -> None:
        nw = sum(len(auto.words(BEAT["lines"][i]["text"])) for i in BEAT["scored"])
        self.assertAlmostEqual(self.m["length.words_per_line"], nw / 4)

    def test_embedding_metrics(self) -> None:
        # generic pool along x; the first two panel lines along y, the next two along z
        vec = {0: [0, 1, 0], 1: [0, 1, 0], 2: [0, 0.6, 0.8], 3: [0, 0, 1], 4: [0, 0, 1]}
        keys, emb = [], []
        for i, v in vec.items():
            keys.append(["e1", i])
            emb.append(v)
        for k in range(len(GENERIC_POOL)):
            keys.append(["__generic__", k])
            emb.append([1, 0, 0])
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "emb.npy", np.array(emb, dtype=np.float32))
            (Path(d) / "emb_keys.json").write_text(json.dumps(keys))
            m = auto.metrics([BEAT], Path(d))["e1"]
        self.assertAlmostEqual(m["specific.1-generic_sim"], 1.0)
        # max cosine to earlier lines: line1 1.0 (= human), line2 0.6, line3 0.8, line4 1.0 -> novel only line2
        self.assertAlmostEqual(m["novelty.share_novel"], 0.25)

    def test_reuse_metrics(self) -> None:
        eps = [_ep([("commentator", "I hear you, but no."), ("guest", "I hear you, and yes.")], eid=f"e{k}")
               for k in range(3)]
        r = auto.reuse_metrics(eps)
        self.assertEqual(r["reuse.top10_opener_share"], 1.0)  # only one opener class
        self.assertAlmostEqual(r["reuse.opener_classes_per100"], 100 / 6)
        self.assertEqual(r["reuse.ack_template_share"], 1.0)
        self.assertTrue(all(math.isnan(v) for v in auto.reuse_metrics([]).values()))

    def test_ack_template(self) -> None:
        for t in ("I really hear Ryan here", "Ryan is right about that", "That's a fair point", "Amy, I agree"):
            self.assertTrue(auto.ACK.match(t), t)
        self.assertFalse(auto.ACK.match("Chicago waiters earn 2 dollars"))


class PerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(0)

    def _texts(self, ep: dict) -> list[str]:
        return [ep["lines"][i]["text"] for i in ep["scored"]]

    def test_metadata(self) -> None:
        p = perturb.perturb(BEAT, "same_opener", self.rng)
        self.assertEqual((p["id"], p["base_id"], p["source"], p["base_source"], p["unit"]),
                         ("e1~same_opener", "e1", "perturbed", "system", False))
        self.assertEqual(BEAT["lines"][1]["text"], "Chicago waiters earn 2 dollars an hour, so tips are the wage.")

    def test_generic_swap(self) -> None:
        p = perturb.perturb(BEAT, "generic_swap", self.rng)
        self.assertGreaterEqual(sum(t in GENERIC_POOL for t in self._texts(p)), 2)

    def test_label_swap(self) -> None:
        p = perturb.perturb(BEAT, "label_swap", self.rng)
        self.assertEqual([p["lines"][i]["speaker"] for i in p["scored"]], ["Amy", "Ryan", "Amy", "Ryan"])
        self.assertEqual(self._texts(p), self._texts(BEAT))

    def test_duplicate_argument(self) -> None:
        t = self._texts(perturb.perturb(BEAT, "duplicate_argument", self.rng))
        self.assertEqual(t[2:], t[:2])

    def test_same_opener(self) -> None:
        t = self._texts(perturb.perturb(BEAT, "same_opener", self.rng))
        self.assertTrue(all(x.startswith(perturb.OPENER) for x in t))
        self.assertEqual(t[0], perturb.OPENER + "chicago waiters earn 2 dollars an hour, so tips are the wage.")

    def test_length_pad(self) -> None:
        t = self._texts(perturb.perturb(BEAT, "length_pad", self.rng))
        self.assertTrue(all(x.endswith(perturb.PAD) for x in t))

    def test_turn_shuffle_breaks_alternation(self) -> None:
        for seed in range(20):
            p = perturb.perturb(BEAT, "turn_shuffle", random.Random(seed))
            roles = [p["lines"][i]["role"] for i in p["scored"]]
            self.assertTrue(any(a == b for a, b in zip(roles, roles[1:])))
            self.assertEqual(sorted(self._texts(p)), sorted(self._texts(BEAT)))

    def test_llm_rewrites_are_not_built_here(self) -> None:
        self.assertIsNone(perturb.perturb(BEAT, "persona_inversion", self.rng))

    def test_offthread_generic(self) -> None:
        perturb.OTHER_POOL["t1"] = {"commentator": ["Mars needs more rockets."], "guest": ["Zoos are cruel."]}
        try:
            t = self._texts(perturb.perturb(BEAT, "offthread_generic", self.rng))
        finally:
            perturb.OTHER_POOL.clear()
        self.assertEqual(t[0], self._texts(BEAT)[0])  # the first line is kept
        self.assertEqual(t[1], "I hear Ryan's point, and zoos are cruel.")
        self.assertEqual(t[2], "I hear Amy's point, and mars needs more rockets.")

    def test_host_template(self) -> None:
        self.assertIsNone(perturb.perturb(BEAT, "host_template", self.rng))
        ep = _ep([("human", "Hi."), ("host", "Why Chicago?"), ("commentator", "Rent.")], host_scored=[1])
        p = perturb.perturb(ep, "host_template", self.rng)
        self.assertEqual(p["lines"][1]["text"], perturb.HOST_TEMPLATE)

    def test_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            perturb.perturb(BEAT, "nope", self.rng)

    def test_every_expected_kind_is_known(self) -> None:
        for kind in set(perturb.EXPECT) | set(perturb.EXPECT_V2):
            if kind in ("strip_reasons", "verbose_pad"):
                continue  # LLM rewrites (ruler.py rewrite)
            perturb.perturb(BEAT, kind, random.Random(0))


class StatsTests(unittest.TestCase):
    def test_alpha_nominal(self) -> None:
        self.assertAlmostEqual(stats.krippendorff_alpha({1: [0, 0], 2: [1, 1], 3: [0, 1]}), 4 / 9)
        self.assertAlmostEqual(stats.krippendorff_alpha({1: [0, 0, None], 2: [1, 1, 1]}), 1.0)
        self.assertTrue(math.isnan(stats.krippendorff_alpha({1: [0, None]})))

    def test_alpha_interval(self) -> None:
        self.assertAlmostEqual(stats.krippendorff_alpha({1: [1, 1], 2: [3, 3], 3: [5, 5]}, "interval"), 1.0)
        self.assertLess(stats.krippendorff_alpha({1: [1, 5], 2: [5, 1]}, "interval"), 0.0)

    def test_bradley_terry(self) -> None:
        s = stats.bradley_terry([("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 1.0)])
        self.assertGreater(s["a"], s["b"])
        self.assertGreater(s["b"], s["c"])
        self.assertAlmostEqual(sum(s.values()), 0.0, places=6)

    def test_bt_bootstrap(self) -> None:
        rows = [{"a": "anchor", "b": "system", "s": 0.9, "topic": t} for t in range(5)]
        rows += [{"a": "system", "b": "perturbed", "s": 0.8, "topic": t} for t in range(5)]
        point, ci = stats.bt_bootstrap(rows, "a", "b", "topic", boot=50)
        for k, v in point.items():
            self.assertLessEqual(ci[k][0], v + 1e-9)
            self.assertGreaterEqual(ci[k][1], v - 1e-9)
        self.assertGreater(point["anchor"], point["perturbed"])

    def test_logistic_length(self) -> None:
        b0, b1 = stats.logistic_length([0.8] * 20, [0.0] * 20)
        self.assertAlmostEqual(b0, math.log(4), delta=0.1)
        self.assertAlmostEqual(b1, 0.0, delta=0.05)

    def test_pearson(self) -> None:
        self.assertAlmostEqual(stats.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertTrue(math.isnan(stats.pearson([1, 2], [1, 2])))
        self.assertTrue(math.isnan(stats.pearson([1, 1, 1], [1, 2, 3])))


class JuryParsingTests(unittest.TestCase):
    def test_probs(self) -> None:
        lp = {0: SimpleNamespace(decoded_token=" Yes", logprob=math.log(0.3)),
              1: SimpleNamespace(decoded_token="no", logprob=math.log(0.1))}
        out = SimpleNamespace(outputs=[SimpleNamespace(logprobs=[lp])])
        self.assertEqual([round(x, 6) for x in jury._probs(out, ["yes", "no"])], [0.75, 0.25])
        self.assertEqual(jury._probs(out, ["a", "b"]), [0.5, 0.5])

    def test_parse_num(self) -> None:
        valid = {1: 4, 2: 5}
        self.assertEqual(jury.parse_num("2", valid), 2)
        self.assertEqual(jury.parse_num(" [1]", valid), 1)
        self.assertIsNone(jury.parse_num("NONE", valid))
        self.assertIsNone(jury.parse_num("7", valid))
        self.assertIsNone(jury.parse_num("", valid))

    def test_judge_scores(self) -> None:
        rows = [
            {"kind": "item", "ep": "e", "line": 1, "dim": "specific", "item": "S1", "reverse": True, "p_yes": 0.2},
            {"kind": "item", "ep": "e", "line": 1, "dim": "specific", "item": "S2", "reverse": False, "p_yes": 0.4},
            {"kind": "item", "ep": "e", "line": 1, "dim": "host", "item": "H1", "reverse": False, "p_yes": 0.9},
            {"kind": "attr", "ep": "e", "line": 1, "a_first": True, "true": "commentator", "p_a": 0.8},
            {"kind": "attr", "ep": "e", "line": 1, "a_first": False, "true": "commentator", "p_a": 0.4},
            {"kind": "item", "ep": "e", "line": 1, "dim": "specific", "item": "S3", "reverse": False, "p_yes": 0.9,
             "retest": True},
        ]
        s = ruler.judge_scores(rows)["e"]
        self.assertAlmostEqual(s["specific"], 0.5)  # S1 passes (reverse, "no"), S2 fails; the retest row is ignored
        self.assertAlmostEqual(s["host.H1"], 1.0)
        self.assertAlmostEqual(s["distinct"], (0.8 + 0.6) / 2)


if __name__ == "__main__":
    unittest.main()
