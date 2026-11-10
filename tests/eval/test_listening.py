"""Listening test maths: response shift R on toy embeddings, judge answer parsing. CPU only, no model."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval import judge_context, responsiveness
from eval.data.minimal_pairs.make_pairs import parse_flags


def _row(slug: str, cond: str, k: int, vec_role: str = "guest") -> dict:
    return {"slug": slug, "topic": slug, "cond": cond, "k": k, "pos": 1, "role": vec_role, "text": f"{cond}{k}",
            "heard": "", "between": [], "log": f"session-{slug}-{cond}-{k}.jsonl"}


def _report(rows: list[dict], emb: list[list[float]], boot: int = 50) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        (out / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        np.save(out / "emb.npy", np.array(emb, dtype=np.float32))
        args = argparse.Namespace(out=str(out), lp_name="lp.jsonl", boot=boot, seed=0)
        with contextlib.redirect_stdout(io.StringIO()):
            responsiveness.cmd_report(args)
        return json.loads((out / "responsiveness.json").read_text())


class ResponseShiftTests(unittest.TestCase):
    """R = mean distance orig-vs-edit / mean distance between orig resamples, summed over topics."""

    # Two orig resamples 0.2 apart (cosine 0.8); the edited reply is far from both, the paraphrase close.
    O1, O2 = [1.0, 0.0], [0.8, 0.6]
    FAR, NEAR = [0.0, 1.0], [0.8, 0.6]

    def _rows_and_emb(self, n_topics: int = 2):
        rows, emb = [], []
        for t in range(n_topics):
            s = f"topic{t}"
            for k, v in enumerate((self.O1, self.O2)):
                rows.append(_row(s, "orig", k))
                emb.append(v)
            rows.append(_row(s, "number", 0))
            emb.append(self.FAR)
            rows.append(_row(s, "paraphrase", 0))
            emb.append(self.NEAR)
        return rows, emb

    def test_r_values(self) -> None:
        rep = _report(*self._rows_and_emb())
        pos1 = rep["positions"]["pos1"]
        # number: distances 1 - 0 = 1.0 and 1 - 0.6 = 0.4 -> 0.7; noise 0.2 -> R = 3.5
        self.assertAlmostEqual(pos1["number"]["R"], 3.5, places=3)
        # paraphrase: distances 0.2 and 0.0 -> 0.1; R = 0.5
        self.assertAlmostEqual(pos1["paraphrase"]["R"], 0.5, places=3)

    def test_identical_topics_give_a_point_ci(self) -> None:
        rep = _report(*self._rows_and_emb())
        lo, hi = rep["positions"]["pos1"]["number"]["ci95"]
        self.assertAlmostEqual(lo, 3.5, places=3)
        self.assertAlmostEqual(hi, 3.5, places=3)

    def test_sums_over_topics_not_mean_of_ratios(self) -> None:
        rows, emb = self._rows_and_emb(1)
        # second topic: noise 0.0 + 1.0 apart orig resamples ([1,0] and [0,1]); edit identical to the first orig
        s = "topic1"
        for k, v in enumerate(([1.0, 0.0], [0.0, 1.0])):
            rows.append(_row(s, "orig", k))
            emb.append(v)
        rows.append(_row(s, "number", 0))
        emb.append([1.0, 0.0])
        rep = _report(rows, emb)
        # topic0: D_edit 0.7, D_noise 0.2; topic1: D_edit (0 + 1) / 2 = 0.5, D_noise 1.0 -> (0.7 + 0.5) / 1.2 = 1.0
        self.assertAlmostEqual(rep["positions"]["pos1"]["number"]["R"], 1.0, places=3)

    def test_landing_share(self) -> None:
        rows, emb = self._rows_and_emb(1)
        rep = _report(rows, emb)
        # orig addresses the commentator; every first line here is the guest's
        self.assertEqual(rep["landing"]["orig"]["first_line_by_addressed"], 0.0)
        self.assertEqual(rep["landing"]["orig"]["n"], 2)

    def test_ci_helper_ignores_nan(self) -> None:
        lo, hi = responsiveness._ci([float("nan"), 1.0, 2.0, 3.0])
        self.assertEqual((lo, hi), (1.0, 2.0))
        self.assertTrue(all(math.isnan(x) for x in responsiveness._ci([float("nan")])))

    def test_room_name_parsing(self) -> None:
        m = responsiveness.ROOM_RE.match("ghost-tipping__number__k3__vB-123-1700000000")
        self.assertEqual((m["slug"], m["cond"], m["k"], m["arm"]), ("tipping", "number", "3", "B"))
        m = responsiveness.ROOM_RE.match("ghost-four-day-week__persona_swap__k0-123-1700000000")
        self.assertEqual((m["slug"], m["cond"], m["arm"]), ("four-day-week", "persona_swap", None))
        self.assertIsNone(responsiveness.ROOM_RE.match("ghost-pfx-tipping-123-1700000000"))


def _out(tokens: dict[str, float]):
    """A fake vLLM output whose first generated position has these top tokens (token -> probability)."""
    lp = {i: SimpleNamespace(decoded_token=tok, logprob=math.log(p)) for i, (tok, p) in enumerate(tokens.items())}
    return SimpleNamespace(outputs=[SimpleNamespace(logprobs=[lp])])


class JudgeParsingTests(unittest.TestCase):
    def test_p_answer_a(self) -> None:
        self.assertAlmostEqual(judge_context.p_answer_a(_out({" A": 0.6, "B": 0.2, "C": 0.1})), 0.75)

    def test_case_and_space_merge(self) -> None:
        # " a" and "A" are the same answer; mass adds up
        self.assertAlmostEqual(judge_context.p_answer_a(_out({"A": 0.3, " a": 0.3, " B": 0.2})), 0.75)

    def test_no_answer_token_is_a_coin_flip(self) -> None:
        self.assertEqual(judge_context.p_answer_a(_out({"The": 0.9})), 0.5)
        self.assertEqual(judge_context.p_answer_yes(_out({"Maybe": 0.9})), 0.5)

    def test_p_answer_yes(self) -> None:
        self.assertAlmostEqual(judge_context.p_answer_yes(_out({"Yes": 0.2, " no": 0.6})), 0.25)


class ScriptFlagTests(unittest.TestCase):
    def test_parse_flags(self) -> None:
        self.assertEqual(parse_flags("TALKSHOW_ADDRESSEE_FLOOR=1, TALKSHOW_UPTAKE_PLANNER=lite,"),
                         {"TALKSHOW_ADDRESSEE_FLOOR": "1", "TALKSHOW_UPTAKE_PLANNER": "lite"})
        self.assertEqual(parse_flags(""), {})


if __name__ == "__main__":
    unittest.main()
