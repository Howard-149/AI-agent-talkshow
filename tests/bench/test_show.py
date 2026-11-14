"""T6 full show (eval/bench/show.py): scripts, episode generation with fake models, counted metrics, judge plumbing."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from eval.bench import build, floor, judge, show, systems

FLAGS = ("TALKSHOW_DIALOGUE_LIBRARY", "TALKSHOW_PANEL_CONCRETE", "TALKSHOW_PANEL_BEST_OF", "TALKSHOW_UPTAKE_PLANNER",
         "TALKSHOW_PANEL_SCAFFOLD_FIX", "TALKSHOW_ADDRESSEE_FLOOR", "TALKSHOW_FACT_PACK", "TALKSHOW_FACT_PACK_K",
         "TALKSHOW_FACT_PACK_TOPIC", "TALKSHOW_GHOST_REPLAY")

SILENT = json.dumps({"think": "", "action": {"decision": "silent", "intent": None, "content": None,
                                             "backchannels": []}})


def speak(text: str) -> str:
    return json.dumps({"think": "", "action": {"decision": "speak", "intent": {"type": "contribute", "goal": "g",
                                                                               "target": None},
                                               "content": text, "backchannels": []}})


class _Env(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"TALKSHOW_TEXT_TOKENIZER": "word"})
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in FLAGS:
            os.environ.pop(k, None)
        self.tmp = Path(tempfile.mkdtemp())

    def gen(self, *argv: str) -> Path:
        out = self.tmp / "_".join(a.strip("-") for a in argv if not a.startswith("-"))[:60]
        with contextlib.redirect_stdout(io.StringIO()):
            show.main(["generate", *argv, "--topics", "tipping,ai-jobs", "--reruns", "2", "--out", str(out)])
        return out


class ScriptTests(unittest.TestCase):
    def test_dev_scripts(self) -> None:
        scripts = show.load_scripts()
        pairs = {t["slug"]: t for t in yaml.safe_load(build.DEV_PAIRS.read_text())["topics"]}
        self.assertEqual({s["slug"] for s in scripts}, set(pairs))
        for s in scripts:
            a1, a2, a3 = s["audience"]
            self.assertEqual((a1, a2), (pairs[s["slug"]]["turn1"], pairs[s["slug"]]["turn2"]["orig"]))
            self.assertTrue(a2.startswith("Ryan, ") and a3.startswith("Amy, "), s["slug"])

    def test_bad_script_refused(self) -> None:
        path = Path(tempfile.mkdtemp()) / "s.yaml"
        path.write_text(yaml.safe_dump({"topics": [{"slug": "x", "topic": "X?", "audience": ["a", "b"]}]}))
        with self.assertRaises(ValueError):
            show.load_scripts(path)
        with self.assertRaises(SystemExit):
            show.load_scripts(topics="no-such-topic")


class GenerateTests(_Env):
    def test_episode_shape(self) -> None:
        for argv in (("--system", "think_action", "--arm", "fast_v2_lib_force"), ("--system", "plain")):
            out = self.gen(*argv, "--fake")
            eps = [json.loads(l) for l in (out / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(eps), 4)
            for e in eps:
                L = e["lines"]
                self.assertEqual(len(e["audience_idx"]), 3)
                self.assertEqual([L[i]["text"] for i in e["audience_idx"]], e["audience"])
                # SLOTS show lines after each audience turn, nothing else but the procedural opening
                gaps = [b - a - 1 for a, b in zip(e["audience_idx"], [*e["audience_idx"][1:], len(L)])]
                self.assertEqual(gaps, [show.SLOTS] * 3)
                self.assertTrue(all(L[i]["role"] in show.PANEL for i in e["scored"]))
                self.assertTrue(all(l.get("procedural") for l in L[: e["audience_idx"][0]]))
            self.assertEqual(json.loads((out / "run.json").read_text())["mode"], "fake")

    def test_arms_differ_in_the_prompts(self) -> None:
        """Each arm's flags reach the run: the planner runs on fast arms only."""
        rec = {}
        for arm in ("fast_v2_lib_force", "production"):
            out = self.gen("--system", "think_action", "--arm", arm, "--dry-run")
            rec[arm] = [json.loads(l) for l in (out / "prompts.jsonl").read_text().splitlines()]
        # the planner runs on fast arms only, once per audience turn (2 topics; a dry run makes 1 rerun)
        self.assertEqual(sum(r["kind"] == "planner" for r in rec["fast_v2_lib_force"]), 2 * 3)
        self.assertEqual(sum(r["kind"] == "planner" for r in rec["production"]), 0)
        # every floor role but the last speaker thinks: 3 after an audience turn, 2 after a show line
        thinks = sum(r["kind"].startswith("think") for r in rec["production"])
        self.assertEqual(thinks, 2 * 3 * (3 + 2 * (show.SLOTS - 1)))

    def test_host_fallback_and_forced_panelist(self) -> None:
        calls = {"controller": 0}

        def respond(r: systems.Request) -> str:
            if r.kind == "controller":
                calls["controller"] += 1
                return json.dumps({"decision": "host_fallback", "selected_role": "host",
                                   "reason_code": "no_suitable_candidate", "reason": "r",
                                   "host_fallback": {"strategy": "invite", "target": "guest",
                                                     "instruction": "Ask Amy for an example."}})
            if r.kind == "think:guest":
                return SILENT
            if r.kind == "think:commentator":
                return SILENT
            if r.kind == "think:host":
                return SILENT
            if r.kind == "host":
                return "Amy, give us one real example."
            if r.kind == "fallback":
                return "A worker line from Amy."
            return systems.FakeBackend().default(r)

        backend = systems.FakeBackend(respond)
        system = systems.make_system("think_action", backend, "fast_v2_lib_force")
        eps = [show.new_episode(s, "x", 0) for s in show.load_scripts(topics="tipping")]
        with contextlib.redirect_stdout(io.StringIO()):
            show.run_think_action(system, backend, eps, slots=2)
        e = show.finish(eps[0])
        L = e["lines"]
        srcs = [L[i + 1].get("source") for i in e["audience_idx"]]
        # nobody bids: after turn 1 (names nobody) the controller directs the host; turns 2 and 3 name Ryan and Amy,
        # who are forced on (ADDRESSEE_FLOOR=force) with the worker draft
        self.assertEqual(srcs, ["host_fallback", "worker_fallback", "worker_fallback"])
        self.assertEqual(L[e["audience_idx"][0] + 1]["text"], "Amy, give us one real example.")
        self.assertEqual([L[e["audience_idx"][n] + 1]["role"] for n in (1, 2)], ["commentator", "guest"])
        self.assertGreater(calls["controller"], 0)

    def test_arbitrate_state_is_not_mutated(self) -> None:
        from agent.interaction.think_action import parse_think_action

        prefix = {"lines": [{"role_id": "host", "speaker": "Lessac", "text": "Welcome.", "procedural": True},
                            {"role_id": "human", "speaker": "Human", "text": "Is tipping out of hand?",
                             "procedural": False}]}
        acts = {"commentator": parse_think_action(speak("Yes it is."), floor_state="free", interval_text="x",
                                                  allow_backchannels=True).action}
        state = {"actual_floor_counts": {"commentator": 2}}
        got = floor.arbitrate(prefix, None, acts, answer=json.dumps(
            {"decision": "speak", "selected_role": "commentator", "reason_code": "high_contribution", "reason": "r"}),
            state=state)
        self.assertEqual(got["outcome"].selected_role, "commentator")
        self.assertEqual(state, {"actual_floor_counts": {"commentator": 2}})
        after = floor.record_grant(got["state"], "commentator")
        self.assertEqual(after["actual_floor_counts"]["commentator"], 3)


class MetricTests(unittest.TestCase):
    def ep(self, ryan: list[str], amy: list[str]) -> dict:
        lines = [{"role": "human", "speaker": "H", "text": "What about tips?"}]
        for r, a in zip(ryan, amy):
            lines += [{"role": "commentator", "speaker": "Ryan", "text": r}, {"role": "guest", "speaker": "Amy", "text": a}]
        e = {"id": "e", "topic_id": "tipping", "topic": "Tipping", "lines": lines}
        return show.finish(e)

    def test_openers_and_self_repetition(self) -> None:
        m = show.counted(self.ep(["Look, tips are a tax.", "Look, wages matter.", "Look, again."],
                                 ["Honestly I disagree.", "My cousin waits tables.", "Honestly no."]))
        self.assertEqual(m["opener.same_first_word_as_own_previous"], 2 / 4)  # Ryan 2 of 2, Amy 0 of 2
        self.assertAlmostEqual(m["opener.top_first_word_share"], (3 / 3 + 2 / 3) / 2)
        self.assertEqual(m["opener.same_first_word_adjacent_lines"], 0.0)
        rep = show.counted(self.ep(["Service charges replace tipping entirely.", "Service charges replace tipping "
                                    "entirely, really."], ["One.", "Two."]))
        # Ryan's second line repeats his first (overlap 1), Amy's does not (0)
        self.assertEqual((rep["self_repeat.max_bigram_jaccard"], rep["self_repeat.share_lines_over_0.2"]), (0.5, 0.5))
        self.assertEqual(rep["floor.panel_lines"], 4)


class CompareTests(_Env):
    def test_requests_judges_and_results(self) -> None:
        a = self.gen("--system", "think_action", "--arm", "fast_v2_lib_force", "--fake")
        b = self.gen("--system", "plain", "--fake")
        cmp_dir = self.tmp / "cmp"
        argv = ["compare", "--runs", f"{a},{b}", "--base", str(a), "--out", str(cmp_dir), "--boot", "20"]
        with contextlib.redirect_stdout(io.StringIO()):
            res = show.main(argv) or json.loads((cmp_dir / "results.json").read_text())
        self.assertEqual(set(res["counted"]), {"think_action-fast_v2_lib_force", "plain"})
        pw = [json.loads(l) for l in (cmp_dir / "judge" / "pairwise.jsonl").read_text().splitlines()]
        up = [json.loads(l) for l in (cmp_dir / "judge" / "uptake.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["kind"] == "systems" for r in pw), 4 * 2)  # 4 episode pairs, both orders
        self.assertEqual(sum(r["kind"] == "perturbation" for r in pw), 4 * len(show.PERTURBATIONS) * 2)
        self.assertEqual(len(up), 2 * 4 * 3)
        self.assertTrue(all(r["messages"][-1]["content"].endswith("A or B.") for r in pw))
        for tag in ("j1", "j2"):
            with contextlib.redirect_stdout(io.StringIO()):
                judge.main(["--requests", str(cmp_dir), "--tag", tag, "--fake", "--rubric-reruns", "all"])
        with contextlib.redirect_stdout(io.StringIO()):
            show.main(argv)
        res = json.loads((cmp_dir / "results.json").read_text())
        j = res["judge"]
        self.assertEqual(j["judges"], ["j1", "j2"])
        self.assertEqual(set(j["bt"]), {"j1", "j2", "pooled"})
        self.assertEqual(set(j["perturbation_gate"]["j1"]), set(show.PERTURBATIONS))
        self.assertIn("agreement_pairwise_alpha", j)
        self.assertEqual(j["uptake"]["j1"]["plain"]["n"], 12)
        self.assertIn("Bradley-Terry", (cmp_dir / "results.md").read_text())


if __name__ == "__main__":
    unittest.main()
