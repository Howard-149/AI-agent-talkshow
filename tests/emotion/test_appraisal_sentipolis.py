"""Sentipolis Fig. 4 harness: replay + scoring with a scripted responder (no LLM)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "eval"))

import appraisal_sentipolis as H  # noqa: E402


def _fixture() -> dict:
    return yaml.safe_load(H.DEFAULT_FIXTURE.read_text(encoding="utf-8"))


def _paper_responder(fx: dict):
    """Answer each call with the paper's Δ for that agent and round (round unit)."""
    names = {a["name"]: aid for aid, a in fx["agents"].items()}
    calls = {aid: 0 for aid in fx["agents"]}

    def respond(system: str, user: str) -> str:
        aid = next(v for k, v in names.items() if f"reactions of {k}" in system)
        d = fx["rounds"][calls[aid]]["paper_delta"][aid]
        calls[aid] += 1
        return f"[pad]: fine {d[0]} {d[1]} {d[2]}"

    return respond


class HarnessTests(unittest.TestCase):
    def test_fixture_sums_match_paper_finals(self) -> None:
        fx = _fixture()
        for aid, a in fx["agents"].items():
            total = [sum(r["paper_delta"][aid][i] for r in fx["rounds"]) for i in range(3)]
            for i in range(3):
                self.assertAlmostEqual(a["initial_pad"][i] + total[i], a["final_pad"][i], places=2)

    def test_perfect_responder_scores_perfectly(self) -> None:
        fx = _fixture()
        trial = H.run_trial(fx, _paper_responder(fx), unit="round", update="plain")
        rep = H.score(fx, [trial])
        for a in rep["agents"].values():
            self.assertEqual(set(a["sign_agreement"].values()), {1.0})
            self.assertEqual(set(a["mae"].values()), {0.0})
            for k in H.AXES:
                self.assertAlmostEqual(a["final_pad_mean_std"][k][0], a["paper_final_pad"][k], places=2)
        self.assertEqual(rep["pleasure_divergence_rate"], 1.0)
        self.assertEqual(rep["parse_failures"], 0)
        self.assertEqual(rep["calls"], 8)

    def test_line_unit_only_appraises_the_other_speaker(self) -> None:
        fx = _fixture()
        seen: list[str] = []

        def respond(system: str, user: str) -> str:
            seen.append(system.split("reactions of ")[1].split(" in a")[0])
            return "[pad]: unmoved 0 0 0"

        H.run_trial(fx, respond, unit="line", update="soft")
        self.assertEqual(len(seen), 8)  # 4 rounds × 2 lines × 1 listener
        self.assertEqual(seen[:2], ["John Lin", "Tom Moreno"])


    def test_exchange_unit_pairs_reply_with_own_line(self) -> None:
        fx = _fixture()
        users: list[tuple[str, str]] = []

        def respond(system: str, user: str) -> str:
            users.append((system.split("reactions of ")[1].split(" in a")[0], user))
            return "[pad]: unmoved 0 0 0"

        H.run_trial(fx, respond, unit="exchange", update="plain")
        self.assertEqual(len(users), 8)
        # First line of the show has nothing to reply to; every later line is a reply.
        self.assertIn("New line", users[0][1])
        tom_r1 = users[1]
        self.assertEqual(tom_r1[0], "Tom Moreno")
        self.assertIn("Tom Moreno's own line and the reply to it", tom_r1[1])
        self.assertIn("Morning, John", tom_r1[1])


if __name__ == "__main__":
    unittest.main()
