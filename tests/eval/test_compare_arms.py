"""compare_arms: first-sentence lengths and the line-length correction."""

from __future__ import annotations

import math
import unittest

from eval import compare_arms as ca


def _arm(name: str, points: list[tuple[float, float]]) -> ca.ArmStats:
    arm = ca.ArmStats(name)
    for x, y in points:
        arm.handover_first_s.append(x)
        arm.handover_parts.append({"tts": y * 0.4, "avatar": y * 0.6})
    return arm


class FirstSentenceTests(unittest.TestCase):
    def test_keeps_only_each_lines_first_chunk(self) -> None:
        rows = [
            {"event": "avatar_chunk_play", "ts": 1.0, "step": "intro_guest:c0", "pcm_duration_sec": 1.2},
            {"event": "avatar_chunk_play", "ts": 3.0, "step": "speech_guest_1:c0", "pcm_duration_sec": 4.0},
            {"event": "avatar_chunk_play", "ts": 7.0, "step": "speech_guest_1:c1", "pcm_duration_sec": 3.0},
            {"event": "line_end", "ts": 9.0},
        ]
        self.assertEqual(
            ca.first_chunks(rows), [(1.0, "intro_guest", 1.2), (3.0, "speech_guest_1", 4.0)]
        )

    def test_sums_the_lines_that_start_in_the_handover(self) -> None:
        chunks = [(0.5, "speech_host_0", 9.0), (1.0, "intro_guest", 1.2), (3.0, "speech_guest_1", 4.0),
                  (20.0, "open_floor", 2.0)]
        self.assertAlmostEqual(ca.first_sentence_s(chunks, 0.8, 3.0), 5.2)
        self.assertTrue(math.isnan(ca.first_sentence_s([], 0.8, 3.0)))


class LengthCorrectionTests(unittest.TestCase):
    def test_arms_with_the_same_cost_per_second_come_out_equal(self) -> None:
        # Silence = 1 + 0.3 s per s of first sentence; B's sentences run 2 s longer.
        a = _arm("A", [(x, 1 + 0.3 * x) for x in (8.0, 10.0, 12.0)])
        b = _arm("B", [(x, 1 + 0.3 * x) for x in (10.0, 12.0, 14.0)])
        ref, out = ca.length_corrected([a, b])
        self.assertAlmostEqual(ref, 11.0)
        self.assertAlmostEqual(out["A"][0], out["B"][0])
        self.assertAlmostEqual(out["A"][1], 0.3)
        self.assertAlmostEqual(out["B"][2], 12.0)

    def test_a_negative_slope_counts_as_none(self) -> None:
        a = _arm("A", [(8.0, 3.0), (10.0, 2.0), (12.0, 1.0)])
        b = _arm("B", [(14.0, 2.0), (16.0, 2.0), (18.0, 2.0)])
        _, out = ca.length_corrected([a, b])
        self.assertEqual(out["A"][1], 0.0)
        self.assertAlmostEqual(out["A"][0], 2.0)  # the raw mean, unmoved

    def test_too_few_handovers_or_no_lengths(self) -> None:
        a = _arm("A", [(8.0, 3.0), (10.0, 2.0)])
        b = _arm("B", [(math.nan, 2.0)] * 4)
        ref, out = ca.length_corrected([a, b])
        self.assertAlmostEqual(ref, 9.0)
        self.assertTrue(math.isnan(out["A"][0]))
        self.assertTrue(math.isnan(out["B"][0]))


if __name__ == "__main__":
    unittest.main()
