"""Unit tests for PAD decay/update and KNN emotion mapping (no MSP CSV required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.emotion.knn_map import PADEmotionKNN
from agent.emotion.msp_anchors import (
    canonicalize_emotion_label,
    normalize_pad_1_to_7,
)
from agent.emotion.pad_state import PADState, decay_pad, decay_then_update


class NormalizeTests(unittest.TestCase):
    def test_endpoints(self) -> None:
        self.assertAlmostEqual(normalize_pad_1_to_7(1.0), -1.0)
        self.assertAlmostEqual(normalize_pad_1_to_7(4.0), 0.0)
        self.assertAlmostEqual(normalize_pad_1_to_7(7.0), 1.0)

    def test_letter_map(self) -> None:
        self.assertEqual(canonicalize_emotion_label("A"), "Anger")
        self.assertEqual(canonicalize_emotion_label("O"), "Vague")
        self.assertEqual(canonicalize_emotion_label("happiness"), "Happiness")
        self.assertEqual(canonicalize_emotion_label("Angry"), "Anger")
        self.assertEqual(canonicalize_emotion_label("Happy"), "Happiness")
        self.assertEqual(canonicalize_emotion_label("Other-Concerned"), "Concerned")
        self.assertEqual(canonicalize_emotion_label("Other-Happy"), "Happiness")
        self.assertEqual(canonicalize_emotion_label("Other"), "Vague")


class PADUpdateTests(unittest.TestCase):
    def test_decay_halves_at_half_life(self) -> None:
        s = PADState(pleasure=1.0, arousal=0.5, dominance=-0.5)
        decay_pad(s, delta_t=10.0, half_life=10.0)
        self.assertAlmostEqual(s.pleasure, 0.5, places=5)
        self.assertAlmostEqual(s.arousal, 0.25, places=5)
        self.assertAlmostEqual(s.dominance, -0.25, places=5)

    def test_decay_then_update(self) -> None:
        s = PADState(pleasure=0.0, arousal=0.0, dominance=0.0)
        decay_then_update(s, (0.4, -0.2, 0.1), delta_t=0.0, half_life=100.0)
        self.assertAlmostEqual(s.pleasure, 0.4)
        self.assertAlmostEqual(s.arousal, -0.2)
        self.assertAlmostEqual(s.dominance, 0.1)


class KNNTests(unittest.TestCase):
    def test_nearest_label(self) -> None:
        pad = np.array(
            [
                [0.8, 0.6, 0.4],  # Happiness-ish
                [-0.7, 0.7, 0.5],  # Anger-ish
                [-0.6, -0.4, -0.3],  # Sadness-ish
            ],
            dtype=np.float64,
        )
        labels = ["Happiness", "Anger", "Sadness"]
        knn = PADEmotionKNN(pad, labels, k=2)
        r = knn.query([0.75, 0.55, 0.35])
        self.assertEqual(r.neighbor_labels[0], "Happiness")
        self.assertIn("Happiness", r.neighbor_labels)

    def test_roundtrip_npz(self) -> None:
        pad = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        labels = np.array(["Neutral", "Happiness"], dtype=object)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a.npz"
            np.savez_compressed(path, pad=pad.astype(np.float32), labels=labels)
            knn = PADEmotionKNN.from_npz(path, k=1)
            self.assertEqual(knn.query([0.9, 0.0, 0.0]).primary_label, "Happiness")


if __name__ == "__main__":
    unittest.main()


class KNNTieTests(unittest.TestCase):
    def test_stacked_ties_vote_by_share(self) -> None:
        # 7 Neutral + 3 Happiness stacked at the origin, one far Anger point.
        pad = np.array([[0.0, 0.0, 0.0]] * 10 + [[0.9, 0.9, 0.9]])
        labels = ["Happiness"] * 3 + ["Neutral"] * 7 + ["Anger"]
        r = PADEmotionKNN(pad, labels, k=3).query([0.0, 0.0, 0.0])
        self.assertEqual(r.primary_label, "Neutral")
        self.assertEqual(r.neighbor_labels, ("Neutral", "Neutral", "Happiness"))

    def test_no_tie_keeps_plain_order(self) -> None:
        pad = np.array([[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0]])
        r = PADEmotionKNN(pad, ["A", "B", "C", "D"], k=3).query([0.0, 0.0, 0.0])
        self.assertEqual(r.neighbor_labels, ("A", "B", "C"))

    def test_tie_is_order_independent(self) -> None:
        pad = np.zeros((6, 3))
        a = PADEmotionKNN(pad, ["X", "Y", "Y", "Z", "Y", "X"], k=3).query([0, 0, 0])
        b = PADEmotionKNN(pad, ["Y", "X", "Z", "Y", "X", "Y"], k=3).query([0, 0, 0])
        self.assertEqual(a.neighbor_labels, b.neighbor_labels)
        self.assertEqual(a.neighbor_labels, ("Y", "Y", "X"))


class SoftLabelTests(unittest.TestCase):
    def test_resolve_no_agreement(self) -> None:
        from agent.emotion.msp_anchors import VOTE_EMOTIONS, resolve_no_agreement

        def votes(**kw):
            return [kw.get(e, 0) for e in VOTE_EMOTIONS]

        self.assertEqual(
            resolve_no_agreement(votes(Fear=2, Contempt=2, Neutral=1)),
            (("Fear", 0.5), ("Contempt", 0.5)),
        )
        # Single top emotion (the tie was with free-text "Other" votes).
        self.assertEqual(resolve_no_agreement(votes(Sadness=2, Anger=1)), (("Sadness", 1.0),))
        self.assertEqual(
            resolve_no_agreement(votes(Anger=1, Sadness=1, Happiness=1, Neutral=1, Fear=1)),
            (("Vague", 1.0),),
        )
        self.assertEqual(resolve_no_agreement(votes()), (("Vague", 1.0),))

    def test_soft_labels_vote_by_weight(self) -> None:
        pad = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.9, 0.9, 0.9]])
        soft = [
            (("Fear", 0.5), ("Contempt", 0.5)),
            (("Fear", 0.5), ("Contempt", 0.5)),
            (("Neutral", 1.0),),
            (("Anger", 1.0),),
        ]
        knn = PADEmotionKNN(pad, [s[0][0] for s in soft], k=3, soft_labels=soft)
        r = knn.query([0.0, 0.0, 0.0])
        # Fear 1.0, Contempt 1.0, Neutral 1.0 → one slot each, nearest first.
        self.assertEqual(sorted(r.neighbor_labels), ["Contempt", "Fear", "Neutral"])
        self.assertEqual(r.primary_label, "Contempt")  # tie on weight+distance → name

    def test_npz_roundtrip_soft(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a.npz"
            np.savez_compressed(
                path,
                pad=np.zeros((2, 3), dtype=np.float32),
                labels=np.array(["Fear", "Neutral"], dtype=object),
                labels_b=np.array(["Contempt", ""], dtype=object),
                weights_a=np.array([0.5, 1.0], dtype=np.float32),
            )
            knn = PADEmotionKNN.from_npz(path, k=2)
            self.assertEqual(knn.soft_labels[0], (("Fear", 0.5), ("Contempt", 0.5)))
            self.assertEqual(knn.soft_labels[1], (("Neutral", 1.0),))
