"""Unit tests for open-vocab emotion, PAD parse, and pad_pipeline materialize."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import numpy as np

from agent.emotion.knn_map import PADEmotionKNN
from agent.emotion.pad_pipeline import (
    apply_pad_delta_from_parsed,
    get_pad_knn,
    materialize_emotion_from_pad,
)
from agent.emotion.state import (
    DEFAULT_EMOTION,
    has_pad_tag,
    normalize_emotion,
    parse_pad_delta,
    strip_pad_tag,
)
from agent.emotion.tts import emotion_instruct


def _load_response_parser():
    """Load response_parser without importing agent.adapters package (livekit)."""
    import sys

    path = (
        Path(__file__).resolve().parents[2]
        / "agent"
        / "adapters"
        / "response_parser.py"
    )
    name = "response_parser_standalone"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _ListLog:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log(self, event: str, **fields) -> None:
        self.rows.append({"event": event, **fields})


@dataclass
class _FakeData:
    role_emotion: dict[str, str] = field(default_factory=dict)
    role_pad: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    role_pad_ts: dict[str, float] = field(default_factory=dict)
    turn_log: _ListLog | None = None
    room_name: str = "test"


class OpenVocabTests(unittest.TestCase):
    def test_normalize_msp_labels(self) -> None:
        self.assertEqual(normalize_emotion("Anger"), "Anger")
        self.assertEqual(normalize_emotion("Concerned"), "Concerned")
        self.assertEqual(normalize_emotion("amused"), "amused")  # legacy

    def test_emotion_instruct_embeds_msp(self) -> None:
        s = emotion_instruct(emotion="Concerned", locale="en")
        self.assertIn("Concerned", s)
        self.assertTrue(s.endswith("<|endofprompt|>"))
        anger = emotion_instruct(emotion="Anger", locale="en")
        self.assertIn("anger", anger.lower())


class PadParseTests(unittest.TestCase):
    def test_parse_pad_delta(self) -> None:
        self.assertEqual(parse_pad_delta("[pad]: 0.1 -0.2 0.0"), (0.1, -0.2, 0.0))
        self.assertEqual(parse_pad_delta("[pad]: +0.5 0 -.1"), (0.5, 0.0, -0.1))
        self.assertEqual(parse_pad_delta("[pad]: 0.2, -0.1, 0.05"), (0.2, -0.1, 0.05))
        self.assertEqual(parse_pad_delta("[pad] 0.1 -0.2 0"), (0.1, -0.2, 0.0))
        self.assertIsNone(parse_pad_delta("no pad here"))

    def test_parse_pad_delta_lenient_formats(self) -> None:
        self.assertEqual(parse_pad_delta("[pad: 0.1 -0.2 0.3]"), (0.1, -0.2, 0.3))
        self.assertEqual(parse_pad_delta("[PAD]: 0.1, 0.2, 0.3"), (0.1, 0.2, 0.3))
        self.assertEqual(
            parse_pad_delta("[pad]: P=0.2 A=-0.1 D=0.05"), (0.2, -0.1, 0.05)
        )
        self.assertEqual(
            parse_pad_delta("[pad]: ΔP=0.2, ΔA=-0.1, ΔD=0"), (0.2, -0.1, 0.0)
        )
        self.assertEqual(parse_pad_delta("[pad]: dP:0.2 dA:0.1 dD:-0.3"), (0.2, 0.1, -0.3))
        self.assertEqual(parse_pad_delta("[pad]: 0.2，−0.1，0"), (0.2, -0.1, 0.0))
        self.assertEqual(parse_pad_delta("[pad]: 3 0 0"), (1.0, 0.0, 0.0))
        self.assertIsNone(parse_pad_delta("[pad]: unchanged"))

    def test_multiword_appraisal_phrase(self) -> None:
        from agent.emotion.state import parse_pad_appraisal

        self.assertEqual(
            parse_pad_appraisal("[pad]: mildly challenged -0.10 0.05 -0.15"),
            ("mildly challenged", (-0.1, 0.05, -0.15)),
        )
        self.assertEqual(
            parse_pad_appraisal("[pad]: a bit uneasy -0.1 0.1 0"),
            ("a bit uneasy", (-0.1, 0.1, 0.0)),
        )
        self.assertEqual(
            parse_pad_appraisal("[pad]: ΔP 0.1 ΔA 0 ΔD 0"), (None, (0.1, 0.0, 0.0))
        )
        # Seen from Gemma on the Sentipolis benchmark.
        self.assertEqual(
            parse_pad_appraisal("[pad]: pleased, confident, in control 0.20 0.15 0.15"),
            ("pleased confident in control", (0.2, 0.15, 0.15)),
        )
        self.assertEqual(
            parse_pad_appraisal("[pad]: Concerned, wary < -0.15> < 0.10> < -0.05>"),
            ("concerned wary", (-0.15, 0.1, -0.05)),
        )
        self.assertEqual(
            parse_pad_appraisal("[pad]: curious, interested <0.10> <0.15> <0.05>"),
            ("curious interested", (0.1, 0.15, 0.05)),
        )

    def test_valence_class_fixes_sign(self) -> None:
        from agent.emotion.state import parse_pad_appraisal, parse_pad_appraisal_full

        r = parse_pad_appraisal_full("[pad]: deeply saddened | unpleasant | 0.35 -0.20 -0.10")
        self.assertEqual((r.word, r.valence, r.raw_p, r.flipped), ("deeply saddened", "unpleasant", 0.35, True))
        self.assertEqual(r.delta, (-0.35, -0.2, -0.1))
        r = parse_pad_appraisal_full("[pad]: validated | pleasant | -0.25 0.10 0.20")
        self.assertEqual(r.delta[0], 0.25)
        r = parse_pad_appraisal_full("[pad]: settled | neutral | 0.05 -0.10 0.00")
        self.assertFalse(r.flipped)
        # Valence written without pipes is split off the phrase.
        r = parse_pad_appraisal_full("[pad]: hurt unpleasant 0.2 0.1 -0.1")
        self.assertEqual((r.word, r.valence, r.delta[0]), ("hurt", "unpleasant", -0.2))
        # Old format (no class) is untouched.
        self.assertEqual(parse_pad_appraisal("[pad]: stung -0.3 0.25 -0.2"), ("stung", (-0.3, 0.25, -0.2)))

    def test_soft_bounded_add(self) -> None:
        from agent.emotion.pad_state import soft_bounded_add

        self.assertAlmostEqual(soft_bounded_add(0.0, 0.2), 0.2)
        self.assertAlmostEqual(soft_bounded_add(0.8, 0.2), 0.84)  # 0.2 × (1 − 0.8)
        self.assertAlmostEqual(soft_bounded_add(0.8, -0.2), 0.6)  # pull-back is full
        self.assertAlmostEqual(soft_bounded_add(-0.5, -0.4), -0.7)
        x = 0.0
        for _ in range(50):
            x = soft_bounded_add(x, 0.3)
        self.assertLess(x, 1.0)

    def test_modal_labels_ties(self) -> None:
        from agent.emotion.knn_map import modal_labels

        self.assertEqual(modal_labels(["Anger", "Surprise", "Contempt"]), ("Anger", "Surprise", "Contempt"))
        self.assertEqual(modal_labels(["Happiness", "Surprise", "Happiness"]), ("Happiness",))

    def test_has_pad_tag(self) -> None:
        self.assertTrue(has_pad_tag("x [pad: 0 0 0]"))
        self.assertTrue(has_pad_tag("[PAD]: nope"))
        self.assertFalse(has_pad_tag("[padding] notes"))
        self.assertFalse(has_pad_tag("launch pad"))

    def test_strip_pad(self) -> None:
        self.assertEqual(strip_pad_tag("Hello. [pad]: 0.1 0 0"), "Hello.")
        self.assertEqual(strip_pad_tag("Hello. [pad: 0.1, 0, 0]"), "Hello.")
        self.assertEqual(strip_pad_tag("Hello. [pad]: P=0.1 A=0 D=0"), "Hello.")

    def test_parse_host_speech_pad(self) -> None:
        mod = _load_response_parser()
        parsed = mod.parse_host_speech(
            "[reply]: Hello there.\n[next]: host\n[pad]: 0.2 -0.1 0.05\n"
        )
        # Dialogue no longer carries PAD; a stray [pad] is still never spoken.
        self.assertEqual(parsed.reply, "Hello there.")
        self.assertNotIn("[pad]", parsed.reply)


class PipelineTests(unittest.TestCase):
    def test_materialize_writes_msp_label(self) -> None:
        pad = np.array(
            [
                [0.8, 0.6, 0.4],
                [-0.7, 0.7, 0.5],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        labels = ["Happiness", "Anger", "Neutral"]
        knn = PADEmotionKNN(pad, labels, k=1)
        data = _FakeData()
        data.role_pad["guest"] = (0.75, 0.55, 0.35)
        with mock.patch(
            "agent.emotion.pad_pipeline.get_pad_knn", return_value=knn
        ):
            tag = materialize_emotion_from_pad(data, "guest", decay=False)
        self.assertEqual(tag, "Happiness")
        self.assertEqual(data.role_emotion["guest"], "Happiness")

    def test_apply_pad_delta_then_materialize(self) -> None:
        pad = np.array([[0.0, 0.0, 0.0], [0.9, 0.5, 0.3]], dtype=np.float64)
        labels = ["Neutral", "Happiness"]
        knn = PADEmotionKNN(pad, labels, k=1)
        data = _FakeData(turn_log=_ListLog())
        with mock.patch(
            "agent.emotion.pad_pipeline.get_pad_knn", return_value=knn
        ), mock.patch.dict(os.environ, {"TALKSHOW_PAD_DELTA_SCALE": "1"}):
            tag = apply_pad_delta_from_parsed(data, "host", (0.85, 0.45, 0.25))
        self.assertEqual(tag, "Happiness")
        self.assertAlmostEqual(data.role_pad["host"][0], 0.85, places=5)
        rows = [r for r in data.turn_log.rows if r["event"] == "pad_map"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["role"], "host")
        self.assertEqual(row["reason"], "delta")
        self.assertEqual(row["source"], "pad")
        self.assertEqual(row["emotion"], "Happiness")
        self.assertEqual(row["emotion_was"], "Neutral")
        self.assertEqual(row["delta"], [0.85, 0.45, 0.25])
        self.assertEqual(row["pad_before"], [0.0, 0.0, 0.0])
        self.assertEqual(row["pad_after"][0], 0.85)
        self.assertIn("Happiness", row["neighbors"])

    def test_delta_scale_amplifies(self) -> None:
        data = _FakeData()
        with mock.patch(
            "agent.emotion.pad_pipeline.get_pad_knn", return_value=None
        ), mock.patch.dict(os.environ, {"TALKSHOW_PAD_DELTA_SCALE": "2.5"}):
            apply_pad_delta_from_parsed(data, "guest", (0.1, -0.2, 0.5))
        self.assertEqual(
            tuple(round(x, 5) for x in data.role_pad["guest"]), (0.25, -0.5, 1.0)
        )

    def test_soft_bound_toggle(self) -> None:
        for env, expected in (({}, 0.84), ({"TALKSHOW_PAD_SOFT_BOUND": "0"}, 1.0)):
            data = _FakeData()
            data.role_pad["guest"] = (0.8, 0.0, 0.0)
            with mock.patch(
                "agent.emotion.pad_pipeline.get_pad_knn", return_value=None
            ), mock.patch.dict(os.environ, env):
                apply_pad_delta_from_parsed(data, "guest", (0.2, 0.0, 0.0))
            self.assertAlmostEqual(data.role_pad["guest"][0], expected, places=5)

    def test_npz_reload_hook(self) -> None:
        pad = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        labels = np.array(["Neutral"], dtype=object)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a.npz"
            np.savez_compressed(path, pad=pad.astype(np.float32), labels=labels)
            with mock.patch.dict(os.environ, {"TALKSHOW_MSP_ANCHORS": str(path)}):
                knn = get_pad_knn(force_reload=True)
                self.assertIsNotNone(knn)
                assert knn is not None
                self.assertEqual(knn.query([0.0, 0.0, 0.0]).primary_label, "Neutral")
            get_pad_knn(force_reload=True)  # reset singleton for other tests


class ShowStatePadTests(unittest.TestCase):
    def test_default_is_neutral_msp(self) -> None:
        self.assertEqual(DEFAULT_EMOTION, "Neutral")

    def test_base_state_includes_role_pad(self) -> None:
        from agent.show_graph.state import base_state

        s = base_state(
            phase="mod_start",
            trigger="t",
            panel_roles=["guest"],
            listen_role="host",
            participants=["host", "guest", "human"],
            role_pad={"host": (0.1, 0.0, -0.1)},
        )
        self.assertEqual(s["role_pad"]["host"], (0.1, 0.0, -0.1))
        self.assertIn("role_emotion", s)


if __name__ == "__main__":
    unittest.main()
