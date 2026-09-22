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

    def test_strip_pad(self) -> None:
        self.assertEqual(strip_pad_tag("Hello. [pad]: 0.1 0 0"), "Hello.")

    def test_parse_host_speech_pad(self) -> None:
        mod = _load_response_parser()
        parsed = mod.parse_host_speech(
            "[reply]: Hello there.\n[next]: host\n[pad]: 0.2 -0.1 0.05\n"
        )
        self.assertEqual(parsed.reply, "Hello there.")
        self.assertEqual(parsed.pad_delta, (0.2, -0.1, 0.05))
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
        ):
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
