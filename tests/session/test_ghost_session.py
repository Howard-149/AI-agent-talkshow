"""Ghost-session inject contract (no LiveKit)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.session.ghost_flags import ghost_turns_allowed, is_ghost_participant
from agent.ui.control_events import parse_control_payload
from eval.ghost_session.client import load_script
from eval.ghost_session.judge_pad import format_pad_judgment, judge_pad_events
from eval.ghost_session.record_frontend import viewer_page_url
from eval.ghost_session.summarize import find_session_log, summarize_session
from eval.log_parse import extract_human_turns, load_events


class ControlEventTests(unittest.TestCase):
    def test_parse_ghost_human_turn(self) -> None:
        ev = parse_control_payload(
            b'{"type":"ghost_human_turn","text":"freeze hiring"}'
        )
        assert ev is not None
        self.assertEqual(ev.type, "ghost_human_turn")
        self.assertEqual(ev.text, "freeze hiring")

    def test_parse_unknown_ignored(self) -> None:
        self.assertIsNone(parse_control_payload(b'{"type":"nope"}'))


class GhostGuardTests(unittest.TestCase):
    def test_allowed_default(self) -> None:
        os.environ.pop("TALKSHOW_GHOST_TURNS", None)
        self.assertTrue(ghost_turns_allowed())

    def test_kill_switch(self) -> None:
        os.environ["TALKSHOW_GHOST_TURNS"] = "0"
        try:
            self.assertFalse(ghost_turns_allowed())
        finally:
            os.environ.pop("TALKSHOW_GHOST_TURNS", None)

    def test_identity_prefix(self) -> None:
        self.assertTrue(
            is_ghost_participant(SimpleNamespace(identity="ghost-guest", metadata=""))
        )
        self.assertFalse(
            is_ghost_participant(SimpleNamespace(identity="howard", metadata=""))
        )

    def test_metadata_flag(self) -> None:
        self.assertTrue(
            is_ghost_participant(
                SimpleNamespace(
                    identity="runner",
                    metadata=json.dumps({"ghost": True, "locale": "en"}),
                )
            )
        )


class ScriptAndLogTests(unittest.TestCase):
    def test_viewer_page_url(self) -> None:
        url = viewer_page_url(
            frontend_url="http://localhost:3000/",
            livekit_url="wss://example.livekit.cloud",
            token="abc",
            locale="en",
        )
        self.assertTrue(url.startswith("http://localhost:3000/custom/?"))
        self.assertIn("record=1", url)
        self.assertIn("token=abc", url)

    def test_track_is_audio_uses_protobuf_int(self) -> None:
        from eval.ghost_session.record_frontend import track_is_audio

        self.assertTrue(track_is_audio(SimpleNamespace(kind=1)))
        self.assertFalse(track_is_audio(SimpleNamespace(kind=2)))
        self.assertFalse(track_is_audio(SimpleNamespace(kind=None)))

    def test_audio_mux_shift_drops_leading_wav(self) -> None:
        from eval.ghost_session.record_frontend import audio_mux_shift_s

        self.assertEqual(audio_mux_shift_s(audio_t0=None, video_t0=1.0), 0.0)
        self.assertAlmostEqual(audio_mux_shift_s(audio_t0=10.0, video_t0=12.5), -2.5)
        self.assertAlmostEqual(audio_mux_shift_s(audio_t0=14.0, video_t0=12.0), 2.0)

    def test_load_smoke_script(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "eval"
            / "ghost_session"
            / "scripts"
            / "smoke.yaml"
        )
        script = load_script(path)
        self.assertEqual(script.locale, "en")
        self.assertGreaterEqual(len(script.turns), 1)
        self.assertIn("hiring", script.turns[0].text.lower())
        swing = load_script(path.parent / "pad_swing.yaml")
        self.assertEqual(len(swing.turns), 2)
        self.assertIn("humiliation", swing.turns[0].text.lower())
        self.assertIn("relieved", swing.turns[1].text.lower())

    def test_extract_ghost_turn_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "session-1.jsonl"
            rows = [
                {"ts": 1.0, "event": "session_start", "room": "talkshow-dev"},
                {"ts": 2.0, "event": "ghost_human_start", "heard": "hello panel"},
                {
                    "ts": 3.0,
                    "event": "ghost_human_done",
                    "heard": "hello panel",
                    "reply": "Ryan, take this.",
                    "raw": "Ryan, take this.",
                    "pad_delta": None,
                },
                {
                    "ts": 4.0,
                    "event": "panel_model_done",
                    "role": "commentator",
                    "reply": "I disagree.",
                    "reply_len": 12,
                    "raw": "I disagree.\n[next]: host\n[pad]: 0.20 -0.10 0.00",
                    "pad_delta": [0.2, -0.1, 0.0],
                },
                {
                    "ts": 4.1,
                    "event": "pad_map",
                    "role": "commentator",
                    "pad_before": [0.0, 0.0, 0.0],
                    "pad_after": [0.2, -0.1, 0.0],
                    "delta": [0.2, -0.1, 0.0],
                    "emotion_was": "Neutral",
                    "emotion": "Concerned",
                    "neighbors": ["Concerned", "Sadness"],
                },
            ]
            log.write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
            )
            turns = extract_human_turns(load_events(log))
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0].heard, "hello panel")
            self.assertEqual(turns[0].host_reply, "Ryan, take this.")
            self.assertEqual(len(turns[0].panel_replies), 1)
            recap = summarize_session(log)
            self.assertIn("hello panel", recap)
            self.assertIn("model_raw:", recap)
            self.assertIn("NO [pad] tag", recap)
            self.assertIn("[pad]: 0.20 -0.10 0.00", recap)
            self.assertIn("pad_map", recap)
            self.assertIn("Neutral→Concerned", recap)
            judged = format_pad_judgment(judge_pad_events(load_events(log)))
            self.assertIn("pad_map: 1", judged)
            self.assertIn("Concerned", judged)
            self.assertEqual(find_session_log(Path(tmp), "talkshow-dev"), log)
            self.assertIsNone(find_session_log(Path(tmp), "other"))

    def test_judge_pad_flags_pleasure_mismatch(self) -> None:
        events = [
            {
                "ts": 1.0,
                "event": "ghost_human_done",
                "heard": "security boxed his desk. public humiliation.",
                "reply": "That walkout was cruel.",
            },
            {
                "ts": 2.0,
                "event": "pad_map",
                "role": "host",
                "pad_before": [0.0, 0.0, 0.0],
                "pad_after": [0.4, 0.1, 0.0],
                "delta": [0.4, 0.1, 0.0],
                "emotion_was": "Neutral",
                "emotion": "Happiness",
                "neighbors": ["Happiness"],
            },
        ]
        report = judge_pad_events(events)
        self.assertEqual(report["n"], 1)
        self.assertEqual(report["rows"][0]["verdict"], "check")


if __name__ == "__main__":
    unittest.main()
