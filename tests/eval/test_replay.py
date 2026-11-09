"""Ghost replay prefixes (agent/session/ghost_replay.py) and the client side. No LiveKit, no model."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent.session.ghost_flags import ghost_replay_enabled
from agent.session.ghost_replay import REPLAY_FLAGS, apply_prefix, snapshot_state
from agent.ui.control_events import parse_control_payload


def live_data():
    """A real TalkShowData (production scenario) after a short exchange, with mood state set."""
    from agent.config import load_scenario
    from agent.data import TalkShowData

    data = TalkShowData(scenario=load_scenario(), runtime=SimpleNamespace())  # type: ignore[arg-type]
    data.room_name, data.turn_log = "test", None
    for role, text in [
        ("host", "Tonight: should we build more nuclear power plants?"),
        ("guest", "Nuclear is the only firm low-carbon power we can scale."),
        ("commentator", "Firm, sure. Affordable, no."),
        ("human", "Vogtle ran 17 billion dollars over budget."),
    ]:
        data.show_history.append(role, text, speaker=role)
    data.role_pad = {"guest": (0.2, -0.1, 0.0), "commentator": (-0.3, 0.4, 0.1)}
    data.role_emotion = {"guest": "calm", "commentator": "irritated"}
    data.role_pad_neighbors = {"guest": ["calm", "content"]}
    data.dialogue_example_index = 3
    return data


def fresh_data():
    from agent.config import load_scenario
    from agent.data import TalkShowData

    data = TalkShowData(scenario=load_scenario(), runtime=SimpleNamespace())  # type: ignore[arg-type]
    data.room_name, data.turn_log = "replay", None
    data.show_history.append("host", "Welcome, I'm your host.", speaker="host")
    return data


class PrefixRoundTripTests(unittest.TestCase):
    def test_snapshot_apply_snapshot(self) -> None:
        snap = json.loads(json.dumps(snapshot_state(live_data())))  # as logged: JSON, tuples become lists
        target = fresh_data()
        apply_prefix(target, snap)
        self.assertEqual(snapshot_state(target), snap)
        self.assertEqual(target.role_pad["guest"], (0.2, -0.1, 0.0))  # tuples again
        self.assertEqual(set(target.role_pad_ts), {"guest", "commentator"})
        self.assertEqual([l.text for l in target.show_history.lines][-1], "Vogtle ran 17 billion dollars over budget.")

    def test_partial_prefix_keeps_the_rest(self) -> None:
        target = live_data()
        apply_prefix(target, {"role_emotion": {"guest": "sad"}})
        self.assertEqual(target.role_emotion, {"guest": "sad"})
        self.assertEqual(len(target.show_history.lines), 4)
        self.assertEqual(target.dialogue_example_index, 3)

    def test_flags_are_whitelisted(self) -> None:
        with mock.patch.dict(os.environ, {}):
            for k in REPLAY_FLAGS:
                os.environ.pop(k, None)
            os.environ.pop("TALKSHOW_SOMETHING_ELSE", None)
            apply_prefix(fresh_data(), {"flags": {"TALKSHOW_ADDRESSEE_FLOOR": 1, "TALKSHOW_PANEL_BEST_OF": "4",
                                                  "TALKSHOW_PANEL_CONCRETE": 0, "TALKSHOW_SOMETHING_ELSE": "1"}})
            self.assertEqual(os.environ["TALKSHOW_ADDRESSEE_FLOOR"], "1")
            self.assertEqual(os.environ["TALKSHOW_PANEL_BEST_OF"], "4")
            self.assertEqual(os.environ["TALKSHOW_PANEL_CONCRETE"], "0")
            self.assertNotIn("TALKSHOW_SOMETHING_ELSE", os.environ)

    def test_persona_swap(self) -> None:
        from agent.config import load_persona_name, load_persona_personality, set_persona_swap

        guest, comm = load_persona_personality("guest"), load_persona_personality("commentator")
        host = load_persona_personality("host")
        try:
            apply_prefix(fresh_data(), {"persona_swap": True})
            swapped = load_persona_personality("guest")
            self.assertNotEqual(swapped, guest)
            self.assertNotIn(load_persona_name("commentator"), swapped)
            self.assertEqual(load_persona_personality("host"), host)
        finally:
            set_persona_swap(False)
        self.assertEqual(load_persona_personality("guest"), guest)
        self.assertEqual(load_persona_personality("commentator"), comm)


class ReplayFlagTests(unittest.TestCase):
    def test_default_off(self) -> None:
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("TALKSHOW_GHOST_REPLAY", None)
            self.assertFalse(ghost_replay_enabled())
            os.environ["TALKSHOW_GHOST_REPLAY"] = "1"
            self.assertTrue(ghost_replay_enabled())

    def test_control_event_carries_prefix(self) -> None:
        ev = parse_control_payload(json.dumps({"type": "ghost_human_turn", "text": "hi", "prefix": {"lines": []}}).encode())
        self.assertEqual(ev.prefix, {"lines": []})
        ev = parse_control_payload(b'{"type":"ghost_human_turn","text":"hi","prefix":"bad"}')
        self.assertIsNone(ev.prefix)
        ev = parse_control_payload(b'{"type":"ghost_human_turn","text":"hi"}')
        self.assertIsNone(ev.prefix)


class ClientScriptTests(unittest.TestCase):
    def test_load_script_with_prefix(self) -> None:
        from eval.ghost_session.client import load_script

        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / "prefix.json"
            pf.write_text(json.dumps({"lines": [{"role_id": "host", "speaker": "Lessac", "text": "Hi"}]}))
            sc = Path(d) / "s.yaml"
            sc.write_text(f"prefix_file: {pf}\npersona_swap: true\nflags: {{TALKSHOW_ADDRESSEE_FLOOR: '1'}}\n"
                          "turns:\n  - text: Ryan, why?\n    topic: tipping\n")
            s = load_script(sc)
        self.assertEqual(s.prefix["lines"][0]["text"], "Hi")
        self.assertTrue(s.prefix["persona_swap"])
        self.assertEqual(s.prefix["flags"], {"TALKSHOW_ADDRESSEE_FLOOR": "1"})
        self.assertEqual([t.text for t in s.turns], ["Ryan, why?"])

    def test_load_script_without_prefix(self) -> None:
        from eval.ghost_session.client import load_script

        with tempfile.TemporaryDirectory() as d:
            sc = Path(d) / "s.yaml"
            sc.write_text("turns:\n  - text: Hello\n")
            self.assertIsNone(load_script(sc).prefix)

    def test_shipped_prefixes_load(self) -> None:
        root = Path(__file__).resolve().parents[2] / "eval" / "data" / "minimal_pairs" / "prefix"
        files = sorted(root.glob("*.json"))
        self.assertEqual(len(files), 30)
        target = fresh_data()
        apply_prefix(target, json.loads(files[0].read_text()))
        self.assertTrue(any(l.role_id in ("guest", "commentator") for l in target.show_history.lines))


if __name__ == "__main__":
    unittest.main()
