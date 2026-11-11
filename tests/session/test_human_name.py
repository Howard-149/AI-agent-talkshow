"""The human guest is called by the name they joined with."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from agent.show import show_history
from agent.show.show_history import (
    HistoryLine,
    human_label,
    human_name,
    human_name_note,
    name_room_human,
    set_human_name,
)


def _room(*participants: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(remote_participants={p.identity: p for p in participants})


def _p(identity: str, name: str = "", meta: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(identity=identity, name=name, metadata=json.dumps(meta) if meta else "")


class HumanNameTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_human_name(None)

    def test_the_room_names_the_human_and_skips_the_agent_and_the_recording_page(self) -> None:
        room = _room(
            _p("agent-1", "Lessac", {"talkshowAgent": True}),
            _p("ghost-viewer", "ghost-viewer", {"ghostViewer": True}),
            _p("ghost-guest", "Howard", {"ghost": True}),
        )
        self.assertEqual(name_room_human(room), "Howard")
        self.assertEqual(human_name(), "Howard")
        self.assertEqual(human_label(), "Howard (human guest)")
        self.assertIn("Howard", human_name_note())
        line = HistoryLine("human", show_history._speaker_label("human"), "hi")
        self.assertEqual(show_history._line_label(line), "Howard (human guest)")

    def test_no_name_of_their_own_keeps_human_guest(self) -> None:
        for participant in (_p("ghost-guest", "ghost-guest"), _p("ghost-guest", "Guest"), _p("u", "")):
            set_human_name("Someone")
            self.assertIsNone(name_room_human(_room(participant)))
            self.assertEqual(human_name(), "Human guest")
            self.assertEqual(human_label(), "Human guest")
            self.assertEqual(human_name_note(), "")


if __name__ == "__main__":
    unittest.main()
