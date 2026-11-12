"""A mood change is shown when the line that caused it is heard, not when it is written."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from agent.emotion import appraisal


class PadDisplaySyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sent: list[dict] = []
        for patcher in (
            mock.patch.object(appraisal, "_send_pad_state", lambda fields: self.sent.append(fields)),
            mock.patch("agent.emotion.role_pad.get_role_pad", lambda data, role: (0.1, 0.2, 0.3)),
            mock.patch.object(appraisal, "get_role_emotion", lambda data, role: "Anger"),
            mock.patch.dict("os.environ", {"TALKSHOW_PAD_DISPLAY_SYNC": "1"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.data = SimpleNamespace(pad_display_pending=[])

    def roles(self) -> list[str]:
        return [fields["role"] for fields in self.sent]

    def test_a_reaction_to_an_ai_line_waits_for_that_line(self) -> None:
        line = ("commentator", "But 'betrayal' is just the symptom, isn't it?")
        appraisal.publish_pad_state(self.data, "guest", feeling="stung", source=line)
        self.assertEqual(self.sent, [])
        appraisal.release_pad_display(self.data, "host", "Another line nobody wrote about.")
        self.assertEqual(self.sent, [])
        appraisal.release_pad_display(self.data, *line)
        self.assertEqual(self.roles(), ["guest"])
        self.assertEqual(self.sent[0]["feeling"], "stung")

    def test_a_reaction_to_the_human_shows_at_once(self) -> None:
        appraisal.publish_pad_state(self.data, "guest", source=("human", "I was fired."))
        appraisal.publish_pad_state(self.data, "guest")
        self.assertEqual(self.roles(), ["guest", "guest"])

    def test_a_line_releases_everything_held_before_it(self) -> None:
        first = ("guest", "It sounds like there's a deep current of cynicism here.")
        second = ("commentator", "Cynicism? No, this is just how the system works.")
        appraisal.publish_pad_state(self.data, "host", source=first)
        appraisal.publish_pad_state(self.data, "host", source=second)
        appraisal.publish_pad_state(self.data, "guest", source=second)
        # The first line was dropped (cut before a sound); the second one plays.
        self.assertEqual(appraisal.release_pad_display(self.data, *second), 3)
        self.assertEqual(self.data.pad_display_pending, [])

    def test_a_cut_line_matches_the_line_it_was_written_as(self) -> None:
        written = ("host", "Wow. That really puts the human cost of these decisions into focus.")
        appraisal.publish_pad_state(self.data, "guest", source=written)
        appraisal.release_pad_display(self.data, "host", "Wow. That really puts the human")
        self.assertEqual(self.roles(), ["guest"])

    def test_off_switch_shows_at_once(self) -> None:
        with mock.patch.dict("os.environ", {"TALKSHOW_PAD_DISPLAY_SYNC": "0"}):
            appraisal.publish_pad_state(self.data, "guest", source=("host", "A line written ahead."))
        self.assertEqual(self.roles(), ["guest"])


if __name__ == "__main__":
    unittest.main()
