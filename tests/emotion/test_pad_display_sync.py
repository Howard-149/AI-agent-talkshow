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
        self.data = SimpleNamespace(pad_display_pending=[], pad_display_played=[])

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

    def test_a_late_reaction_to_a_playing_line_shows_at_once(self) -> None:
        line = ("guest", "Oh, I'm genuinely glad he landed somewhere better.")
        appraisal.publish_pad_state(self.data, "guest", source=line)  # speaker relaxes
        appraisal.release_pad_display(self.data, *line)  # line starts playing
        appraisal.publish_pad_state(self.data, "host", feeling="relieved", source=line)
        self.assertEqual(self.roles(), ["guest", "host"])
        self.assertEqual(self.data.pad_display_pending, [])

    def test_rollback_drops_changes_from_lines_never_heard(self) -> None:
        dropped = ("host", "Oh, come on, obedience training is bleak, isn't it?")
        appraisal.publish_pad_state(self.data, "guest", feeling="bitter", source=dropped)
        self.data.scenario = SimpleNamespace(turn_control=SimpleNamespace(listen_role="host"))
        with mock.patch(
            "agent.panel.panel_context.panel_speaker_roles", lambda scenario: ["guest", "commentator"]
        ):
            appraisal.resync_pad_display(self.data)
        self.assertEqual(self.data.pad_display_pending, [])
        self.assertEqual(self.roles(), ["host", "guest", "commentator"])
        self.assertNotIn("feeling", self.sent[1])
        # The next line to play releases nothing from the dropped script.
        self.assertEqual(appraisal.release_pad_display(self.data, *dropped), 0)

    def test_off_switch_shows_at_once(self) -> None:
        with mock.patch.dict("os.environ", {"TALKSHOW_PAD_DISPLAY_SYNC": "0"}):
            appraisal.publish_pad_state(self.data, "guest", source=("host", "A line written ahead."))
        self.assertEqual(self.roles(), ["guest"])


class DroppedLineAppraisalTests(unittest.TestCase):
    def test_appraisal_of_a_rolled_back_line_is_not_applied(self) -> None:
        import asyncio

        heard, cut = object(), object()
        data = SimpleNamespace(show_history=SimpleNamespace(lines=[heard]), turn_log=None)
        applied: list[str] = []
        result = SimpleNamespace(role="guest")
        with mock.patch.object(appraisal, "run_appraisal", mock.AsyncMock(return_value=result)), \
                mock.patch.object(appraisal, "apply_appraisal", lambda d, u, r: applied.append(u.text)):
            for line, text in ((heard, "heard"), (cut, "cut"), (None, "untracked")):
                utt = appraisal.Utterance(speaker="host", text=text, line=line)
                asyncio.run(appraisal._appraise_chained(data, "guest", utt, None))
        self.assertEqual(applied, ["heard", "untracked"])


if __name__ == "__main__":
    unittest.main()
