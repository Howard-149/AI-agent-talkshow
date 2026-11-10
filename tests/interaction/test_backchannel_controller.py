from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.interaction.backchannel_controller import (
    BackchannelCandidate,
    BackchannelController,
)
from agent.interaction.think_action import (
    Backchannel,
    CommunicativeIntent,
    parse_think_action,
)


def candidate(role: str, content: str, segment: int = 0) -> BackchannelCandidate:
    event = Backchannel(
        CommunicativeIntent("acknowledge", "show listening", None), content, 3, segment
    )
    return BackchannelCandidate(role, event, 20, segment)


class BackchannelParserTests(unittest.TestCase):
    def test_keeps_only_one_short_feedback_candidate(self) -> None:
        raw = '''{"think":"A clause ended.","action":{"decision":"silent","intent":null,
          "content":null,"backchannels":[
            {"intent":{"type":"acknowledge","goal":"show listening","target":null},
             "content":"That is helpful.","after_word_index":3},
            {"intent":{"type":"support","goal":"agree","target":null},
             "content":"Exactly.","after_word_index":4}]}}'''
        result = parse_think_action(raw, floor_state="occupied", interval_text="A point ended.")
        self.assertEqual([item.content for item in result.action.backchannels], ["That is helpful."])

    def test_rejects_long_or_non_feedback_backchannels(self) -> None:
        raw = '''{"think":"x","action":{"decision":"silent","intent":null,
          "content":null,"backchannels":[
            {"intent":{"type":"challenge","goal":"argue","target":null},
             "content":"I disagree.","after_word_index":1},
            {"intent":{"type":"acknowledge","goal":"listen","target":null},
             "content":"That is exactly what I was thinking.","after_word_index":1}]}}'''
        result = parse_think_action(raw, floor_state="occupied", interval_text="Well.")
        self.assertEqual(result.action.backchannels, ())


class BackchannelControllerTests(unittest.TestCase):
    def test_audio_publishers_are_separate_per_role_and_locale(self) -> None:
        from agent.adapters.locale_audio import get_backchannel_audio_publisher

        with patch("agent.adapters.locale_audio._backchannel", {}):
            english = get_backchannel_audio_publisher("host", "en")
            chinese = get_backchannel_audio_publisher("host", "zh")

        self.assertNotEqual(english.track_name, chinese.track_name)
        self.assertEqual(english.track_name, "talkshow-backchannel-en-host")
        self.assertEqual(chinese.track_name, "talkshow-backchannel-zh-host")

    def test_approves_only_one_candidate_per_speaker_cue(self) -> None:
        controller = BackchannelController()
        first = candidate("guest", "Interesting.")
        second = candidate("commentator", "I agree.")

        selected, decisions = controller.arbitrate(1, [first, second], now=10.0)

        self.assertIs(selected, first)
        self.assertEqual([reason for _, reason in decisions], [
            "approved_pending_playback", "one_candidate_per_speaker_turn"
        ])

    def test_actual_playback_enforces_global_cooldown_and_ack_repeat(self) -> None:
        controller = BackchannelController()
        with patch.dict("os.environ", {"TALKSHOW_BACKCHANNEL_COOLDOWN_S": "12"}):
            self.assertEqual(controller.try_start(1, "host", "Right.", now=10.0), (True, "played"))
            self.assertEqual(
                controller.try_start(2, "guest", "Exactly.", now=15.0),
                (False, "global_cooldown"),
            )
            self.assertEqual(
                controller.try_start(2, "guest", "Exactly.", now=23.0),
                (False, "repeated_feedback"),
            )


if __name__ == "__main__":
    unittest.main()