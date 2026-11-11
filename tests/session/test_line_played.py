"""A line's played flag (line_end UI event) and the UI queued for a line that never played."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from agent.show.talkshow_data import TalkShowData


def _chunk() -> SimpleNamespace:
    return SimpleNamespace(
        tts_ms=1, bake_ms=1, duration_sec=1.0, preroll_frames=0, expected_frames=25,
        gen_fps=25.0, url="", word_ends=None,
    )


class AudioHeardTests(unittest.IsolatedAsyncioTestCase):
    async def speak(self, text: str, *, start_audio: bool) -> list[str]:
        from agent.session import session_handoff as sh

        heard: list[str] = []

        async def prepare(**_kw):
            return _chunk()

        async def play(*_a, on_audio_start=None, **_kw):
            if start_audio and on_audio_start is not None:
                on_audio_start()

        data = SimpleNamespace(room_name="r", queue_speech_ui=lambda *a, **k: None)
        with (
            mock.patch.object(sh, "_prepare_chunk", prepare),
            mock.patch.object(sh, "_play_prepared_chunk", play),
            mock.patch.object(sh, "cleanup_speech_slots", lambda *_a: None),
            mock.patch.dict("os.environ", {"TALKSHOW_TTS_ENGINE": "piper"}),
        ):
            await sh._speak_avatar_chunked(
                None, data, bridge=object(), speak_role="host", text=text, step="s",
                tts_cfg=SimpleNamespace(engine="piper"), turn_log=None, video_pub=None,
                use_lk_video=False, emit_ui=False,
                on_audio_heard=lambda: heard.append("heard"),
            )
        return heard

    async def test_a_one_sentence_line_reports_its_audio(self) -> None:
        # Sentence events need two sentences; whether the line was heard must not.
        heard = await self.speak("So, to start things off, I'd like to hear from you.", start_audio=True)
        self.assertEqual(heard, ["heard"])

    async def test_no_audio_no_report(self) -> None:
        self.assertEqual(await self.speak("One line.", start_audio=False), [])


class DropSpeechUiTests(unittest.TestCase):
    def test_a_line_that_never_played_leaves_no_queued_ui(self) -> None:
        data = TalkShowData.__new__(TalkShowData)
        from collections import deque

        data.pending_speech_ui = deque()
        data.announced_transcripts = set()
        data.queue_speech_ui("guest", "Cut before a sound.", step="speech_guest_1")
        data.queue_speech_ui("guest", "", step="speech_guest_1:c1")
        data.announced_transcripts.add(("guest", "speech_guest_1"))
        data.queue_speech_ui("commentator", "Next line.", step="speech_commentator_0")

        data.drop_speech_ui("guest", "speech_guest_1")

        # The next audio's UI is the next line's, not the dropped one's.
        self.assertEqual(data.pop_pending_speech_ui()[0], "commentator")
        self.assertIsNone(data.pop_pending_speech_ui())
        self.assertEqual(data.announced_transcripts, set())


if __name__ == "__main__":
    unittest.main()
