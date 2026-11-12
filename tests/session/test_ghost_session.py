"""Ghost-session inject contract (no LiveKit)."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import unittest.mock
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
        first, second = swing.turns
        self.assertEqual((first.after_lines, first.when, first.emotion), (1, "line_end", "Anger"))
        self.assertEqual((second.after_lines, second.when, second.after_sec), (4, "mid_line", 3.0))

    def test_script_names_the_guest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "named.yaml"
            path.write_text("name: Howard\nturns:\n  - text: hi\n", encoding="utf-8")
            self.assertEqual(load_script(path).name, "Howard")
            path.write_text("turns:\n  - text: hi\n", encoding="utf-8")
            self.assertEqual(load_script(path).name, "Guest")

    def test_script_rejects_an_unknown_cut_in_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("turns:\n  - text: hi\n    when: whenever\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_script(path)

    def test_guest_mix_delays_each_turn_onto_the_video_clock(self) -> None:
        from eval.ghost_session.record_frontend import guest_mix_filter

        mix = guest_mix_filter([(Path("a.wav"), 12.0), (Path("b.wav"), 40.5)], video_t0=10.0)
        self.assertEqual(
            mix,
            "[2:a]adelay=2000:all=1[g0];[3:a]adelay=30500:all=1[g1];"
            "[1:a][g0][g1]amix=inputs=3:normalize=0:duration=first[aout]",
        )

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


class GhostVoiceTests(unittest.TestCase):
    def test_reference_prefers_the_emotion_take(self) -> None:
        from eval.ghost_session.voice import reference_wav

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with unittest.mock.patch.dict(os.environ, {"GHOST_VOICE_DIR": tmp}):
                with self.assertRaises(FileNotFoundError):
                    reference_wav("Anger")
                (folder / "Neutral.wav").write_bytes(b"")
                self.assertEqual(reference_wav("Anger").name, "Neutral.wav")
                (folder / "Anger.wav").write_bytes(b"")
                self.assertEqual(reference_wav("anger").name, "Anger.wav")
                self.assertEqual(reference_wav("").name, "Neutral.wav")


class CutInTimingTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        from eval.ghost_session.client import GhostRoomClient

        return GhostRoomClient()

    def line(self, client, role: str, step: str) -> None:
        client.on_ui_event({"type": "transcript", "role": role, "step": step, "text": "x"})

    async def test_line_end_waits_for_the_line_to_finish(self) -> None:
        from eval.ghost_session.client import ScriptedTurn

        client = self.client()
        turn = ScriptedTurn(text="hi", after_lines=2, when="line_end", max_wait_sec=5)
        waiter = asyncio.ensure_future(client.wait_for_turn(turn, lines_before=0))
        self.line(client, "host", "opening")
        self.line(client, "host", "opening")  # the same line announced twice counts once
        client.on_ui_event({"type": "role_active", "role": "host"})
        self.line(client, "guest", "speech_guest_1")
        client.on_ui_event({"type": "role_active", "role": "guest"})
        await asyncio.sleep(0.3)
        self.assertFalse(waiter.done())  # the second line is still playing
        client.on_ui_event({"type": "role_idle"})
        client.on_ui_event({"type": "line_end", "role": "host", "step": "opening"})
        client.on_ui_event({"type": "line_end", "role": "guest", "step": "speech_guest_1"})
        self.assertEqual(await asyncio.wait_for(waiter, 3), "line_end")

    async def test_line_end_waits_for_audio_announced_before_it_plays(self) -> None:
        from eval.ghost_session.client import ScriptedTurn

        client = self.client()
        client.idle_since -= 60  # the client has been idle since it joined
        turn = ScriptedTurn(text="hi", after_lines=1, when="line_end", max_wait_sec=5)
        waiter = asyncio.ensure_future(client.wait_for_turn(turn, lines_before=0))
        # The worker announces a line's transcript before its audio starts.
        self.line(client, "host", "opening")
        await asyncio.sleep(1.2)
        self.assertFalse(waiter.done())  # the opening has not even started playing
        client.on_ui_event({"type": "role_active", "role": "host"})
        await asyncio.sleep(0.3)
        self.assertFalse(waiter.done())
        client.on_ui_event({"type": "role_idle"})
        client.on_ui_event({"type": "line_end", "role": "host", "step": "opening"})
        self.assertEqual(await asyncio.wait_for(waiter, 3), "line_end")

    async def test_line_end_waits_for_the_whole_line_and_the_next_line_of_the_beat(self) -> None:
        from eval.ghost_session.client import ScriptedTurn

        client = self.client()
        turn = ScriptedTurn(text="hi", after_lines=1, when="line_end", max_wait_sec=5)
        waiter = asyncio.ensure_future(client.wait_for_turn(turn, lines_before=0))
        self.line(client, "host", "opening")
        client.on_ui_event({"type": "role_active", "role": "host"})
        client.on_ui_event({"type": "role_idle"})
        await asyncio.sleep(1.2)
        self.assertFalse(waiter.done())  # idle between the opening's sentences
        client.on_ui_event({"type": "role_active", "role": "host"})
        client.on_ui_event({"type": "role_idle"})
        client.on_ui_event({"type": "line_end", "role": "host", "step": "opening"})
        # The host's invite is announced right after the opening ends.
        self.line(client, "host", "speech_host_0")
        await asyncio.sleep(1.2)
        self.assertFalse(waiter.done())  # the invite is about to play
        client.on_ui_event({"type": "role_active", "role": "host"})
        client.on_ui_event({"type": "role_idle"})
        client.on_ui_event({"type": "line_end", "role": "host", "step": "speech_host_0"})
        self.assertEqual(await asyncio.wait_for(waiter, 3), "line_end")

    def test_a_step_id_used_again_in_a_later_beat_is_a_new_line(self) -> None:
        client = self.client()
        self.line(client, "guest", "speech_guest_1")
        client.on_ui_event({"type": "line_end", "role": "guest", "step": "speech_guest_1", "played": False})
        # The guest cut in; the next beat numbers its lines from 0 again.
        self.line(client, "guest", "speech_guest_1")
        self.assertEqual(client.agent_lines, 2)
        self.assertEqual(client.lines_ended, 1)
        client.on_ui_event({"type": "line_end", "role": "guest", "step": "speech_guest_1"})
        client.on_ui_event({"type": "line_end", "role": "guest", "step": "speech_guest_1"})
        self.assertEqual(client.lines_ended, 2)  # a repeated line_end counts once

    async def test_mid_line_cuts_in_while_the_line_plays(self) -> None:
        from eval.ghost_session.client import ScriptedTurn

        client = self.client()
        turn = ScriptedTurn(text="hi", after_lines=1, when="mid_line", after_sec=0.1, max_wait_sec=5)
        waiter = asyncio.ensure_future(client.wait_for_turn(turn, lines_before=0))
        self.line(client, "commentator", "speech_commentator_2")
        client.on_ui_event({"type": "role_active", "role": "commentator"})
        self.assertEqual(await asyncio.wait_for(waiter, 3), "mid_line")
        self.assertTrue(client.speaking)

    async def test_gives_up_waiting_after_max_wait(self) -> None:
        from eval.ghost_session.client import ScriptedTurn

        client = self.client()
        turn = ScriptedTurn(text="hi", after_lines=3, max_wait_sec=0.2)
        self.assertEqual(await client.wait_for_turn(turn, lines_before=0), "timeout")

    async def test_speech_goes_out_in_real_time_frames_then_silence(self) -> None:
        from eval.ghost_session import client as ghost
        from eval.ghost_session.voice import Speech

        client = self.client()
        frames: list[int] = []

        class Source:
            async def capture_frame(self, frame):
                frames.append(len(bytes(frame.data)))
                await asyncio.sleep(0)

        client._source = Source()
        mic = asyncio.ensure_future(client._mic_loop())
        try:
            speech = Speech(pcm=bytes(2 * 2400), sample_rate=24000)  # 0.1 s
            await asyncio.wait_for(client.say(speech, Path("turn.wav")), 3)
        finally:
            mic.cancel()
        samples = ghost.MIC_RATE * ghost.FRAME_MS // 1000
        self.assertEqual(client.spoken[0][0], Path("turn.wav"))
        self.assertGreaterEqual(len(frames), 10)  # 0.1 s at 10 ms frames, resampled to 48 kHz
        self.assertTrue(all(n == 2 * samples for n in frames))


class GhostViewerWaitTests(unittest.IsolatedAsyncioTestCase):
    def participant(self, identity: str, **meta):
        from types import SimpleNamespace

        return SimpleNamespace(identity=identity, metadata=json.dumps(meta) if meta else "")

    async def test_opening_waits_for_the_recording_page_only_when_asked(self) -> None:
        from types import SimpleNamespace

        from agent.session.ghost_flags import wait_for_ghost_viewer

        with unittest.mock.patch.dict(os.environ, {"TALKSHOW_GHOST_VIEWER_GRACE_S": "0"}):
            plain = SimpleNamespace(remote_participants={"g": self.participant("ghost-guest", ghost=True)})
            self.assertFalse(await wait_for_ghost_viewer(plain, timeout_sec=1))

            room = SimpleNamespace(
                remote_participants={"g": self.participant("ghost-guest", ghost=True, awaitViewer=True)}
            )
            waiter = asyncio.ensure_future(wait_for_ghost_viewer(room, timeout_sec=5))
            await asyncio.sleep(0.3)
            self.assertFalse(waiter.done())
            room.remote_participants["v"] = self.participant("ghost-viewer", ghostViewer=True)
            self.assertTrue(await asyncio.wait_for(waiter, 2))

    def test_the_recording_ghost_asks_the_worker_to_wait(self) -> None:
        import jwt as pyjwt  # ships with livekit-api

        from eval.ghost_session.client import mint_ghost_token

        env = {"LIVEKIT_URL": "wss://x", "LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "s" * 32}
        with unittest.mock.patch.dict(os.environ, env):
            _, recording = mint_ghost_token(room="r", identity="guest", locale="en", await_viewer=True)
            _, plain = mint_ghost_token(room="r", identity="guest", locale="en")
        meta = lambda tok: json.loads(pyjwt.decode(tok, options={"verify_signature": False})["metadata"])
        self.assertTrue(meta(recording)["awaitViewer"])
        self.assertNotIn("awaitViewer", meta(plain))
