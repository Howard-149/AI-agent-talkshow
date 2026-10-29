"""Fish PCM routing, progressive playback, cancellation and safe failures (offline)."""
import asyncio
from contextlib import aclosing
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError
from livekit.agents.types import APIConnectOptions

from agent.adapters.fish_client import DEFAULT_FISH_VOICES, FishSettings, fish_voice_for_role, iter_fish_pcm
from agent.adapters.fish_tts import FishTTS
from agent.adapters.tts_factory import build_tts_from_config
from agent.adapters.tts_synthesize import resolve_tts_engine, synthesize_pcm_for_config
from agent.config import LocaleTTSConfig, load_config, load_persona_tts


class Parts(httpx.AsyncByteStream):
    def __init__(self, chunks, *, error=None, block=False):
        self.chunks, self.error, self.block = chunks, error, block
        self.closed = False
        self.gate = asyncio.Event()

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.block:
            await self.gate.wait()
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


class FishSettingsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"FISH_API_KEY": "test-backend-key"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_distinct_role_voices_match_existing_demo(self):
        voices = [fish_voice_for_role(r) for r in ("host", "guest", "commentator")]
        self.assertEqual(len(set(voices)), 3)
        self.assertEqual(voices[0], DEFAULT_FISH_VOICES["host"])

    def test_locale_override_then_role_override(self):
        os.environ.update(FISH_REFERENCE_ID_HOST="role-voice", FISH_REFERENCE_ID_EN_HOST="english-voice")
        self.assertEqual(fish_voice_for_role("host", "en"), "english-voice")
        self.assertEqual(fish_voice_for_role("host", "zh"), "role-voice")

    def test_free_model_and_balanced_explicit_defaults(self):
        cfg = FishSettings.from_env("host")
        self.assertEqual((cfg.model, cfg.latency, cfg.sample_rate), ("s2.1-pro-free", "balanced", 44100))
        self.assertNotIn("test-backend-key", repr(cfg))

    def test_missing_key_fails_instead_of_falling_back(self):
        os.environ.pop("FISH_API_KEY")
        with self.assertRaisesRegex(ValueError, "FISH_API_KEY"):
            FishSettings.from_env("host")

    def test_invalid_options_fail_locally(self):
        for option in ({"model": "s2.1-pro-fre"}, {"latency": "balenced"}, {"sample_rate": 22050},
                       {"timeout_s": 0}, {"reference_id": ""}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                FishSettings(**{"api_key": "test-backend-key", "reference_id": "voice", **option})

    def test_fish_engine_and_factory(self):
        os.environ["TALKSHOW_TTS_ENGINE"] = "fish"
        self.assertEqual(resolve_tts_engine(LocaleTTSConfig("piper", "file.onnx", 22050)), "fish")
        cfg = load_persona_tts("guest", load_config(), locale="en")
        self.assertEqual((cfg.engine, cfg.sample_rate, cfg.model_path), ("fish", 44100, DEFAULT_FISH_VOICES["guest"]))
        adapter = build_tts_from_config(cfg, role="guest")
        self.assertIsInstance(adapter, FishTTS)
        self.assertEqual((adapter.provider, adapter.model), ("fish", "s2.1-pro-free"))

    def test_piper_still_explicitly_available(self):
        os.environ["TALKSHOW_TTS_ENGINE"] = "piper"
        self.assertEqual(resolve_tts_engine(), "piper")


class FishStreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict(os.environ, {"FISH_API_KEY": "test-backend-key"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.settings = FishSettings.from_env("guest")

    async def collect(self, source, *, status=200, content_type="audio/pcm", captured=None):
        def handler(request):
            if captured is not None:
                captured.append(request)
            return httpx.Response(status, headers={"Content-Type": content_type}, stream=source)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return b"".join([chunk async for chunk in iter_fish_pcm("Hello, Ryan.", self.settings, client=client)])

    async def test_payload_and_odd_chunk_boundaries(self):
        requests = []
        source = Parts([b"\x01", b"\x02\x03", b"\x04"])
        self.assertEqual(await self.collect(source, captured=requests), b"\x01\x02\x03\x04")
        self.assertTrue(source.closed)
        self.assertEqual(str(requests[0].url), "https://api.fish.audio/v1/tts")
        self.assertEqual(requests[0].headers["model"], "s2.1-pro-free")
        body = json.loads(requests[0].content)
        self.assertEqual(body["format"], "pcm")
        self.assertEqual(body["latency"], "balanced")
        self.assertEqual(body["sample_rate"], 44100)
        self.assertEqual(body["reference_id"], DEFAULT_FISH_VOICES["guest"])
        self.assertNotIn("api_key", body)

    async def test_http_error_redacted_and_no_auth_retry(self):
        with self.assertRaises(APIStatusError) as caught:
            await self.collect(Parts([b"upstream echoed test-backend-key"]), status=401)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn("test-backend-key", str(caught.exception))

    async def test_temporary_status_retryable(self):
        with self.assertRaises(APIStatusError) as caught:
            await self.collect(Parts([]), status=429)
        self.assertTrue(caught.exception.retryable)

    async def test_reject_json_instead_of_playing_error_as_pcm(self):
        with self.assertRaisesRegex(APIError, "raw PCM"):
            await self.collect(Parts([b"{}"]), content_type="application/json")

    async def test_empty_and_truncated_audio_fail(self):
        for chunks in ([], [b"\x01"]):
            with self.subTest(chunks=chunks), self.assertRaises(APIError):
                await self.collect(Parts(chunks))

    async def test_audio_arrives_before_response_finishes_and_closes_early(self):
        source = Parts([b"\x01\x02"], block=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, headers={"Content-Type": "audio/pcm"}, stream=source))) as client:
            async with aclosing(iter_fish_pcm("Hello", self.settings, client=client)) as stream:
                self.assertEqual(await asyncio.wait_for(anext(stream), 1), b"\x01\x02")
                self.assertFalse(source.closed)
            self.assertTrue(source.closed)

    async def test_cancel_waiting_on_network_closes_response(self):
        source = Parts([], block=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, headers={"Content-Type": "audio/pcm"}, stream=source))) as client:
            async def consume():
                async with aclosing(iter_fish_pcm("Hello", self.settings, client=client)) as stream:
                    async for _ in stream:
                        pass
            task = asyncio.create_task(consume())
            await asyncio.sleep(0.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(source.closed)

    async def test_midstream_connection_failure_does_not_repeat_speech(self):
        with self.assertRaises(APIConnectionError) as caught:
            await self.collect(Parts([b"\x01\x02"], error=httpx.ReadError("private error body")))
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn("private error body", str(caught.exception))

    async def test_timeout_before_audio_is_retryable(self):
        with self.assertRaises(APITimeoutError) as caught:
            await self.collect(Parts([], error=httpx.ReadTimeout("private details")))
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn("private details", str(caught.exception))

    async def test_livekit_frames_progressively_emitted_and_cancelled(self):
        source = Parts([b"\x01\x02" * 4410], block=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, headers={"Content-Type": "audio/pcm"}, stream=source))) as client:
            log = Mock()
            adapter = FishTTS(LocaleTTSConfig("fish", DEFAULT_FISH_VOICES["guest"], 44100), role="guest", client=client, turn_log=log)
            async with adapter.synthesize("Hello", conn_options=APIConnectOptions(max_retry=0)) as stream:
                ev = await asyncio.wait_for(anext(stream), 2)
                self.assertEqual((ev.frame.sample_rate, ev.frame.num_channels), (44100, 1))
                self.assertGreater(len(ev.frame.data), 0)
                self.assertFalse(source.closed)
            self.assertTrue(source.closed)
            self.assertEqual(log.log.call_args.kwargs["status"], "cancelled")

    async def test_empty_input_does_not_request_network(self):
        handler = Mock()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual([c async for c in iter_fish_pcm(" ", self.settings, client=client)], [])
        handler.assert_not_called()

    async def test_one_shot_uses_fish_without_piper(self):
        with patch.dict(os.environ, {"TALKSHOW_TTS_ENGINE": "fish"}), patch(
            "agent.adapters.fish_client.synthesize_fish_pcm", new_callable=AsyncMock, return_value=(b"1234", 44100, 1)
        ) as synth:
            result = await synthesize_pcm_for_config(LocaleTTSConfig("piper", "file.onnx", 22050), "Hello", role="guest")
        self.assertEqual(result, (b"1234", 44100, 1))
        self.assertEqual(synth.call_args.args[1].reference_id, DEFAULT_FISH_VOICES["guest"])


if __name__ == "__main__":
    unittest.main()
