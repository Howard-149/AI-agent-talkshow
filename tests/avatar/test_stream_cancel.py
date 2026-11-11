"""An avatar stream nobody will play is closed, freeing the one-at-a-time sidecar."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


class FakeSidecar:
    """Yields meta then frames slowly, like the sidecar rendering; records closing."""

    def __init__(self, frames: int = 200) -> None:
        self.frames = frames
        self.sent = 0
        self.closed = threading.Event()

    def __call__(self, **_kwargs):
        def gen():
            try:
                yield ("meta", {"width": 4, "height": 4, "fps": 25.0, "expected_frames": self.frames})
                for _ in range(self.frames):
                    time.sleep(0.005)
                    self.sent += 1
                    yield ("frame", b"\x00" * 64)
            finally:
                self.closed.set()

        return gen()


class StreamCancelTests(unittest.IsolatedAsyncioTestCase):
    async def synthesize(self, sidecar: FakeSidecar):
        from agent.adapters import dystream_bridge as bridge_mod

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        wav = Path(tmp.name) / "a.wav"
        wav.write_bytes(b"")
        env = {"TALKSHOW_AVATAR_VIDEO_WIDTH": "4", "TALKSHOW_AVATAR_VIDEO_HEIGHT": "4"}
        with mock.patch.object(bridge_mod, "run_dystream_synthesize_stream", sidecar), \
                mock.patch.dict("os.environ", env):
            bridge = bridge_mod.AvatarBridge()
            return await bridge._synthesize_stream(
                role="host", step="s:c2", slot="c2", portrait_path=Path("p.png"),
                wav_path=wav, t0=time.monotonic(), pcm_duration_sec=8.0,
            )

    async def test_cancel_closes_the_stream_before_it_renders_everything(self) -> None:
        sidecar = FakeSidecar(frames=200)
        result = await self.synthesize(sidecar)
        result.cancel()
        await asyncio.wait_for(result._reader_task, 5)
        self.assertTrue(sidecar.closed.is_set())
        self.assertLess(sidecar.sent, 200)

    async def test_a_played_stream_runs_to_the_end(self) -> None:
        sidecar = FakeSidecar(frames=30)
        result = await self.synthesize(sidecar)
        await asyncio.wait_for(result._reader_task, 5)
        self.assertEqual(sidecar.sent, 30)


if __name__ == "__main__":
    unittest.main()
