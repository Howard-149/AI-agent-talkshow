"""The recording's audio mixes every show track on one clock (overlaps stay audible)."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from eval.ghost_session.record_frontend import RoomAudioTap


def _write(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RoomAudioTap.RATE)
        wf.writeframes(samples.astype(np.int16).tobytes())


class MixTests(unittest.TestCase):
    def test_a_later_track_lands_at_its_own_time_and_overlaps(self) -> None:
        rate = RoomAudioTap.RATE
        with tempfile.TemporaryDirectory() as tmp:
            tap = RoomAudioTap(Path(tmp) / "show.wav")
            speaker = Path(tmp) / "a.wav"
            backchannel = Path(tmp) / "b.wav"
            _write(speaker, np.full(2 * rate, 1000))  # 2 s line
            _write(backchannel, np.full(rate // 2, 500))  # "mm-hmm", 0.5 s
            tap._mix([(10.0, speaker), (11.0, backchannel)])  # backchannel 1 s into the line

            with wave.open(str(tap.path)) as wf:
                mix = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.assertEqual(len(mix), 2 * rate)
        self.assertEqual(mix[rate // 2], 1000)  # line alone
        self.assertEqual(mix[rate + rate // 4], 1500)  # both at once
        self.assertEqual(mix[rate + 3 * rate // 4], 1000)
        self.assertEqual(tap.started_at, 10.0)

    def test_the_mix_clips_instead_of_wrapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tap = RoomAudioTap(Path(tmp) / "show.wav")
            a, b = Path(tmp) / "a.wav", Path(tmp) / "b.wav"
            _write(a, np.full(100, 30000))
            _write(b, np.full(100, 30000))
            tap._mix([(0.0, a), (0.0, b)])
            with wave.open(str(tap.path)) as wf:
                mix = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.assertTrue((mix == 32767).all())


if __name__ == "__main__":
    unittest.main()
