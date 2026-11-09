"""Skipping the wav2vec pass DyStream runs per window and then discards."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from avatar import dystream_skip
from avatar.dystream_job import stream_payload


class FakeDyStream:
    """Shaped like DyStream f587541's one_clip_only_inference: it computes features
    from the window, then replaces them with the precomputed ones."""

    def __init__(self) -> None:
        self.inner_calls = 0

    def get_audio2face_fea(self, audio, prev_audio, n):
        self.inner_calls += 1
        return [x * 100 for x in audio]

    def get_audio2face_fea_other(self, audio, prev_audio, n):
        self.inner_calls += 1
        return [x * 200 for x in audio]

    def one_clip_only_inference(self, per_compute_audio_feature, audio_self, per_compute_audio_other_feature=None):
        use_pre_compute_audio_feature = True
        audio2face_fea = self.get_audio2face_fea(audio_self, None, 3)
        audio2face_fea_other = self.get_audio2face_fea_other(audio_self, None, 3)
        audio_features = audio2face_fea
        audio_other_features = audio2face_fea_other
        if use_pre_compute_audio_feature:
            audio_features = per_compute_audio_feature
            audio_other_features = per_compute_audio_other_feature
        return [a + b for a, b in zip(audio_features, audio_other_features)]


class ChangedDyStream(FakeDyStream):
    """A version that uses its own window features: never skip."""

    def one_clip_only_inference(self, per_compute_audio_feature, audio_self, per_compute_audio_other_feature=None):
        return self.get_audio2face_fea(audio_self, None, 3)


def _run(model, skip: bool):
    with dystream_skip.unused_wav2vec_skipped(model, skip) as skipped:
        out = model.one_clip_only_inference([1, 2, 3], [9, 9, 9], [10, 20, 30])
    return out, skipped


class SkipTests(unittest.TestCase):
    def test_same_output_without_the_inner_calls(self) -> None:
        plain, model = FakeDyStream(), FakeDyStream()
        expected, skipped = _run(plain, skip=False)
        self.assertFalse(skipped)
        self.assertEqual(plain.inner_calls, 2)
        got, skipped = _run(model, skip=True)
        self.assertTrue(skipped)
        self.assertEqual(got, expected)
        self.assertEqual(model.inner_calls, 0)

    def test_methods_come_back_after_the_render(self) -> None:
        model = FakeDyStream()
        _run(model, skip=True)
        self.assertNotIn("get_audio2face_fea", model.__dict__)
        model.one_clip_only_inference([1], [9], [1])
        self.assertEqual(model.inner_calls, 2)

    def test_a_profiler_wrapper_is_kept(self) -> None:
        model = FakeDyStream()
        wrapper = mock.Mock(return_value=[0])
        model.get_audio2face_fea = wrapper
        _run(model, skip=True)
        self.assertIs(model.__dict__["get_audio2face_fea"], wrapper)
        wrapper.assert_not_called()

    def test_a_model_that_uses_its_own_features_is_left_alone(self) -> None:
        model = ChangedDyStream()
        out, skipped = _run(model, skip=True)
        self.assertFalse(skipped)
        self.assertEqual(out, [900, 900, 900])


class RequestTests(unittest.TestCase):
    def test_the_field_is_sent_only_when_on(self) -> None:
        common = dict(portrait=Path("p.png"), audio_wav=Path("a.wav"), denoising_steps=5, npz=None, cache_key="host")
        self.assertNotIn("skip_unused_wav2vec", stream_payload(**common))
        self.assertIs(stream_payload(**common, skip_unused_wav2vec=True)["skip_unused_wav2vec"], True)

    def test_the_agent_flag(self) -> None:
        from agent.adapters.dystream_bridge import avatar_skip_unused_wav2vec

        with mock.patch.dict(os.environ, {"TALKSHOW_AVATAR_SKIP_UNUSED_WAV2VEC": "1"}):
            self.assertTrue(avatar_skip_unused_wav2vec())
        with mock.patch.dict(os.environ, {"TALKSHOW_AVATAR_SKIP_UNUSED_WAV2VEC": "0"}):
            self.assertFalse(avatar_skip_unused_wav2vec())


if __name__ == "__main__":
    unittest.main()
