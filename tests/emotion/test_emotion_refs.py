"""Per-emotion CosyVoice reference selection (COSYVOICE_EMOTION_REFS)."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parents[2]


def _load_voices():
    """Load cosyvoice_voices without importing the agent.adapters package (livekit)."""
    spec = importlib.util.spec_from_file_location("cosyvoice_voices", REPO / "agent" / "adapters" / "cosyvoice_voices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EmotionPromptWavTest(unittest.TestCase):
    def setUp(self):
        self.voices = _load_voices()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "guest").mkdir()
        for name in ("Neutral", "Anger", "Sadness"):
            (root / "guest" / f"{name}.wav").write_bytes(b"RIFF")
        self.root = root
        self.voices.EMOTION_REFS_DIR = root

    def tearDown(self):
        self.tmp.cleanup()

    def pick(self, emotion, role="guest", flag="1"):
        with mock.patch.dict(os.environ, {"COSYVOICE_EMOTION_REFS": flag}):
            return self.voices.emotion_prompt_wav("/piper/en_guest.wav", role=role, emotion=emotion)

    def test_flag_off_keeps_default(self):
        self.assertEqual(self.pick("Anger", flag="0"), "/piper/en_guest.wav")

    def test_emotion_with_clip(self):
        self.assertEqual(self.pick("Anger"), str(self.root / "guest" / "Anger.wav"))
        self.assertEqual(self.pick("sadness"), str(self.root / "guest" / "Sadness.wav"))

    def test_unmapped_label_uses_same_speakers_neutral(self):
        for emotion in ("Fear", "Contempt", "Vague", None, "amused"):
            self.assertEqual(self.pick(emotion), str(self.root / "guest" / "Neutral.wav"), emotion)

    def test_role_without_clips_keeps_default(self):
        self.assertEqual(self.pick("Anger", role="host"), "/piper/en_guest.wav")


class ZeroShotPromptTextTest(unittest.TestCase):
    def test_transcript_next_to_the_ref(self):
        voices = _load_voices()
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "Anger.wav"
            wav.write_bytes(b"RIFF")
            self.assertEqual(voices.zero_shot_prompt_text(str(wav)), "")
            wav.with_suffix(".txt").write_text("crawl into some hole somewhere\n", encoding="utf-8")
            self.assertEqual(voices.zero_shot_prompt_text(str(wav)),
                             "You are a helpful assistant.<|endofprompt|>crawl into some hole somewhere")


class BakeTranscriptTest(unittest.TestCase):
    def test_msp_markers_are_dropped(self):
        spec = importlib.util.spec_from_file_location("bake", REPO / "deploy" / "bake_msp_emotion_refs.py")
        bake = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bake)
        self.assertEqual(bake.clean_transcript("[parado 00:08:08] and john. hi, john."), "and john. hi, john")
        self.assertEqual(bake.clean_transcript("you should like [00:00:59] something else"),
                         "you should like something else")


class MspEmotionRefsConfigTest(unittest.TestCase):
    def test_every_role_has_one_speaker_and_the_five_clips(self):
        cfg = yaml.safe_load((REPO / "config" / "msp_emotion_refs.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(cfg["roles"]), {"host", "guest", "commentator"})
        for role, spec in cfg["roles"].items():
            self.assertEqual(set(spec["refs"]), {"Neutral", "Happiness", "Anger", "Sadness", "Surprise"}, role)
            self.assertTrue(all(n.endswith(".wav") for n in spec["refs"].values()), role)
        speakers = [spec["speaker"] for spec in cfg["roles"].values()]
        self.assertEqual(len(speakers), len(set(speakers)))


if __name__ == "__main__":
    unittest.main()
