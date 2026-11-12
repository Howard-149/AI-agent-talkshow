"""Offline model name -> local snapshot directory (eval/hf_local.py). CPU, a fake cache directory."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.hf_local import local_snapshot

NAME = "Qwen/Qwen3-Embedding-0.6B"


class LocalSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = Path(tempfile.mkdtemp())
        self.repo = self.cache / "models--Qwen--Qwen3-Embedding-0.6B"
        self.snap = self.repo / "snapshots" / "abc123"
        self.snap.mkdir(parents=True)
        (self.snap / "config.json").write_text("{}")  # no .gitattributes / README.md, as in the copied cache

    def env(self, **kv):
        base = {"HF_HUB_CACHE": str(self.cache), "HF_HUB_OFFLINE": "1"}
        base.update(kv)
        return mock.patch.dict(os.environ, base)

    def test_offline_resolves_the_snapshot(self) -> None:
        with self.env():
            self.assertEqual(local_snapshot(NAME), str(self.snap))

    def test_refs_main_wins(self) -> None:
        other = self.repo / "snapshots" / "zzz999"
        other.mkdir()
        (other / "config.json").write_text("{}")
        (self.repo / "refs").mkdir()
        (self.repo / "refs" / "main").write_text("abc123\n")
        with self.env():
            self.assertEqual(local_snapshot(NAME), str(self.snap))

    def test_falls_back_to_the_name(self) -> None:
        with self.env(HF_HUB_OFFLINE="0"):
            self.assertEqual(local_snapshot(NAME), NAME)  # online: the hub resolves it
        with self.env():
            self.assertEqual(local_snapshot("Qwen/Not-Cached"), "Qwen/Not-Cached")
            self.assertEqual(local_snapshot(str(self.snap)), str(self.snap))  # already a path
        (self.snap / "config.json").unlink()
        with self.env():
            self.assertEqual(local_snapshot(NAME), NAME)  # a snapshot without config.json is not used

    def test_hf_home_default(self) -> None:
        home = Path(tempfile.mkdtemp())
        (home / "hub").symlink_to(self.cache)
        with mock.patch.dict(os.environ, {"HF_HOME": str(home), "HF_HUB_OFFLINE": "1"}):
            os.environ.pop("HF_HUB_CACHE", None)
            self.assertEqual(Path(local_snapshot(NAME)).name, "abc123")


if __name__ == "__main__":
    unittest.main()
