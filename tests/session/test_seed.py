"""TALKSHOW_SEED: seeded LLM requests and repeatable host line picks."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from agent.adapters import gemma_mm_client as gmc
from agent.config import show_seed
from agent.floor import host_lines


def _without_seed() -> mock._patch_dict:
    env = {k: v for k, v in os.environ.items() if k != "TALKSHOW_SEED"}
    return mock.patch.dict(os.environ, env, clear=True)


class SeedTests(unittest.TestCase):
    def test_unset_or_not_a_number_means_no_seed(self) -> None:
        with _without_seed():
            self.assertIsNone(show_seed())
        for raw in ("", "  ", "abc"):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {"TALKSHOW_SEED": raw}):
                self.assertIsNone(show_seed())
        with mock.patch.dict(os.environ, {"TALKSHOW_SEED": " 42 "}):
            self.assertEqual(show_seed(), 42)

    def test_the_seed_goes_into_each_request(self) -> None:
        payload: dict = {"temperature": 0.45}
        with mock.patch.dict(os.environ, {"TALKSHOW_SEED": "7"}):
            gmc._add_seed(payload)
        self.assertEqual(payload, {"temperature": 0.45, "seed": 7})

    def test_no_seed_leaves_the_request_unseeded(self) -> None:
        payload: dict = {"temperature": 0.45}
        with _without_seed():
            gmc._add_seed(payload)
        self.assertEqual(payload, {"temperature": 0.45})

    def test_line_picks_repeat_from_the_same_seed(self) -> None:
        options = ["a", "b", "c", "d"]

        def picks(seed: int) -> list[str]:
            host_lines.reset_line_picks(seed)
            return [host_lines._pick("intro", options) for _ in range(12)]

        self.assertEqual(picks(7), picks(7))
        self.assertNotEqual(picks(7), picks(8))


if __name__ == "__main__":
    unittest.main()
