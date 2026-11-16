"""session_start records every setting an arms preset or a ghost replay can change."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest import mock

from agent.session.ghost_replay import REPLAY_FLAGS
from agent.telemetry import run_meta

ROOT = Path(__file__).resolve().parents[2]
# Emotion-only settings that the PAD arms files set; they don't change timing.
NOT_LOGGED = {
    "TALKSHOW_MOOD_WORDING",
    "TALKSHOW_PAD_EXPRESSION_AROUSAL_DECAY",
    "TALKSHOW_PAD_SOFT_BOUND",
}


def arms_vars() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "deploy" / "arms").glob("*.arms"):
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0]
            names.update(re.findall(r"\b([A-Z][A-Z0-9_]*)=", line))
    return names


class RunMetaTests(unittest.TestCase):
    def test_every_arms_flag_is_logged(self) -> None:
        missing = arms_vars() - set(run_meta._LATENCY_ENV) - NOT_LOGGED
        self.assertFalse(missing, f"set by deploy/arms but not logged: {sorted(missing)}")

    def test_every_replayed_flag_is_logged(self) -> None:
        missing = set(REPLAY_FLAGS) - set(run_meta._LATENCY_ENV)
        self.assertFalse(missing, f"replayed but not logged: {sorted(missing)}")

    def test_no_secret_is_logged(self) -> None:
        # session logs get copied around and shared: keys and passwords must never land in them
        secret = re.compile(r"API_KEY|_KEY$|SECRET|PASSWORD|_TOKEN$")
        leaked = [name for name in run_meta._LATENCY_ENV if secret.search(name)]
        self.assertFalse(leaked, f"looks like a secret: {leaked}")
        env = {"TALKSHOW_TEXT_LLM_API_KEY": "sk-test", "TYPESAFE_API_KEY": "tk-test"}
        with mock.patch.dict(os.environ, env), mock.patch.object(run_meta, "_run", return_value=""):
            logged = run_meta.run_metadata()["env"]
        self.assertFalse(set(env) & set(logged))

    def test_think_action_pacing_is_logged(self) -> None:
        env = {"TALKSHOW_TEXT_TOKEN_INTERVAL": "32", "TALKSHOW_THINK_LEAN": "1"}
        with mock.patch.dict(os.environ, env), mock.patch.object(run_meta, "_run", return_value=""):
            logged = run_meta.run_metadata()["env"]
        self.assertEqual(logged["TALKSHOW_TEXT_TOKEN_INTERVAL"], "32")
        self.assertEqual(logged["TALKSHOW_THINK_LEAN"], "1")


if __name__ == "__main__":
    unittest.main()
