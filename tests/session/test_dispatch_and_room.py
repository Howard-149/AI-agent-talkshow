"""Explicit agent dispatch in ghost tokens."""

from __future__ import annotations

import base64
import json
import os
import unittest
from unittest import mock

from eval.ghost_session.client import mint_ghost_token

_LK_ENV = {
    "LIVEKIT_URL": "wss://example.livekit.cloud",
    "LIVEKIT_API_KEY": "key",
    "LIVEKIT_API_SECRET": "secret" * 6,
}


def _claims(jwt: str) -> dict:
    payload = jwt.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


class GhostTokenDispatchTests(unittest.TestCase):
    def test_no_agent_name_means_automatic_dispatch(self) -> None:
        with mock.patch.dict(os.environ, _LK_ENV, clear=False):
            os.environ.pop("TALKSHOW_AGENT_NAME", None)
            _, token = mint_ghost_token(room="r1", identity="ghost-a", locale="en")
        self.assertNotIn("roomConfig", _claims(token))

    def test_agent_name_requests_that_agent(self) -> None:
        env = {**_LK_ENV, "TALKSHOW_AGENT_NAME": "talkshow-test"}
        with mock.patch.dict(os.environ, env, clear=False):
            _, token = mint_ghost_token(room="r1", identity="ghost-a", locale="en")
        agents = _claims(token)["roomConfig"]["agents"]
        self.assertEqual([a["agentName"] for a in agents], ["talkshow-test"])


if __name__ == "__main__":
    unittest.main()
