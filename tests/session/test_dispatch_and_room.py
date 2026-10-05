"""Explicit agent dispatch in ghost tokens, per-run ghost rooms, room delete on exit."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from agent.session import session_lifecycle
from eval.ghost_session.client import default_room, mint_ghost_token

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


class GhostRoomTests(unittest.TestCase):
    def test_default_room_is_new_per_run(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TALKSHOW_ROOM", None)
            with mock.patch("eval.ghost_session.client.time.time", return_value=1791180140.5):
                self.assertEqual(default_room(), "talkshow-ghost-1791180140")

    def test_talkshow_room_overrides(self) -> None:
        with mock.patch.dict(os.environ, {"TALKSHOW_ROOM": "talkshow-dev"}, clear=False):
            self.assertEqual(default_room(), "talkshow-dev")


class _FakeJobCtx:
    def __init__(self) -> None:
        self.deleted: list[str | None] = []
        self.shutdown_reasons: list[str] = []

    def delete_room(self, room_name: str | None = None):
        self.deleted.append(room_name)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    def shutdown(self, reason: str = "") -> None:
        self.shutdown_reasons.append(reason)


def _inactive_session() -> SimpleNamespace:
    return SimpleNamespace(is_running=False, shutdown=lambda drain=False: None)


def _active_session() -> SimpleNamespace:
    calls: list[bool] = []
    return SimpleNamespace(
        is_running=True,
        _activity=object(),
        _close_session_atask=None,
        shutdown=lambda drain=False: calls.append(drain),
        calls=calls,
    )


class DeleteRoomOnExitTests(unittest.TestCase):
    def _run(self, session, env: dict[str, str]) -> _FakeJobCtx:
        job_ctx = _FakeJobCtx()
        data = SimpleNamespace(
            shutdown_event=asyncio.Event(),
            panel_chain_running=True,
            panel_followup_pending=False,
            speak_line_busy=False,
            panel_followup_deferred=False,
        )
        with mock.patch.dict(os.environ, env, clear=False):
            asyncio.run(
                session_lifecycle.shutdown_session_when_alone(
                    session, data, room_name="r1", reason="test", job_ctx=job_ctx
                )
            )
        return job_ctx

    def test_deletes_room_then_shuts_down(self) -> None:
        job_ctx = self._run(_active_session(), {"TALKSHOW_SESSION_DRAIN_ON_LEAVE": "0"})
        self.assertEqual(job_ctx.deleted, ["r1"])
        self.assertEqual(job_ctx.shutdown_reasons, ["test"])

    def test_deletes_room_when_session_already_closed(self) -> None:
        job_ctx = self._run(_inactive_session(), {})
        self.assertEqual(job_ctx.deleted, ["r1"])

    def test_opt_out(self) -> None:
        job_ctx = self._run(_active_session(), {"TALKSHOW_DELETE_ROOM_ON_EXIT": "0"})
        self.assertEqual(job_ctx.deleted, [])
        self.assertEqual(job_ctx.shutdown_reasons, ["test"])

    def test_drain_keeps_room(self) -> None:
        job_ctx = self._run(_active_session(), {"TALKSHOW_SESSION_DRAIN_ON_LEAVE": "1"})
        self.assertEqual(job_ctx.deleted, [])


if __name__ == "__main__":
    unittest.main()
