"""The LLM host's next pick, made while a panelist's line plays, is used only if it
was made from the request the host would send when the pick is due."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

import agent.agents  # noqa: F401  (import order, as in the other tests)
from agent.floor import host_floor
from agent.floor.host_decide import CALL_ON, DecideRequest, DecisionAhead, HostDecision, Raise


class Log:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []

    def log(self, event: str, **fields: Any) -> None:
        self.rows.append((event, fields))

    def events(self, event: str) -> list[dict[str, Any]]:
        return [f for e, f in self.rows if e == event]


def request(**changes: Any) -> DecideRequest:
    fields: dict[str, Any] = dict(
        prompt="Who speaks next?",
        system="You are Lessac, the host.",
        history=[{"role": "user", "content": "Ryan: hiring freezes backfire."}],
        tools=[{"type": "function", "function": {"name": CALL_ON}}],
        names={"guest": "Amy"},
        raises=(Raise("Amy", True),),
        human=Raise("Human guest", False),
        turns={"Ryan": 1},
        human_must_go=False,
    )
    fields.update(changes)
    return DecideRequest(**fields)


AMY = HostDecision(role="guest", line="Amy, does that match what you saw?", tool=CALL_ON)


class SameAsTests(unittest.TestCase):
    def test_equal_requests_match(self) -> None:
        self.assertTrue(request().same_as(request(names={"guest": "Amy"}, turns={"Ryan": 1})))

    def test_any_input_the_model_sees_breaks_the_match(self) -> None:
        for changes in (
            {"prompt": "Who speaks next? Amy raised a hand."},
            {"history": [{"role": "user", "content": "Ryan: and one more thing."}]},
            {"human_must_go": True},
            {"tools": []},
        ):
            with self.subTest(changes=list(changes)):
                self.assertFalse(request().same_as(request(**changes)))


class DecisionMadeAheadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log = Log()
        self.data = SimpleNamespace(turn_log=self.log, room_name="r")

    def _run(self, ahead_req: DecideRequest | None, now_req: DecideRequest, *, fail: bool = False) -> Any:
        async def go() -> Any:
            async def answer() -> HostDecision:
                return AMY

            async def made_ahead() -> DecisionAhead:
                if fail:
                    raise RuntimeError("poll failed")
                return DecisionAhead(request=ahead_req, task=asyncio.ensure_future(answer()))

            task = None if ahead_req is None and not fail else asyncio.ensure_future(made_ahead())
            return await host_floor._decision_made_ahead(self.data, task, now_req)

        return asyncio.run(go())

    def test_a_matching_pick_is_used(self) -> None:
        self.assertEqual(self._run(request(), request()), AMY)
        self.assertEqual(len(self.log.events("host_decide_ahead_used")), 1)

    def test_a_pick_from_a_different_request_is_dropped(self) -> None:
        got = self._run(request(), request(prompt="Who speaks next? The guest raised a hand."))
        self.assertIs(got, host_floor._NOT_AHEAD)
        self.assertEqual([f["reason"] for f in self.log.events("host_decide_ahead_dropped")], ["changed"])

    def test_a_failed_pick_ahead_is_dropped(self) -> None:
        self.assertIs(self._run(None, request(), fail=True), host_floor._NOT_AHEAD)
        self.assertEqual([f["reason"] for f in self.log.events("host_decide_ahead_dropped")], ["error"])

    def test_none_started(self) -> None:
        self.assertIs(self._run(None, request()), host_floor._NOT_AHEAD)
        self.assertEqual(self.log.rows, [])


if __name__ == "__main__":
    unittest.main()
