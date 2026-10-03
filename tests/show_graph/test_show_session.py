"""Long-lived session graph (phase 3): events, beats, reset, ordering, recovery."""

from __future__ import annotations

import asyncio
import os
import unittest
from typing import Any
from unittest import mock



def _parsed_turn_cls() -> Any:
    """ParsedTurn without importing agent.adapters/__init__ (needs livekit.agents)."""
    try:
        from agent.adapters.response_parser import ParsedTurn

        return ParsedTurn
    except ImportError:
        import importlib.util
        import sys
        from pathlib import Path

        name = "agent.adapters.response_parser"  # real name: checkpoints resolve by module path
        path = Path(__file__).resolve().parents[2] / "agent" / "adapters" / "response_parser.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.ParsedTurn


ParsedTurn = _parsed_turn_cls()
from agent.show_graph.session import ShowSession
from agent.show_graph.state import FloorPick
from tests.show_graph.test_show_graph import PANEL, FakeActuators


class RecordingHooks:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []

    async def before_beat(self, event: dict[str, Any]) -> None:
        self.calls.append(("before", event["kind"], None))

    async def after_beat(self, event: dict[str, Any], result: dict[str, Any] | None) -> None:
        self.calls.append(("after", event["kind"], (result or {}).get("last_outcome")))


def _session(act: FakeActuators, **kw: Any) -> ShowSession:
    return ShowSession(act, panel_roles=PANEL, listen_role="host", session_id="test", **kw)


HUMAN = {"heard": "Should we freeze senior hiring?", "parsed": ParsedTurn(heard="x", reply="y")}


class SessionLifecycleTests(unittest.TestCase):
    def test_opening_human_idle_stop(self) -> None:
        async def go() -> tuple[list[Any], ShowSession, FakeActuators, RecordingHooks]:
            act = FakeActuators(
                picks=[
                    FloorPick("human", "queue_fifo"),  # opening → straight to the human
                    FloorPick("guest", "queue_fifo"),  # human beat
                    FloorPick("human", "queue_fifo"),
                    FloorPick(None, ""),  # idle beat: nobody picked
                ]
            )
            hooks = RecordingHooks()
            show = _session(act, hooks=hooks)
            show.start()
            r1 = await show.post("opening", text="Welcome!", texts={"en": "Welcome!"})
            r2 = await show.post("human", event=HUMAN)
            r3 = await show.post("moderate", trigger="idle_wait")
            await show.stop()
            return [r1, r2, r3], show, act, hooks

        (r1, r2, r3), show, act, hooks = asyncio.run(go())
        self.assertEqual(r1["last_outcome"], "human")
        self.assertEqual(r2["last_outcome"], "human")
        self.assertEqual(r3["last_outcome"], "")
        self.assertEqual(r3["beats"], 3)
        self.assertFalse(show.running)
        self.assertEqual(
            [k for k, _ in act.kinds("speak")],
            [
                "welcome", "grant_human",  # opening (session_start skips the open-floor line)
                "host_reply", "open_floor", "intro", "panelist", "open_floor", "grant_human",
                "open_floor",  # idle: nobody raised
            ],
        )
        self.assertEqual(
            hooks.calls,
            [
                ("before", "opening", None), ("after", "opening", "human"),
                ("before", "human", None), ("after", "human", "human"),
                ("before", "moderate", None), ("after", "moderate", ""),
            ],
        )

    def test_per_beat_counters_reset(self) -> None:
        async def go() -> list[dict[str, Any]]:
            act = FakeActuators(
                picks=[
                    FloorPick("guest", "queue_fifo"), FloorPick("human", "x"),
                    FloorPick("commentator", "queue_fifo"), FloorPick("human", "x"),
                ]
            )
            show = _session(act)
            show.start()
            out = [await show.post("human", event=HUMAN), await show.post("human", event=HUMAN)]
            await show.stop()
            return out

        first, second = asyncio.run(go())
        self.assertEqual(first["turn_idx"], 1)
        self.assertEqual(second["turn_idx"], 1)  # not 2: reset at the start of the beat
        self.assertEqual(second["spoken_roles"], ["commentator"])
        self.assertEqual(len(second["spoken"]), len(first["spoken"]))
        self.assertEqual(second["beats"], 2)

    def test_events_run_in_order_one_at_a_time(self) -> None:
        async def go() -> FakeActuators:
            act = FakeActuators(picks=[FloorPick("human", "x")] * 3, speak_s=0.02)
            show = _session(act)
            show.start()
            futs = [show.post("moderate", trigger=f"t{i}") for i in range(3)]
            await asyncio.gather(*futs)
            await show.stop()
            return act

        act = asyncio.run(go())
        # Each beat: open_floor line then grant — never interleaved across beats.
        self.assertEqual(
            [k for k, _ in act.kinds("speak")], ["open_floor", "grant_human"] * 3
        )
        self.assertEqual(act.kinds("return_to_host"), ["t0", "t1", "t2"])


class SessionRobustnessTests(unittest.TestCase):
    def test_shutdown_stops_and_releases_queued_events(self) -> None:
        async def go() -> tuple[Any, ShowSession, RecordingHooks]:
            down = asyncio.Event()
            hooks = RecordingHooks()
            act = FakeActuators(picks=[FloorPick("human", "x")] * 2, speak_s=0.05)
            show = _session(act, hooks=hooks, shutdown=down)
            show.start()
            first = show.post("moderate", trigger="a")
            queued = show.post("human", event=HUMAN)
            await asyncio.sleep(0.01)
            down.set()  # session over while the first beat runs
            await first
            result = await asyncio.wait_for(queued, timeout=2)
            await show.stop()
            return result, show, hooks

        result, show, hooks = asyncio.run(go())
        self.assertIsNone(result)
        self.assertFalse(show.running)
        # The dropped human event still got after_beat (releases its busy claim).
        self.assertIn(("after", "human", None), hooks.calls)
        self.assertNotIn(("before", "human", None), hooks.calls)

    def test_failed_beat_restarts_graph_and_keeps_serving(self) -> None:
        class Boom(FakeActuators):
            async def generate_panelist_line(self, role, step):
                raise RuntimeError("vLLM down")

        async def go() -> tuple[Any, Any, ShowSession]:
            act = Boom(picks=[FloorPick("guest", "queue_fifo"), FloorPick("human", "x")])
            show = _session(act)
            show.start()
            failed = await show.post("moderate", trigger="a")
            ok = await show.post("moderate", trigger="b")
            await show.stop()
            return failed, ok, show

        failed, ok, show = asyncio.run(go())
        self.assertIsNone(failed)
        self.assertEqual(ok["last_outcome"], "human")
        self.assertEqual(show.beats, 1)

    def test_checkpoint_types_allowed_in_strict_mode(self) -> None:
        async def go() -> dict[str, Any]:
            act = FakeActuators(picks=[FloorPick("guest", "queue_fifo"), FloorPick("human", "x")])
            show = _session(act)
            show.start()
            # human_event carries a ParsedTurn; extra_utterances / appraise jobs carry Utterances.
            r = await show.post("human", event=HUMAN)
            r2 = await show.post("moderate", trigger="again")
            await show.stop()
            return {"r": r, "r2": r2}

        with mock.patch.dict(os.environ, {"LANGGRAPH_STRICT_MSGPACK": "true"}):
            out = asyncio.run(go())
        self.assertIsNotNone(out["r"])
        self.assertIsNotNone(out["r2"])


if __name__ == "__main__":
    unittest.main()
