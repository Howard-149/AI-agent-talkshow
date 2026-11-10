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


class OpeningTopicTests(unittest.TestCase):
    def run_opening(self, **event: Any) -> FakeActuators:
        async def go() -> FakeActuators:
            act = FakeActuators(picks=[FloorPick("human", "queue_fifo")])
            show = _session(act)
            show.start()
            await show.post("opening", text="Welcome!", texts={"en": "Welcome!"}, **event)
            await show.stop()
            return act

        return asyncio.run(go())

    def test_host_announces_topic_right_after_welcome(self) -> None:
        act = self.run_opening(topic="Today's discussion topic: AI in schools")
        self.assertEqual(
            [k for k, _ in act.kinds("speak")], ["welcome", "topic", "grant_human"]
        )
        self.assertIn(("host", "Today's discussion topic: AI in schools", False), act.history)

    def test_no_topic_no_extra_line(self) -> None:
        act = self.run_opening(topic="")
        self.assertEqual([k for k, _ in act.kinds("speak")], ["welcome", "grant_human"])

    def test_generated_opening_is_one_on_record_line(self) -> None:
        act = self.run_opening(
            topic="Today's discussion topic: AI in schools", generated=True
        )
        self.assertEqual([k for k, _ in act.kinds("speak")], ["welcome", "grant_human"])
        self.assertIn(("host", "Welcome!", False), act.history)


class TopicConfigTests(unittest.TestCase):
    def test_topic_env_prefers_new_name_and_keeps_old_one(self) -> None:
        import importlib
        from unittest.mock import patch

        importlib.import_module("agent.agents")  # before agent.floor.host_floor (import cycle)
        from agent.floor.host_floor import show_topic, topic_announcement

        with patch.dict(os.environ, {"TALKSHOW_TOPIC": "", "TALKSHOW_TEXT_TOPIC": "Old"}):
            self.assertEqual(show_topic(), "Old")
        with patch.dict(os.environ, {"TALKSHOW_TOPIC": "New", "TALKSHOW_TEXT_TOPIC": "Old"}):
            self.assertEqual(topic_announcement(), "Today's discussion topic: New")
        with patch.dict(os.environ, {"TALKSHOW_TOPIC": "", "TALKSHOW_TEXT_TOPIC": ""}):
            self.assertEqual(topic_announcement(), "")
            self.assertEqual(show_topic("Default"), "Default")


class HostOpeningLineTests(unittest.TestCase):
    def opening(self, reply: str | Exception, topic: str = "AI in schools") -> str:
        import importlib
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        importlib.import_module("agent.agents")  # before agent.floor.host_floor (import cycle)
        from agent.floor import host_floor

        client = SimpleNamespace(
            complete_text=AsyncMock(
                side_effect=reply if isinstance(reply, Exception) else None,
                return_value=None if isinstance(reply, Exception) else reply,
            )
        )
        data = SimpleNamespace(runtime=SimpleNamespace(gemma_client=client))
        with (
            patch.dict(os.environ, {"TALKSHOW_TOPIC": topic, "TALKSHOW_TEXT_TOPIC": ""}),
            patch.object(host_floor, "session_welcome_line", return_value=("Welcome!", {})),
            patch("agent.config.load_persona_instructions", return_value="host persona"),
            patch("agent.config.load_persona_name", return_value="Lessac"),
        ):
            return asyncio.run(host_floor.host_opening_line(data))

    def test_opening_is_the_generated_reply_only(self) -> None:
        text = self.opening(
            "[heard]: You are Lessac, opening today's show.\n"
            "[reply]: Welcome in! Today we ask whether AI belongs in schools."
        )
        self.assertEqual(text, "Welcome in! Today we ask whether AI belongs in schools.")

    def test_opening_falls_back_to_canned_lines(self) -> None:
        canned = "Welcome! Today's discussion topic: AI in schools"
        self.assertEqual(self.opening("[heard]: You are Lessac, opening the show."), canned)
        self.assertEqual(self.opening(RuntimeError("api down")), canned)

    def test_no_topic_keeps_canned_welcome(self) -> None:
        self.assertEqual(self.opening("Unused.", topic=""), "Welcome!")

    def test_opening_invites_first_panelist(self) -> None:
        import importlib
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        importlib.import_module("agent.agents")  # before agent.floor.host_floor (import cycle)
        from agent.floor import host_floor

        client = SimpleNamespace(complete_text=AsyncMock(return_value="Hi, I'm Lessac. Ryan?"))
        interaction = SimpleNamespace(pending_invitation_target=None, opening_invite_role=None)
        data = SimpleNamespace(
            runtime=SimpleNamespace(gemma_client=client),
            scenario=SimpleNamespace(
                turn_control=SimpleNamespace(
                    order=["human", "host", "commentator", "guest"], listen_role="host"
                )
            ),
            floor_interaction=interaction,
        )
        with (
            patch.dict(os.environ, {"TALKSHOW_TOPIC": "AI in schools", "TALKSHOW_TEXT_TOPIC": ""}),
            patch.object(host_floor, "session_welcome_line", return_value=("Welcome!", {})),
            patch("agent.config.load_persona_instructions", return_value="host persona"),
            patch("agent.config.load_persona_name", side_effect=lambda role: role.title()),
        ):
            text = asyncio.run(host_floor.host_opening_line(data))

        self.assertEqual(text, "Hi, I'm Lessac. Ryan?")
        self.assertEqual(interaction.opening_invite_role, "commentator")
        self.assertEqual(interaction.pending_invitation_target, "commentator")
        prompt = client.complete_text.await_args.args[0]
        self.assertIn("introduce yourself", prompt)
        self.assertIn("inviting Commentator", prompt)


class HostClosingLineTests(unittest.TestCase):
    def closing(self, reply: str | Exception) -> str:
        import importlib
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        importlib.import_module("agent.agents")  # before agent.floor.host_floor (import cycle)
        from agent.floor import host_floor

        client = SimpleNamespace(
            complete_text=AsyncMock(
                side_effect=reply if isinstance(reply, Exception) else None,
                return_value=None if isinstance(reply, Exception) else reply,
            )
        )
        data = SimpleNamespace(
            runtime=SimpleNamespace(gemma_client=client),
            show_history=SimpleNamespace(prior_messages=lambda perspective: []),
        )
        with (
            patch.dict(os.environ, {"TALKSHOW_TOPIC": "AI in schools", "TALKSHOW_TEXT_TOPIC": ""}),
            patch("agent.config.load_persona_instructions", return_value="host persona"),
            patch("agent.config.load_persona_name", return_value="Lessac"),
        ):
            return asyncio.run(host_floor.host_closing_line(data))

    def test_closing_is_the_generated_reply(self) -> None:
        text = self.closing("[reply]: We found real promise and real risk. Thanks, all.")
        self.assertEqual(text, "We found real promise and real risk. Thanks, all.")

    def test_closing_fallback_asks_nothing(self) -> None:
        from agent.floor.host_floor import HOST_CLOSE_FALLBACK

        self.assertEqual(self.closing(RuntimeError("api down")), HOST_CLOSE_FALLBACK)
        self.assertEqual(self.closing(""), HOST_CLOSE_FALLBACK)
        self.assertNotIn("?", HOST_CLOSE_FALLBACK)


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

    def test_beat_failing_during_shutdown_stops_quietly(self) -> None:
        class Closing(FakeActuators):
            async def generate_panelist_line(self, role, step):
                raise RuntimeError("AgentSession is closing, cannot use say()")

        class ClosingHooks(RecordingHooks):
            def shutting_down(self) -> bool:
                return True

        async def go() -> tuple[Any, Any, ShowSession]:
            act = Closing(picks=[FloorPick("guest", "queue_fifo")] * 4)
            show = _session(act, hooks=ClosingHooks())
            show.start()
            first = show.post("moderate", trigger="a")
            queued = show.post("moderate", trigger="b")
            return await first, await queued, show

        first, queued, show = asyncio.run(go())
        self.assertIsNone(first)
        self.assertIsNone(queued)  # released, not run against a dead session
        self.assertFalse(show.running)
        self.assertEqual(show.beats, 0)

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
