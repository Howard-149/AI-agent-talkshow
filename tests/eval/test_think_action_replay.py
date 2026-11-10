"""Replay and offline evaluation on the Think & Action floor. No LiveKit, no model."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, Mock

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent.interaction import ThinkActionInteraction
from agent.interaction.think_action import Action, FloorOutcome
from agent.session.ghost_replay import replay_before_human_line, snapshot_state
from eval.think_action_prompt import PLAN_PLACEHOLDER, last_interval, spoken_content, think_request

FLAGS = (
    "TALKSHOW_GHOST_REPLAY",
    "TALKSHOW_ADDRESSEE_FLOOR",
    "TALKSHOW_UPTAKE_PLANNER",
    "TALKSHOW_PANEL_SCAFFOLD_FIX",
    "TALKSHOW_DIALOGUE_LIBRARY",
    "TALKSHOW_PANEL_CONCRETE",
)
PREFIX = {
    "lines": [
        {"role_id": "host", "speaker": "Lessac", "text": "Tonight: rent control.", "procedural": False},
        {"role_id": "guest", "speaker": "Amy", "text": "Rent control keeps families in place.", "procedural": False},
        {"role_id": "commentator", "speaker": "Ryan", "text": "And freezes the supply.", "procedural": False},
    ],
    "floor": {"actual_floor_counts": {"guest": 1, "commentator": 1}, "fallback_history": []},
}


class _Consumer:
    def stop(self):
        return None, False


class _Env(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {"TALKSHOW_TEXT_TOKENIZER": "word"})
        env.start()
        self.addCleanup(env.stop)
        for name in FLAGS:
            os.environ.pop(name, None)


class ThinkRequestTests(_Env):
    def test_matches_the_live_think_call(self) -> None:
        from agent.panel.panel_prompts import panelist_personality_block

        req = think_request(PREFIX, "guest", human_text="Amy, rents rose 40 percent here.")
        msgs = req["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": panelist_personality_block("guest")})
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("Human guest: Amy, rents rose 40 percent here.", msgs[-1]["content"])
        self.assertIn("Current floor holder: Human guest (human).", msgs[-1]["content"])
        self.assertIn("Latest public interval: Amy, rents rose 40 percent here.\n", msgs[-1]["content"])
        self.assertTrue(msgs[-1]["content"].endswith("speak is valid only when free."))
        self.assertEqual((req["temperature"], req["max_tokens"]), (0.2, 768))
        self.assertEqual(req["speaker"], "human")

    def test_flags_reach_the_offline_prompt(self) -> None:
        from agent.panel.concrete import CONCRETE

        os.environ.update({"TALKSHOW_UPTAKE_PLANNER": "lite", "TALKSHOW_PANEL_CONCRETE": "1"})
        req = think_request(PREFIX, "guest", human_text="Rents rose 40 percent.", plan="- 40 percent")
        last = req["messages"][-1]["content"]
        self.assertIn("extracted from their latest line):\n- 40 percent\n", last)
        self.assertTrue(last.endswith(CONCRETE))
        trace = think_request(PREFIX, "guest", human_text="Rents rose 40 percent.", plan=PLAN_PLACEHOLDER)
        self.assertIn(PLAN_PLACEHOLDER, trace["messages"][-1]["content"])

    def test_after_a_panel_line_the_speaker_is_that_panelist(self) -> None:
        req = think_request(PREFIX, "guest")
        self.assertEqual(req["speaker"], "commentator")
        self.assertEqual(req["interval_text"], last_interval("And freezes the supply."))
        self.assertIn("Current floor holder: Ryan (commentator).", req["messages"][-1]["content"])

    def test_spoken_content(self) -> None:
        speak = json.dumps({"think": "t", "action": {"decision": "speak", "intent": {
            "type": "answer", "goal": "g", "target": None}, "content": "Forty percent is real.", "backchannels": []}})
        self.assertEqual(spoken_content(speak, "x"), ("speak", "Forty percent is real."))
        silent = json.dumps({"think": "t", "action": {"decision": "silent", "intent": None, "content": None}})
        self.assertEqual(spoken_content(silent, "x"), ("silent", ""))
        self.assertEqual(spoken_content("not json", "x"), ("silent", ""))

    def test_model_only_trace_path(self) -> None:
        from eval.model_only import build_think_request

        req = build_think_request({"lines": PREFIX["lines"]}, "Is that fair, Ryan?", "commentator", False)
        self.assertEqual(req["panel_roles"], ["commentator", "guest"])
        self.assertIn("Is that fair, Ryan?", req["messages"][-1]["content"])


def _replay_data():
    from agent.config import load_scenario
    from agent.data import TalkShowData

    data = TalkShowData(scenario=load_scenario(), runtime=SimpleNamespace())  # type: ignore[arg-type]
    data.room_name = "ghost-replay"
    data.turn_log = Mock()
    data.show_history.append("host", "Welcome to the show.", speaker="host")
    return data


class ThinkActionReplayTests(_Env):
    def test_prefix_replaces_the_show_and_resets_the_floor(self) -> None:
        os.environ["TALKSHOW_GHOST_REPLAY"] = "1"
        data = _replay_data()
        interaction = ThinkActionInteraction(_Consumer())
        interaction.actions["guest"] = Action("silent", None, None)
        interaction.actual_floor_counts = {"host": 5}
        interaction.pending_invitation_target = "guest"
        interaction.opening_invite_role = "guest"
        data.floor_interaction = interaction
        replay_before_human_line(data, interaction, PREFIX)
        self.assertEqual([l.text for l in data.show_history.lines], [l["text"] for l in PREFIX["lines"]])
        self.assertEqual(interaction.actions, {})
        self.assertEqual(interaction.actual_floor_counts, {"guest": 1, "commentator": 1})
        self.assertIsNone(interaction.pending_invitation_target)
        self.assertIsNone(interaction.opening_invite_role)
        kind, fields = data.turn_log.log.call_args.args[0], data.turn_log.log.call_args.kwargs
        self.assertEqual(kind, "history_snapshot")
        self.assertEqual(fields["floor"]["actual_floor_counts"], {"guest": 1, "commentator": 1})
        self.assertEqual(json.loads(json.dumps(snapshot_state(data)))["floor"], PREFIX["floor"])

    def test_replay_off_is_a_no_op(self) -> None:
        data = _replay_data()
        interaction = ThinkActionInteraction(_Consumer())
        replay_before_human_line(data, interaction, PREFIX)
        self.assertEqual([l.text for l in data.show_history.lines], ["Welcome to the show."])
        data.turn_log.log.assert_not_called()


class GhostTurnThinkActionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {"TALKSHOW_GHOST_TURNS": "1"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("TALKSHOW_GHOST_REPLAY", None)

    async def run_turn(self, data, prefix=None):
        from agent.session import ghost_turn

        submitted = AsyncMock(return_value=True)
        controller = SimpleNamespace(is_host_moderated_mode=lambda: True)
        with (
            mock.patch("agent.interaction.human_may_speak", lambda data: True),
            mock.patch("agent.show_graph.submit_human_turn", submitted),
        ):
            ok = await ghost_turn.run_ghost_human_turn(None, data, controller, text="Amy, is that fair?",
                                                       prefix=prefix)
        return ok, submitted

    def data(self):
        return SimpleNamespace(turn_log=None, room_name="r", active_role="host", touch_activity=Mock())

    async def test_prefix_travels_with_the_turn_when_replay_is_on(self) -> None:
        os.environ["TALKSHOW_GHOST_REPLAY"] = "1"
        ok, submitted = await self.run_turn(self.data(), prefix=PREFIX)
        self.assertTrue(ok)
        event = submitted.await_args.args[3]
        self.assertEqual(event["prefix"], PREFIX)
        self.assertEqual(event["done_event"], "ghost_human_done")

    async def test_prefix_dropped_when_replay_is_off(self) -> None:
        _ok, submitted = await self.run_turn(self.data(), prefix=PREFIX)
        self.assertNotIn("prefix", submitted.await_args.args[3])

    async def test_repeat_of_the_last_ghost_turn_is_dropped(self) -> None:
        data = self.data()
        self.assertTrue((await self.run_turn(data))[0])
        ok, submitted = await self.run_turn(data)
        self.assertFalse(ok)
        submitted.assert_not_awaited()

    async def test_commit_human_turn_applies_the_prefix_before_the_human_line(self) -> None:
        from agent.show_graph.actuators import LiveActuators

        os.environ["TALKSHOW_GHOST_REPLAY"] = "1"
        data = _replay_data()
        interaction = ThinkActionInteraction(_Consumer())
        interaction.react_to_human = AsyncMock(return_value=FloorOutcome("speak", "guest"))
        data.floor_interaction = interaction

        async def record(data, *, heard, **kw):
            data.show_history.append("human", heard)

        act = LiveActuators(None, data, SimpleNamespace(listen_role=lambda: "host"))
        with (
            mock.patch("agent.session.human_turn.record_human_line", record),
            mock.patch("agent.floor.floor_control.apply_floor_next", lambda data, role: None),
            mock.patch("agent.emotion.appraisal.snapshot_context", return_value=None),
        ):
            await act.commit_human_turn({"heard": "Is that fair?", "done_event": "ghost_human_done",
                                         "prefix": PREFIX})
        self.assertEqual([l.text for l in data.show_history.lines],
                         [l["text"] for l in PREFIX["lines"]] + ["Is that fair?"])


class GhostClientTests(unittest.TestCase):
    def test_default_floor_wait_is_think_action(self) -> None:
        from eval.ghost_session.client import _build_parser

        self.assertEqual(_build_parser().parse_args([]).floor_wait, "none")
        self.assertEqual(_build_parser().parse_args(["--floor-wait", "grant"]).floor_wait, "grant")


if __name__ == "__main__":
    unittest.main()
