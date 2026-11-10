"""Dialogue quality fixes on Think & Action panel turns (all flags default off)."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.interaction import ThinkActionInteraction
from agent.interaction.think_action import Action, CommunicativeIntent, FloorOutcome
from agent.panel.concrete import CONCRETE
from agent.panel.think_prompt import HEADER, RESPOND_TO_HUMAN, think_prompt_additions
from agent.panel.uptake import PLANNER_SYSTEM
from agent.show.show_history import ShowHistory

importlib.import_module("agent.agents")  # before agent.session.session_handoff (import cycle)

FLOOR_ROLES = ["host", "commentator", "guest"]
FLAGS = (
    "TALKSHOW_ADDRESSEE_FLOOR",
    "TALKSHOW_UPTAKE_PLANNER",
    "TALKSHOW_PANEL_SCAFFOLD_FIX",
    "TALKSHOW_DIALOGUE_LIBRARY",
    "TALKSHOW_PANEL_CONCRETE",
)
ALL_ON = {
    "TALKSHOW_UPTAKE_PLANNER": "lite",
    "TALKSHOW_PANEL_SCAFFOLD_FIX": "1",
    "TALKSHOW_DIALOGUE_LIBRARY": "panel_concrete",
    "TALKSHOW_PANEL_CONCRETE": "1",
}
PLAN = "- 40 percent of renters\n- addressed: Amy"
THINK_JSON = json.dumps(
    {
        "think": "x",
        "action": {
            "decision": "speak",
            "intent": {"type": "answer", "goal": "answer", "target": None},
            "content": "My point.",
            "backchannels": [],
        },
    }
)
SPEAK = Action("speak", CommunicativeIntent("answer", "answer", None), "My point.")
SILENT = Action("silent", None, None)


class _NullConsumer:
    async def submit(self, data, cue):
        return None

    def disclose(self, *args):
        return None

    def close(self, cue, end):
        cue.end = end

    def stop(self):
        return None, False

    def idle(self):
        return True

    def ahead(self):
        return 0


def make_data() -> SimpleNamespace:
    calls: list[dict] = []

    async def complete_text(prompt, **kw):
        calls.append({"prompt": prompt, **kw})
        if kw.get("system_prompt") == PLANNER_SYSTEM:
            return PLAN
        return THINK_JSON

    return SimpleNamespace(
        show_history=ShowHistory(),
        scenario=SimpleNamespace(turn_control=SimpleNamespace(listen_role="host", order=["human", *FLOOR_ROLES])),
        runtime=SimpleNamespace(
            gemma_client=SimpleNamespace(log_text_only_event=Mock(), complete_text=complete_text, calls=calls)
        ),
        room_name="test",
        turn_log=None,
    )


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {})
        env.start()
        self.addCleanup(env.stop)
        for name in FLAGS:
            os.environ.pop(name, None)
        roles = patch.object(ThinkActionInteraction, "_floor_roles", staticmethod(lambda data: FLOOR_ROLES))
        roles.start()
        self.addCleanup(roles.stop)

    async def think(self, data, role: str, speaker: str = "human", text: str = "") -> dict:
        interaction = ThinkActionInteraction(_NullConsumer())
        history = ShowHistory(lines=list(data.show_history.lines))
        await interaction._think_and_act(
            data,
            role,
            history=history,
            speaker=speaker,
            interval_text=text,
            visible_text=text,
            checkpoint=1,
            floor_state="free",
        )
        return [c for c in data.runtime.gemma_client.calls if c.get("system_prompt") != PLANNER_SYSTEM][-1]


class ThinkPromptFlagsOffTests(_Base):
    async def test_flags_off_prompt_is_unchanged(self) -> None:
        data = make_data()
        data.show_history.append("human", "Amy, rents rose 40 percent here.")
        on = await self.think(data, "guest", text="Amy, rents rose 40 percent here.")
        with patch("agent.panel.think_prompt.think_prompt_additions", AsyncMock(return_value=("", {}))):
            ref = await self.think(make_data_with(data), "guest", text="Amy, rents rose 40 percent here.")
        self.assertEqual(on["prompt"], ref["prompt"])
        self.assertEqual(on["system_prompt"], ref["system_prompt"])
        self.assertEqual(on["history_messages"], ref["history_messages"])
        self.assertTrue(on["prompt"].endswith("speak is valid only when free."))
        self.assertNotIn(HEADER, on["prompt"])
        self.assertEqual(len(data.runtime.gemma_client.calls), 1)  # no planner call
        kinds = [c.args[1] for c in data.runtime.gemma_client.log_text_only_event.call_args_list]
        self.assertFalse(any(k.startswith("think additions") for k in kinds))

    async def test_additions_are_empty_for_every_role_when_flags_are_off(self) -> None:
        data = make_data()
        data.show_history.append("human", "Amy, what do you think?")
        for role in FLOOR_ROLES:
            self.assertEqual(await think_prompt_additions(data, role, data.show_history), ("", {}))


def make_data_with(src: SimpleNamespace) -> SimpleNamespace:
    data = make_data()
    data.show_history = ShowHistory(lines=list(src.show_history.lines))
    return data


class ThinkPromptBlocksTests(_Base):
    async def test_blocks_reach_the_panelist_think_prompt_after_a_human_turn(self) -> None:
        os.environ.update(ALL_ON)
        data = make_data()
        data.show_history.append("guest", "Housing is about people.")
        data.show_history.append("human", "Amy, rents here rose 40 percent in two years.")
        call = await self.think(data, "guest", text="Amy, rents here rose 40 percent in two years.")
        prompt = call["prompt"]
        self.assertIn(HEADER, prompt)
        self.assertIn(RESPOND_TO_HUMAN, prompt)
        self.assertIn("The human guest's concrete points (extracted from their latest line):\n" + PLAN, prompt)
        self.assertIn("Make your content engage the item that matters most", prompt)
        self.assertIn("Style example (do not copy its topic)", prompt)
        self.assertTrue(prompt.endswith(CONCRETE))
        self.assertNotIn("[engages]", prompt)
        planner = [c for c in data.runtime.gemma_client.calls if c.get("system_prompt") == PLANNER_SYSTEM]
        self.assertEqual(len(planner), 1)
        self.assertIn("rents here rose 40 percent", planner[0]["prompt"])

    async def test_full_planner_mode_uses_the_lite_instruction(self) -> None:
        os.environ["TALKSHOW_UPTAKE_PLANNER"] = "1"
        data = make_data()
        data.show_history.append("human", "Rents rose 40 percent.")
        prompt = (await self.think(data, "commentator", text="Rents rose 40 percent."))["prompt"]
        self.assertIn("Make your content engage", prompt)
        self.assertNotIn("[engages]", prompt)

    async def test_host_line_after_the_human_still_counts_as_a_human_turn(self) -> None:
        os.environ.update(ALL_ON)
        data = make_data()
        data.show_history.append("human", "Rents rose 40 percent.")
        data.show_history.append("host", "Strong claim. Panel?")
        prompt = (await self.think(data, "guest", speaker="host", text="Strong claim. Panel?"))["prompt"]
        self.assertIn(RESPOND_TO_HUMAN, prompt)

    async def test_after_a_panelist_answered_only_the_style_blocks_remain(self) -> None:
        os.environ.update(ALL_ON)
        data = make_data()
        data.show_history.append("human", "Rents rose 40 percent.")
        data.show_history.append("commentator", "Forty percent is a cartel, not a market.")
        prompt = (await self.think(data, "guest", speaker="commentator", text="Forty"))["prompt"]
        self.assertNotIn(RESPOND_TO_HUMAN, prompt)
        self.assertNotIn("concrete points", prompt)
        self.assertTrue(prompt.endswith(CONCRETE))
        self.assertFalse(any(c.get("system_prompt") == PLANNER_SYSTEM for c in data.runtime.gemma_client.calls))

    async def test_host_think_gets_nothing(self) -> None:
        os.environ.update(ALL_ON)
        data = make_data()
        data.show_history.append("human", "Rents rose 40 percent.")
        prompt = (await self.think(data, "host", text="Rents rose 40 percent."))["prompt"]
        self.assertNotIn(HEADER, prompt)

    async def test_late_plan_is_skipped(self) -> None:
        os.environ["TALKSHOW_UPTAKE_PLANNER"] = "lite"
        data = make_data()
        data.show_history.append("human", "Rents rose 40 percent.")
        never = asyncio.get_running_loop().create_future()
        data.uptake_plan_for = "Rents rose 40 percent."
        data.uptake_plan_task = never
        with patch("agent.panel.uptake.PLAN_WAIT_TIMEOUT_S", 0.01):
            text, info = await think_prompt_additions(data, "guest", data.show_history)
        self.assertEqual(text, "")
        self.assertFalse(info["plan_used"])
        never.cancel()

    async def test_commit_human_turn_starts_the_planner(self) -> None:
        from agent.show_graph.actuators import LiveActuators

        interaction = ThinkActionInteraction(_NullConsumer())
        interaction.react_to_human = AsyncMock(return_value=FloorOutcome("speak", "guest"))
        data = make_data()
        data.floor_interaction = interaction
        controller = SimpleNamespace(listen_role=lambda: "host")
        act = LiveActuators(None, data, controller)
        act.listeners = lambda speaker: []  # no appraisal of the human line here
        started = Mock()
        with (
            patch("agent.panel.uptake.start_uptake_plan", started),
            patch("agent.session.human_turn.record_human_line", AsyncMock()),
            patch("agent.floor.floor_control.apply_floor_next", lambda data, role: None),
            patch("agent.emotion.appraisal.snapshot_context", return_value=None),
        ):
            await act.commit_human_turn({"heard": "Amy, is that fair?"})
        started.assert_called_once_with(data, "Amy, is that fair?")


class AddresseeFloorTests(_Base):
    def make(self, history: list[tuple[str, str]], actions: dict[str, Action]):
        interaction = ThinkActionInteraction(_NullConsumer())
        interaction.actions.update(actions)
        data = make_data()
        for role, text in history:
            data.show_history.append(role, text)
        controller = AsyncMock(return_value='{"decision":"speak","selected_role":"commentator",'
                                             '"reason_code":"high_contribution","reason":"r"}')
        data.runtime.gemma_client.complete_text = controller
        return interaction, data, controller

    async def arbitrate(self, interaction, data) -> FloorOutcome:
        candidates = [r for r in FLOOR_ROLES if (a := interaction.actions.get(r)) and a.decision == "speak"]
        return await interaction.arbitrate_free(
            data, candidates=candidates, history=data.show_history, interval=1
        )

    async def test_named_panelist_who_chose_speak_answers_first(self) -> None:
        os.environ["TALKSHOW_ADDRESSEE_FLOOR"] = "1"
        interaction, data, controller = self.make(
            [("human", "Amy, do you buy that?")], {"commentator": SPEAK, "guest": SPEAK}
        )
        outcome = await self.arbitrate(interaction, data)
        self.assertEqual((outcome.decision, outcome.selected_role), ("speak", "guest"))
        self.assertEqual(outcome.reason_code, "directed_response")
        controller.assert_not_awaited()
        self.assertIs(interaction.last_floor_outcome, outcome)

    async def test_flag_off_leaves_the_choice_to_the_controller(self) -> None:
        interaction, data, controller = self.make(
            [("human", "Amy, do you buy that?")], {"commentator": SPEAK, "guest": SPEAK}
        )
        outcome = await self.arbitrate(interaction, data)
        self.assertEqual(outcome.selected_role, "commentator")
        controller.assert_awaited_once()

    async def test_named_panelist_who_chose_silent_is_not_forced(self) -> None:
        os.environ["TALKSHOW_ADDRESSEE_FLOOR"] = "1"
        interaction, data, controller = self.make(
            [("human", "Amy, do you buy that?")], {"commentator": SPEAK, "guest": SILENT}
        )
        outcome = await self.arbitrate(interaction, data)
        self.assertEqual(outcome.selected_role, "commentator")
        controller.assert_awaited_once()
        data.runtime.gemma_client.log_text_only_event.assert_any_call(
            "floor", "addressee", addressed_role="guest", decision="silent", picked=False
        )

    async def test_host_line_in_between_keeps_the_addressee(self) -> None:
        os.environ["TALKSHOW_ADDRESSEE_FLOOR"] = "1"
        interaction, data, _ = self.make(
            [("human", "What do you make of that, Amy?"), ("host", "Good question.")],
            {"commentator": SPEAK, "guest": SPEAK},
        )
        self.assertEqual((await self.arbitrate(interaction, data)).selected_role, "guest")

    async def test_no_preference_once_a_panelist_answered(self) -> None:
        os.environ["TALKSHOW_ADDRESSEE_FLOOR"] = "1"
        interaction, data, controller = self.make(
            [("human", "Amy, do you buy that?"), ("guest", "I do.")], {"commentator": SPEAK}
        )
        await self.arbitrate(interaction, data)
        controller.assert_awaited_once()

    async def test_react_to_human_picks_the_named_panelist(self) -> None:
        """The path every human turn takes (live, ghost and text-only sessions)."""
        os.environ["TALKSHOW_ADDRESSEE_FLOOR"] = "1"
        interaction, data, controller = self.make([("human", "Amy, do you buy that?")], {})

        async def think(data, role, **kw):
            interaction.actions[role] = SPEAK if role != "host" else SILENT
            return interaction.actions[role]

        interaction._think_and_act = think
        outcome = await interaction.react_to_human(data, "Amy, do you buy that?")
        self.assertEqual(outcome.selected_role, "guest")
        controller.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
