"""Golden command-sequence tests for LangGraph floor planner (no LiveKit)."""

from __future__ import annotations

import unittest

from agent.show_graph.commands import command_ops
from agent.show_graph.graph import plan_moderation_step, plan_panel_step
from agent.show_graph.state import base_state


def _mod(**kwargs):
    base = base_state(
        phase="mod_start",
        trigger="turn_0",
        panel_roles=["commentator", "guest"],
        listen_role="host",
        participants=["host", "commentator", "guest", "human"],
        max_depth=8,
        max_turns=12,
    )
    base.update(kwargs)
    return base


class ModerationPlannerTests(unittest.TestCase):
    def test_open_floor_speak_when_queue_empty(self) -> None:
        state = _mod(phase="mod_start", skip_open_floor=False, queue=[])
        state = plan_moderation_step(state)
        self.assertEqual(state["phase"], "hand_raise")
        self.assertEqual(command_ops(state["pending_commands"]), ["open_floor_speak"])

    def test_open_floor_skip_when_queue_nonempty(self) -> None:
        state = _mod(phase="mod_start", skip_open_floor=False, queue=["guest"])
        state = plan_moderation_step(state)
        self.assertEqual(state["phase"], "hand_raise")
        self.assertEqual(
            command_ops(state["pending_commands"]), ["open_floor_skip_log"]
        )

    def test_session_start_skips_open_floor(self) -> None:
        state = _mod(
            phase="mod_start",
            trigger="session_start",
            skip_open_floor=True,
            queue=[],
        )
        state = plan_moderation_step(state)
        self.assertEqual(state["phase"], "hand_raise")
        self.assertEqual(command_ops(state["pending_commands"]), [])

    def test_hand_raise_emits_round(self) -> None:
        state = _mod(phase="hand_raise", queue=[])
        state = plan_moderation_step(state)
        self.assertEqual(state["phase"], "after_raise")
        self.assertEqual(command_ops(state["pending_commands"]), ["hand_raise_round"])

    def test_after_raise_fifo_panelist(self) -> None:
        state = _mod(
            phase="after_raise",
            next_role="guest",
            grant_reason="queue_fifo",
            trigger="turn_0",
        )
        state = plan_moderation_step(state)
        self.assertEqual(state["phase"], "done")
        self.assertEqual(state["outcome"], "guest")
        self.assertEqual(command_ops(state["pending_commands"]), ["panelist_beat"])
        self.assertEqual(state["pending_commands"][0]["role"], "guest")
        self.assertFalse(state["pending_commands"][0]["skip_intro"])

    def test_after_raise_direct_call_skips_intro(self) -> None:
        state = _mod(
            phase="after_raise",
            next_role="commentator",
            grant_reason="host_direct_no_raises",
        )
        state = plan_moderation_step(state)
        self.assertEqual(command_ops(state["pending_commands"]), ["panelist_beat"])
        self.assertTrue(state["pending_commands"][0]["skip_intro"])

    def test_after_raise_human(self) -> None:
        state = _mod(phase="after_raise", next_role="human", grant_reason="queue_fifo")
        state = plan_moderation_step(state)
        self.assertEqual(state["outcome"], "human")
        self.assertEqual(command_ops(state["pending_commands"]), ["grant_human"])

    def test_after_raise_close(self) -> None:
        state = _mod(phase="after_raise", next_role="close")
        state = plan_moderation_step(state)
        self.assertEqual(state["outcome"], "close")
        self.assertEqual(command_ops(state["pending_commands"]), [])

    def test_max_depth_short_circuits(self) -> None:
        state = _mod(phase="mod_start", depth=8, max_depth=8)
        state = plan_moderation_step(state)
        self.assertEqual(state["phase"], "done")
        self.assertEqual(command_ops(state["pending_commands"]), [])


class PanelPlannerTests(unittest.TestCase):
    def test_direct_panelist_speak(self) -> None:
        state = _mod(
            phase="panel_consume",
            floor_next="guest",
            turn_idx=0,
            trigger="after_human",
        )
        # base_state used mod defaults; override phase already set
        state = plan_panel_step(state)
        self.assertEqual(state["phase"], "panel_consume")
        self.assertEqual(command_ops(state["pending_commands"]), ["speak_direct_panelist"])
        self.assertEqual(state["turn_idx"], 1)

    def test_human_next_tag(self) -> None:
        state = _mod(phase="panel_consume", floor_next="human")
        state = plan_panel_step(state)
        self.assertEqual(state["phase"], "done")
        self.assertEqual(command_ops(state["pending_commands"]), ["grant_human"])

    def test_host_pending_goes_moderate(self) -> None:
        state = _mod(phase="panel_consume", floor_next="host", turn_idx=2)
        state = plan_panel_step(state)
        self.assertEqual(state["phase"], "panel_moderate")
        self.assertEqual(state["trigger"], "turn_2")
        self.assertEqual(command_ops(state["pending_commands"]), [])

    def test_empty_floor_next_goes_moderate(self) -> None:
        """After-human path seeds host; empty consume also opens moderation."""
        state = _mod(phase="panel_consume", floor_next="", turn_idx=0)
        state = plan_panel_step(state)
        self.assertEqual(state["phase"], "panel_moderate")

    def test_close_to_host_close_command(self) -> None:
        state = _mod(phase="panel_consume", floor_next="close")
        state = plan_panel_step(state)
        self.assertEqual(state["phase"], "panel_close")
        state = plan_panel_step(state)
        self.assertEqual(state["phase"], "done")
        self.assertEqual(command_ops(state["pending_commands"]), ["host_close"])

    def test_max_turns_closes(self) -> None:
        state = _mod(phase="panel_consume", floor_next="guest", turn_idx=12, max_turns=12)
        state = plan_panel_step(state)
        self.assertEqual(state["phase"], "panel_close")


if __name__ == "__main__":
    unittest.main()
