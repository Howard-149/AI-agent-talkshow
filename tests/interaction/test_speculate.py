"""Thinking ahead on a partial human turn (agent/interaction/speculate.py)."""

from __future__ import annotations

import asyncio
import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.interaction import ThinkActionInteraction
from agent.interaction.floor_interaction import Thought
from agent.interaction.speculate import covers
from agent.interaction.think_action import Action, CommunicativeIntent, FloorOutcome
from agent.show.show_history import ShowHistory

importlib.import_module("agent.agents")  # before agent.session.session_handoff (import cycle)

FLOOR_ROLES = ["host", "commentator", "guest"]
PARTIAL = "I use AI for every homework now and I think I stopped"
FINAL = "I use AI for every homework now and I think I stopped learning. Is that bad?"


def _action(role: str, content: str) -> Action:
    if role != "guest":
        return Action("silent", None, None)
    return Action("speak", CommunicativeIntent("answer", "answer", None), content)


class CoversTests(unittest.TestCase):
    def test_a_few_trailing_words_are_fine(self) -> None:
        ok, missing = covers(PARTIAL, FINAL)
        self.assertTrue(ok)
        self.assertEqual(missing, 4)

    def test_too_much_said_after_the_partial(self) -> None:
        final = PARTIAL + " and also my sister thinks the whole school system should ban it entirely"
        self.assertFalse(covers(PARTIAL, final)[0])

    def test_a_partial_the_final_does_not_contain(self) -> None:
        self.assertFalse(covers("we should talk about the weather today", FINAL)[0])

    def test_too_short_to_think_on(self) -> None:
        self.assertFalse(covers("I use", "I use it")[0])


class SpeculationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for patcher in (
            patch.dict(
                os.environ, {"TALKSHOW_TEXT_TOKENIZER": "word", "TALKSHOW_SPECULATE_ROLES": "all"}
            ),
            patch.object(
                ThinkActionInteraction, "_floor_roles", staticmethod(lambda data: FLOOR_ROLES)
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        consumer = SimpleNamespace(stop=lambda: (None, False))
        self.interaction = ThinkActionInteraction(consumer)
        self.data = SimpleNamespace(
            show_history=ShowHistory(),
            floor_interaction=self.interaction,
            panel_chain_running=False,
            scenario=SimpleNamespace(turn_control=SimpleNamespace(listen_role="host", order=FLOOR_ROLES)),
            runtime=SimpleNamespace(
                gemma_client=SimpleNamespace(log_text_only_event=Mock(), complete_text=AsyncMock())
            ),
        )
        self.thought_on: list[tuple[str, str]] = []

        async def think(data, role, **kw):
            self.thought_on.append((role, kw["visible_text"]))
            history = kw["history"]
            self.assertEqual(history.lines[-1].role_id, "human")
            return Thought(SimpleNamespace(action=_action(role, f"reply to: {kw['visible_text']}"), think="t"))

        self.interaction._think = think
        self.interaction.arbitrate_free = AsyncMock(return_value=FloorOutcome("speak", "guest"))

    async def test_a_covering_speculation_is_used_instead_of_thinking_again(self) -> None:
        self.interaction.speculate_on_partial(self.data, PARTIAL)
        await asyncio.sleep(0)
        self.data.show_history.append("human", FINAL)

        outcome = await self.interaction.react_to_human(self.data, FINAL)

        self.assertEqual(outcome.selected_role, "guest")
        self.assertEqual(sorted(self.thought_on), sorted((role, PARTIAL) for role in FLOOR_ROLES))
        self.assertEqual(self.interaction.consume_proposed_reply("guest"), f"reply to: {PARTIAL}")
        self.assertEqual(self.interaction._speculations, [])

    async def test_a_finished_speculation_beats_a_newer_one_still_thinking(self) -> None:
        self.interaction.speculate_on_partial(self.data, PARTIAL)
        await self.interaction._speculations[-1].task  # done
        slow = asyncio.Event()
        fast_think = self.interaction._think

        async def think(data, role, **kw):
            await slow.wait()
            return await fast_think(data, role, **kw)

        self.interaction._think = think
        self.interaction.speculate_on_partial(self.data, PARTIAL + " learning")
        newer = self.interaction._speculations[-1].task
        self.data.show_history.append("human", FINAL)

        await asyncio.wait_for(self.interaction.react_to_human(self.data, FINAL), timeout=1)

        self.assertEqual(self.interaction.consume_proposed_reply("guest"), f"reply to: {PARTIAL}")
        await asyncio.sleep(0)
        self.assertTrue(newer.cancelled())

    async def test_a_stale_speculation_is_dropped(self) -> None:
        self.interaction.speculate_on_partial(self.data, "we should talk about the weather today then")
        await asyncio.sleep(0)
        self.data.show_history.append("human", FINAL)

        await self.interaction.react_to_human(self.data, FINAL)

        self.assertEqual(
            sorted(text for _, text in self.thought_on if text == FINAL), [FINAL] * len(FLOOR_ROLES)
        )
        self.assertEqual(self.interaction.consume_proposed_reply("guest"), f"reply to: {FINAL}")

    async def test_speculation_leaves_floor_state_alone(self) -> None:
        self.interaction.speculate_on_partial(self.data, PARTIAL)
        await self.interaction._speculations[-1].task
        self.assertEqual(self.interaction.actions, {})
        self.assertEqual(self.data.show_history.lines, [])

    async def test_speculation_survives_the_barge_in_rollback(self) -> None:
        # The rollback restores a checkpoint taken before the human spoke; on 11/17 it
        # restored the empty speculation list too, so no speculation was ever used.
        checkpoint = self.interaction._checkpoint(self.data)
        self.interaction.speculate_on_partial(self.data, PARTIAL)
        self.interaction._restore(self.data, checkpoint)
        self.assertEqual(len(self.interaction._speculations), 1)

    async def test_an_abandoned_turn_cancels_speculation(self) -> None:
        self.interaction.speculate_on_partial(self.data, PARTIAL)
        task = self.interaction._speculations[-1].task
        self.interaction.human_turn_abandoned()
        await asyncio.sleep(0)
        self.assertTrue(task.cancelled())
        self.assertEqual(self.interaction._speculations, [])


class OneResponderTests(SpeculationTests):
    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {"TALKSHOW_SPECULATE_ROLES": "one"})
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_only_the_likely_responder_thinks_ahead_and_answers(self) -> None:
        # Ryan (commentator) spoke last, so Amy (guest) thinks ahead; she has a reply.
        self.data.show_history.append("commentator", "Earlier line.")
        self.interaction.speculate_on_partial(self.data, PARTIAL)
        await asyncio.sleep(0)
        self.data.show_history.append("human", FINAL)

        await self.interaction.react_to_human(self.data, FINAL)

        self.assertEqual(self.thought_on, [("guest", PARTIAL)])
        self.assertEqual(self.interaction.actions["commentator"].decision, "silent")
        self.assertEqual(self.interaction.actions["host"].decision, "silent")
        self.assertEqual(self.interaction.consume_proposed_reply("guest"), f"reply to: {PARTIAL}")

    async def test_a_silent_responder_means_everyone_thinks(self) -> None:
        # Amy spoke last: Ryan thinks ahead, and (in this fake) stays silent.
        self.data.show_history.append("guest", "Earlier line.")
        self.interaction.speculate_on_partial(self.data, PARTIAL)
        await asyncio.sleep(0)
        self.data.show_history.append("human", FINAL)

        await self.interaction.react_to_human(self.data, FINAL)

        self.assertEqual(self.thought_on[0], ("commentator", PARTIAL))
        self.assertEqual(sorted(r for r, t in self.thought_on if t == FINAL), sorted(FLOOR_ROLES))

    # The parent tests assume every panelist thinks ahead.
    test_a_covering_speculation_is_used_instead_of_thinking_again = None
    test_a_finished_speculation_beats_a_newer_one_still_thinking = None


class LikelyResponderTests(unittest.TestCase):
    def test_a_named_panelist_answers(self) -> None:
        from agent.interaction.speculate import likely_responder

        with patch("agent.config.load_persona_name", lambda r: {"guest": "Amy", "commentator": "Ryan", "host": "Claire"}[r]):
            self.assertEqual(
                likely_responder("Ryan, do you really think so", FLOOR_ROLES, ["commentator"], "host"),
                "commentator",
            )
            self.assertEqual(
                likely_responder("I really think so", FLOOR_ROLES, ["guest", "commentator"], "host"),
                "guest",
            )


if __name__ == "__main__":
    unittest.main()
