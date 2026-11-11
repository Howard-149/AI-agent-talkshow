"""Floor rules for Think & Action interrupts (TALKSHOW_FLOOR_RULES): pure gate + interaction wiring."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from agent.floor.floor_rules import (
    HOST_OCCUPIED_GUIDANCE,
    QUOTE_RULE,
    InterruptRequest,
    floor_rules_enabled,
    gate_interrupts,
    quote_heard,
)
from agent.interaction.think_action import Action, CommunicativeIntent, parse_think_action
from agent.show.show_history import ShowHistory
from tests.interaction.test_floor_interaction import SPEAK, FloorInteractionTestCase

# 12 words: past the 3 s minimum speaking time at 2.6 words/s (7.8 words).
SPOKEN = "Cars should be banned downtown because every study shows traffic deaths double"
CHALLENGE = CommunicativeIntent("challenge", "dispute the claim", "guest")
MODERATE = CommunicativeIntent("facilitate", "back to topic", "guest")


def interrupt(quote: str | None, intent: CommunicativeIntent = CHALLENGE) -> Action:
    return Action("interrupt", intent, "Hold on, that's not right.", (), quote)


def gate(requests, *, text=SPOKEN, now_s=100.0, last_ai_cut_s=None, chain_cuts=0):
    return gate_interrupts(
        requests,
        host_role="host",
        visible_text=text,
        now_s=now_s,
        last_ai_cut_s=last_ai_cut_s,
        chain_cuts=chain_cuts,
    )


class FlagTests(unittest.TestCase):
    def test_on_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TALKSHOW_FLOOR_RULES", None)
            self.assertTrue(floor_rules_enabled())
        for off in ("0", "false", "off"):
            with patch.dict(os.environ, {"TALKSHOW_FLOOR_RULES": off}):
                self.assertFalse(floor_rules_enabled())


class QuoteTests(unittest.TestCase):
    def test_ignores_case_punctuation_and_quote_marks(self) -> None:
        self.assertTrue(quote_heard("“Every study shows”", SPOKEN))
        self.assertTrue(quote_heard("traffic deaths double.", SPOKEN))

    def test_rejects_paraphrase_stitching_and_empty(self) -> None:
        self.assertFalse(quote_heard("all research proves", SPOKEN))
        self.assertFalse(quote_heard("cars should traffic deaths", SPOKEN))
        self.assertFalse(quote_heard("", SPOKEN))
        self.assertFalse(quote_heard(None, SPOKEN))


class GateTests(unittest.TestCase):
    def test_ungrounded_request_is_blocked(self) -> None:
        result = gate([InterruptRequest("guest", None), InterruptRequest("commentator", "not said")])
        self.assertEqual(result.to_controller, [])
        self.assertEqual({d.rule for d in result.decisions}, {"quote_not_heard"})

    def test_grounded_panelist_goes_to_controller(self) -> None:
        result = gate([InterruptRequest("commentator", "every study shows")])
        self.assertIsNone(result.granted)
        self.assertEqual(result.to_controller, ["commentator"])

    def test_minimum_speaking_time(self) -> None:
        result = gate([InterruptRequest("commentator", "cars should")], text="Cars should be")
        self.assertEqual(result.decisions[0].rule, "minimum_speaking_time")
        host = gate([InterruptRequest("host", "cars should")], text="Cars should be")
        self.assertIsNone(host.granted)  # the speaker gets a fair start, even from the host

    def test_chain_limit_and_cooldown_block_panelists(self) -> None:
        request = [InterruptRequest("commentator", "every study shows")]
        self.assertEqual(gate(request, chain_cuts=1).decisions[0].rule, "chain_limit")
        self.assertEqual(gate(request, now_s=20, last_ai_cut_s=10).decisions[0].rule, "ai_cooldown")
        self.assertEqual(gate(request, now_s=30, last_ai_cut_s=10).to_controller, ["commentator"])

    def test_host_moderation_outranks_panelists_and_skips_ai_limits(self) -> None:
        result = gate(
            [
                InterruptRequest("commentator", "every study shows"),
                InterruptRequest("host", "traffic deaths double"),
            ],
            chain_cuts=1,
            now_s=11,
            last_ai_cut_s=10,
        )
        self.assertEqual(result.granted, "host")
        self.assertEqual(result.to_controller, [])
        by_role = {d.role: d for d in result.decisions}
        self.assertEqual(by_role["host"].priority, 80)
        self.assertEqual(by_role["commentator"].status, "blocked")

    def test_host_outranks_a_panelist_that_passed(self) -> None:
        result = gate(
            [
                InterruptRequest("commentator", "every study shows"),
                InterruptRequest("host", "traffic deaths double"),
            ]
        )
        self.assertEqual(result.granted, "host")
        rules = {d.role: d.rule for d in result.decisions}
        self.assertEqual(rules["commentator"], "moderator_priority")


class ParserTests(unittest.TestCase):
    def raw(self, decision: str, **extra) -> str:
        action = {
            "decision": decision,
            "intent": {"type": "challenge", "goal": "dispute", "target": None},
            "content": "Not so fast.",
            **extra,
        }
        return json.dumps({"think": "t", "action": action})

    def test_quote_is_kept_for_interrupt_only(self) -> None:
        parsed = parse_think_action(
            self.raw("interrupt", quote=" every study "), floor_state="occupied", interval_text="x"
        )
        self.assertEqual(parsed.action.quote, "every study")
        spoken = parse_think_action(
            self.raw("speak", quote="every study"), floor_state="free", interval_text="x"
        )
        self.assertIsNone(spoken.action.quote)
        missing = parse_think_action(self.raw("interrupt"), floor_state="occupied", interval_text="x")
        self.assertIsNone(missing.action.quote)


CONTROLLER_PICKS_COMMENTATOR = json.dumps(
    {"decision": "interrupt", "selected_role": "commentator", "reason_code": "timely_intervention", "reason": "r"}
)


class InteractionRuleTests(FloorInteractionTestCase):
    def setUp(self) -> None:
        super().setUp()
        rules = patch.dict(os.environ, {"TALKSHOW_FLOOR_RULES": "1"})
        rules.start()
        self.addCleanup(rules.stop)

    def listeners_return(self, interaction, actions: dict[str, Action]) -> None:
        async def think(data, role, **kw):
            interaction.actions[role] = actions.get(role, Action("silent", None, None))
            return interaction.actions[role]

        interaction._think_and_act = think

    async def occupied_interval(self, interaction, data):
        interaction._final_interval = False
        data.show_history.append("guest", SPOKEN)
        return await interaction._update_listeners(
            data,
            speaker="guest",
            interval_text="traffic deaths double",
            visible_text=SPOKEN,
            panel_roles=["commentator", "guest"],
            step="s",
            checkpoint=3,
        )

    async def test_ungrounded_interrupt_never_reaches_the_controller(self) -> None:
        interaction, _, data = self.make()
        self.listeners_return(interaction, {"commentator": interrupt(None)})

        result = await self.occupied_interval(interaction, data)

        self.assertIsNone(result)
        data.runtime.gemma_client.complete_text.assert_not_awaited()
        titles = [c.args[1] for c in data.runtime.gemma_client.log_text_only_event.call_args_list]
        self.assertIn("floor rules - interval 3", titles)

    async def test_grounded_panelist_is_judged_by_the_controller(self) -> None:
        interaction, _, data = self.make()
        interaction.rule_clock_words = 40
        self.listeners_return(interaction, {"commentator": interrupt("every study shows")})
        data.runtime.gemma_client.complete_text.return_value = CONTROLLER_PICKS_COMMENTATOR

        result = await self.occupied_interval(interaction, data)

        self.assertEqual(result, "commentator")
        data.runtime.gemma_client.complete_text.assert_awaited_once()
        self.assertEqual(interaction.rule_chain_cuts, 1)
        self.assertAlmostEqual(interaction.rule_last_ai_cut_s, (40 + 3) / 2.6)

    async def test_host_moderation_cut_skips_the_controller_and_returns_the_floor(self) -> None:
        interaction, _, data = self.make()
        self.listeners_return(
            interaction,
            {
                "commentator": interrupt("every study shows"),
                "host": interrupt("traffic deaths double", MODERATE),
            },
        )

        result = await self.occupied_interval(interaction, data)

        self.assertEqual(result, "host")
        data.runtime.gemma_client.complete_text.assert_not_awaited()
        self.assertEqual(interaction.rule_return_to, "guest")
        self.assertEqual(interaction.rule_chain_cuts, 1)
        self.assertEqual(interaction.last_floor_outcome.reason_code, "moderator_cut")

        # At the end of the host's line, the stopped speaker gets the floor back.
        interaction.actions = {"guest": SPEAK, "commentator": SPEAK}
        outcome = await interaction.arbitrate_free(
            data, candidates=["guest", "commentator"], history=data.show_history, interval=1
        )
        self.assertEqual(outcome.selected_role, "guest")
        self.assertEqual(outcome.reason_code, "return_after_moderation")
        data.runtime.gemma_client.complete_text.assert_not_awaited()
        self.assertIsNone(interaction.rule_return_to)

    async def test_a_cut_that_names_another_panelist_hands_them_the_floor(self) -> None:
        from agent.config import load_persona_name

        interaction, _, data = self.make()
        interaction.rule_return_to = "guest"
        interaction.actions = {"guest": SPEAK, "commentator": SPEAK}
        guest, commentator = load_persona_name("guest"), load_persona_name("commentator")
        data.show_history.append("guest", SPOKEN)
        data.show_history.append("host", f"{guest}, good point. {commentator}, what do you think?")

        outcome = await interaction.arbitrate_free(
            data, candidates=["guest", "commentator"], history=data.show_history, interval=1
        )

        self.assertEqual(outcome.selected_role, "commentator")
        self.assertEqual(outcome.reason_code, "directed_response")
        data.runtime.gemma_client.complete_text.assert_not_awaited()

    async def test_a_named_panelist_who_chose_silent_goes_to_the_controller(self) -> None:
        from agent.config import load_persona_name

        interaction, _, data = self.make()
        interaction.rule_return_to = "guest"
        interaction.actions = {"guest": SPEAK}
        data.show_history.append("host", f"{load_persona_name('commentator')}, your view?")
        data.runtime.gemma_client.complete_text.return_value = json.dumps(
            {"decision": "speak", "selected_role": "guest", "reason_code": "high_contribution", "reason": "r"}
        )

        await interaction.arbitrate_free(
            data, candidates=["guest"], history=data.show_history, interval=1
        )

        data.runtime.gemma_client.complete_text.assert_awaited_once()
        prompt = data.runtime.gemma_client.complete_text.await_args.args[0]
        self.assertIn("Pending host invitation target (role ID): 'commentator'", prompt)

    async def test_returned_speaker_who_chose_silent_is_not_forced(self) -> None:
        interaction, _, data = self.make()
        interaction.rule_return_to = "guest"
        interaction.actions = {"commentator": SPEAK}
        data.runtime.gemma_client.complete_text.return_value = json.dumps(
            {"decision": "speak", "selected_role": "commentator", "reason_code": "high_contribution", "reason": "r"}
        )

        outcome = await interaction.arbitrate_free(
            data, candidates=["commentator"], history=data.show_history, interval=1
        )

        self.assertEqual(outcome.selected_role, "commentator")
        data.runtime.gemma_client.complete_text.assert_awaited_once()

    async def test_the_interrupters_line_cannot_be_cut_by_a_panelist(self) -> None:
        interaction, _, data = self.make()
        interaction.rule_chain_cuts = 1  # the guest's line took over after a cut
        self.listeners_return(interaction, {"commentator": interrupt("every study shows")})

        self.assertIsNone(await self.occupied_interval(interaction, data))
        data.runtime.gemma_client.complete_text.assert_not_awaited()

    async def test_an_uncut_line_ends_the_chain(self) -> None:
        interaction, _, data = self.make()
        interaction.rule_chain_cuts = 1
        interaction.actions = {}
        self.listeners_return(interaction, {})
        interaction._final_interval = True
        data.runtime.gemma_client.complete_text.return_value = "{}"

        await interaction._update_listeners(
            data, speaker="guest", interval_text="x", visible_text=SPOKEN,
            panel_roles=["commentator", "guest"], step="s", checkpoint=4,
        )

        self.assertEqual(interaction.rule_chain_cuts, 0)

    async def test_disclosed_words_advance_the_speech_clock_and_roll_back(self) -> None:
        interaction, _, data = self.make()
        checkpoint = interaction._checkpoint(data)
        await self.speak(interaction, data, "guest", "one two three four five")
        self.assertEqual(interaction.rule_clock_words, 5)
        interaction.rule_chain_cuts = 1
        interaction._restore(data, checkpoint)
        self.assertEqual((interaction.rule_clock_words, interaction.rule_chain_cuts), (0, 0))


class InteractionFlagOffTests(FloorInteractionTestCase):
    async def test_ungrounded_interrupt_still_reaches_the_controller(self) -> None:
        interaction, _, data = self.make()
        os.environ["TALKSHOW_FLOOR_RULES"] = "0"  # restored by the base class's patch.dict

        async def think(data, role, **kw):
            interaction.actions[role] = interrupt(None) if role == "commentator" else Action("silent", None, None)
            return interaction.actions[role]

        interaction._think_and_act = think
        data.runtime.gemma_client.complete_text.return_value = CONTROLLER_PICKS_COMMENTATOR
        data.show_history.append("guest", SPOKEN)

        result = await interaction._update_listeners(
            data, speaker="guest", interval_text="x", visible_text=SPOKEN,
            panel_roles=["commentator", "guest"], step="s", checkpoint=1,
        )

        self.assertEqual(result, "commentator")
        self.assertEqual(interaction.rule_chain_cuts, 0)  # nothing is tracked with the rules off


class ThinkPromptTests(FloorInteractionTestCase):
    async def prompt_for(self, role: str, floor_state: str, *, rules: bool):
        interaction, _, data = self.make()
        data.runtime.gemma_client.complete_text.return_value = "{}"
        env = {"TALKSHOW_FLOOR_RULES": "1" if rules else "0"}
        with patch.dict(os.environ, env):
            await interaction._think_and_act(
                data, role, history=ShowHistory(), speaker="guest", interval_text="x",
                visible_text=SPOKEN, checkpoint=1, floor_state=floor_state,
            )
        call = data.runtime.gemma_client.complete_text.await_args
        return call.args[0], call.kwargs["system_prompt"]

    async def test_rules_off_prompt_has_no_rule_text(self) -> None:
        prompt, _ = await self.prompt_for("host", "occupied", rules=False)
        self.assertNotIn('"quote"', prompt)
        self.assertNotIn(QUOTE_RULE, prompt)
        self.assertNotIn(HOST_OCCUPIED_GUIDANCE, prompt)
        self.assertIn("witty reaction", prompt)

    async def test_host_moderates_while_occupied(self) -> None:
        prompt, system = await self.prompt_for("host", "occupied", rules=True)
        self.assertIn(HOST_OCCUPIED_GUIDANCE, prompt)
        self.assertNotIn("witty reaction", prompt)
        self.assertIn(QUOTE_RULE, prompt)
        self.assertTrue(prompt.endswith("speak is valid only when free."))
        self.assertIn("host and moderator", system)

    async def test_free_floor_prompt_asks_for_no_quote(self) -> None:
        on, system = await self.prompt_for("commentator", "free", rules=True)
        off, _ = await self.prompt_for("commentator", "free", rules=False)
        self.assertEqual(on, off)  # the content a selected speaker says is written as before
        self.assertNotIn("host and moderator", system)

    async def test_panelists_keep_their_guidance_and_must_quote(self) -> None:
        prompt, system = await self.prompt_for("commentator", "occupied", rules=True)
        self.assertIn("witty reaction", prompt)
        self.assertIn('"quote"', prompt)
        self.assertNotIn("host and moderator", system)


if __name__ == "__main__":
    unittest.main()
