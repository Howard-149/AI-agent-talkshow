"""Think & Action floor interaction: free-running script, barge-in rollback, speech consumers."""

from __future__ import annotations

import asyncio
import importlib
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.interaction import AVSpeechConsumer, Cue, ThinkActionInteraction
from agent.interaction.think_action import Action, Backchannel, CommunicativeIntent, FloorOutcome
from agent.show.show_history import ShowHistory

importlib.import_module("agent.agents")  # before agent.session.session_handoff (import cycle)

FLOOR_ROLES = ["host", "commentator", "guest"]
LINE_A = "one two three four five six"
LINE_B = "seven eight nine ten"
SPEAK = Action("speak", CommunicativeIntent("contribute", "add", None), "My point.")


class RecordingConsumer:
    """Consumer stub that never plays: the generator must not wait for it."""

    def __init__(self) -> None:
        self.cues: list[Cue] = []
        self.disclosed: list[tuple[Cue, int, str, list]] = []
        self.position = None
        self.waiting = 0

    async def submit(self, data, cue):
        self.cues.append(cue)

    def disclose(self, cue, end, chunk, backchannels):
        self.disclosed.append((cue, end, chunk, backchannels))

    def close(self, cue, end):
        cue.end = end

    def stop(self):
        return self.position, self.position is not None

    def idle(self):
        return True

    def ahead(self):
        return self.waiting

    async def drain(self):
        return None


def make_data(interaction: ThinkActionInteraction) -> SimpleNamespace:
    return SimpleNamespace(
        show_history=ShowHistory(),
        floor_interaction=interaction,
        panel_chain_running=True,
        role_emotion={"guest": "neutral"},
        scenario=SimpleNamespace(
            turn_control=SimpleNamespace(listen_role="host", order=FLOOR_ROLES)
        ),
        runtime=SimpleNamespace(
            gemma_client=SimpleNamespace(log_text_only_event=Mock(), complete_text=AsyncMock())
        ),
    )


class FloorInteractionTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for patcher in (
            patch.dict(os.environ, {"TALKSHOW_TEXT_TOKENIZER": "word"}),
            patch.object(
                ThinkActionInteraction, "_floor_roles", staticmethod(lambda data: FLOOR_ROLES)
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make(self) -> tuple[ThinkActionInteraction, RecordingConsumer, SimpleNamespace]:
        consumer = RecordingConsumer()
        interaction = ThinkActionInteraction(consumer)
        interaction.token_interval = 2
        return interaction, consumer, make_data(interaction)

    async def speak(self, interaction, data, role: str, text: str, on_interval=None) -> None:
        """Record then disclose a line, like prepare_speak → speak in the show graph."""
        data.show_history.append(role, text)
        calls = {"n": 0}

        async def listeners(data, **kw):
            calls["n"] += 1
            interaction._interval_backchannels = []
            if on_interval is not None:
                on_interval(calls["n"])
            return None

        interaction._update_listeners = listeners
        await interaction.disclose_panel_line(data, {"role": role, "text": text})


class HostFallbackSpeechTests(FloorInteractionTestCase):
    async def test_internal_strategy_label_is_removed_from_spoken_line(self) -> None:
        interaction, _, data = self.make()
        interaction.host_fallback_instruction = "Invite the guest to share a perspective."
        interaction.host_fallback_strategy = "invite"
        interaction.host_fallback_target = "guest"
        data.runtime.gemma_client.complete_text.return_value = (
            "Host fallback strategy: invite. Would you share your perspective?"
        )

        text = await interaction.host_fallback_line(data, "host")

        self.assertEqual(text, "Would you share your perspective?")
        self.assertNotIn("Host fallback strategy", text)

    async def test_echoed_instruction_under_heard_is_not_spoken(self) -> None:
        interaction, _, data = self.make()
        interaction.host_fallback_instruction = "Frame the question around benefits and risks."
        interaction.host_fallback_strategy = "reframe"
        data.runtime.gemma_client.complete_text.return_value = (
            "[heard]: Controller instruction: Frame the question around benefits and risks.\n"
            "[reply]: Let's weigh both sides of AI coding."
        )

        text = await interaction.host_fallback_line(data, "host")

        self.assertEqual(text, "Let's weigh both sides of AI coding.")

    async def test_heard_without_reply_falls_back_to_canned_line(self) -> None:
        interaction, _, data = self.make()
        interaction.host_fallback_instruction = "Frame the question around benefits and risks."
        interaction.host_fallback_strategy = "reframe"
        data.runtime.gemma_client.complete_text.return_value = (
            "[heard]: Controller instruction: Frame the question around benefits and risks."
        )

        text = await interaction.host_fallback_line(data, "host")

        self.assertNotIn("Controller instruction", text)
        self.assertNotIn("[heard]", text)


class FreeRunningScriptTests(FloorInteractionTestCase):
    def test_backchannel_may_target_any_word_of_the_line(self) -> None:
        text = "A complete first sentence. A second sentence continues."
        segment_at = ThinkActionInteraction._backchannel_segment_at
        self.assertEqual(segment_at(text, len("A complete first")), 0)
        self.assertEqual(segment_at(text, len("A complete first sentence.")), 0)
        self.assertEqual(segment_at(text, len("A complete first sentence. A second")), 1)
        self.assertEqual(segment_at(text, len(text)), 1)

    async def test_listener_backchannels_are_arbitrated_once_per_speaker_cue(self) -> None:
        interaction, consumer, data = self.make()
        text = "A complete first sentence. A second sentence continues."
        first_event = Backchannel(
            CommunicativeIntent("acknowledge", "recognize the point", None),
            "Good point.",
            len("A complete first sentence.".split()),
        )
        second_event = Backchannel(
            CommunicativeIntent("support", "show agreement", None),
            "I agree.",
            len("A complete first sentence.".split()),
        )
        responses = iter(
            [
                Action("silent", None, None, (first_event,)),
                Action("silent", None, None, (second_event,)),
            ]
        )
        interaction._think_and_act = AsyncMock(side_effect=lambda *args, **kwargs: next(responses))
        interaction.arbitrate_free = AsyncMock(return_value=FloorOutcome("host_fallback"))
        interaction._final_interval = True

        await interaction._update_listeners(
            data,
            cue=Cue("guest", text, "speech_guest_0"),
            speaker="guest",
            interval_text=text,
            visible_text=text,
            panel_roles=["commentator"],
            step="speech_guest_0",
            checkpoint=1,
        )

        self.assertEqual(len(interaction._interval_backchannels), 1)
        role, accepted = interaction._interval_backchannels[0]
        self.assertEqual((role, accepted.content, accepted.target_segment_index), ("host", "Good point.", 0))
        self.assertEqual(len(consumer.disclosed), 0)

    async def test_lines_are_decided_without_waiting_for_delivery(self) -> None:
        interaction, consumer, data = self.make()
        await self.speak(interaction, data, "guest", LINE_A)
        await self.speak(interaction, data, "commentator", LINE_B)

        self.assertEqual([cue.text for cue in consumer.cues], [LINE_A, LINE_B])
        self.assertEqual([cue.end for cue in consumer.cues], [len(LINE_A), len(LINE_B)])
        self.assertEqual(len(consumer.disclosed), 5)

    async def test_backchannels_carry_their_offset_in_the_line(self) -> None:
        interaction, consumer, data = self.make()
        nod = Backchannel(CommunicativeIntent("acknowledge", "nod", None), "Right.", 1)

        def first_interval_nod(n: int) -> None:
            if n == 1:
                interaction._interval_backchannels = [("host", nod)]

        await self.speak(interaction, data, "guest", LINE_A, first_interval_nod)

        self.assertEqual(consumer.disclosed[0][3], [("host", nod, len("one"))])
        self.assertEqual(data.show_history.lines[-1].text, "Right.")


class LookaheadCapTests(FloorInteractionTestCase):
    async def test_generator_holds_three_lines_ahead_of_playback(self) -> None:
        interaction, consumer, data = self.make()
        consumer.waiting = 3
        queued = asyncio.create_task(interaction.play_line(data, "host", "Next.", step="x"))
        await asyncio.sleep(0.12)
        self.assertEqual(consumer.cues, [])  # held: three lines already wait to be played

        consumer.waiting = 2
        await asyncio.wait_for(queued, timeout=1)
        self.assertEqual([cue.text for cue in consumer.cues], ["Next."])

    async def test_human_barge_in_releases_a_held_generator(self) -> None:
        interaction, consumer, data = self.make()
        consumer.waiting = 3
        held = asyncio.create_task(interaction.play_line(data, "host", "Next.", step="x"))
        await asyncio.sleep(0.06)
        interaction.human_started_speaking(data)
        await asyncio.wait_for(held, timeout=1)
        self.assertEqual(consumer.cues, [])

    async def test_cap_is_configurable_and_zero_disables_it(self) -> None:
        interaction, consumer, data = self.make()
        consumer.waiting = 50
        with patch.dict(os.environ, {"TALKSHOW_MAX_LOOKAHEAD_TURNS": "0"}):
            await asyncio.wait_for(
                interaction.play_line(data, "host", "Next.", step="x"), timeout=1
            )
        self.assertEqual(len(consumer.cues), 1)


class BargeInRollbackTests(FloorInteractionTestCase):
    async def two_lines(self):
        interaction, consumer, data = self.make()
        nod = Backchannel(CommunicativeIntent("acknowledge", "nod", None), "Right.", 1)

        def guest_line(n: int) -> None:
            interaction._interval_backchannels = [("host", nod)] if n in (1, 3) else []
            if n == 3:
                interaction.actions["commentator"] = SPEAK
                data.role_emotion["guest"] = "excited"

        await self.speak(interaction, data, "guest", LINE_A, guest_line)
        await self.speak(interaction, data, "commentator", LINE_B)
        return interaction, consumer, data

    async def test_mid_line_barge_in_drops_everything_unheard(self) -> None:
        interaction, consumer, data = await self.two_lines()
        line_a = consumer.cues[0]
        consumer.position = (line_a, len("one two three"))

        interaction.human_started_speaking(data)
        self.assertTrue(interaction.human_floor_pending)
        interaction.apply_pending_rollback(data)

        self.assertEqual(
            [(line.role_id, line.text) for line in data.show_history.lines],
            [("guest", "one two three"), ("host", "Right.")],
        )
        # Think & decisions made after the heard point are gone, and so is the mood.
        self.assertEqual(interaction.actions, {})
        self.assertEqual(data.role_emotion, {"guest": "neutral"})
        self.assertEqual(interaction._rethink, ("guest", "one two three"))

    async def test_barge_in_between_lines_keeps_the_finished_line(self) -> None:
        interaction, consumer, data = await self.two_lines()
        consumer.position = (consumer.cues[1], 0)  # line B queued, not yet heard

        interaction.human_started_speaking(data)
        interaction.apply_pending_rollback(data)

        self.assertEqual(
            [line.text for line in data.show_history.lines], [LINE_A, "Right.", "Right."]
        )
        self.assertEqual(interaction.actions, {"commentator": SPEAK})
        self.assertEqual(data.role_emotion, {"guest": "excited"})
        self.assertIsNone(interaction._rethink)

    async def test_generator_stops_writing_once_the_human_holds_the_floor(self) -> None:
        interaction, consumer, data = self.make()

        def barge_in_at_second_interval(n: int) -> None:
            if n == 2:
                interaction.human_started_speaking(data)

        await self.speak(interaction, data, "guest", LINE_A, barge_in_at_second_interval)
        self.assertEqual(len(consumer.disclosed), 1)
        await interaction.play_line(data, "host", "Floor's open.", step="x")
        self.assertEqual(len(consumer.cues), 1)

    async def test_resumed_show_rethinks_on_the_cut_line(self) -> None:
        interaction, consumer, data = await self.two_lines()
        consumer.position = (consumer.cues[0], len("one two three"))
        interaction.human_started_speaking(data)
        self.assertFalse(interaction.human_turn_abandoned())  # still talking: keep the floor
        interaction.human_stopped_speaking(data)
        self.assertTrue(interaction.human_turn_abandoned())
        interaction.apply_pending_rollback(data)
        interaction._react_free = AsyncMock(return_value=FloorOutcome("speak", "commentator"))

        outcome = await interaction.select_free_floor_outcome(data, interval=1)

        self.assertEqual(outcome.selected_role, "commentator")
        interaction._react_free.assert_awaited_once_with(data, "guest", "one two three")


    async def test_line_heard_in_full_before_listeners_decided_is_rethought(self) -> None:
        interaction, consumer, data = self.make()
        data.show_history.append("guest", LINE_A)
        interaction._update_listeners = AsyncMock(return_value=None)
        cue = interaction._new_cue(data, "guest", LINE_A, "s")  # queued, never closed
        consumer.position = (cue, len(LINE_A))

        interaction.human_started_speaking(data)
        interaction.apply_pending_rollback(data)

        self.assertEqual(interaction._rethink, ("guest", LINE_A))

    async def test_rollback_chain_is_bounded(self) -> None:
        interaction, _, data = self.make()
        for i in range(40):
            await interaction.play_line(data, "host", f"line {i}", step="x")
        depth, cue = 0, interaction._last_cue
        while cue.mark.prev is not None:
            depth, cue = depth + 1, cue.mark.prev
        self.assertLessEqual(depth, 16)


class HumanFloorTests(FloorInteractionTestCase):
    async def test_transcript_without_vad_still_takes_the_floor(self) -> None:
        interaction, consumer, data = self.make()
        interaction.human_turn_submitted(data)
        self.assertTrue(interaction.human_floor_pending)
        interaction._react_free = AsyncMock(return_value=FloorOutcome("speak", "guest"))
        interaction.human_turn_taken()
        await interaction.react_to_human(data, "Typed line.")
        self.assertFalse(interaction.human_floor_pending)

    async def test_human_who_keeps_talking_keeps_the_floor_after_the_reaction(self) -> None:
        interaction, _, data = self.make()
        interaction._react_free = AsyncMock(return_value=FloorOutcome("speak", "guest"))
        interaction.human_started_speaking(data)
        interaction.human_stopped_speaking(data)
        interaction.human_turn_submitted(data)  # first sentence transcribed
        interaction.human_started_speaking(data)  # ... and they go on
        interaction.human_turn_taken()

        await interaction.react_to_human(data, "First sentence.")
        self.assertTrue(interaction.human_floor_pending)

        interaction.human_stopped_speaking(data)
        interaction.human_turn_submitted(data)  # second sentence already queued
        interaction.human_turn_taken()
        await interaction.react_to_human(data, "First sentence.")
        self.assertFalse(interaction.human_floor_pending)

    async def test_queued_transcript_keeps_the_floor_after_the_reaction(self) -> None:
        interaction, _, data = self.make()
        interaction._react_free = AsyncMock(return_value=FloorOutcome("speak", "guest"))
        interaction.human_turn_submitted(data)
        interaction.human_turn_submitted(data)
        interaction.human_turn_taken()
        await interaction.react_to_human(data, "One.")
        self.assertTrue(interaction.human_floor_pending)

    async def test_floor_is_released_when_no_transcript_follows(self) -> None:
        interaction, _, data = self.make()
        abandoned: list[object] = []
        with (
            patch.dict(os.environ, {"TALKSHOW_BARGE_IN_RELEASE_S": "0.05"}),
            patch(
                "agent.show_graph.runner.abandon_human_turn",
                lambda d: abandoned.append(interaction.human_turn_abandoned()),
            ),
        ):
            interaction.human_started_speaking(data)
            interaction.human_stopped_speaking(data)
            await asyncio.sleep(0.15)
        self.assertEqual(len(abandoned), 1)
        self.assertFalse(interaction.human_floor_pending)

    async def test_transcript_cancels_the_release(self) -> None:
        interaction, _, data = self.make()
        with (
            patch.dict(os.environ, {"TALKSHOW_BARGE_IN_RELEASE_S": "0.05"}),
            patch("agent.show_graph.runner.abandon_human_turn", Mock()) as abandon,
        ):
            interaction.human_started_speaking(data)
            interaction.human_stopped_speaking(data)
            interaction.human_turn_submitted(data)
            await asyncio.sleep(0.15)
        abandon.assert_not_called()
        self.assertTrue(interaction.human_floor_pending)


class HumanReactionTests(FloorInteractionTestCase):
    async def test_every_agent_thinks_on_the_transcript_then_floor_is_arbitrated(self) -> None:
        interaction, _, data = self.make()
        interaction.human_floor_pending = True
        interaction.discussion_complete = True
        data.show_history.append("human", "What about rural schools?")
        seen: list[tuple[str, str, str]] = []

        async def think(data, role, **kw):
            seen.append((role, kw["speaker"], kw["floor_state"]))
            decision = "speak" if role == "guest" else "silent"
            intent = CommunicativeIntent("answer", "answer", None) if role == "guest" else None
            interaction.actions[role] = Action(
                decision, intent, "Rural access matters." if role == "guest" else None
            )
            return interaction.actions[role]

        interaction._think_and_act = think
        interaction.arbitrate_free = AsyncMock(return_value=FloorOutcome("speak", "guest"))

        outcome = await interaction.react_to_human(data, "What about rural schools?")

        self.assertEqual(outcome.selected_role, "guest")
        self.assertEqual(
            sorted(seen), sorted((role, "human", "free") for role in FLOOR_ROLES)
        )
        self.assertEqual(
            interaction.arbitrate_free.await_args.kwargs["candidates"], ["guest"]
        )
        self.assertFalse(interaction.human_floor_pending)
        self.assertFalse(interaction.discussion_complete)
        self.assertEqual(interaction.consume_proposed_reply("guest"), "Rural access matters.")


class EmotionHookTests(FloorInteractionTestCase):
    async def test_queued_line_keeps_the_emotion_it_was_queued_with(self) -> None:
        interaction, consumer, data = self.make()
        interaction.cue_emotion = lambda data, role: data.role_emotion[role]
        data.role_emotion["guest"] = "Sadness"
        await interaction.play_line(data, "guest", "heard line", step="a")
        data.role_emotion["guest"] = "Anger"  # a later appraisal must not change it

        self.assertEqual(consumer.cues[0].emotion, "Sadness")

    async def test_finished_line_is_appraised_before_its_last_interval(self) -> None:
        interaction, _, data = self.make()
        order: list[str] = []

        async def complete(data, speaker, text):
            order.append(f"complete:{speaker}:{text}")

        interaction.on_line_complete = complete
        await self.speak(
            interaction, data, "guest", LINE_A, on_interval=lambda n: order.append(f"interval:{n}")
        )

        self.assertEqual(
            order, ["interval:1", "interval:2", f"complete:guest:{LINE_A}", "interval:3"]
        )

    async def test_cut_line_is_not_reported_finished(self) -> None:
        interaction, _, data = self.make()
        interaction.on_line_complete = AsyncMock()

        async def listeners(data, **kw):
            interaction._interval_backchannels = []
            return "commentator"  # approved interrupt at the first interval

        interaction._update_listeners = listeners
        data.show_history.append("guest", LINE_A)
        await interaction.disclose_panel_line(data, {"role": "guest", "text": LINE_A})

        interaction.on_line_complete.assert_not_awaited()

    async def test_mood_goes_into_the_think_and_host_prompts(self) -> None:
        interaction, _, data = self.make()
        interaction.mood_block = lambda data, role: f"Your current mood: gloomy ({role})."
        complete = data.runtime.gemma_client.complete_text
        complete.return_value = (
            '{"think": "x", "action": {"decision": "silent", "intent": null, '
            '"content": null, "backchannels": []}}'
        )
        with patch("agent.panel.panel_prompts.panelist_personality_block", return_value="persona"):
            await interaction._think_and_act(
                data, "guest", history=data.show_history, speaker="commentator",
                interval_text="x", visible_text="x", checkpoint=1, floor_state="free",
            )
        self.assertIn("Your current mood: gloomy (guest).", complete.await_args.args[0])

        interaction.host_fallback_instruction = "Invite the guest."
        interaction.host_fallback_strategy = "invite"
        complete.return_value = "Amy, your view?"
        await interaction.host_fallback_line(data, "host")
        self.assertIn("Your current mood: gloomy (host).", complete.await_args.args[0])


class ActuatorTests(FloorInteractionTestCase):
    def make_actuators(self, interaction):
        from agent.show_graph.actuators import LiveActuators

        data = make_data(interaction)
        data.shutdown_event = asyncio.Event()
        data.turn_log = None
        controller = SimpleNamespace(
            listen_role=lambda: "host", panel_speaker_roles=lambda: ["commentator", "guest"]
        )
        return LiveActuators(None, data, controller), data

    def test_pending_human_stops_the_beat_and_its_writes(self) -> None:
        interaction = ThinkActionInteraction(RecordingConsumer())
        act, data = self.make_actuators(interaction)
        self.assertFalse(act.should_stop())
        interaction.human_floor_pending = True
        self.assertTrue(act.should_stop())
        self.assertIsNone(act.record_line({"role": "guest", "text": "late line"}))
        self.assertEqual(data.show_history.lines, [])

    async def test_human_turn_routes_floor_to_selected_agent_without_host_reply(self) -> None:
        interaction = ThinkActionInteraction(RecordingConsumer())
        interaction.react_to_human = AsyncMock(return_value=FloorOutcome("speak", "commentator"))
        act, data = self.make_actuators(interaction)
        act.listeners = lambda speaker: []
        routed: list[str] = []
        with (
            patch("agent.session.human_turn.record_human_line", AsyncMock()),
            patch(
                "agent.floor.floor_control.apply_floor_next",
                lambda data, role: routed.append(role),
            ),
            patch("agent.emotion.appraisal.snapshot_context", return_value=None),
        ):
            reply, utt = await act.commit_human_turn({"heard": "Is that fair?"})

        self.assertEqual(reply, "")
        self.assertIsNone(utt)  # appraised here already; the graph has nothing left to fan out
        self.assertEqual(routed, ["commentator"])
        interaction.react_to_human.assert_awaited_once_with(data, "Is that fair?")

    async def test_human_line_is_recorded_after_the_rollback(self) -> None:
        interaction = ThinkActionInteraction(RecordingConsumer())
        interaction.react_to_human = AsyncMock(return_value=FloorOutcome("speak", "guest"))
        act, data = self.make_actuators(interaction)
        act.listeners = lambda speaker: []
        data.show_history.append("guest", "heard line")
        await interaction.play_line(data, "guest", "heard line", step="a")
        data.show_history.append("host", "never heard")
        await interaction.play_line(data, "host", "never heard", step="b")
        interaction.consumer.position = (interaction.consumer.cues[1], 0)
        interaction.human_started_speaking(data)

        async def record(data, *, heard, **kw):
            data.show_history.append("human", heard)

        with (
            patch("agent.session.human_turn.record_human_line", record),
            patch("agent.floor.floor_control.apply_floor_next", lambda data, role: None),
            patch("agent.emotion.appraisal.snapshot_context", return_value=None),
        ):
            await act.commit_human_turn({"heard": "Is that fair?"})

        self.assertEqual(
            [line.text for line in data.show_history.lines], ["heard line", "Is that fair?"]
        )

    async def test_human_words_are_appraised_before_anyone_reacts(self) -> None:
        interaction = ThinkActionInteraction(RecordingConsumer())
        order: list[str] = []

        async def react(data, heard):
            order.append("react")
            return FloorOutcome("speak", "guest")

        interaction.react_to_human = react
        act, _ = self.make_actuators(interaction)

        async def appraise(utt, role):
            order.append(f"appraise:{role}:{utt.text}")
            return {"role": role}

        act.appraise = appraise
        act.listeners = lambda speaker: ["host", "guest"]
        with (
            patch("agent.session.human_turn.record_human_line", AsyncMock()),
            patch("agent.floor.floor_control.apply_floor_next", lambda data, role: None),
            patch("agent.emotion.appraisal.snapshot_context", return_value=None),
        ):
            await act.commit_human_turn({"heard": "Is that fair?"})

        self.assertEqual(
            order, ["appraise:host:Is that fair?", "appraise:guest:Is that fair?", "react"]
        )

    async def test_human_appraisal_applies_while_the_human_holds_the_floor(self) -> None:
        from agent.emotion.appraisal import Utterance

        interaction = ThinkActionInteraction(RecordingConsumer())
        interaction.human_floor_pending = True
        act, _ = self.make_actuators(interaction)
        applied: list[str] = []
        result = SimpleNamespace(delta=(0.1, 0.0, 0.0), word="moved")
        with (
            patch("agent.emotion.appraisal.run_appraisal", AsyncMock(return_value=result)),
            patch(
                "agent.emotion.appraisal.apply_appraisal",
                lambda data, utt, result: applied.append(utt.speaker),
            ),
        ):
            await act.appraise(Utterance("human", "Is that fair?", 1, None), "guest")
            agent = await act.appraise(Utterance("host", "written past the barge-in", 2, None), "guest")

        self.assertEqual(applied, ["human"])
        self.assertTrue(agent["barge_in"])

    async def test_finished_line_is_appraised_once_inside_its_disclosure(self) -> None:
        from agent.emotion.appraisal import Utterance

        interaction = ThinkActionInteraction(RecordingConsumer())
        act, _ = self.make_actuators(interaction)
        appraised: list[tuple[str, str]] = []

        async def appraise(utt, role):
            appraised.append((role, utt.text))
            return {"role": role}

        async def disclose(data, line, on_start=None):
            await interaction.on_line_complete(data, "guest", "Full line.")

        act.appraise = appraise
        act.listeners = lambda speaker: ["host", "commentator"]
        interaction.disclose_panel_line = disclose
        interaction.latest_line_text = lambda role: "Full line."
        act._deferred_utterance = Utterance(speaker="guest", text="Full line.", seq=1, context=None)
        with patch("agent.emotion.appraisal.snapshot_context", return_value=None):
            await act.speak({"role": "guest", "kind": "panelist"})

        self.assertEqual(appraised, [("host", "Full line."), ("commentator", "Full line.")])

    async def test_free_floor_host_fallback_selects_host(self) -> None:
        interaction = ThinkActionInteraction(RecordingConsumer())
        interaction.select_free_floor_outcome = AsyncMock(
            return_value=FloorOutcome("host_fallback", "host", host_strategy="reframe")
        )
        act, _ = self.make_actuators(interaction)
        pick = await act.select_next_speaker(0, [])
        self.assertEqual((pick.role, pick.reason), ("host", "host_fallback"))


class AVSpeechConsumerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        gap = patch.dict(os.environ, {"TALKSHOW_TURN_GAP_S": "0", "TALKSHOW_TURN_GAP_JITTER_S": "0"})
        gap.start()
        self.addCleanup(gap.stop)
        self.session = SimpleNamespace(on=Mock(), interrupt=Mock())
        self.data = SimpleNamespace(panel_chain_running=True)
        self.spoken: list[str] = []

    def fake_speech(self, seconds: float):
        async def speak_panel_line(session, data, *, speak_role, text, **kw):
            self.spoken.append(text)
            consumer._on_agent_state(SimpleNamespace(new_state="speaking"))
            await asyncio.sleep(seconds)

        consumer = self.consumer
        return patch("agent.session.session_handoff.speak_panel_line", speak_panel_line)

    async def test_cues_play_in_order_without_blocking_the_generator(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        with self.fake_speech(0.01):
            for text in ("first line", "second line"):
                cue = Cue("guest", text, "s")
                await self.consumer.submit(self.data, cue)  # returns at once
                self.consumer.close(cue, len(text))
            self.assertFalse(self.consumer.idle())
            await self.consumer.drain()
        self.assertEqual(self.spoken, ["first line", "second line"])

    async def test_stop_reports_heard_position_and_drops_the_queue(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        with self.fake_speech(10):
            first, second = Cue("guest", "one two three", "s"), Cue("host", "later", "s")
            await self.consumer.submit(self.data, first)
            await self.consumer.submit(self.data, second)
            while not self.spoken:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.01)
            (cue, heard), cut = self.consumer.stop()

        self.assertIs(cue, first)
        self.assertGreater(heard, 0)
        self.assertTrue(cut)
        self.assertTrue(self.consumer.idle())
        self.session.interrupt.assert_called_once_with(force=True)
        self.assertEqual(self.spoken, ["one two three"])

    async def test_playback_uses_the_emotion_fixed_when_queued(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        seen: list[str | None] = []

        async def speak_panel_line(session, data, *, speak_role, text, **kw):
            seen.append(kw.get("emotion"))

        with patch("agent.session.session_handoff.speak_panel_line", speak_panel_line):
            cue = Cue("guest", "one two", "s", emotion="Sadness")
            await self.consumer.submit(self.data, cue)
            self.consumer.close(cue, len("one two"))
            await self.consumer.drain()

        self.assertEqual(seen, ["Sadness"])

    async def test_script_cut_decided_before_playback_plays_only_the_prefix(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        with self.fake_speech(0.01):
            cue = Cue("guest", "one two three four", "s")
            await self.consumer.submit(self.data, cue)
            self.consumer.close(cue, len("one two"))
            await self.consumer.drain()
        self.assertEqual(self.spoken, ["one two"])

    async def test_transcript_is_announced_before_the_audio_and_not_repeated(self) -> None:
        from agent.show.talkshow_data import TalkShowData

        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        self.data.announced_transcripts = set()
        order: list[str] = []

        async def announce(role, text, **kw):
            order.append(f"transcript:{text}:stream={kw.get('stream')}")

        async def speak_panel_line(session, data, *, speak_role, text, **kw):
            order.append(f"audio:{text}")

        with (
            patch("agent.ui.ui_events.emit_transcript", announce),
            patch("agent.session.session_handoff.speak_panel_line", speak_panel_line),
        ):
            cue = Cue("guest", "one two", "speech_guest_0")
            await self.consumer.submit(self.data, cue)
            self.consumer.close(cue, len("one two"))
            await self.consumer.drain()

        self.assertEqual(order, ["transcript:one two:stream=True", "audio:one two"])
        self.assertEqual(self.data.announced_transcripts, {("guest", "speech_guest_0")})
        data = TalkShowData.__new__(TalkShowData)
        data.announced_transcripts = set(self.data.announced_transcripts)
        self.assertTrue(data.take_announced_transcript("guest", "speech_guest_0"))
        self.assertFalse(data.take_announced_transcript("guest", "speech_guest_0"))

    async def test_interrupt_cut_tells_the_ui_who_interrupted_and_what_was_heard(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        emitted = AsyncMock()
        with self.fake_speech(0.01), patch("agent.ui.ui_events.emit_interrupt", emitted):
            cue = Cue("guest", "one two three four", "s")
            await self.consumer.submit(self.data, cue)
            cue.interrupted_by = "commentator"
            self.consumer.close(cue, len("one two"))
            await self.consumer.drain()
            await asyncio.sleep(0)
        emitted.assert_awaited_once_with("guest", "commentator", heard_text="one two")

    async def test_cut_without_interrupter_sends_no_interrupt_event(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        emitted = AsyncMock()
        with self.fake_speech(0.01), patch("agent.ui.ui_events.emit_interrupt", emitted):
            cue = Cue("guest", "one two three four", "s")
            await self.consumer.submit(self.data, cue)
            self.consumer.close(cue, len("one two"))
            await self.consumer.drain()
            await asyncio.sleep(0)
        emitted.assert_not_awaited()

    async def test_backchannel_is_announced_to_the_ui_as_it_plays(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        published: list[float] = []

        async def play_pcm(pcm, rate, *, on_started=None):
            published.append(rate)
            if on_started is not None:
                on_started(time.monotonic())

        publisher = SimpleNamespace(play_pcm=play_pcm)
        emitted = AsyncMock()
        cue = Cue("guest", "one two three", "s")
        cue.playout_started_at = time.monotonic()
        with (
            patch(
                "agent.interaction.consumers._prepare_backchannel",
                AsyncMock(return_value=(publisher, b"\x00\x00" * 24000, 24000)),
            ),
            patch("agent.ui.ui_events.emit_backchannel", emitted),
        ):
            await self.consumer._backchannel(cue, "host", "mm-hmm", 0, 0)
            await asyncio.sleep(0)
        emitted.assert_awaited_once_with(
            "host", "mm-hmm", speaker_role="guest", duration_s=1.0
        )
        self.assertEqual(published, [24000])

    async def test_script_cut_decided_during_playback_stops_there(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=20.0)
        with self.fake_speech(10):
            cue = Cue("guest", "one two three four five six seven eight", "s")
            await self.consumer.submit(self.data, cue)
            while not self.spoken:
                await asyncio.sleep(0.01)
            self.consumer.close(cue, len("one two"))
            await asyncio.wait_for(self.consumer.drain(), timeout=2)
        self.assertEqual(self.consumer._last, (cue, len("one two")))
        self.session.interrupt.assert_called_once_with(force=True)

    async def play_two(self, first_end: int | None = None, second_role: str = "host") -> float:
        """Seconds between the first line ending and the second one starting."""
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        spans: list[list[float]] = []

        async def speak_panel_line(session, data, **kw):
            loop = asyncio.get_running_loop()
            spans.append([loop.time()])
            await asyncio.sleep(0.05)
            spans[-1].append(loop.time())

        with patch("agent.session.session_handoff.speak_panel_line", speak_panel_line):
            first, second = Cue("guest", "one two three", "s"), Cue(second_role, "reply", "s")
            await self.consumer.submit(self.data, first)
            self.consumer.close(first, first_end if first_end is not None else len(first.text))
            await self.consumer.submit(self.data, second)
            self.consumer.close(second, len(second.text))
            await self.consumer.drain()
        return spans[1][0] - spans[0][1]

    async def test_next_speaker_waits_a_gap_with_jitter(self) -> None:
        with patch.dict(
            os.environ, {"TALKSHOW_TURN_GAP_S": "0.2", "TALKSHOW_TURN_GAP_JITTER_S": "0.1"}
        ):
            gap = await self.play_two()
        self.assertGreaterEqual(gap, 0.19)
        self.assertLess(gap, 0.4)

    async def test_interrupter_takes_over_without_a_gap(self) -> None:
        with patch.dict(os.environ, {"TALKSHOW_TURN_GAP_S": "0.5"}):
            gap = await self.play_two(first_end=len("one two"))
        self.assertLess(gap, 0.1)

    async def test_same_speaker_goes_on_without_a_gap(self) -> None:
        with patch.dict(os.environ, {"TALKSHOW_TURN_GAP_S": "0.5"}):
            gap = await self.play_two(second_role="guest")
        self.assertLess(gap, 0.1)

    def test_turn_gap_is_base_plus_jitter(self) -> None:
        from agent.interaction.consumers import turn_gap_s

        with patch.dict(os.environ, {"TALKSHOW_TURN_GAP_S": "1", "TALKSHOW_TURN_GAP_JITTER_S": "0.5"}):
            gaps = [turn_gap_s() for _ in range(200)]
        self.assertTrue(all(1.0 <= g <= 1.5 for g in gaps))
        self.assertGreater(max(gaps) - min(gaps), 0.2)

    async def test_progress_clock_starts_without_a_new_speaking_transition(self) -> None:
        self.session.agent_state = "speaking"
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)

        async def speak_panel_line(session, data, **kw):
            await asyncio.sleep(10)  # no agent_state_changed event

        with patch("agent.session.session_handoff.speak_panel_line", speak_panel_line):
            await self.consumer.submit(self.data, Cue("guest", "one two three", "s"))
            await asyncio.sleep(0.15)  # the playout loop starts the clock on its own
            self.assertEqual(self.consumer.heard_chars(), len("one two three"))
            self.consumer.stop()

    async def test_position_during_cue_start_is_not_the_previous_line(self) -> None:
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=1000.0)
        starting = asyncio.Event()
        release = asyncio.Event()

        async def on_start():
            starting.set()
            await release.wait()

        with self.fake_speech(0.01):
            first = Cue("guest", "a much longer first line here", "s")
            await self.consumer.submit(self.data, first)
            self.consumer.close(first, len(first.text))
            second = Cue("host", "next", "s", on_start=[on_start])
            await self.consumer.submit(self.data, second)
            await starting.wait()
            (cue, heard), _ = self.consumer.stop()
        self.assertIs(cue, second)
        self.assertEqual(heard, 0)

    async def play_backchannels(
        self, cue: Cue, *targets: tuple[str, int], drive=None
    ) -> list[tuple[str, float]]:
        """Run backchannels for ``(text, offset)`` targets; returns (text, start time)."""
        self.consumer = AVSpeechConsumer(self.session, words_per_sec=10.0)
        played: list[tuple[str, float]] = []

        async def prepare(data, role, text):
            async def play_pcm(pcm, rate, *, on_started=None):
                played.append((text, time.monotonic()))
                if on_started is not None:
                    on_started(time.monotonic())

            return SimpleNamespace(play_pcm=play_pcm), b"\x00\x00", 24000

        with (
            patch("agent.interaction.consumers._prepare_backchannel", prepare),
            patch("agent.ui.ui_events.emit_backchannel", AsyncMock()),
        ):
            tasks = [
                asyncio.create_task(self.consumer._backchannel(cue, "host", text, offset))
                for text, offset in targets
            ]
            if drive is not None:
                await drive()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        return played

    async def test_mid_sentence_backchannel_waits_for_projected_playback(self) -> None:
        cue = Cue("guest", "one two three four five six", "s")
        cue.playout_started_at = time.monotonic()
        played = await self.play_backchannels(cue, ("Wow.", len("one two three")))
        self.assertEqual([text for text, _ in played], ["Wow."])
        # Three words at 10 words/s after playback started.
        self.assertAlmostEqual(played[0][1] - cue.playout_started_at, 0.3, delta=0.08)

    async def test_mid_sentence_backchannel_uses_the_sentences_audio_duration(self) -> None:
        cue = Cue("guest", "one two three four five six", "s")
        cue.playout_started_at = time.monotonic()
        # The sentence's audio lasts 1.2s, slower than the configured 10 words/s.
        cue.started_segments[0] = (0.0, cue.playout_started_at, 1.2, None)
        played = await self.play_backchannels(cue, ("Wow.", len("one two three")))
        self.assertEqual([text for text, _ in played], ["Wow."])
        # Three of six words: halfway through the sentence's real audio, not 0.3s.
        self.assertAlmostEqual(played[0][1] - cue.playout_started_at, 0.6, delta=0.08)

    async def test_mid_sentence_backchannel_plays_when_its_aligned_word_ends(self) -> None:
        cue = Cue("guest", "one two three four five six", "s")
        cue.playout_started_at = time.monotonic()
        aligned = asyncio.get_running_loop().create_future()
        # "three" ends 0.9s into the audio: later than both rate and duration share predict.
        aligned.set_result([0.1, 0.2, 0.9, 1.0, 1.1, 1.2])
        cue.started_segments[0] = (0.0, cue.playout_started_at, 1.2, aligned)
        played = await self.play_backchannels(cue, ("Wow.", len("one two three")))
        self.assertEqual([text for text, _ in played], ["Wow."])
        self.assertAlmostEqual(played[0][1] - cue.playout_started_at, 0.9, delta=0.08)

    async def test_backchannel_at_a_sentence_end_plays_on_its_playback_event(self) -> None:
        cue = Cue("guest", "First sentence ends here. Second sentence goes on.", "s")
        cue.playout_started_at = time.monotonic()

        async def first_sentence_played() -> None:
            await asyncio.sleep(0.05)
            cue.played_segments[0] = (0.05, time.monotonic())

        # The sentence played far faster than the configured rate predicts.
        played = await self.play_backchannels(
            cue, ("Right.", len("First sentence ends here.")), drive=first_sentence_played
        )
        self.assertEqual([text for text, _ in played], ["Right."])
        self.assertAlmostEqual(played[0][1] - cue.playout_started_at, 0.05, delta=0.04)

    async def test_backchannel_is_dropped_when_late_cut_or_stopped(self) -> None:
        text = "First sentence ends here. Second sentence goes on."
        late = Cue("guest", text, "s")
        late.playout_started_at = time.monotonic() - 5.0
        late.played_segments[0] = (1.0, time.monotonic() - 4.0)  # playback passed it
        cut = Cue("guest", text, "s", end=len("First"))
        cut.playout_started_at = time.monotonic()
        stopped = Cue("guest", text, "s")
        stopped.playout_done.set()

        for cue in (late, cut, stopped):
            self.assertEqual(await self.play_backchannels(cue, ("Hm.", len("First sentence"))), [])

    async def test_backchannel_after_the_last_word_plays_when_the_line_ends(self) -> None:
        cue = Cue("guest", "Only one sentence here.", "s")
        cue.playout_started_at = time.monotonic() - 10.0  # configured rate says long overdue

        async def line_ends() -> None:
            await asyncio.sleep(0.05)
            cue.playout_ended_at = time.monotonic()
            cue.playout_done.set()

        played = await self.play_backchannels(cue, ("Nice.", len(cue.text)), drive=line_ends)
        self.assertEqual([text for text, _ in played], ["Nice."])


if __name__ == "__main__":
    unittest.main()


class LiveWholeLineLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_whole_lines_reach_the_model_output_log(self) -> None:
        interaction = Mock()
        data = SimpleNamespace(floor_interaction=interaction)
        consumer = AVSpeechConsumer(Mock(), words_per_sec=1000.0)
        consumer._worker = Mock(done=Mock(return_value=False))  # keep playout idle

        await consumer.submit(data, Cue("host", "Welcome, I'm Lessac.", "opening", whole=True))
        await consumer.submit(data, Cue("host", "Amy, go ahead.", "intro_guest_x", whole=True))
        await consumer.submit(data, Cue("guest", "Panel line.", "speech_guest_0"))

        interaction.log_speech_start.assert_called_once_with(data, "host", "Welcome, I'm Lessac.")
        self.assertEqual(interaction.flush_pending_selection.call_count, 2)


class BackchannelGainTests(unittest.TestCase):
    def test_gain_scales_and_clips(self) -> None:
        import numpy as np

        from agent.interaction.consumers import _amplify_pcm

        pcm = np.array([1000, -1000, 30000, -30000], dtype=np.int16).tobytes()
        louder = np.frombuffer(_amplify_pcm(pcm, 1.6), dtype=np.int16)
        self.assertEqual(louder.tolist(), [1600, -1600, 32767, -32768])
        self.assertEqual(_amplify_pcm(pcm, 1.0), pcm)

    def test_gain_comes_from_env(self) -> None:
        from agent.interaction.consumers import DEFAULT_BACKCHANNEL_GAIN, backchannel_gain

        with patch.dict(os.environ, {"TALKSHOW_BACKCHANNEL_GAIN": "2"}):
            self.assertEqual(backchannel_gain(), 2.0)
        with patch.dict(os.environ, {"TALKSHOW_BACKCHANNEL_GAIN": "loud"}):
            self.assertEqual(backchannel_gain(), DEFAULT_BACKCHANNEL_GAIN)


class TurnCapGraceTests(FloorInteractionTestCase):
    async def test_grace_turn_only_after_an_interrupt_and_blocks_new_ones(self) -> None:
        from agent.show_graph.actuators import LiveActuators

        interaction, _, data = self.make()
        actuators = LiveActuators.__new__(LiveActuators)
        actuators.interaction = interaction

        interaction.last_line_interrupted = False
        self.assertFalse(actuators.grant_interrupt_grace_turn())
        self.assertTrue(interaction.interrupts_allowed)

        interaction.interrupted_role = "guest"
        line = {"role": "host", "text": "Who should set that standard"}
        self.assertEqual(interaction.next_role_after_line(data, line), "guest")
        self.assertTrue(actuators.grant_interrupt_grace_turn())
        self.assertFalse(interaction.interrupts_allowed)

        interaction._think_and_act = AsyncMock(return_value=None)
        interaction.arbitrate_occupied = AsyncMock(return_value="commentator")
        interaction._final_interval = False
        result = await interaction._update_listeners(
            data,
            speaker="guest",
            interval_text="An answer.",
            visible_text="An answer.",
            panel_roles=["guest", "commentator"],
            step="speech_guest_12",
            checkpoint=1,
        )
        self.assertIsNone(result)
        interaction.arbitrate_occupied.assert_not_awaited()


class InterruptHandoffTests(FloorInteractionTestCase):
    make_actuators = ActuatorTests.make_actuators

    async def run_speak(self, interrupted: str | None) -> tuple[asyncio.Event, bool]:
        from agent.emotion.appraisal import Utterance

        interaction = ThinkActionInteraction(RecordingConsumer())
        act, data = self.make_actuators(interaction)
        released = asyncio.Event()

        async def disclose(data, line, on_start=None):
            interaction.interrupted_role = interrupted

        async def slow_appraise(utt, role):
            await released.wait()
            return {"role": role}

        interaction.disclose_panel_line = disclose
        interaction.latest_line_text = lambda role: "Who should set that standard"
        act._deferred_utterance = Utterance(speaker="guest", text="full", seq=1, context=None)
        act.appraise = slow_appraise
        act.listeners = lambda speaker: ["host", "commentator"]
        with patch("agent.emotion.appraisal.snapshot_context", return_value=None):
            speak = asyncio.ensure_future(act.speak({"role": "guest", "kind": "panelist"}))
            await asyncio.sleep(0.05)
            returned_early = speak.done()
            released.set()
            await speak
            await asyncio.gather(*act._background)
        return released, returned_early

    async def test_interrupted_line_hands_over_without_waiting_for_appraisals(self) -> None:
        _, returned_early = await self.run_speak("commentator")
        self.assertTrue(returned_early)

    async def test_normal_line_still_waits_for_appraisals(self) -> None:
        _, returned_early = await self.run_speak(None)
        self.assertFalse(returned_early)


class ControllerSelectionTests(FloorInteractionTestCase):
    """Gemma names the panelist ("Ryan") where the prompt asks for the role ID."""

    INTERRUPT = Action(
        "interrupt", CommunicativeIntent("challenge", "push back", "Amy"), "But the policy allowed it."
    )

    async def test_an_interrupt_approved_by_persona_name_is_granted(self) -> None:
        interaction, _consumer, data = self.make()
        interaction.actions = {"commentator": self.INTERRUPT}
        data.runtime.gemma_client.complete_text = AsyncMock(
            return_value='```json\n{"decision": "interrupt", "selected_role": "Ryan", '
            '"reason_code": "timely_intervention", "reason": "timely"}\n```'
        )
        with patch("agent.config.load_persona_name", lambda role: {"commentator": "Ryan"}.get(role, role)):
            selected = await interaction.arbitrate_occupied(
                data, speaker="guest", visible_text="When I think about it,",
                candidates=["commentator"], interval=1, history=None,
            )
        self.assertEqual(selected, "commentator")

    async def test_a_name_that_is_no_candidate_is_still_rejected(self) -> None:
        interaction, _consumer, data = self.make()
        interaction.actions = {"commentator": self.INTERRUPT}
        data.runtime.gemma_client.complete_text = AsyncMock(
            return_value='{"decision": "interrupt", "selected_role": "Amy", "reason": "x"}'
        )
        with patch("agent.config.load_persona_name", lambda role: {"commentator": "Ryan"}.get(role, role)):
            selected = await interaction.arbitrate_occupied(
                data, speaker="guest", visible_text="When I think about it,",
                candidates=["commentator"], interval=1, history=None,
            )
        self.assertIsNone(selected)


class HostInviteTargetTests(FloorInteractionTestCase):
    """The panelist role "guest" is Amy; the human guest is "human"."""

    def tearDown(self) -> None:
        from agent.show.show_history import set_human_name

        set_human_name(None)

    def test_a_target_resolves_by_role_id_or_name(self) -> None:
        from agent.interaction.floor_interaction import _roster, _target_role
        from agent.show.show_history import set_human_name

        set_human_name("Howard")
        roster = _roster(FLOOR_ROLES, "host")
        self.assertIn("AI panelist", roster["guest"])
        self.assertIn("Howard", roster["human"])
        self.assertEqual(_target_role("guest", roster), "guest")
        self.assertEqual(_target_role("Amy", roster), "guest")
        self.assertEqual(_target_role("Howard", roster), "human")
        self.assertEqual(_target_role("human", roster), "human")
        self.assertIsNone(_target_role("Bob", roster))

    async def test_the_controller_can_invite_the_human(self) -> None:
        from agent.show.show_history import set_human_name

        set_human_name("Howard")
        interaction, _consumer, data = self.make()
        data.runtime.gemma_client.complete_text = AsyncMock(
            return_value='{"decision": "host_fallback", "selected_role": "host", '
            '"reason_code": "no_suitable_candidate", "reason": "nobody applied", '
            '"host_fallback": {"strategy": "invite", "target": "human", '
            '"instruction": "Ask Howard for his story."}}'
        )
        outcome = await interaction.arbitrate_free(
            data, candidates=[], history=data.show_history, interval=0
        )
        prompt = data.runtime.gemma_client.complete_text.await_args.args[0]
        self.assertIn("guest = Amy, an AI panelist", prompt)
        self.assertIn("human = the human guest, Howard", prompt)
        self.assertEqual(
            (outcome.decision, outcome.host_strategy, outcome.host_target),
            ("host_fallback", "invite", "human"),
        )

        data.runtime.gemma_client.complete_text = AsyncMock(return_value="Howard, tell us.")
        with patch("agent.config.load_persona_instructions", lambda *a, **k: "host persona"):
            await interaction.host_fallback_line(data, "host")
        line_prompt = data.runtime.gemma_client.complete_text.await_args.args[0]
        self.assertIn("Invite the human guest, Howard, to speak.", line_prompt)
        self.assertIsNone(interaction.pending_invitation_target)  # Amy is not owed the floor
