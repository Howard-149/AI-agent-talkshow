import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace

from agent.session.text_only_interaction import (
    SpeakingIntent,
    TextOnlyInteraction,
    parse_speaking_intent,
    token_chunk_ends,
)



def _without_timestamps(text: str) -> str:
    """Drop the per-entry ``time:`` lines so block layout can be compared exactly."""
    import re

    return re.sub(r'^time: "[^"]*"\n', "", text, flags=re.MULTILINE)

class TextOnlyInteractionTests(unittest.TestCase):
    def test_tokenizer_loading_allows_hub_resolution_and_is_cached(self) -> None:
        loader = Mock(return_value=object())
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=loader)
        )
        with patch.dict("sys.modules", {"transformers": transformers}):
            interaction = TextOnlyInteraction()
            tokenizer = interaction._get_tokenizer()
            self.assertIs(interaction._get_tokenizer(), tokenizer)

        loader.assert_called_once()
        self.assertFalse(loader.call_args.kwargs.get("local_files_only", False))

    def test_token_chunk_ends_use_fixed_token_intervals(self) -> None:
        text = "abcdefghij"
        offsets = [(index, index + 1) for index in range(len(text))]

        self.assertEqual(token_chunk_ends(text, offsets, 3), [3, 6, 9, 10])

    def test_openai_text_chunking_does_not_load_huggingface_tokenizer(self) -> None:
        interaction = TextOnlyInteraction()
        interaction.token_interval = 2
        with patch.dict("os.environ", {"TALKSHOW_TEXT_TOKENIZER": "word"}):
            with patch.object(interaction, "_get_tokenizer") as load_tokenizer:
                ends = interaction._chunk_ends("Hello, world!")

        self.assertEqual(ends, [6, 13])
        load_tokenizer.assert_not_called()

    def test_parse_speaking_intent_clamps_scores(self) -> None:
        intent = parse_speaking_intent(
            "[want_to_speak]: true\n[time_cost_s]: 12\n[urgency]: 8"
        )

        self.assertEqual(intent, SpeakingIntent(True, 9, 8))

    def test_private_reply_is_retained_only_when_interested(self) -> None:
        raw = (
            "[want_to_speak]: true\n[time_cost_s]: 2\n[urgency]: 5\n"
            "[proposed_reply]: We should examine the incentives.\n"
            "A concrete example would help."
        )
        self.assertEqual(
            parse_speaking_intent(raw).proposed_reply,
            "We should examine the incentives.\nA concrete example would help.",
        )
        self.assertEqual(
            parse_speaking_intent(raw.replace(": true", ": false")).proposed_reply,
            "",
        )

    def test_public_block_starts_with_speech_not_full_utterance_draft(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from agent.telemetry.model_output_logger import ModelOutputLogger

        with TemporaryDirectory() as directory:
            output = ModelOutputLogger(Path(directory))
            data = SimpleNamespace(
                runtime=SimpleNamespace(gemma_client=SimpleNamespace(log_text_only_event=output.event))
            )
            TextOnlyInteraction.log_speech_start(
                data,
                "commentator",
                "Actually disclosed wording.",
            )
            text = _without_timestamps(output.path.read_text(encoding="utf-8"))

        self.assertTrue(text.startswith('[speech]\nspeaker: "Ryan"'))
        self.assertNotIn("[draft]", text)
        self.assertNotIn("[speech_end]", text)
        self.assertIn('role: "commentator"', text)
        self.assertIn('text: "Actually disclosed wording."', text)
        self.assertNotIn("step:", text)

    def test_event_headers_and_selection_block_spacing(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from agent.telemetry.model_output_logger import ModelOutputLogger

        with TemporaryDirectory() as directory:
            output = ModelOutputLogger(Path(directory))
            output.event("floor", "selection", selected_speaker="Amy", selected_role="guest")
            output.event(
                "interval",
                "interval 1",
                content="Hello.",
                complete_content="Hello there.",
            )
            output.event("guest", "draft - interval 1", proposed_reply="A reply.")
            output.event("topic", "topic", topic="Should AI be used in education?")
            output.event("guest", "speech", text="Hello.")
            raw = output.path.read_text(encoding="utf-8")
            text = _without_timestamps(raw)

        self.assertEqual(raw.count("\ntime: "), 5)
        self.assertIn("[selection]\nselected_speaker: \"Amy\"", text)
        self.assertNotIn('speaker: "Floor"', text)
        self.assertIn('selected_role: "guest"\n\n\n[interval 1]', text)
        interval_block = text.split("[interval 1]\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn('content: "Hello."', interval_block)
        self.assertIn('complete_content: "Hello there."', interval_block)
        self.assertIn(
            'complete_content: "Hello there."\n\n[draft - interval 1]',
            text,
        )
        self.assertIn('speaker: "Amy"\nrole: "guest"', text)
        draft_block = text.split("[draft - interval 1]\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn('speaker: "Amy"', draft_block)
        self.assertIn('proposed_reply: "A reply."', draft_block)
        self.assertNotIn("content:", draft_block)
        self.assertNotIn("complete_content:", draft_block)
        self.assertIn('topic: "Should AI be used in education?"', text)
        self.assertNotIn("[topic]", text)

    def test_interval_and_threshold_are_configurable(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TALKSHOW_TEXT_TOKEN_INTERVAL": "16",
                "TALKSHOW_TEXT_INTERRUPTION_THRESHOLD": "8",
            },
        ):
            interaction = TextOnlyInteraction()

        self.assertEqual(interaction.token_interval, 16)
        self.assertEqual(interaction.interruption_threshold, 8)

    def test_free_floor_uses_readiness_and_interruption_uses_urgency(self) -> None:
        interaction = TextOnlyInteraction()
        interaction.intents = {
            "guest": SpeakingIntent(True, 5, 9),
            "commentator": SpeakingIntent(True, 2, 7),
        }

        self.assertEqual(
            interaction.select_free_floor(["guest", "commentator"]),
            "commentator",
        )
        self.assertEqual(
            interaction.select_interruption("host", ["guest", "commentator"]),
            "guest",
        )
        interaction.intents["guest"] = SpeakingIntent(True, 1, 7)
        self.assertIsNone(
            interaction.select_interruption("host", ["guest", "commentator"])
        )

    def test_text_only_free_floor_can_select_host_by_readiness(self) -> None:
        interaction = TextOnlyInteraction()
        interaction.intents = {
            "commentator": SpeakingIntent(True, 2, 6),
            "host": SpeakingIntent(True, 2, 7),
        }

        self.assertEqual(interaction.select_free_floor(["host", "commentator"]), "host")
        interaction.intents["guest"] = SpeakingIntent(True, 1, 6)
        self.assertIsNone(
            interaction.select_interruption("host", ["guest", "commentator"])
        )

    def test_interrupted_draft_is_truncated_to_disclosed_text(self) -> None:
        line = SimpleNamespace(role_id="guest", text="public prefix plus private tail")
        data = SimpleNamespace(show_history=SimpleNamespace(lines=[line]))

        TextOnlyInteraction._truncate_public_line(data, "guest", "public prefix")

        self.assertEqual(data.show_history.lines[-1].text, "public prefix")

    def test_candidate_is_consumed_once(self) -> None:
        interaction = TextOnlyInteraction()
        interaction.intents["guest"] = SpeakingIntent(True, 2, 3, "Prepared reply.")
        self.assertEqual(interaction.consume_proposed_reply("guest"), "Prepared reply.")
        self.assertEqual(interaction.consume_proposed_reply("guest"), "")
        interaction.intents["guest"] = SpeakingIntent(False, 9, 0, "Do not say this.")
        self.assertEqual(interaction.consume_proposed_reply("guest"), "")


class ThinkCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_disclosure_block_orders_speech_two_drafts_per_interval_then_selection(self) -> None:
        events = []

        class FakeTurnController:
            def __init__(self, scenario, data):
                pass

            def panel_speaker_roles(self):
                return ["guest", "commentator"]

        async def switch_to_role(*args, **kwargs):
            return None

        async def update_listeners(data, **kwargs):
            events.append(
                (
                    "interval",
                    f"interval {kwargs['checkpoint']}",
                    kwargs["interval_text"],
                    kwargs["visible_text"],
                )
            )
            for role in ("guest", "host"):
                events.append(
                    (
                        role,
                        f"draft - interval {kwargs['checkpoint']}",
                        None,
                        None,
                    )
                )

        interaction = TextOnlyInteraction()
        interaction._chunk_ends = Mock(return_value=[5, 10])
        interaction._update_listeners = update_listeners
        logger = SimpleNamespace(
            log_text_only_event=lambda role, kind, **fields: events.append(
                (role, kind, fields.get("text"), None)
            )
        )
        data = SimpleNamespace(
            active_role="commentator",
            scenario=SimpleNamespace(
                turn_control=SimpleNamespace(listen_role="host")
            ),
            show_history=SimpleNamespace(lines=[]),
            runtime=SimpleNamespace(gemma_client=SimpleNamespace(log_text_only_event=logger.log_text_only_event)),
        )
        handoff = SimpleNamespace(switch_to_role=switch_to_role)
        turn_controller = SimpleNamespace(TurnController=FakeTurnController)
        with patch.dict(
            "sys.modules",
            {
                "agent.session.session_handoff": handoff,
                "agent.floor.turn_controller": turn_controller,
            },
        ):
            await interaction.disclose_panel_line(
                data,
                {
                    "role": "commentator",
                    "text": "abcdefghij",
                    "step": "speech_commentator_0",
                    "next_tag": "host",
                },
            )
        interaction.log_selection(
            data,
            selected_role="guest",
            reason="time_cost_s",
            floor_state="free",
            candidates=["guest"],
        )
        self.assertEqual(
            [event[1] for event in events],
            [
                "speech",
                "interval 1",
                "draft - interval 1",
                "draft - interval 1",
                "interval 2",
                "draft - interval 2",
                "draft - interval 2",
                "selection",
            ],
        )
        self.assertEqual(events[0][2], "abcdefghij")
        self.assertEqual(events[1][2:], ("abcde", "abcde"))
        self.assertEqual(events[2][0], "guest")
        self.assertEqual(events[3][0], "host")
        self.assertEqual(events[4][2:], ("fghij", "abcdefghij"))
        self.assertEqual(events[-1][0], "floor")

    async def test_over_threshold_selects_after_interval_without_emitting_next_interval(self) -> None:
        events = []

        class FakeTurnController:
            def __init__(self, scenario, data):
                pass

            def panel_speaker_roles(self):
                return ["guest", "commentator"]

        async def switch_to_role(*args, **kwargs):
            return None

        async def update_listeners(data, **kwargs):
            checkpoint = kwargs["checkpoint"]
            data.runtime.gemma_client.log_text_only_event(
                "interval", f"interval {checkpoint}",
                content=kwargs["interval_text"],
                complete_content=kwargs["visible_text"],
            )
            for listener in ("guest", "host"):
                data.runtime.gemma_client.log_text_only_event(
                    listener, f"draft - interval {checkpoint}",
                    proposed_reply="candidate",
                )
            if checkpoint == 1:
                interaction.intents["guest"] = SpeakingIntent(
                    True, 2, interaction.interruption_threshold + 1, "candidate"
                )

        def log_event(role, kind, **fields):
            events.append((role, kind, fields))

        interaction = TextOnlyInteraction()
        interaction._chunk_ends = Mock(return_value=[5, 10])
        interaction._update_listeners = update_listeners
        data = SimpleNamespace(
            active_role="commentator",
            scenario=SimpleNamespace(
                turn_control=SimpleNamespace(listen_role="host")
            ),
            show_history=SimpleNamespace(
                lines=[SimpleNamespace(role_id="commentator", text="abcdefghij")]
            ),
            runtime=SimpleNamespace(
                gemma_client=SimpleNamespace(log_text_only_event=log_event)
            ),
        )
        with patch.dict(
            "sys.modules",
            {
                "agent.session.session_handoff": SimpleNamespace(
                    switch_to_role=switch_to_role
                ),
                "agent.floor.turn_controller": SimpleNamespace(
                    TurnController=FakeTurnController
                ),
            },
        ):
            await interaction.disclose_panel_line(
                data,
                {
                    "role": "commentator",
                    "text": "abcdefghij",
                    "step": "speech_commentator_0",
                },
            )

        self.assertEqual(
            [kind for _, kind, _ in events],
            ["speech", "interval 1", "draft - interval 1", "draft - interval 1", "selection"],
        )
        self.assertEqual(events[-1][2]["selected_role"], "guest")
        self.assertEqual(data.show_history.lines[-1].text, "abcde")

    async def test_over_threshold_on_final_interval_does_not_mark_floor_occupied(self) -> None:
        events = []

        class FakeTurnController:
            def __init__(self, scenario, data):
                pass

            def panel_speaker_roles(self):
                return ["guest", "commentator"]

        async def switch_to_role(*args, **kwargs):
            return None

        async def update_listeners(data, **kwargs):
            checkpoint = kwargs["checkpoint"]
            events.append(
                (
                    "interval",
                    f"interval {checkpoint}",
                    {
                        "content": kwargs["interval_text"],
                        "complete_content": kwargs["visible_text"],
                    },
                )
            )
            for listener in ("guest", "host"):
                events.append((listener, f"draft - interval {checkpoint}"))
            if checkpoint == 2:
                interaction.intents["guest"] = SpeakingIntent(
                    True, 1, interaction.interruption_threshold + 1, "candidate"
                )

        def log_event(role, kind, **fields):
            events.append((role, kind, fields))

        interaction = TextOnlyInteraction()
        interaction._chunk_ends = Mock(return_value=[5, 10])
        interaction._update_listeners = update_listeners
        data = SimpleNamespace(
            active_role="commentator",
            scenario=SimpleNamespace(
                turn_control=SimpleNamespace(listen_role="host")
            ),
            show_history=SimpleNamespace(
                lines=[SimpleNamespace(role_id="commentator", text="abcdefghij")]
            ),
            runtime=SimpleNamespace(
                gemma_client=SimpleNamespace(log_text_only_event=log_event)
            ),
        )
        with patch.dict(
            "sys.modules",
            {
                "agent.session.session_handoff": SimpleNamespace(
                    switch_to_role=switch_to_role
                ),
                "agent.floor.turn_controller": SimpleNamespace(
                    TurnController=FakeTurnController
                ),
            },
        ):
            await interaction.disclose_panel_line(
                data,
                {
                    "role": "commentator",
                    "text": "abcdefghij",
                    "step": "speech_commentator_0",
                },
            )

        self.assertIsNone(interaction.interrupted_role)
        self.assertEqual(data.show_history.lines[-1].text, "abcdefghij")
        self.assertEqual(
            [event[1] for event in events if len(event) == 3 and isinstance(event[2], dict)],
            ["speech", "interval 1", "interval 2"],
        )

    async def test_gemma_text_response_uses_current_logger_api(self) -> None:
        from agent.adapters.gemma_mm_client import GemmaMMClient

        response = SimpleNamespace(
            status_code=200,
            raise_for_status=Mock(),
            json=Mock(
                return_value={
                    "choices": [
                        {
                            "message": {
                                "content": "[reply]: Ryan, your turn.\n[next]: commentator"
                            }
                        }
                    ]
                }
            ),
        )
        config = SimpleNamespace(
            base_url="http://localhost/v1",
            api_key="test",
            model="test-model",
            max_tokens=128,
        )
        client = GemmaMMClient(config, system_prompt="test")
        output_logger = Mock()
        client.set_output_logger(output_logger)
        client._http = SimpleNamespace(post=AsyncMock(return_value=response))

        with patch.dict(
            "os.environ",
            {
                "OPENAI_USE_MAX_COMPLETION_TOKENS": "0",
                "OPENAI_USE_DEFAULT_SAMPLING": "0",
            },
        ):
            raw = await client.complete_text("prompt", raw=True, output_role="host")

        self.assertIn("[reply]: Ryan, your turn.", raw)
        payload = client._http.post.await_args.kwargs["json"]
        self.assertIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertEqual(payload["temperature"], 0.45)
        output_logger.log.assert_not_called()

        client._http.post.reset_mock()
        with patch.dict(
            "os.environ",
            {
                "OPENAI_USE_MAX_COMPLETION_TOKENS": "1",
                "OPENAI_USE_DEFAULT_SAMPLING": "1",
            },
        ):
            await client.complete_text("prompt", raw=True, output_role="host")
        payload = client._http.post.await_args.kwargs["json"]
        self.assertIn("max_completion_tokens", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)

    async def test_guest_checkpoint_updates_host_and_commentator(self) -> None:
        from agent.show.show_history import ShowHistory

        history = ShowHistory()
        history.append("guest", "public prefix hidden tail")
        data = SimpleNamespace(
            show_history=history,
            scenario=SimpleNamespace(
                turn_control=SimpleNamespace(listen_role="host")
            ),
            runtime=SimpleNamespace(
                gemma_client=SimpleNamespace(log_text_only_event=Mock())
            ),
        )
        interaction = TextOnlyInteraction()
        interaction._update_intent = AsyncMock()
        await interaction._update_listeners(
            data,
            speaker="guest",
            interval_text="public prefix",
            visible_text="public prefix",
            panel_roles=["guest", "commentator"],
            step="speech_guest_0",
            checkpoint=1,
        )
        calls = interaction._update_intent.call_args_list
        data.runtime.gemma_client.log_text_only_event.assert_called_once_with(
            "interval",
            "interval 1",
            content="public prefix",
            complete_content="public prefix",
        )
        self.assertEqual({call.args[1] for call in calls}, {"host", "commentator"})
        for call in calls:
            self.assertEqual(call.kwargs["history"].lines[-1].text, "public prefix")
            self.assertEqual(call.kwargs["checkpoint"], 1)
        self.assertEqual(history.lines[-1].text, "public prefix hidden tail")

    async def test_listener_draft_is_logged_with_observed_interval(self) -> None:
        from agent.show.show_history import HistoryLine, ShowHistory

        raw = (
            "[want_to_speak]: true\n[time_cost_s]: 2\n[urgency]: 5\n"
            "[proposed_reply]: Let us connect this to a concrete example."
        )
        event = Mock()

        async def complete_text(prompt, **kwargs):
            self.assertIn("[proposed_reply]", prompt)
            self.assertIn("You are Lessac (host).", prompt)
            self.assertIn("The current floor holder is Amy (guest).", prompt)
            self.assertIn("Apply these four turn-taking rules:", prompt)
            self.assertIn("A name alone is not an invitation.", prompt)
            self.assertIn("time_cost_s=9, urgency=0", prompt)
            self.assertIn("[time_cost_s]: 0-9 seconds", prompt)
            self.assertIn(
                "Since your last public speaking turn, 1 other completed public speaking turns have occurred.",
                prompt,
            )
            self.assertIn("NOT elapsed or expected model-generation latency", prompt)
            self.assertIn("If 2-3 turns have passed", prompt)
            self.assertIn("If 4 or more turns have passed", prompt)
            self.assertIn("set time_cost_s to 0-1 in the usual case", prompt)
            self.assertIn("Do not keep assigning 3 or more solely", prompt)
            self.assertIn("discussion has materially shifted", prompt)
            self.assertIn("baseline preparation from its complexity", prompt)
            self.assertEqual(kwargs["max_tokens"], 384)
            self.assertNotIn("output_role", kwargs)
            return raw

        data = SimpleNamespace(
            runtime=SimpleNamespace(
                gemma_client=SimpleNamespace(
                    complete_text=complete_text,
                    log_text_only_event=event,
                )
            )
        )
        interaction = TextOnlyInteraction()
        history = ShowHistory(
            lines=[
                HistoryLine("host", "Lessac", "Earlier host turn."),
                HistoryLine("commentator", "Ryan", "Earlier panelist turn."),
                HistoryLine("guest", "Amy", "Current visible prefix."),
            ]
        )
        await interaction._update_intent(
            data,
            "host",
            history=history,
            speaker="guest",
            interval_text=" new public fragment ",
            visible_text=" earlier fragment new public fragment ",
            step="speech_guest_0",
            checkpoint=2,
        )
        event.assert_called_once_with(
            "host",
            "draft - interval 2",
            think_id=1,
            want_to_speak=True,
            urgency=5,
            time_cost_s=2,
            proposed_reply="Let us connect this to a concrete example.",
        )
        self.assertEqual(interaction.intents["host"].proposed_reply, "Let us connect this to a concrete example.")

    def test_selection_records_actual_choice_and_pending_think_outcomes(self) -> None:
        event = Mock()
        data = SimpleNamespace(
            scenario=SimpleNamespace(
                turn_control=SimpleNamespace(listen_role="host")
            ),
            runtime=SimpleNamespace(gemma_client=SimpleNamespace(log_text_only_event=event)),
        )
        interaction = TextOnlyInteraction()
        interaction.intents = {
            "guest": SpeakingIntent(True, 3, 2, "Amy's candidate", think_id=11),
            "commentator": SpeakingIntent(True, 1, 4, "Ryan's candidate", think_id=12),
            "host": SpeakingIntent(True, 1, 5, "Host candidate", think_id=13),
        }
        interaction._pending_thinks = list(interaction.intents.items())

        interaction.log_selection(
            data,
            selected_role="commentator",
            reason="time_cost_s",
            floor_state="free",
            candidates=["guest", "commentator"],
        )

        event.assert_called_once()
        call = event.call_args
        self.assertEqual(call.args, ("floor", "selection"))
        self.assertEqual(call.kwargs["selected_speaker"], "Ryan")
        self.assertEqual(call.kwargs["selected_role"], "commentator")
        self.assertEqual(
            call.kwargs["eligible_candidates"],
            [
                {"speaker": "Amy", "role": "guest"},
                {"speaker": "Ryan", "role": "commentator"},
            ],
        )
        decisions = {item["role"]: item for item in call.kwargs["decisions"]}
        self.assertEqual(decisions["guest"]["speaker"], "Amy")
        self.assertEqual(decisions["commentator"]["speaker"], "Ryan")
        self.assertEqual(decisions["guest"]["outcome"], "not_selected")
        self.assertEqual(decisions["commentator"]["outcome"], "selected")
        self.assertFalse(decisions["host"]["eligible"])
        self.assertEqual(decisions["host"]["outcome"], "not_eligible")

    def test_close_selection_does_not_resolve_control_role_as_persona(self) -> None:
        event = Mock()
        data = SimpleNamespace(
            runtime=SimpleNamespace(gemma_client=SimpleNamespace(log_text_only_event=event))
        )
        interaction = TextOnlyInteraction()

        interaction.log_selection(
            data,
            selected_role="close",
            reason="no_panelists_left",
            floor_state="free",
            candidates=[],
        )

        event.assert_called_once_with(
            "floor",
            "selection",
            floor_state="free",
            reason="no_panelists_left",
            selected_speaker="close",
            selected_role="close",
            eligible_candidates=[],
            decisions=[],
        )

    def test_host_direct_call_repeats_spoken_roles_only_in_text_only_mode(self) -> None:
        from agent.floor.host_direct_call import _host_call_candidates

        panel_roles = ["guest", "commentator"]
        spoken_roles = set(panel_roles)

        self.assertEqual(
            _host_call_candidates(panel_roles, spoken_roles, allow_repeat=True),
            panel_roles,
        )
        self.assertEqual(
            _host_call_candidates(panel_roles, spoken_roles, allow_repeat=False),
            [],
        )

    async def test_direct_trigger_is_structured(self) -> None:
        event = Mock()
        data = SimpleNamespace(
            scenario=SimpleNamespace(
                turn_control=SimpleNamespace(listen_role="host")
            ),
            runtime=SimpleNamespace(gemma_client=SimpleNamespace(log_text_only_event=event)),
        )
        interaction = TextOnlyInteraction()
        interaction.log_trigger(data, "no_intentions_2")
        interaction.log_selection(
            data,
            selected_role="host",
            reason="host_direct_call",
            floor_state="free",
            candidates=["host"],
            include_pending=False,
            consume_trigger=False,
        )
        interaction.log_selection(
            data,
            selected_role="guest",
            reason="host_direct_no_raises",
            floor_state="free",
            candidates=[],
            defer=True,
        )

        event.assert_called_once()
        self.assertEqual(event.call_args.args[1], "selection")
        TextOnlyInteraction.log_speech_start(data, "host", "Opening question")
        interaction.flush_pending_selection(data)

        self.assertEqual(
            [call.args[1] for call in event.call_args_list],
            ["selection", "speech", "selection"],
        )
        self.assertEqual(event.call_args_list[0].kwargs["selected_role"], "host")
        self.assertEqual(event.call_args_list[0].kwargs.get("trigger"), None)
        self.assertEqual(event.call_args_list[2], unittest.mock.call(
            "floor",
            "selection",
            floor_state="free",
            reason="host_direct_no_raises",
            selected_speaker="Amy",
            selected_role="guest",
            eligible_candidates=[],
            trigger="no_intentions_2",
            decisions=[],
        ))


if __name__ == "__main__":
    unittest.main()