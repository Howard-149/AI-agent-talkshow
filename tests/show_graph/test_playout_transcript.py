import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent.ui.playout_transcript import PlayoutTranscriptOutput


class PlayoutTranscriptTests(unittest.IsolatedAsyncioTestCase):
    def setup_sink(self):
        turn = SimpleNamespace(epoch=1, role="guest", step="opening", heard=[], winner=None)
        broker = SimpleNamespace(active=turn, demo_stop=False, emit=AsyncMock())
        return PlayoutTranscriptOutput(SimpleNamespace(reactive_session=broker)), broker, turn

    async def test_cumulative_words_and_sentence_boundaries(self):
        sink, broker, turn = self.setup_sink()
        await sink.capture_text("Students ")
        self.assertEqual(broker.emit.call_args.kwargs["text"], "Students")
        await sink.capture_text("can learn.")
        self.assertEqual(broker.emit.call_args.kwargs["text"], "Students can learn.")
        sink.flush()
        turn.heard.append("Students can learn.")
        await sink.capture_text("But ")
        self.assertEqual(broker.emit.call_args.kwargs["text"], "Students can learn. But")
        self.assertEqual(broker.emit.call_args.kwargs["utterance_id"], "reactive-1")

    async def test_no_unplayed_tail_after_interrupt_or_stop(self):
        sink, broker, turn = self.setup_sink()
        turn.winner = object()
        await sink.capture_text("Never spoken")
        turn.winner = None
        broker.demo_stop = True
        await sink.capture_text("Never spoken")
        broker.demo_stop = False
        broker.active = None
        await sink.capture_text("Never spoken")
        broker.emit.assert_not_awaited()

    async def test_new_epoch_does_not_reuse_previous_role_text(self):
        sink, broker, turn = self.setup_sink()
        await sink.capture_text("Amy ")
        broker.active = SimpleNamespace(epoch=2, role="commentator", step="reply", heard=[], winner=None)
        await sink.capture_text("Ryan ")
        self.assertEqual(broker.emit.call_args.kwargs["text"], "Ryan")
        self.assertEqual(broker.emit.call_args.kwargs["role"], "commentator")
        self.assertEqual(broker.emit.call_args.kwargs["utterance_id"], "reactive-2")
