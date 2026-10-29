"""Forward playback-synchronized LiveKit words, never an unspoken LLM draft.

This sink sits AFTER LiveKit's TranscriptSynchronizer via next_in_chain.
Fish does not provide word timestamps: pacing is the SDK's audio-based estimate,
not a second ASR request or a browser typewriter animation.
"""
from __future__ import annotations

from typing import Any
from livekit.agents.voice.io import TextOutput

from agent.config import load_persona_name


class PlayoutTranscriptOutput(TextOutput):
    def __init__(self, data: Any):
        super().__init__(label="Talkshow live captions", next_in_chain=None)
        self.data = data
        self._epoch: int | None = None
        self._sentence = ""

    async def capture_text(self, text: str) -> None:
        broker = getattr(self.data, "reactive_session", None)
        turn = broker.active if broker is not None else None
        if turn is None or turn.winner is not None or broker.demo_stop:
            return
        if self._epoch != turn.epoch:
            self._epoch, self._sentence = turn.epoch, ""
        self._sentence += str(text)
        caption = " ".join([*turn.heard, self._sentence.strip()]).strip()
        if not caption:
            return
        await broker.emit(
            "transcript_partial", utterance_id=f"reactive-{turn.epoch}",
            role=turn.role, speaker=load_persona_name(turn.role), epoch=turn.epoch,
            text=caption, final=False, step=turn.step, source="playout_sync",
        )

    def flush(self) -> None:
        # Each session.say() is one sentence. Keep the epoch, clear only its
        # sentence buffer; completed sentences live in ActiveTurn.heard.
        self._sentence = ""
