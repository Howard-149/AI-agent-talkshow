"""Show interaction: Think & Action text generation decoupled from speech delivery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.interaction.consumers import (
    AVSpeechConsumer,
    ConsoleSpeechConsumer,
    Cue,
    SpeechConsumer,
)
from agent.interaction.floor_interaction import ThinkActionInteraction, human_barge_in_enabled

if TYPE_CHECKING:
    from livekit.agents import AgentSession


def ensure_floor_interaction(data: Any, session: AgentSession | None = None) -> Any:
    """The session's floor interaction; created on first use (console when no session)."""
    interaction = getattr(data, "floor_interaction", None)
    if interaction is None:
        consumer = AVSpeechConsumer(session) if session is not None else ConsoleSpeechConsumer()
        interaction = ThinkActionInteraction(consumer)
        data.floor_interaction = interaction
    return interaction


def human_barge_in(data: Any) -> None:
    """The human started speaking: take the floor from whoever holds it."""
    interaction = getattr(data, "floor_interaction", None)
    if isinstance(interaction, ThinkActionInteraction) and human_barge_in_enabled():
        interaction.human_started_speaking(data)


def human_speech_ended(data: Any) -> None:
    """VAD end of the human's speech (arms the no-transcript release)."""
    interaction = getattr(data, "floor_interaction", None)
    if isinstance(interaction, ThinkActionInteraction) and human_barge_in_enabled():
        interaction.human_stopped_speaking(data)


def human_transcript_arrived(data: Any) -> None:
    """A human transcript is handed to the show: it takes the floor even if VAD missed it."""
    interaction = getattr(data, "floor_interaction", None)
    if isinstance(interaction, ThinkActionInteraction) and human_barge_in_enabled():
        interaction.human_turn_submitted(data)


def apply_pending_rollback(data: Any) -> None:
    """Between show beats: drop script generated past a human barge-in."""
    interaction = getattr(data, "floor_interaction", None)
    if isinstance(interaction, ThinkActionInteraction):
        interaction.apply_pending_rollback(data)


def playout_idle(data: Any) -> bool:
    """True when no generated script is waiting to be (or being) delivered."""
    interaction = getattr(data, "floor_interaction", None)
    consumer = getattr(interaction, "consumer", None)
    return consumer is None or consumer.idle()


def human_may_speak(data: Any) -> bool:
    """With Think & Action floor control the human never waits for a floor grant."""
    return isinstance(getattr(data, "floor_interaction", None), ThinkActionInteraction)


__all__ = [
    "AVSpeechConsumer",
    "ConsoleSpeechConsumer",
    "Cue",
    "SpeechConsumer",
    "ThinkActionInteraction",
    "apply_pending_rollback",
    "ensure_floor_interaction",
    "human_barge_in",
    "human_barge_in_enabled",
    "human_may_speak",
    "human_speech_ended",
    "human_transcript_arrived",
    "playout_idle",
]
