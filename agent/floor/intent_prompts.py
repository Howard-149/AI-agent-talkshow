"""Listener intent prompts. No model or LiveKit dependencies.

The live reactive session and the offline floor evaluation (eval/floor_eval)
build requests from these functions, so offline results measure the prompts
that actually run in the show.
"""
from __future__ import annotations

from agent.panel.panel_prompts import panelist_personality_block

_PANELIST_RULES = (
    "You are listening to a live talk show. Decide independently whether to request the floor. "
    "The current speaker has NOT finished. You only know the fully played sentences shown below. "
    "Do not invent later claims. Interrupt only for a direct disagreement or factual correction. "
    "Use queue for a useful addition, none for agreement or repetition. "
    "For an interrupt, evidence MUST be an exact short quote from the played sentences. "
    "Your speech must address that quote in first person as yourself, in 2 or 3 short sentences. "
    "Never write another character's dialogue. Output ONLY a JSON object with keys "
    'action ("interrupt", "queue", "none"), kind ("rebuttal", "correction", "addition"), '
    "reason, evidence, speech. No markdown."
)

# Opposing stances for the homework demo only; natural shows pass no stance.
DEMO_STANCES = {
    "commentator": (
        "For this debate your position is that homework should demonstrate the student's "
        "own understanding, and simply reading or submitting AI answers does not show mastery. "
        "Assess the actual played claim against that position. You may still choose queue "
        "or none if it does not directly conflict."
    ),
    "guest": (
        "For this debate you favor learning from worked AI examples. "
        "Acknowledge shallow-understanding risks without instantly abandoning "
        "that benefit. Assess only the actual played claim; queue or none is fine."
    ),
}


def panelist_intent_system(role: str, *, stance: str = "") -> str:
    system = f"{panelist_personality_block(role)}\n\n{_PANELIST_RULES}"
    return f"{system}\n{stance}" if stance else system


def moderator_intent_system(role: str, *, topic: str = "") -> str:
    topic_line = (f"The show's topic is: {topic}" if topic
                  else "Judge the show's topic from the conversation so far.")
    return (
        f"{panelist_personality_block(role)}\n\n"
        "You are the moderator, listening to a panelist who has NOT finished. "
        "You only know the fully played sentences shown below. "
        f"{topic_line} "
        "Cut in ONLY if the speaker has clearly drifted away from that topic or makes a "
        "personal attack on another panelist. Never cut in to take a side, add your own "
        "opinion, or correct a fact; the panelists handle disagreement. If in doubt, choose none. "
        "For an interrupt, evidence MUST be an exact short quote from the played sentences "
        "that shows the drift or attack. Your speech must steer back to the topic politely, "
        "address the speaker by name, and be 1 or 2 short sentences. Output ONLY a JSON object "
        'with keys action ("interrupt" or "none"), kind ("moderation"), reason, evidence, '
        "speech. No markdown."
    )


def intent_prompt(speaker_name: str, heard: str, *, moderator: bool = False) -> str:
    question = (
        "Has the speaker clearly left the topic or attacked someone, so that you must step in now?"
        if moderator else
        "Is there a concrete claim you disagree with enough to interrupt, a new point to queue, "
        "or nothing worth adding? Do not request just to keep the show going."
    )
    return f"Current speaker: {speaker_name}\nFully played sentences:\n{heard}\n\n{question}"
