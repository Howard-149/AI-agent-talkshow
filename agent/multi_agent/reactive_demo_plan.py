"""Bounded demo pacing, not scripted dialogue or forced interruption decisions.

Only the speaking order is fixed. Every reply uses the shared *played* history.
The opening may be interrupted, queued or left alone by Ryan's real intent poll.
After that exchange, protect the replies from further AI cuts so the audience can
hear the argument develop. Human priority still applies throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from agent.config import load_persona_name
from agent.panel.panel_prompts import panelist_personality_block


@dataclass(frozen=True)
class DemoBeat:
    role: str
    step: str
    instruction: str
    kind: str = "panelist"


RYAN_FIRST_REPLY = DemoBeat(
    "commentator", "demo_ryan_reply",
    "Answer Amy's actual played argument. Your position is that homework must "
    "show the student's own understanding, not just recognition of an AI answer. "
    "Offer one concrete challenge, without pretending you interrupted if you did not.",
)

DEMO_FOLLOWUPS = (
    DemoBeat(
        "guest", "demo_amy_response",
        "Respond directly to Ryan's latest objection, not to your unspoken opening. "
        "You favor learning from worked AI examples. Defend one practical benefit "
        "with a concrete example, while acknowledging his concern about shallow "
        "understanding. Do not immediately abandon your position or merely agree.",
    ),
    DemoBeat(
        "host", "demo_host_followup",
        "As Lessac, the moderator, briefly identify the actual disagreement "
        "between Amy and Ryan without picking a winner. Then ask Ryan by name "
        "for one practical homework rule that would distinguish learning from "
        "copying. This is a middle-of-discussion follow-up, NOT the ending: do "
        "not say 'floor is yours', 'back to you' or ask the human a question.",
        "host_reply",
    ),
    DemoBeat(
        "commentator", "demo_ryan_proposal",
        "Answer Lessac's actual question with one specific, workable homework "
        "rule. Address Amy's benefit as well as your concern about understanding. "
        "Do not repeat your earlier analogy or restart the debate.",
    ),
    DemoBeat(
        "guest", "demo_amy_final",
        "Respond to Ryan's proposed rule. Say concretely what you could accept "
        "and one trade-off or refinement that protects the benefit of AI examples. "
        "This is your final contribution; do not restart your opening argument.",
    ),
    DemoBeat(
        "host", "demo_host_wrap",
        "Close this short discussion as Lessac. Summarize one real point of "
        "agreement and one remaining trade-off from what Amy and Ryan actually "
        "said. Do not invent agreement. End with a natural short thank-you to "
        "both panelists. Do not call on another speaker or say 'floor is yours'.",
        "close",
    ),
)


async def generate_demo_line(data, beat: DemoBeat) -> dict:
    """One fresh persona reply; no canned dialogue and no unplayed drafts."""
    name = load_persona_name(beat.role)
    system = (
        f"{panelist_personality_block(beat.role)}\n\n"
        f"You speak only as {name} in a live English talk show. "
        "The topic is: Should students use AI for homework? "
        "The transcript contains only fully played sentences; never invent an "
        "unfinished sentence, unseen claim, or another character's next answer. "
        "Use conversational English, contractions and a specific response to "
        "what was just said. No markdown, lecture, stage directions, speaker labels or "
        "control tags. Write only your spoken words: two short sentences, "
        "about 30 to 45 words total.\n\n"
        + beat.instruction
    )
    t0 = time.monotonic()
    text = await data.runtime.gemma_client.complete_text(
        "Your turn now. Continue the actual discussion using your role brief.",
        system_prompt=system,
        history_messages=data.show_history.prior_messages(perspective=beat.role),
        temperature=0.55, max_tokens=256,
    )
    text = text.strip()
    if not text:
        raise ValueError("empty demo continuation")
    if data.turn_log is not None:
        data.turn_log.log(
            "reactive_demo_model_done", room=data.room_name, role=beat.role,
            step=beat.step, reply=text, history_version=len(data.show_history.lines),
            model_latency_s=round(time.monotonic() - t0, 3),
        )
    return {"role": beat.role, "text": text, "step": beat.step,
            "kind": beat.kind, "procedural": False,
            "reason": "demo_discussion_complete" if beat.kind == "close" else "demo_followup"}
