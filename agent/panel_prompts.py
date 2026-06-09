from __future__ import annotations

from agent.config import load_persona_name
from agent.show_history import HUMAN_LABEL
from agent.show_context import (
    host_audio_user_hint,
    host_close_card,
    panel_host_gemma_addon,
    panelist_card,
)

PANEL_HOST_AUDIO_USER_HINT = host_audio_user_hint()

PANEL_OPENING_LINE = (
    "Welcome — I'm Lessac, your host. When you have a topic, take the floor; "
    "I'll tee it up for Ryan and Amy, then back to you."
)


def panelist_role_brief(role: str) -> str:
    name = load_persona_name(role)
    if role == "commentator":
        amy = load_persona_name("guest")
        return (
            f"You are {name} (commentator). Your turn now — speak 2–4 sentences out loud. "
            f'Satisfy what "{HUMAN_LABEL}" asked in the transcript — not {amy}, who has '
            f"not spoken yet unless you see a line labeled \"{amy}:\"."
        )
    if role == "guest":
        return (
            f"You are {name} (guest). Your turn now — speak 2–4 sentences out loud. "
            "Add a second concrete contribution; do not repeat the commentator."
        )
    return f"You are {name}. Speak in character."


def _panel_identity_guard(role: str, *, other_has_spoken: bool) -> str:
    """Disambiguate Human guest (real person) vs Amy (AI guest panelist)."""
    amy = load_persona_name("guest")
    host = load_persona_name("host")
    if role == "commentator":
        if other_has_spoken:
            return (
                f"\nThe real person in the room is \"{HUMAN_LABEL}\" — not {amy}. "
                f"Only attribute words to {amy} if a transcript line is labeled \"{amy}:\"."
            )
        return (
            f"\nThe real person in the room is \"{HUMAN_LABEL}\" — NOT {amy}. "
            f"{amy} has not spoken yet; do not open with \"{amy}, you…\" or thank {amy} "
            f"for what {HUMAN_LABEL} said. Respond to {HUMAN_LABEL}'s request; you may "
            f"mention that {amy} will speak next, without treating her as if she already did."
        )
    if role == "guest":
        ryan = load_persona_name("commentator")
        return (
            f"\nThe real person in the room is \"{HUMAN_LABEL}\" — not you ({amy}). "
            f"Build on {ryan} if helpful; satisfy what {HUMAN_LABEL} asked for."
        )
    return f"\nYou are {host}; the human is \"{HUMAN_LABEL}\"."


def panelist_system_for_text(role: str, *, other_has_spoken: bool = False) -> str:
    name = load_persona_name(role)
    host = load_persona_name("host")
    ryan = load_persona_name("commentator")
    amy = load_persona_name("guest")
    if role == "commentator":
        role_line = "color commentator"
        other = amy
    elif role == "guest":
        role_line = "guest panelist (AI)"
        other = ryan
    else:
        role_line = role
        other = ryan
    return (
        f"You are {name}, the {role_line} on a live English talk-show panel. "
        "The messages above are the running transcript — lines labeled with your "
        f"name are yours; lines labeled {other} or {host} are other AI speakers; "
        f'lines labeled "{HUMAN_LABEL}:" are the real human — never confuse them '
        f"with {amy} (she is a separate AI guest, not the human). "
        f"Speak in first person as {name} (use I/me/my, never \"{name} has\" or "
        f"\"I think {name}\"). You may address {other} by name only when they have "
        f"actually spoken in the transcript. "
        "Fulfill the human guest's latest request in what you say. "
        "If they asked for examples, include the full example in your speech. "
        "Plain English. After your spoken lines, always output [next]: "
        "commentator | guest | host — use commentator/guest ONLY if you pass the floor "
        "to them by name; otherwise host."
        f"{_panel_identity_guard(role, other_has_spoken=other_has_spoken)}"
    )


def panel_speech_prompt(
    *,
    role: str,
    name: str,
    host_name: str,
    latest_human: str,
    closing: bool = False,
) -> str:
    if closing:
        scene = host_close_card()
        task = "Close this round and return the floor to the human."
    else:
        scene = panelist_card(role=role, name=name, prior_lines=[])
        task = panelist_role_brief(role)

    human_block = ""
    if latest_human.strip():
        human_block = (
            f"\nHuman guest's latest message (you must satisfy THIS in your speech):\n"
            f'"{latest_human.strip()}"\n'
        )

    return f"""Live talk-show panel — your speaking turn.
{human_block}
{scene}

{task}

Rules:
- You are {name} — first person only (I/me), never third person ({name} said…).
- "{HUMAN_LABEL}" is the real person; {load_persona_name("guest")} is the AI guest — different people.
- Read all messages above plus the latest human message.
- Perform the request; do not only discuss whether it is hard to perform.
- Do not ask the human questions; their floor is frozen until the host closes the round.
- Do NOT say you are {host_name} unless you are the host closing.

Output exactly:
[reply]: <your spoken lines>
[next]: commentator | guest | host
- [next:commentator|guest] ONLY if you explicitly hand off to them by name in [reply]
- Otherwise [next:host] — the host will moderate who speaks next
"""


# Re-export for config.py
__all__ = [
    "panel_host_gemma_addon",
    "PANEL_HOST_AUDIO_USER_HINT",
    "PANEL_OPENING_LINE",
    "panel_speech_prompt",
    "panelist_system_for_text",
]
