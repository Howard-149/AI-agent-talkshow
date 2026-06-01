from __future__ import annotations

from agent.config import load_persona_name
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
        return (
            f"You are {name} (commentator). Your turn now — speak 2–4 sentences out loud. "
            "Satisfy the human guest's latest request using the transcript above."
        )
    if role == "guest":
        return (
            f"You are {name} (guest). Your turn now — speak 2–4 sentences out loud. "
            "Add a second concrete contribution; do not repeat the commentator."
        )
    return f"You are {name}. Speak in character."


def panelist_system_for_text(role: str) -> str:
    name = load_persona_name(role)
    if role == "commentator":
        role_line = "color commentator"
    elif role == "guest":
        role_line = "guest"
    else:
        role_line = role
    other = load_persona_name(
        "guest" if role == "commentator" else "commentator"
    )
    return (
        f"You are {name}, the {role_line} on a live English talk-show panel. "
        "The messages above are the running transcript — lines labeled with your "
        f"name are yours; lines labeled {other} or Lessac are other speakers. "
        f"Speak in first person as {name} (use I/me/my, never \"{name} has\" or "
        f"\"I think {name}\"). You may address {other} by name, but do not speak "
        f"as if you are {other} commenting on {name}. "
        "Fulfill the human guest's latest request in what you say. "
        "If they asked for examples, include the full example in your speech. "
        "Output only spoken lines. Plain English. No labels or markdown."
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
- Read all messages above plus the latest human message.
- Perform the request; do not only discuss whether it is hard to perform.
- Do not ask the human questions; their floor is frozen until the host closes the round.
- Do NOT say you are {host_name} unless you are the host closing.
"""


# Re-export for config.py
__all__ = [
    "panel_host_gemma_addon",
    "PANEL_HOST_AUDIO_USER_HINT",
    "PANEL_OPENING_LINE",
    "panel_speech_prompt",
    "panelist_system_for_text",
]
