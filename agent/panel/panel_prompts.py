"""Build Gemma system/user prompts for host audio turns and panelist speech."""

from __future__ import annotations

from agent.config import (
    ScenarioConfig,
    load_persona_name,
    load_persona_personality,
    load_scenario,
)
from agent.panel.panel_context import (
    any_other_panelist_spoken,
    format_name_list,
    panel_room_roster,
    next_tag_options,
    other_panelist_names,
    panel_opening_line,
    role_label,
    unspoken_panelist_names,
)
from agent.show.show_history import HUMAN_LABEL
from agent.show.show_context import (
    host_audio_user_hint,
    host_close_card,
    panel_host_gemma_addon,
    panelist_card,
)

PANEL_HOST_AUDIO_USER_HINT = host_audio_user_hint()


def panelist_personality_block(role: str) -> str:
    """Personality traits from persona yaml — used for speaking and hand-raise polls."""
    personality = load_persona_personality(role)
    if personality:
        return personality
    name = load_persona_name(role)
    return f"You are {name} on a live English talk-show panel."


def panelist_hand_raise_system(role: str) -> str:
    """System prompt for parallel hand-raise polls — includes personality, not just name."""
    personality = panelist_personality_block(role)
    return (
        f"{personality}\n\n"
        "Hand-raise decision (stay in character):\n"
        "- Read the transcript; decide if you want the floor next.\n"
        "- You may speak more than once this round.\n"
        "- If yes, you may propose a [topic] angle that fits how YOU would enter the discussion.\n"
        "Output ONLY:\n"
        "[raise]: yes | no\n"
        "[topic]: optional angle (if yes)\n"
        "[reason]: optional one short phrase (if yes)"
    )


def panelist_role_brief(
    role: str,
    *,
    scenario: ScenarioConfig,
    history: object,
    panel_roles: list[str],
) -> str:
    name = load_persona_name(role)
    label = role_label(role)
    others = format_name_list(other_panelist_names(role, panel_roles))
    unspoken = unspoken_panelist_names(history, panel_roles)
    unspoken_others = [n for n in unspoken if n != name]

    if any_other_panelist_spoken(history, role, panel_roles):
        return (
            f"Your turn, {name}. The panel is mid-discussion — respond to the latest "
            f"speaker or the thread under debate; add your {label.lower()} angle."
        )
    if unspoken_others:
        not_yet = format_name_list(unspoken_others)
        return (
            f"Your turn, {name}. The host opened the topic — give your first take on "
            f"what the room is discussing. {not_yet} may not have spoken yet; "
            f"do not talk as if they already did."
        )
    return (
        f"Your turn, {name}. Join the live discussion — you may agree with, push back on, "
        f"or extend what {others} or the host said, or deepen the topic on the table."
    )


def _panel_identity_guard(
    role: str,
    *,
    scenario: ScenarioConfig,
    panel_roles: list[str],
    history: object,
) -> str:
    host = load_persona_name("host")
    unspoken_others = [
        load_persona_name(r)
        for r in panel_roles
        if r != role and not history.role_has_spoken(r)  # type: ignore[union-attr]
    ]

    extra = ""
    if unspoken_others:
        extra = (
            f"\nThese panelists have not spoken yet this round: "
            f"{format_name_list(unspoken_others)} — do not attribute their views to them."
        )

    return f"\n{panel_room_roster(scenario)}{extra}"


def panelist_system_prompt(
    role: str,
    *,
    scenario: ScenarioConfig,
    panel_roles: list[str],
    history: object,
    dialogue_hint: str = "",
    data: object | None = None,
) -> str:
    name = load_persona_name(role)
    host = load_persona_name("host")

    personality = panelist_personality_block(role)

    tag_opts = next_tag_options(panel_roles, include_host=True)
    mechanics = (
        "Panel mechanics:\n"
        f"- {panel_room_roster(scenario)}\n"
        "- Each transcript line is tagged with its speaker — match your reply to that person or thread.\n"
        f"- When the thread is the guest's opening take, engage {HUMAN_LABEL}'s claim (in the transcript).\n"
        f"- When a panelist just spoke, you may name them and answer their point.\n"
        f"- When {host} last spoke, that line is moderation or summary — the underlying claim is under "
        f"{HUMAN_LABEL} or another panelist.\n"
        "- Match tone to stance: build when you agree; push back clearly when you disagree.\n"
        f"- Speak in first person as {name} (I/me/my).\n"
        "- If the human's floor is FROZEN, do not ask them direct questions — they are listening.\n"
        f"- Plain English. After your spoken lines, output [next]: {tag_opts} — "
        "use a panelist role ONLY if you pass the floor to them by name; otherwise host.\n"
        "- Also output [emotion]: matching your [reply] tone (closed set in the user prompt)."
        f"{_panel_identity_guard(role, scenario=scenario, panel_roles=panel_roles, history=history)}"
    )

    mood_block = ""
    if data is not None:
        from agent.emotion import emotion_prompt_block

        mood_block = f"\n\n{emotion_prompt_block(data, role)}"  # type: ignore[arg-type]

    hint_block = f"\n\n{dialogue_hint.strip()}" if dialogue_hint.strip() else ""
    return f"{personality}\n\n{mechanics}{mood_block}{hint_block}"


def panel_speech_prompt(
    *,
    role: str,
    name: str,
    host_name: str,
    scenario: ScenarioConfig,
    panel_roles: list[str],
    history: object,
    latest_human: str,
    closing: bool = False,
    data: object | None = None,
) -> str:
    from agent.emotion import emotion_output_lines, emotion_prompt_block

    tag_opts = next_tag_options(panel_roles, include_host=True)
    if closing:
        scene = host_close_card(scenario)
        task = "Close this round and return the floor to the human."
    else:
        scene = panelist_card(role=role, name=name, scenario=scenario)
        task = panelist_role_brief(
            role, scenario=scenario, history=history, panel_roles=panel_roles
        )

    topic_block = ""
    if latest_human.strip():
        topic_block = (
            f'\n{HUMAN_LABEL} said: "{latest_human.strip()}"\n'
            f"Engage this guest line when it is still the thread on the table.\n"
        )

    mood = ""
    if data is not None:
        mood = f"\n{emotion_prompt_block(data, role)}\n"  # type: ignore[arg-type]

    return f"""Live talk-show panel — your speaking turn.
{topic_block}
{scene}
{mood}
{task}

Rules:
- You are {name} — first person only (I/me), never third person ({name} said…).
- {panel_room_roster(scenario)}
- Read the full transcript; your line should fit the current discussion, not ignore what others said.
- You may agree, disagree, or build on another panelist — this is conversation, not a solo Q&A.
- Do not ask the human direct questions while their floor is frozen.
- Do NOT say you are {host_name} unless you are the host closing.

Output exactly:
[reply]: <your spoken lines>
[next]: {tag_opts}
{emotion_output_lines()}
- Pass [next:<role>] ONLY if you explicitly hand off to that panelist by name in [reply]
- Otherwise [next:host] — the host will moderate who speaks next
"""


# Re-export for config.py
__all__ = [
    "panel_host_gemma_addon",
    "PANEL_HOST_AUDIO_USER_HINT",
    "panel_opening_line",
    "panel_speech_prompt",
    "panelist_hand_raise_system",
    "panelist_personality_block",
    panel_room_roster,
    "panelist_system_prompt",
]
