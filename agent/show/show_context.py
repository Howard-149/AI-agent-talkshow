"""Scene/persona cards and Gemma hint blocks for host and panelist prompts."""

from __future__ import annotations

import re

from agent.config import load_persona_name
from agent.data import TalkShowData
from agent.panel.panel_context import (
    format_name_list,
    panel_room_roster,
    name_to_panel_role,
    next_tag_options,
    panel_opening_line,
    panel_role_name_map,
    panel_speaker_names,
    panel_speaker_roles,
    role_label,
    session_welcome_line,
    other_panelist_names,
    unspoken_panelist_names,
    any_other_panelist_spoken,
)
from agent.show.show_history import HUMAN_LABEL


def scene_card(
    *,
    you: str,
    you_role: str,
    just_spoke: str,
    up_next: str,
    human_floor: str,
) -> str:
    """Shared 'who has the floor' block for every Gemma call in a panel round."""
    frozen = (
        "OPEN — the human may respond."
        if human_floor == "open"
        else "FROZEN — the human is listening; they are NOT your conversation partner this beat."
    )
    return f"""
SCENE (live panel — keep dialogue natural):
- You are {you} ({you_role}).
- Who just spoke: {just_spoke}
- Floor control: {up_next}
- Human guest floor: {frozen}
- Speak as if everyone is in the same room; address whoever last spoke in the transcript.
- If the human is FROZEN, talking TO them (especially questions) is unnatural — they cannot answer yet.
- If the human is OPEN, you may invite them back.
""".strip()


def host_after_human_card(scenario) -> str:
    host = load_persona_name("host")
    panel = format_name_list(panel_speaker_names(scenario))
    return scene_card(
        you=host,
        you_role="host / moderator",
        just_spoke="the human guest",
        up_next="host moderates — panelists may raise hands; you grant the floor",
        human_floor="frozen",
    ) + (
        f"\nYour beat: 1–2 sentences to the ROOM.\n"
        f"- Name a panelist ({panel}) to speak next → matching [next:<role>]\n"
        f"- Open floor for volunteers → [next:host]"
    )


def host_audio_user_hint() -> str:
    from agent.config import load_scenario
    from agent.emotion import EMOTION_CHOICES, emotion_output_lines

    scenario = load_scenario()
    panel_roles = panel_speaker_roles(scenario)
    tags = next_tag_options(panel_roles, include_host=True) + " | human | close"
    return (
        f"{host_after_human_card(scenario)}\n\n"
        "Listen to the human's audio. Output exactly:\n"
        "[heard]: <transcript>\n"
        "[reply]: <short host tee-up to the room>\n"
        f"[next]: {tags}\n"
        f"{emotion_output_lines()}\n\n"
        "[next] rules:\n"
        "- You named a panelist to speak next → their role id\n"
        "- Open floor, no one picked → host (panel raises hands)\n"
        "- If [reply] names someone but you forget the tag, the show may still route from your tee-up\n"
        f"[reply] must acknowledge what the human raised and match your [emotion] ({EMOTION_CHOICES}). "
        "Not a 1-on-1 interview."
    )


def panelist_card(
    *,
    role: str,
    name: str,
    scenario,
    prior_lines: list[tuple[str, str]] | None = None,
) -> str:
    host = load_persona_name("host")
    panel_roles = panel_speaker_roles(scenario)
    label = role_label(role)

    if prior_lines:
        last = prior_lines[-1]
        just_spoke = f"{last[1]} ({last[0]})"
    else:
        just_spoke = "the host or an earlier speaker (see transcript)"

    others = [n for n in other_panelist_names(role, panel_roles)]
    others_text = format_name_list(others) if others else "other panelists"

    return scene_card(
        you=name,
        you_role=label,
        just_spoke=just_spoke,
        up_next="host moderates who speaks next after you",
        human_floor="frozen",
    ) + (
        f"\nJoin the discussion on the topic in the transcript — respond to {just_spoke}, "
        f"{others_text}, or the guest's take as fits the thread.\n"
        f"{panel_room_roster(scenario)}"
    )


def host_close_card(scenario) -> str:
    host = load_persona_name("host")
    return scene_card(
        you=host,
        you_role="host / moderator",
        just_spoke="the panel (see transcript for who spoke last)",
        up_next="return floor to the human guest",
        human_floor="open",
    ) + "\nYour beat: 1–2 sentences; return the floor to the human."


def panel_host_gemma_addon() -> str:
    from agent.config import load_scenario

    scenario = load_scenario()
    host = load_persona_name("host")
    panel = format_name_list(panel_speaker_names(scenario))
    return f"""
PANEL SHOW — floor awareness (not a rule list; keep the scene coherent):

You are {host}, moderator. This is a recorded panel, not a private chat with the human.

After the human speaks, you tee up the room; panelists ({panel}) join via hand-raise queue
and your grants — order is NOT fixed. Until you return the floor, the human is listening.

{host_after_human_card(scenario)}
""".strip()


def _panel_tee_up_fallback(human_heard: str, scenario) -> str:
    panel = format_name_list(panel_speaker_names(scenario))
    topic = (human_heard or "that").strip()[:120]
    if topic.startswith("("):
        topic = "today's topic"
    return f"On {topic} — let's hear from {panel}."


def sanitize_host_panel_reply(reply: str, human_heard: str, scenario) -> str:
    """Strip handoff tags, fix empty/tag-only lines, reshape interview tone."""
    from agent.adapters.response_parser import _strip_handoff_tags
    from agent.floor.floor_parser import strip_speech_control_tags

    text, _ = _strip_handoff_tags(reply)
    text = strip_speech_control_tags(text).strip()

    fallback = _panel_tee_up_fallback(human_heard, scenario)
    panel_names = {load_persona_name(r).lower() for r in panel_speaker_roles(scenario)}

    if not text or len(text) < 12 or text.lower().startswith("[handoff"):
        return fallback

    # Reply opens by praising a single panelist only — wrong beat (human just spoke)
    first_word = text.split()[0].lower().rstrip(",.") if text.split() else ""
    if first_word in panel_names and "?" not in text:
        mentions_panel = sum(1 for n in panel_names if n in text.lower())
        if mentions_panel == 1:
            return fallback

    if "?" in text:
        parts = re.split(r"(?<=[.!?])\s+", text)
        kept = [p for p in parts if "?" not in p]
        if not kept:
            return fallback
        text = " ".join(kept).strip()

    return text


def host_used_tee_fallback(reply: str, human_heard: str, scenario) -> bool:
    """True when sanitize substituted the generic tee-up template."""
    return reply.strip() == _panel_tee_up_fallback(human_heard, scenario)
