from __future__ import annotations

import re

from agent.config import load_persona_name


def scene_card(
    *,
    you: str,
    you_role: str,
    just_spoke: str,
    up_next: list[str],
    human_floor: str,
) -> str:
    """Shared 'who has the floor' block for every Gemma call in a panel round."""
    next_line = " → ".join(up_next) if up_next else "(end of round)"
    frozen = (
        "OPEN — the human may respond."
        if human_floor == "open"
        else "FROZEN — the human is listening; they are NOT your conversation partner this beat."
    )
    return f"""
SCENE (live panel — keep dialogue natural):
- You are {you} ({you_role}).
- Who just spoke: {just_spoke}
- Who speaks NEXT (in order): {next_line}
- Human guest floor: {frozen}
- Speak as if everyone is in the same room: address whoever actually speaks next.
- If the human is FROZEN, talking TO them (especially questions) is unnatural — they cannot answer yet.
- If the human is OPEN, you may invite them back.
""".strip()


def host_after_human_card() -> str:
    host = load_persona_name("host")
    ryan = load_persona_name("commentator")
    amy = load_persona_name("guest")
    return scene_card(
        you=host,
        you_role="host / moderator",
        just_spoke="the human guest",
        up_next=[ryan, amy, f"{host} (close)", "human guest"],
        human_floor="frozen",
    ) + (
        f"\nYour beat: 1–2 sentences to the ROOM. End by calling on {ryan} to speak next "
        f"(name them once). If the human asked for examples or demonstrations, say the "
        f"panel will show them out loud — do not ask {ryan} an open question they cannot "
        f"answer while frozen. Do not ask the human anything."
    )


def host_audio_user_hint() -> str:
    return (
        f"{host_after_human_card()}\n\n"
        "Listen to the human's audio. Output [heard]: and [reply]:.\n"
        "[reply] must acknowledge what the human asked, match what they want from the "
        "panel (discussion vs concrete examples), then call on "
        f"{load_persona_name('commentator')} to speak next in the same paragraph. "
        "No [handoff:…] tags. Not a 1-on-1 interview."
    )


def panelist_card(
    *,
    role: str,
    name: str,
    prior_lines: list[tuple[str, str]] | None = None,
) -> str:
    host = load_persona_name("host")
    ryan = load_persona_name("commentator")
    amy = load_persona_name("guest")

    if role == "commentator":
        return scene_card(
            you=name,
            you_role="commentator",
            just_spoke=f"{host} (moderator)",
            up_next=[amy, f"{host} (close)", "human guest"],
            human_floor="frozen",
        ) + (
            "\nDeliver what the human asked for in your spoken lines. "
            "If they wanted examples, say the actual example — do not only analyze humor."
        )
    if role == "guest":
        just = (
            f"{ryan} (commentator)"
            if prior_lines
            else "the panel (see conversation above)"
        )
        return scene_card(
            you=name,
            you_role="guest",
            just_spoke=just,
            up_next=[f"{host} (close)", "human guest"],
            human_floor="frozen",
        ) + (
            "\nDeliver what the human asked for; build on the commentator if needed. "
            "If they wanted examples, your lines must include a full concrete example."
        )
    return scene_card(
        you=name,
        you_role=role,
        just_spoke="the panel",
        up_next=["human guest"],
        human_floor="open",
    )


def host_close_card() -> str:
    host = load_persona_name("host")
    amy = load_persona_name("guest")
    return scene_card(
        you=host,
        you_role="host / moderator",
        just_spoke=f"{amy} (guest)",
        up_next=["human guest"],
        human_floor="open",
    ) + "\nYour beat: 1–2 sentences; return the floor to the human."


def panel_host_gemma_addon() -> str:
    host = load_persona_name("host")
    ryan = load_persona_name("commentator")
    amy = load_persona_name("guest")
    return f"""
PANEL SHOW — floor awareness (not a rule list; keep the scene coherent):

You are {host}, moderator. This is a recorded panel, not a private chat with the human.

Each human turn triggers a fixed sequence: human → you (short) → {ryan} → {amy} → you (close) → human.
Until {ryan} and {amy} finish, the human is an audience member, not your dialogue partner.

{host_after_human_card()}
""".strip()


def _panel_tee_up_fallback(human_heard: str) -> str:
    ryan = load_persona_name("commentator")
    amy = load_persona_name("guest")
    topic = (human_heard or "that").strip()[:120]
    if topic.startswith("("):
        topic = "today's topic"
    return f"On {topic} — let's hear from {ryan}, then {amy}."


def sanitize_host_panel_reply(reply: str, human_heard: str) -> str:
    """Strip handoff tags, fix empty/tag-only lines, reshape interview tone."""
    from agent.adapters.response_parser import _strip_handoff_tags

    text, _ = _strip_handoff_tags(reply)
    text = text.strip()

    ryan = load_persona_name("commentator")
    amy = load_persona_name("guest")
    fallback = _panel_tee_up_fallback(human_heard)

    if not text or len(text) < 12 or text.lower().startswith("[handoff"):
        return fallback

    # Reply is only praising a panelist — wrong beat (human just spoke)
    if re.match(rf"^{re.escape(ryan)}\b", text, re.I) and amy.lower() not in text.lower():
        return fallback

    if "?" in text:
        parts = re.split(r"(?<=[.!?])\s+", text)
        kept = [p for p in parts if "?" not in p]
        if not kept:
            return fallback
        text = " ".join(kept).strip()

    return text
