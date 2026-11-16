"""The shared show frame for every role's think call (TALKSHOW_SHOW_FRAME=1, default OFF).

Before this, a panelist's system prompt was only its own card ("You are Ryan. On this show: color commentator ..."):
nothing said what kind of show it is, and with the lean think prompt the canned host line that names the topic is
dropped from the transcript, so panelists did not see the topic either. The host's card called it a debate. The frame
says, once and for everyone: this is a live talk show, not a debate or a speech; who is on it; today's topic; how
people talk on this show. Each role's own card follows it.
"""

from __future__ import annotations

import os
import re
from typing import Any

_TOPIC_LINE = re.compile(r"talk about (.+?)\s*[—–-]+\s*the floor is yours", re.IGNORECASE)

FRAME = (
    "This is {show}a live TV talk show in front of a studio audience. It is a conversation, not a debate and not a "
    "lecture.{topic}\n"
    "On the show: {host} hosts; {commentator} and {guest} are on the panel; a guest from the audience joins in.\n"
    "How people talk on this show: with each other, not at each other. React to what was just said, in your own "
    "voice. Keep it conversational and short, the way people talk on TV: no speeches, no lists, no closing "
    "summaries. Disagreement is welcome and can be playful. A quick joke, a vivid example or a real story lands "
    "better than one more argument. The audience should enjoy the conversation and learn something from it."
)


def show_frame_enabled() -> bool:
    return os.environ.get("TALKSHOW_SHOW_FRAME", "0").strip().lower() in ("1", "true", "yes", "on")


def topic_of(history: Any) -> str:
    """Today's topic: TALKSHOW_TOPIC if set, else the canned host line that hands over the floor ("You wanted to talk
    about <topic> - the floor is yours"), else ""."""
    from agent.show.show_topic import show_topic

    topic = show_topic()
    if topic:
        return topic
    for line in getattr(history, "lines", []) or []:
        m = _TOPIC_LINE.search(getattr(line, "text", "") or "")
        if m:
            return m.group(1).strip()
    return ""


def show_frame(history: Any) -> str:
    from agent.config import load_persona_name

    topic = topic_of(history)
    name = os.environ.get("TALKSHOW_SHOW_NAME", "").strip()
    return FRAME.format(show=f"{name}, " if name else "",
                        topic=f" Today's topic: {topic}." if topic else "",
                        host=load_persona_name("host"), commentator=load_persona_name("commentator"),
                        guest=load_persona_name("guest"))


def with_frame(system_prompt: str, history: Any) -> str:
    """``system_prompt`` with the show frame in front (unchanged when the flag is off)."""
    if not show_frame_enabled():
        return system_prompt
    return f"{show_frame(history)}\n\n{system_prompt}"
