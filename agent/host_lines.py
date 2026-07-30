"""Fixed host moderation lines with light random variation.

Keeps canned cues (intro / open floor / human grant) from sounding identical
every beat. English only — git-tracked.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

# Avoid immediate repeats across calls in the same process.
_last_line: dict[str, str] = {}


def _pick(kind: str, options: Sequence[str]) -> str:
    choices = list(options)
    if not choices:
        return ""
    last = _last_line.get(kind)
    if last and len(choices) > 1:
        choices = [c for c in choices if c != last] or list(options)
    line = random.choice(choices)
    _last_line[kind] = line
    return line


# {name} = panelist display name (Amy, Ryan, …)
_INTRO_SPEAKER = (
    "{name}, you're up.",
    "{name}, over to you.",
    "{name}, take it away.",
    "{name}, the floor is yours.",
    "Let's hear from {name}.",
    "{name}, go ahead.",
    "All right {name} — you're on.",
    "{name}, what do you think?",
)

_OPEN_FLOOR = (
    "Let's open the floor — who wants to weigh in?",
    "Floor's open — anyone want to jump in?",
    "Who wants to take this next?",
    "I'm opening the floor — raise a hand if you've got something.",
    "All right, open floor — who'd like to speak?",
    "Anyone care to weigh in? Floor's open.",
)

_HUMAN_FLOOR = (
    "You have the floor — go ahead whenever you're ready.",
    "Over to you — whenever you're ready.",
    "The mic is yours — go ahead.",
    "All yours — take your time.",
    "Floor's yours — jump in whenever you're ready.",
)

_HUMAN_FLOOR_TOPIC = (
    "You wanted to talk about {topic} — the floor is yours, go ahead.",
    "You flagged {topic} — over to you.",
    "About {topic}: the mic is yours whenever you're ready.",
    "You wanted {topic} — go ahead.",
)

_DIRECT_CALL_FALLBACK = (
    "{name}, you're up — what would you like to add?",
    "{name}, over to you — what are you thinking?",
    "{name}, jump in — what's your take?",
    "{name}, the floor is yours — anything to add?",
)


def host_intro_speaker_line(name: str) -> str:
    """Verbal grant before a panelist speaks."""
    return _pick("intro", _INTRO_SPEAKER).format(name=name)


def host_open_floor_line() -> str:
    return _pick("open_floor", _OPEN_FLOOR)


def host_human_floor_line(*, topic: str | None = None) -> str:
    topic = (topic or "").strip()
    if topic:
        return _pick("human_topic", _HUMAN_FLOOR_TOPIC).format(topic=topic)
    return _pick("human", _HUMAN_FLOOR)


def host_direct_call_fallback_line(name: str) -> str:
    """When Gemma fails to produce a direct-call line."""
    return _pick("direct_fallback", _DIRECT_CALL_FALLBACK).format(name=name)
