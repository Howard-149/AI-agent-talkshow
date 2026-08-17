"""Parse Gemma [heard]/[reply]/[handoff] tags from multimodal model output."""

from __future__ import annotations

import re
from dataclasses import dataclass

from functools import lru_cache

_HEARD_RE = re.compile(r"\[heard\]\s*:\s*(.*?)(?=\[reply\]\s*:|$)", re.IGNORECASE | re.DOTALL)
_REPLY_RE = re.compile(r"\[reply\]\s*:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)
_HANDOFF_TAG_RE = re.compile(r"\[handoff:([^\]]+)\]", re.IGNORECASE)


@lru_cache(maxsize=1)
def _handoff_role_lookup() -> tuple[frozenset[str], dict[str, str]]:
    from agent.panel.panel_context import floor_valid_next_roles, role_name_aliases

    valid = floor_valid_next_roles()
    aliases = role_name_aliases(include_host=True)
    return valid, aliases


def _normalize_handoff_role(raw: str) -> str | None:
    key = raw.strip().lower()
    valid, aliases = _handoff_role_lookup()
    if key in valid:
        return key
    return aliases.get(key)


@dataclass(frozen=True)
class ParsedTurn:
    heard: str
    reply: str
    handoff_to: str | None = None
    next_speaker: str | None = None
    emotion: str | None = None


def _handoff_as_next(handoff_to: str | None) -> bool:
    if not handoff_to:
        return False
    valid, _ = _handoff_role_lookup()
    return handoff_to in valid


def _strip_handoff_tags(reply: str) -> tuple[str, str | None]:
    """Remove all [handoff:…] tags; never speak them. Last valid tag wins for tooling."""
    handoff_to: str | None = None
    for m in _HANDOFF_TAG_RE.finditer(reply):
        role = _normalize_handoff_role(m.group(1))
        if role:
            handoff_to = role
    clean = _HANDOFF_TAG_RE.sub("", reply)
    clean = re.sub(r"\s+", " ", clean).strip(" .")
    return clean, handoff_to


def parse_heard_reply(text: str) -> ParsedTurn:
    from agent.emotion import parse_emotion_tag, strip_emotion_tag
    from agent.floor.floor_parser import parse_next_speaker_tag, strip_speech_control_tags

    text = text.strip()
    next_speaker = parse_next_speaker_tag(text)
    emotion = parse_emotion_tag(text)
    heard_m = _HEARD_RE.search(text)
    reply_m = _REPLY_RE.search(text)

    heard = heard_m.group(1).strip() if heard_m else ""
    reply = reply_m.group(1).strip() if reply_m else ""

    if not reply:
        # Model ignored format — treat full output as reply for MVP robustness
        reply = strip_speech_control_tags(text)
    if not heard:
        heard = ""

    reply, handoff_to = _strip_handoff_tags(reply)
    reply = strip_speech_control_tags(reply)
    reply = strip_emotion_tag(reply)
    if not next_speaker and _handoff_as_next(handoff_to):
        next_speaker = handoff_to
    return ParsedTurn(
        heard=heard,
        reply=reply,
        handoff_to=handoff_to,
        next_speaker=next_speaker,
        emotion=emotion,
    )


def parse_host_speech(text: str) -> ParsedTurn:
    """Parse host text-only Gemma output: optional [reply]: block + [next]/[emotion] tags."""
    text = text.strip()
    if not text:
        return ParsedTurn(heard="", reply="", next_speaker=None, emotion=None)
    if _HEARD_RE.search(text) or _REPLY_RE.search(text):
        return parse_heard_reply(text)
    from agent.emotion import parse_emotion_tag
    from agent.floor.floor_parser import parse_next_speaker_tag, strip_speech_control_tags

    next_speaker = parse_next_speaker_tag(text)
    emotion = parse_emotion_tag(text)
    spoken, handoff_to = _strip_handoff_tags(strip_speech_control_tags(text))
    if not next_speaker and _handoff_as_next(handoff_to):
        next_speaker = handoff_to
    return ParsedTurn(
        heard="",
        reply=spoken,
        handoff_to=handoff_to,
        next_speaker=next_speaker,
        emotion=emotion,
    )


def is_noise_heard(heard: str) -> bool:
    """True when STT should not start a user turn (silence, placeholder, noise)."""
    text = heard.strip()
    if len(text) < 2:
        return True
    lower = text.lower()
    noise_markers = (
        "(audio understood",
        "(no audio",
        "(silence",
        "[silence]",
        "[no speech]",
        "[inaudible]",
        "no speech detected",
        "no audible speech",
        "inaudible",
        "silence.",
        "silence)",
        "...",
        "…",
        "n/a",
        "none",
        "unknown",
        "unable to transcribe",
        "could not hear",
        "couldn't hear",
        "did not hear",
        "didn't hear",
        "no input",
        "empty audio",
    )
    if any(m in lower for m in noise_markers):
        return True
    # Gemma sometimes invents a short greeting from noise
    if len(text) <= 24 and lower in {
        "hello",
        "hi",
        "hey",
        "thank you",
        "thanks",
        "ok",
        "okay",
        "yes",
        "no",
        "hmm",
        "um",
        "uh",
    }:
        return True
    return False
