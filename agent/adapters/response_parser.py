from __future__ import annotations

import re
from dataclasses import dataclass

_HEARD_RE = re.compile(r"\[heard\]\s*:\s*(.*?)(?=\[reply\]\s*:|$)", re.IGNORECASE | re.DOTALL)
_REPLY_RE = re.compile(r"\[reply\]\s*:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)
_HANDOFF_TAG_RE = re.compile(r"\[handoff:([^\]]+)\]", re.IGNORECASE)
_VALID_HANDOFF = frozenset({"host", "guest", "commentator"})
# Model sometimes uses display names instead of role ids
_HANDOFF_ALIASES: dict[str, str] = {
    "lessac": "host",
    "ryan": "commentator",
    "amy": "guest",
}


@dataclass(frozen=True)
class ParsedTurn:
    heard: str
    reply: str
    handoff_to: str | None = None
    next_speaker: str | None = None


def _normalize_handoff_role(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in _VALID_HANDOFF:
        return key
    return _HANDOFF_ALIASES.get(key)


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
    from agent.floor_parser import parse_next_speaker_tag, strip_next_tag

    text = text.strip()
    next_speaker = parse_next_speaker_tag(text)
    heard_m = _HEARD_RE.search(text)
    reply_m = _REPLY_RE.search(text)

    heard = heard_m.group(1).strip() if heard_m else ""
    reply = reply_m.group(1).strip() if reply_m else ""

    if not reply:
        # Model ignored format — treat full output as reply for MVP robustness
        reply = strip_next_tag(text)
    if not heard:
        heard = ""

    reply, handoff_to = _strip_handoff_tags(reply)
    reply = strip_next_tag(reply)
    if not next_speaker and handoff_to in ("commentator", "guest", "host"):
        next_speaker = handoff_to
    return ParsedTurn(
        heard=heard, reply=reply, handoff_to=handoff_to, next_speaker=next_speaker
    )


def parse_host_speech(text: str) -> ParsedTurn:
    """Parse host text-only Gemma output: optional [reply]: block + [next]: tag."""
    text = text.strip()
    if not text:
        return ParsedTurn(heard="", reply="", next_speaker=None)
    if _HEARD_RE.search(text) or _REPLY_RE.search(text):
        return parse_heard_reply(text)
    from agent.floor_parser import parse_next_speaker_tag, strip_next_tag

    next_speaker = parse_next_speaker_tag(text)
    spoken, handoff_to = _strip_handoff_tags(strip_next_tag(text))
    if not next_speaker and handoff_to in ("commentator", "guest", "host"):
        next_speaker = handoff_to
    return ParsedTurn(heard="", reply=spoken, handoff_to=handoff_to, next_speaker=next_speaker)


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
