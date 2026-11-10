"""Deterministic rules for Think & Action interrupts (on by default; ``TALKSHOW_FLOOR_RULES=0`` turns them off).

Listeners propose interrupts and the LLM floor controller judges whether one is worth it.
With the rules on, hard limits run first, so the controller only sees requests that pass:

- grounding: an interrupt quotes the words it reacts to, and the quote must appear in the
  speech disclosed so far (case, punctuation and quote marks ignored; no paraphrase);
- minimum speaking time: the speaker gets a fair start before anyone may cut in;
- moderator priority: the host only cuts in to moderate (clear drift or a personal attack,
  see ``HOST_OCCUPIED_GUIDANCE``); a grounded host cut is granted without a controller
  call, outranks panelists, and skips the AI cooldown and chain cap;
- chain cap: after one approved cut, the line that took over cannot be cut by a panelist
  (the host may still moderate); the chain ends when a line finishes uncut;
- AI cooldown: panelist cuts are spaced out on the speech clock.

Time is measured in spoken words (the script runs ahead of playback, so wall time does not
tell how much the audience has heard): seconds = words / ``TALKSHOW_SPEECH_WORDS_PER_SEC``.
No model calls; the gate itself (``gate_interrupts``) is pure.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

DEFAULT_MIN_SPEECH_S = 3.0
DEFAULT_AI_COOLDOWN_S = 15.0
DEFAULT_MAX_CHAIN_CUTS = 1

HOST_PRIORITY = 80
PANELIST_PRIORITY = 50

_WORD_RE = re.compile(r"\S+")

# Replaces the occupied-floor guidance of the host's think prompt when the rules are on.
HOST_OCCUPIED_GUIDANCE = (
    "When occupied, you are the moderator, not a debater. Request interrupt ONLY if the "
    "speaker has clearly drifted away from the show's topic (judge the topic from the "
    "conversation so far) or makes a personal attack on another participant. Never "
    "interrupt to take a side, add your own opinion, or correct a fact; the panelists "
    "handle disagreement. If in doubt, stay silent. For an interrupt, use intent type "
    "facilitate; your content must steer back to the topic politely, address the speaker "
    "by name, and be 1 or 2 short sentences. "
)

# Added to every listener's think prompt when the rules are on.
QUOTE_RULE = (
    " For interrupt, also return \"quote\": an exact short quote (a few consecutive words, "
    "copied verbatim) from the complete visible speech prefix that your interruption "
    "responds to. An interrupt without a verbatim quote of words already spoken is ignored."
)
QUOTE_SCHEMA_LINE = '    "quote": "exact words from the visible speech prefix (interrupt only) or null",\n'


def floor_rules_enabled() -> bool:
    """On unless TALKSHOW_FLOOR_RULES is 0 / false / no / off."""
    return os.environ.get("TALKSHOW_FLOOR_RULES", "1").strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def min_speech_s() -> float:
    return _env_float("TALKSHOW_FLOOR_MIN_SPEECH_S", DEFAULT_MIN_SPEECH_S)


def ai_cooldown_s() -> float:
    return _env_float("TALKSHOW_FLOOR_AI_COOLDOWN_S", DEFAULT_AI_COOLDOWN_S)


def max_chain_cuts() -> int:
    return int(_env_float("TALKSHOW_FLOOR_MAX_CHAIN_CUTS", DEFAULT_MAX_CHAIN_CUTS))


def words_per_sec() -> float:
    """The speaking rate the speech consumer uses to place rollbacks and backchannels."""
    # Imported here: agent.interaction imports this module.
    from agent.interaction.consumers import DEFAULT_SPEECH_WORDS_PER_SEC

    wps = _env_float("TALKSHOW_SPEECH_WORDS_PER_SEC", DEFAULT_SPEECH_WORDS_PER_SEC)
    return wps if wps > 0 else DEFAULT_SPEECH_WORDS_PER_SEC


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def host_identity(name: str) -> str:
    """The host's think system prompt (the host persona has no ``personality`` text)."""
    return (
        f"You are {name}, the host and moderator of this live talk show. You are "
        "professionally neutral on the debate topic: you run the show, keep the discussion "
        "on track and bring in every voice; you are not one of the debaters."
    )


def _normalized_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")))


def quote_heard(quote: str | None, heard: str) -> bool:
    """The quote's words appear contiguously in ``heard``.

    Case, punctuation and wrapping quotation marks are ignored (models often return
    "quoted" text); paraphrases and stitched fragments are not.
    """
    words = _normalized_words(quote or "")
    return bool(words) and f" {words} " in f" {_normalized_words(heard)} "


@dataclass(frozen=True)
class InterruptRequest:
    role: str
    quote: str | None


@dataclass(frozen=True)
class RuleDecision:
    role: str
    status: Literal["granted", "to_controller", "blocked"]
    rule: str
    priority: int


@dataclass(frozen=True)
class GateResult:
    granted: str | None  # a host cut granted without the controller
    to_controller: list[str]  # panelists the controller may still pick from
    decisions: list[RuleDecision]


def gate_interrupts(
    requests: list[InterruptRequest],
    *,
    host_role: str,
    visible_text: str,
    now_s: float,
    last_ai_cut_s: float | None,
    chain_cuts: int,
) -> GateResult:
    """Apply the rules to this interval's interrupt requests (in listener order)."""
    spoken_s = word_count(visible_text) / words_per_sec()
    decisions: list[RuleDecision] = []
    granted: str | None = None
    for request in requests:
        is_host = request.role == host_role
        priority = HOST_PRIORITY if is_host else PANELIST_PRIORITY
        if not quote_heard(request.quote, visible_text):
            decisions.append(RuleDecision(request.role, "blocked", "quote_not_heard", priority))
        elif spoken_s < min_speech_s():
            decisions.append(RuleDecision(request.role, "blocked", "minimum_speaking_time", priority))
        elif is_host:
            if granted is None:
                granted = request.role
            decisions.append(RuleDecision(request.role, "granted", "moderator_cut", priority))
        elif chain_cuts >= max_chain_cuts():
            decisions.append(RuleDecision(request.role, "blocked", "chain_limit", priority))
        elif last_ai_cut_s is not None and now_s - last_ai_cut_s < ai_cooldown_s():
            decisions.append(RuleDecision(request.role, "blocked", "ai_cooldown", priority))
        else:
            decisions.append(RuleDecision(request.role, "to_controller", "grounded_request", priority))
    if granted is not None:
        # A moderator cut outranks every panelist request in the same interval.
        decisions = [
            RuleDecision(d.role, "blocked", "moderator_priority", d.priority)
            if d.status == "to_controller"
            else d
            for d in decisions
        ]
        return GateResult(granted, [], decisions)
    to_controller = [d.role for d in decisions if d.status == "to_controller"]
    return GateResult(None, to_controller, decisions)
