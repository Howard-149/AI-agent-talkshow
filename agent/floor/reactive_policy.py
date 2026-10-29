"""Deterministic floor arbitration. No model or LiveKit dependencies.

An LLM proposes an intent; only this policy can authorize a cut. Epochs are
monotonic turn IDs, not transcript length (the transcript has a rolling cap).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re


@dataclass(frozen=True)
class FloorRequest:
    role: str
    epoch: int
    action: str
    kind: str = "addition"
    reason: str = ""
    evidence: str = ""
    speech: str = ""
    source: str = "llm"


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")))


def quote_heard(evidence: str, heard: str) -> bool:
    """The quote's words appear contiguously in the played speech.

    Case, punctuation and wrapping quotation marks are ignored (small models
    often return “quoted” evidence); paraphrases and stitched fragments are not.
    """
    quote = _words(evidence)
    return bool(quote) and f" {quote} " in f" {_words(heard)} "


@dataclass(frozen=True)
class Decision:
    status: str  # granted | queued | rejected | none
    rule: str
    priority: int = 0


def parse_request(raw: str, *, role: str, epoch: int, moderator: str = "host") -> FloorRequest:
    """Strict JSON and allowlisted enums; malformed responses fail closed."""
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("request must be a JSON object")
    action = obj.get("action")
    kind = obj.get("kind", "moderation" if role == moderator else "addition")
    if action not in {"interrupt", "queue", "none"}:
        raise ValueError("invalid action")
    if kind not in {"rebuttal", "correction", "addition", "moderation"}:
        raise ValueError("invalid kind")
    # Only the moderator may claim a moderation cut, and the moderator never debates.
    if (kind == "moderation") != (role == moderator):
        raise ValueError("kind not allowed for this role")
    fields = {}
    for key, limit in {"reason": 240, "evidence": 500, "speech": 900}.items():
        value = obj.get(key, "")
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"invalid {key}")
        fields[key] = value.strip()
    if action != "none" and (not fields["reason"] or not fields["speech"]):
        raise ValueError("a request needs a reason and a speech")
    # role/epoch/source come from trusted orchestration, never the model response.
    return FloorRequest(role=role, epoch=epoch, action=action, kind=kind, **fields)


class ReactivePolicy:
    def __init__(self, *, min_speech_s: float = 3.0, cooldown_s: float = 15.0):
        self.min_speech_s = min_speech_s
        self.cooldown_s = cooldown_s
        self.last_ai_cut = float("-inf")

    def decide(
        self, request: FloorRequest, *, epoch: int, speaker: str,
        heard: str, elapsed_s: float, now: float, eligible_roles: list[str],
        allow_ai_cut: bool = True, human_pending: bool = False,
    ) -> Decision:
        if request.epoch != epoch:
            return Decision("rejected", "stale_epoch")
        if request.source == "human" and request.role == "human":
            return Decision("granted", "human_priority", 100)
        if request.role == speaker or request.role not in eligible_roles:
            return Decision("rejected", "ineligible_role")
        if request.action == "none":
            return Decision("none", "no_new_point")
        if human_pending:
            return Decision("queued", "human_priority", 50)
        moderation = request.kind == "moderation"
        if moderation and request.action != "interrupt":
            # The moderator already owns the next turn; it has no queue slot.
            return Decision("none", "moderator_waits")
        if request.action == "queue" or request.kind == "addition":
            return Decision("queued", "wait_for_turn_end", 10)
        if not quote_heard(request.evidence, heard):
            return Decision("rejected", "evidence_not_heard")
        if moderation:
            # Moderation outranks debate cuts and is not limited by the AI
            # cooldown or chain cap, but the speaker still gets a fair start.
            if elapsed_s < self.min_speech_s:
                return Decision("queued", "minimum_speaking_time", 80)
            return Decision("granted", "moderator_cut", 80)
        if not allow_ai_cut:
            return Decision("queued", "queue_only_or_chain_limit", 50)
        if elapsed_s < self.min_speech_s:
            return Decision("queued", "minimum_speaking_time", 50)
        if now - self.last_ai_cut < self.cooldown_s:
            return Decision("queued", "cooldown", 50)
        return Decision("granted", "grounded_rebuttal", 50)

    def commit_cut(self, request: FloorRequest, *, now: float) -> None:
        if request.source == "llm":
            self.last_ai_cut = now
