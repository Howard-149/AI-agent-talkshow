"""T2f floor items: both panelists bid on the same audience line, and the show's own free-floor selection picks.

For the Think & Action system the selection is the production code: ``ThinkActionInteraction.arbitrate_free`` with
the panelists' parsed think actions, including the addressee priority (TALKSHOW_ADDRESSEE_FLOOR) and the LLM floor
controller. It runs against a client that records the controller request instead of sending it (as
eval/think_action_prompt.py does for think calls), so the controller prompt and sampling are the live ones. Run it once
with no answer to get the controller request (or no request, when a rule picked without the controller), then again
with the controller's answer to get the outcome.

Differences from the live show, by design: only the two panelists bid (the host does not think on this turn), and
the floor history (grants, fallbacks, invitations) starts empty.

Plain systems have no floor: they get one floor question per item (FLOOR_QUESTION), so every system has a T2f score.
The two kinds are measured differently and the report says which applied.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

FLOOR_SYSTEM = "You decide who speaks next on a live talk show panel."
FLOOR_QUESTION = "Who should speak next: {a} or {b}? Answer with one name."
PANEL = ("commentator", "guest")


class _ControllerClient:
    """Stands in for GemmaMMClient in arbitrate_free: records the controller request, answers with ``answer``."""

    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.requests: list[dict[str, Any]] = []
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def complete_text(self, prompt: str, **kw: Any) -> str:
        self.requests.append({"prompt": prompt, **kw})
        return self.answer if self.answer is not None else "{}"

    def log_text_only_event(self, role: str, kind: str, **fields: Any) -> None:
        self.events.append((role, kind, fields))


def arbitrate(prefix: dict, human_text: str, actions: dict[str, Any], answer: str | None = None) -> dict[str, Any]:
    """The production free-floor selection after ``human_text`` with the panelists' ``actions``.

    Returns {"outcome": FloorOutcome, "controller": {"messages", "temperature", "max_tokens"} or None,
    "candidates": [roles that chose speak], "addressee": event fields or None}."""
    from agent.adapters.gemma_mm_client import GemmaMMClient
    from agent.interaction import ConsoleSpeechConsumer, ThinkActionInteraction
    from agent.show.show_history import ShowHistory
    from eval.think_action_prompt import _data_for

    data = _data_for(prefix, human_text, "")
    client = _ControllerClient(answer)
    data.runtime.gemma_client = client
    inter = ThinkActionInteraction(ConsoleSpeechConsumer())
    inter.actions = dict(actions)
    candidates = [r for r in PANEL if r in actions and actions[r].decision == "speak"]
    history = ShowHistory(lines=list(data.show_history.lines))
    outcome = asyncio.run(inter.arbitrate_free(data, candidates=candidates, history=history, interval=1))
    controller = None
    if client.requests:
        (req,) = client.requests
        controller = {
            "messages": GemmaMMClient._build_messages(
                None,  # type: ignore[arg-type]  # uses no instance state
                req["system_prompt"],
                history_messages=req.get("history_messages") or [],
                tail_user=req["prompt"],
            ),
            "temperature": req["temperature"],
            "max_tokens": req["max_tokens"],
        }
    addressee = next((f for _r, kind, f in client.events if kind == "addressee"), None)
    return {"outcome": outcome, "controller": controller, "candidates": candidates, "addressee": addressee}


def plain_floor_messages(obs: dict) -> list[dict]:
    """The plain systems' floor question: the transcript and the audience line, then FLOOR_QUESTION."""
    from eval.bench.systems import AUDIENCE_LABEL

    names = obs.get("names", {})
    lines = [f"{AUDIENCE_LABEL if h['role'] == 'human' else names.get(h['role'], h['speaker'])}: {h['text']}"
             for h in obs["history"]]
    if obs.get("audience_turn"):
        lines.append(f"{AUDIENCE_LABEL}: {obs['audience_turn']}")
    question = FLOOR_QUESTION.format(a=names.get("commentator", "commentator"), b=names.get("guest", "guest"))
    user = f"Topic: {obs['topic']}\n\nTranscript so far:\n" + "\n".join(lines) + f"\n\n{question}"
    return [{"role": "system", "content": FLOOR_SYSTEM}, {"role": "user", "content": user}]


def parse_name(text: str, names: dict[str, str]) -> str | None:
    """The panel role whose name the answer gives first (None: neither, or unreadable)."""
    hits = []
    for role in PANEL:
        m = re.search(rf"\b{re.escape(names.get(role, role))}\b", text or "", re.I)
        if m:
            hits.append((m.start(), role))
    return min(hits)[1] if hits else None
