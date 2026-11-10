"""Offline Think & Action think prompts for a replayed history.

With the Think & Action floor the selected panelist speaks the ``content`` of its own think call (the panel worker only
drafts when there is no proposed reply). To measure what Think & Action panelists would say without the live stack, this
module builds the exact think call the show makes: it runs the production ``ThinkActionInteraction._think_and_act``
against a client that records the request instead of sending it. Prompt, system prompt, speaker-perspective history,
temperature and token budget are therefore those of the live show, and the dialogue quality flags (TALKSHOW_*) are read
from the environment as in the live show (the uptake planner's output is passed in, see ``plan``).

The call modelled is a panelist's reaction to a finished turn on a free floor:
- after a human turn (``react_to_human``): speaker = human, interval = the whole transcript;
- after a panel line (its final interval): speaker = that line's role, interval = the line's last chunk.

Differences from the live show, by design: one think per panelist (no earlier intervals, so no interrupt or
backchannel history), no floor controller (the caller fixes who speaks), mood = prefix state.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

PLAN_PLACEHOLDER = "- (uptake planner output for this human line)"


class _RecordingClient:
    """Stands in for GemmaMMClient: records the think request, answers the planner with ``plan``."""

    def __init__(self, plan: str) -> None:
        self.plan = plan
        self.requests: list[dict[str, Any]] = []
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def complete_text(self, prompt: str, **kw: Any) -> str:
        from agent.panel.uptake import PLANNER_SYSTEM

        if kw.get("system_prompt") == PLANNER_SYSTEM:
            return self.plan
        self.requests.append({"prompt": prompt, **kw})
        return "{}"

    def log_text_only_event(self, role: str, kind: str, **fields: Any) -> None:
        self.events.append((role, kind, fields))


def _data_for(prefix: dict, human_text: str | None, plan: str, topic_id: str | None = None) -> Any:
    from agent.config import load_scenario
    from agent.data import TalkShowData
    from agent.session.ghost_replay import apply_prefix

    data = TalkShowData(scenario=load_scenario(), runtime=None)  # type: ignore[arg-type]
    data.room_name = "think-action-offline"
    data.turn_log = None
    data.fact_pack_topic = topic_id  # read by the fact pack block (TALKSHOW_FACT_PACK)
    apply_prefix(data, {k: v for k, v in prefix.items() if k not in ("persona_swap", "flags")})
    if human_text is not None:
        data.show_history.append("human", human_text)
    data.runtime = SimpleNamespace(gemma_client=_RecordingClient(plan))  # type: ignore[assignment]
    return data


def last_interval(text: str) -> str:
    """The final interval of a disclosed line, as the live show splits it (token interval, tokenizer from env)."""
    from agent.interaction import ConsoleSpeechConsumer, ThinkActionInteraction

    ends = ThinkActionInteraction(ConsoleSpeechConsumer())._chunk_ends(text)
    start = ends[-2] if len(ends) > 1 else 0
    return text[start:]


def think_request(
    prefix: dict,
    role: str,
    *,
    human_text: str | None = None,
    plan: str = "",
    interval_text: str | None = None,
    topic_id: str | None = None,
) -> dict[str, Any]:
    """The think call ``role`` makes on a free floor after the last turn of ``prefix`` (+ ``human_text`` if given).

    Returns {"messages", "temperature", "max_tokens", "interval_text", "speaker", "data"}; ``messages`` is what
    GemmaMMClient sends (system + history, the instruction folded into a trailing user turn).
    ``plan``: the uptake planner's output for the human line (used only when TALKSHOW_UPTAKE_PLANNER is on).
    ``interval_text``: override the final interval of a panel line (default: split as the live show does).
    ``topic_id``: the fact pack topic (TALKSHOW_FACT_PACK); "fact_pack" in the result has the facts offered.
    """
    from agent.adapters.gemma_mm_client import GemmaMMClient
    from agent.config import set_persona_swap
    from agent.interaction import ConsoleSpeechConsumer, ThinkActionInteraction
    from agent.show.show_history import ShowHistory

    set_persona_swap(bool(prefix.get("persona_swap")))
    try:
        data = _data_for(prefix, human_text, plan, topic_id)
        lines = [line for line in data.show_history.lines if not line.procedural]
        if not lines:
            raise ValueError("think_request needs at least one spoken line to react to")
        speaker, visible = lines[-1].role_id, lines[-1].text
        if interval_text is None:
            interval_text = visible if speaker == "human" else last_interval(visible)
        interaction = ThinkActionInteraction(ConsoleSpeechConsumer())
        asyncio.run(
            interaction._think_and_act(
                data,
                role,
                history=ShowHistory(lines=list(data.show_history.lines)),
                speaker=speaker,
                interval_text=interval_text,
                visible_text=visible,
                checkpoint=1,
                floor_state="free",
            )
        )
    finally:
        set_persona_swap(False)
    (req,) = data.runtime.gemma_client.requests
    messages = GemmaMMClient._build_messages(
        None,  # type: ignore[arg-type]  # uses no instance state
        req["system_prompt"],
        history_messages=req["history_messages"],
        tail_user=req["prompt"],
    )
    additions = next((f for _r, kind, f in data.runtime.gemma_client.events if kind.startswith("think additions")), {})
    return {
        "fact_pack": {k: v for k, v in additions.items() if k.startswith("fact_pack")},
        "messages": messages,
        "temperature": req["temperature"],
        "max_tokens": req["max_tokens"],
        "interval_text": interval_text,
        "speaker": speaker,
        "data": data,
    }


def spoken_content(raw: str, interval_text: str) -> tuple[str, str]:
    """(decision, the words the show would speak) from a think response; text is "" unless the panelist chose speak."""
    from agent.floor.floor_parser import strip_speech_control_tags
    from agent.interaction.think_action import parse_think_action

    action = parse_think_action(raw, floor_state="free", interval_text=interval_text, allow_backchannels=True).action
    if action.decision != "speak":
        return action.decision, ""
    return action.decision, strip_speech_control_tags(action.content or "").strip()
