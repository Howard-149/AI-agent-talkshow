"""Experimental token-interval interaction used only by Text-Only sessions."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_INTERVAL = 8
DEFAULT_INTERRUPTION_THRESHOLD = 7
_WANT_RE = re.compile(r"\[want_to_speak\]\s*:\s*(true|false)\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\[(?:time_cost_s|time_cost)\]\s*:\s*(\d+)\b", re.IGNORECASE)
_URGENCY_RE = re.compile(r"\[urgency\]\s*:\s*(\d+)\b", re.IGNORECASE)
_PROPOSED_REPLY_RE = re.compile(
    r"\[proposed_reply\]\s*:\s*(.*?)(?=\n\s*\[\w+\]\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class SpeakingIntent:
    want_to_speak: bool
    time_cost_s: int
    urgency: int
    proposed_reply: str = ""
    think_id: int = 0


def token_chunk_ends(
    text: str, token_offsets: list[tuple[int, int]], interval: int
) -> list[int]:
    interval = max(1, interval)
    ends: list[int] = []
    token_count = 0
    for _, end in token_offsets:
        if end <= 0:
            continue
        token_count += 1
        if token_count % interval == 0 and (not ends or end > ends[-1]):
            ends.append(end)
    if text and (not ends or ends[-1] < len(text)):
        ends.append(len(text))
    return ends


def parse_speaking_intent(text: str) -> SpeakingIntent:
    want_match = _WANT_RE.search(text)
    time_match = _TIME_RE.search(text)
    urgency_match = _URGENCY_RE.search(text)
    reply_match = _PROPOSED_REPLY_RE.search(text)
    want_to_speak = bool(want_match and want_match.group(1).lower() == "true")
    return SpeakingIntent(
        want_to_speak=want_to_speak,
        time_cost_s=min(9, int(time_match.group(1))) if time_match else 9,
        urgency=min(9, int(urgency_match.group(1))) if urgency_match else 0,
        proposed_reply=reply_match.group(1).strip() if want_to_speak and reply_match else "",
    )


def speaking_recency_context(history: Any, role: str, current_speaker: str) -> str:
    public_turns = [line for line in history.lines if not line.procedural]
    if public_turns and public_turns[-1].role_id == current_speaker:
        public_turns.pop()

    last_spoken = next(
        (index for index in range(len(public_turns) - 1, -1, -1)
         if public_turns[index].role_id == role),
        None,
    )
    if last_spoken is None:
        return (
            "You have not spoken in the visible conversation; "
            f"{len(public_turns)} other public speaking turns have occurred."
        )

    turns_waited = len(public_turns) - last_spoken - 1
    return (
        "Since your last public speaking turn, "
        f"{turns_waited} other completed public speaking turns have occurred."
    )


class TextOnlyInteraction:
    def __init__(self) -> None:
        self.intents: dict[str, SpeakingIntent] = {}
        self.interrupted_role: str | None = None
        self._think_sequence = 0
        self._pending_thinks: list[tuple[str, SpeakingIntent]] = []
        self._pending_trigger: str | None = None
        self._pending_selection: tuple[dict[str, object], str | None, list[str]] | None = None
        self.token_interval = max(
            1, int(os.environ.get("TALKSHOW_TEXT_TOKEN_INTERVAL", DEFAULT_TOKEN_INTERVAL))
        )
        self.interruption_threshold = min(
            9,
            max(
                0,
                int(
                    os.environ.get(
                        "TALKSHOW_TEXT_INTERRUPTION_THRESHOLD",
                        DEFAULT_INTERRUPTION_THRESHOLD,
                    )
                ),
            ),
        )
        self._tokenizer: Any | None = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_path = os.environ.get("VLLM_MODEL_PATH") or os.environ.get(
                "VLLM_MODEL", "google/gemma-4-E4B-it"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        return self._tokenizer

    def _chunk_ends(self, text: str) -> list[int]:
        if os.environ.get("TALKSHOW_TEXT_TOKENIZER") == "word":
            offsets = [
                match.span()
                for match in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)
            ]
            return token_chunk_ends(text, offsets, self.token_interval)

        encoded = self._get_tokenizer()(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return token_chunk_ends(
            text, list(encoded["offset_mapping"]), self.token_interval
        )

    def select_interruption(self, speaker: str, panel_roles: list[str]) -> str | None:
        candidates = [
            (role, self.intents[role])
            for role in panel_roles
            if role != speaker
            and role in self.intents
            and self.intents[role].want_to_speak
            and self.intents[role].urgency > self.interruption_threshold
        ]
        if not candidates:
            return None
        order = {role: index for index, role in enumerate(panel_roles)}
        return min(
            candidates,
            key=lambda item: (-item[1].urgency, item[1].time_cost_s, order[item[0]]),
        )[0]

    def select_free_floor(self, panel_roles: list[str]) -> str | None:
        candidates = [
            role
            for role in panel_roles
            if role in self.intents and self.intents[role].want_to_speak
        ]
        if not candidates:
            return None
        order = {role: index for index, role in enumerate(panel_roles)}
        return min(candidates, key=lambda role: (self.intents[role].time_cost_s, order[role]))

    def log_selection(
        self,
        data: Any,
        *,
        selected_role: str | None,
        reason: str,
        floor_state: str,
        candidates: list[str],
        defer: bool = False,
        include_pending: bool = True,
        consume_trigger: bool = True,
    ) -> None:
        from agent.config import load_persona_name

        event = data.runtime.gemma_client.log_text_only_event
        trigger = self._pending_trigger if consume_trigger else None
        if trigger and consume_trigger:
            self._pending_trigger = None
        fields: dict[str, object] = {
            "floor_state": floor_state,
            "reason": reason,
            "selected_speaker": load_persona_name(selected_role)
            if selected_role and selected_role not in {"close", "human"}
            else selected_role,
            "selected_role": selected_role,
            "eligible_candidates": [
                {"speaker": load_persona_name(role), "role": role}
                for role in candidates
            ],
        }
        if trigger:
            fields["trigger"] = trigger
        if defer:
            self._pending_selection = (fields, selected_role, list(candidates))
            return
        decisions = (
            self._selection_decisions(selected_role, candidates)
            if include_pending
            else []
        )
        if include_pending:
            self._pending_thinks.clear()
        event("floor", "selection", **fields, decisions=decisions)

    def _selection_decisions(
        self, selected_role: str | None, candidates: list[str]
    ) -> list[dict[str, object]]:
        from agent.config import load_persona_name

        return [
            {
                "think_id": intent.think_id,
                "speaker": load_persona_name(role),
                "role": role,
                "latest": self.intents.get(role) is not None
                and self.intents[role].think_id == intent.think_id,
                "eligible": role in candidates and intent.want_to_speak,
                "want_to_speak": intent.want_to_speak,
                "urgency": intent.urgency,
                "time_cost_s": intent.time_cost_s,
                "outcome": (
                    "superseded"
                    if role in self.intents
                    and self.intents[role].think_id != intent.think_id
                    else "selected"
                    if role == selected_role
                    else "not_selected"
                    if intent.want_to_speak and role in candidates
                    else "not_eligible"
                    if intent.want_to_speak
                    else "silent"
                ),
            }
            for role, intent in self._pending_thinks
        ]

    def flush_pending_selection(self, data: Any) -> None:
        pending, self._pending_selection = self._pending_selection, None
        if pending is None:
            return
        fields, selected_role, candidates = pending
        decisions = self._selection_decisions(selected_role, candidates)
        self._pending_thinks.clear()
        data.runtime.gemma_client.log_text_only_event(
            "floor", "selection", **fields, decisions=decisions
        )

    @staticmethod
    def log_speech_start(
        data: Any,
        role: str,
        text: str,
    ) -> None:
        data.runtime.gemma_client.log_text_only_event(
            role, "speech", text=text.strip()
        )

    @staticmethod
    def log_topic(data: Any, text: str) -> None:
        prefix = "Today's discussion topic:"
        topic = text.removeprefix(prefix).strip() if text.startswith(prefix) else text.strip()
        data.runtime.gemma_client.log_text_only_event("topic", "topic", topic=topic)

    def log_trigger(self, data: Any, trigger_type: str) -> None:
        self._pending_trigger = trigger_type

    def clear_intent(self, role: str) -> None:
        self.intents.pop(role, None)

    def consume_proposed_reply(self, role: str) -> str:
        intent = self.intents.pop(role, None)
        if intent is None or not intent.want_to_speak:
            return ""
        return intent.proposed_reply.strip()

    def consume_interruption(self) -> str | None:
        role, self.interrupted_role = self.interrupted_role, None
        return role

    async def disclose_panel_line(
        self,
        data: Any,
        line: dict[str, Any],
        *,
        allow_interruptions: bool = True,
    ) -> None:
        from agent.config import load_persona_name
        from agent.floor.floor_parser import strip_speech_control_tags
        from agent.floor.turn_controller import TurnController
        from agent.session.session_handoff import switch_to_role

        role = line["role"]
        text = strip_speech_control_tags(line.get("text", "")).strip()
        if not text:
            return
        await switch_to_role(
            None, data, role, reason=f"panel:{line.get('step', 'text_only')}"
        )

        boundaries = self._chunk_ends(text)
        chunks: list[str] = []
        start = 0
        for end in boundaries:
            chunks.append(text[start:end])
            start = end

        name = load_persona_name(role)
        step = line.get("step", "text_only")
        self.log_speech_start(data, role, text)
        print(f"\n{name}: ", end="", flush=True)
        controller = TurnController(data.scenario, data)
        panel_roles = controller.panel_speaker_roles()
        floor_roles = [data.scenario.turn_control.listen_role, *panel_roles]
        for index, chunk in enumerate(chunks):
            print(chunk, end="", flush=True)
            visible_text = text[: boundaries[index]]
            await self._update_listeners(
                data,
                speaker=role,
                interval_text=chunk,
                visible_text=visible_text,
                panel_roles=panel_roles,
                step=step,
                checkpoint=index + 1,
            )
            if allow_interruptions and index + 1 < len(chunks):
                interrupted = self.select_interruption(role, floor_roles)
                if interrupted:
                    self._truncate_public_line(data, role, visible_text)
                    self.interrupted_role = interrupted
                    print(flush=True)
                    self.log_selection(
                        data,
                        selected_role=interrupted,
                        reason="urgency_threshold",
                        floor_state="occupied",
                        candidates=[
                            candidate
                            for candidate in floor_roles
                            if candidate != role
                            and candidate in self.intents
                            and self.intents[candidate].want_to_speak
                            and self.intents[candidate].urgency > self.interruption_threshold
                        ],
                    )
                    logger.info(
                        "text-only floor interrupted speaker=%s by=%s urgency=%d",
                        role,
                        interrupted,
                        self.intents[interrupted].urgency,
                    )
                    return
        print(flush=True)
        self.flush_pending_selection(data)

    async def _update_listeners(
        self,
        data: Any,
        *,
        speaker: str,
        interval_text: str,
        visible_text: str,
        panel_roles: list[str],
        step: str,
        checkpoint: int,
    ) -> None:
        from agent.show.show_history import ShowHistory

        history = ShowHistory(lines=list(data.show_history.lines))
        if history.lines and history.lines[-1].role_id == speaker:
            history.lines[-1] = replace(history.lines[-1], text=visible_text)
        else:
            history.append(speaker, visible_text)
        listener_roles = list(dict.fromkeys([data.scenario.turn_control.listen_role, *panel_roles]))
        listeners = [role for role in listener_roles if role != speaker]
        data.runtime.gemma_client.log_text_only_event(
            "interval",
            f"interval {checkpoint}",
            content=interval_text.strip(),
            complete_content=visible_text.strip(),
        )
        await asyncio.gather(
            *(
                self._update_intent(
                    data,
                    role,
                    history=history,
                    speaker=speaker,
                    interval_text=interval_text,
                    step=step,
                    checkpoint=checkpoint,
                    visible_text=visible_text,
                )
                for role in listeners
            )
        )

    async def _update_intent(
        self,
        data: Any,
        role: str,
        *,
        history: ShowHistory,
        speaker: str,
        interval_text: str,
        step: str,
        checkpoint: int,
        visible_text: str,
    ) -> None:
        from agent.config import load_persona_name
        from agent.panel.panel_prompts import panelist_personality_block

        previous = self.intents.get(role)
        recency = speaking_recency_context(history, role, speaker)
        prompt = (
            f"You are {load_persona_name(role)} ({role}).\n"
            f"The current floor holder is {load_persona_name(speaker)} ({speaker}).\n"
            f"The latest public interval from {speaker}:\n{interval_text}\n\n"
            f"Speaking-turn recency: {recency}\n"
            f"Your previous private intention: {previous or 'none'}.\n"
            "Update your private decision using the public conversation and this new interval.\n"
            "Apply these four turn-taking rules:\n"
            "1. If you are directly addressed and expected to answer or take the floor, "
            "set want_to_speak=true.\n"
            "2. If you are not addressed but already hold the floor, continue speaking.\n"
            "3. If you are not addressed and do not hold the floor, remain silent: "
            "want_to_speak=false.\n"
            "4. If your name or views are merely mentioned without inviting your response, "
            "remain silent: want_to_speak=false.\n"
            "A name alone is not an invitation. Agreement, disagreement, a useful addition, "
            "or a general question to the room does not override these rules. "
            "Use the visible conversation to determine whether an invitation is still unanswered; "
            "do not invent an invitation from an unfinished fragment or keep an old intention "
            "after its invitation has been answered or withdrawn.\n"
            "want_to_speak=true means a response is expected, not permission to interrupt. "
            "Normally wait until the current speaker finishes. "
            "When remaining silent, output time_cost_s=9, urgency=0 and an empty proposed_reply.\n"
            "time_cost_s is your estimate of cognitive preparation still needed before you "
            "could give a coherent response if given the floor; it is NOT elapsed or expected "
            "model-generation latency. Output an integer from 0 to 9; lower means more ready. "
            "Estimate the response's baseline preparation from its complexity, then calibrate it "
            "with speaking-turn recency. If 0-1 other public turns have occurred since you last "
            "spoke, apply no waiting-time discount; if you spoke very recently, treat this as a "
            "new contribution rather than carrying over old preparation. If 2-3 turns have passed "
            "and your want_to_speak intention and coherent reply have persisted, normally lower "
            "the baseline by about 1-2 points. If 4 or more turns have passed and you have "
            "persistently wanted to speak with a still-relevant, coherent reply, set time_cost_s "
            "to 0-1 in the usual case: you have had substantial time to prepare while listening. "
            "Do not keep assigning 3 or more solely because the reply is polished or thoughtful. "
            "Use a higher value when the discussion has materially shifted, the old reply is now "
            "stale, or the latest turn requires genuinely new synthesis; a simple relevant reply "
            "can remain ready quickly. Treat these bands as strong guidance, not arithmetic: "
            "readiness and context still matter. This is not a fairness bonus and does not by "
            "itself mean you should speak or interrupt.\n"
            "urgency is how necessary it is to intervene before the current speaker finishes, "
            "not how much you want to speak. Use an integer from 0 to 9.\n"
            "If want_to_speak is true, write the actual words you would say next in "
            "proposed_reply: a concise, coherent in-character response to the visible conversation. "
            "This is a private candidate, not a public utterance or an explanation of your reasoning. "
            "Revise it when new public information changes your response. "
            "If want_to_speak is false, leave proposed_reply empty.\n"
            "Output only these four fields:\n"
            "[want_to_speak]: true | false\n"
            "[time_cost_s]: 0-9 seconds\n"
            "[urgency]: 0-9\n"
            "[proposed_reply]: <what you would say, or empty>"
        )
        try:
            raw = await data.runtime.gemma_client.complete_text(
                prompt,
                system_prompt=panelist_personality_block(role),
                history_messages=history.prior_messages(perspective=role),
                raw=True,
                max_tokens=384,
                temperature=0.2,
                reasoning_effort="low",
            )
            parsed = parse_speaking_intent(raw)
            self._think_sequence += 1
            intent = replace(parsed, think_id=self._think_sequence)
            self.intents[role] = intent
            self._pending_thinks.append((role, intent))
            think_fields: dict[str, object] = {
                "think_id": intent.think_id,
                "want_to_speak": intent.want_to_speak,
                "urgency": intent.urgency,
                "time_cost_s": intent.time_cost_s,
                "proposed_reply": intent.proposed_reply or None,
            }
            data.runtime.gemma_client.log_text_only_event(
                role, f"draft - interval {checkpoint}", **think_fields
            )
        except Exception:
            logger.exception("text-only intention update failed role=%s", role)

    @staticmethod
    def _truncate_public_line(data: Any, role: str, visible_text: str) -> None:
        lines = data.show_history.lines
        if lines and lines[-1].role_id == role:
            lines[-1].text = visible_text.strip()
        else:
            data.show_history.append(role, visible_text)