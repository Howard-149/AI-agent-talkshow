"""Think & Action floor interaction — the text generator of the show.

Every spoken agent line is disclosed to the other agents interval by interval.
After each interval every listener makes one Think & Action call (private think,
floor decision, candidate content, backchannels) and a central controller
decides: continue / approve one interrupt while the floor is occupied, or pick
a submitted speaker / direct the host once it is free. This runs at model speed,
ahead of playback: the words are delivered later by a ``SpeechConsumer``
(console or TTS + avatar video) from the queued script.

A human can take the floor at any time: ``human_started_speaking`` stops
delivery and records where the audience was; everything generated past that
point (lines, backchannels, think & decisions, mood) is rolled back, and
``react_to_human`` runs the same think & decision on the finished transcript.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from agent.interaction.consumers import (
    ConsoleSpeechConsumer,
    Cue,
    Position,
    SpeechConsumer,
    _char_end_after_words,
)
from agent.interaction.backchannel_controller import (
    BackchannelCandidate,
    BackchannelController,
)
from agent.interaction.think_action import (
    HOST_STRATEGIES,
    Action,
    Backchannel,
    FloorOutcome,
    _json_object,
    parse_think_action,
)
from agent.session.text_only_interaction import TextOnlyInteraction

logger = logging.getLogger(__name__)


def _speaker_name(role: str) -> str:
    from agent.config import load_persona_name

    return "Human guest" if role == "human" else load_persona_name(role)


def _index_of(lines: list[Any], line: Any) -> int | None:
    return next((i for i, candidate in enumerate(lines) if candidate is line), None)


@dataclass(eq=False)
class _Mark:
    """Generator checkpoint around one cue, for rolling back unheard script."""

    line: Any  # the cue's history line (None: not recorded in history)
    anchor: Any  # last history line when the cue was queued
    prev: Cue | None
    start: dict[str, Any]
    end: dict[str, Any] | None = None
    backchannels: list[tuple[int, Any]] = field(default_factory=list)  # (char offset, line)


# Interaction fields that are configuration or barge-in bookkeeping, not show state.
_NOT_CHECKPOINTED = frozenset(
    {
        "consumer",
        "human_floor_pending",
        "_barge_in_cut_beat",
        "_rollback_to",
        "_rethink",
        "_last_cue",
        "_human_speaking",
        "_human_turns_waiting",
        "_release_handle",
        "_tokenizer",
        "_think_sequence",
        "token_interval",
        "interruption_threshold",
    }
)
_EMOTION_FIELDS = ("role_emotion", "role_pad", "role_pad_ts")
# Cues kept reachable for rollback (each links to the previous one).
_CUE_HISTORY = 16
DEFAULT_BARGE_IN_RELEASE_S = 12.0


DEFAULT_MAX_LOOKAHEAD_TURNS = 3


def max_lookahead_turns() -> int:
    """Speaker lines the script may run ahead of playback (backchannels don't count; 0 = no cap)."""
    try:
        return max(0, int(os.environ.get("TALKSHOW_MAX_LOOKAHEAD_TURNS", DEFAULT_MAX_LOOKAHEAD_TURNS)))
    except ValueError:
        return DEFAULT_MAX_LOOKAHEAD_TURNS


def human_barge_in_enabled() -> bool:
    return os.environ.get("TALKSHOW_HUMAN_BARGE_IN", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


# Occupied-floor controller: reason codes allowed with a "continue" decision.
_OCCUPIED_CONTINUE_CODES = frozenset(
    {"no_candidates", "insufficient_immediate_value", "low_contribution", "poor_timing"}
)


class ThinkActionInteraction(TextOnlyInteraction):
    """Interval-disclosed Think & Action floor control (generator side)."""

    def __init__(self, consumer: SpeechConsumer | None = None) -> None:
        super().__init__()
        self.consumer: SpeechConsumer = consumer or ConsoleSpeechConsumer()
        self.public_utterance_count = 0
        self.actions: dict[str, Action] = {}
        self.thoughts: dict[str, str] = {}
        self._interval_backchannels: list[tuple[str, Backchannel]] = []
        self.backchannel_controller = BackchannelController()
        self._speech_line: Any = None
        self._final_interval = False
        self.last_floor_outcome: FloorOutcome | None = None
        self.host_fallback_instruction: str | None = None
        self.host_fallback_strategy: str | None = None
        self.host_fallback_target: str | None = None
        self.host_fallback_close_spoken = False
        self.discussion_complete = False
        self.pending_invitation_target: str | None = None
        # Panelist the host's opening invited; the first free-floor pick goes to them.
        self.opening_invite_role: str | None = None
        # The last routed line was cut by an approved interrupt (the interrupter is next).
        self.last_line_interrupted = False
        # False while a line must finish uncut (the grace turn at the turn cap).
        self.interrupts_allowed = True
        self.actual_floor_counts: dict[str, int] = {}
        self.fallback_history: list[dict[str, str | None]] = []
        self.host_fallback_streak = 0
        # Set when a human starts speaking; cleared once the transcript is handled.
        self.human_floor_pending = False
        self._barge_in_cut_beat = False
        # Where the audience was when the human barged in; applied at the next safe point.
        self._rollback_to: Position | None = None
        # A cut line to re-think on (speaker, heard prefix) when the show resumes.
        self._rethink: tuple[str, str] | None = None
        self._last_cue: Cue | None = None
        # VAD says the human is talking right now.
        self._human_speaking = False
        # Human transcripts handed to the show but not yet reacted to.
        self._human_turns_waiting = 0
        self._release_handle: asyncio.TimerHandle | None = None

    @staticmethod
    def log_speech_start(data: Any, role: str, text: str) -> None:
        TextOnlyInteraction.log_speech_start(data, role, text)
        interaction = getattr(data, "floor_interaction", None)
        if isinstance(interaction, ThinkActionInteraction):
            interaction.public_utterance_count += 1

    def select_interruption(self, speaker: str, panel_roles: list[str]) -> str | None:
        candidates = [
            role
            for role in panel_roles
            if role != speaker
            and self.actions.get(role) is not None
            and self.actions[role].decision == "interrupt"
        ]
        return candidates[0] if candidates else None

    def select_free_floor(self, panel_roles: list[str]) -> str | None:
        return next(
            (
                role
                for role in panel_roles
                if (action := self.actions.get(role)) is not None
                and action.decision == "speak"
            ),
            None,
        )

    def consume_proposed_reply(self, role: str) -> str:
        action = self.actions.pop(role, None)
        self.intents.pop(role, None)
        if action is None or action.decision not in {"speak", "interrupt"}:
            return ""
        return action.content or ""

    def record_floor_grant(self, role: str) -> None:
        if role not in {"human", "close"}:
            self.actual_floor_counts[role] = self.actual_floor_counts.get(role, 0) + 1
        if role == self.host_fallback_target:
            self.host_fallback_target = None

    async def arbitrate_occupied(
        self,
        data: Any,
        *,
        speaker: str,
        visible_text: str,
        candidates: list[str],
        interval: int,
        history: Any | None = None,
    ) -> str | None:
        """Ask the controller at an occupied interval boundary when someone requested an
        interrupt; with no request the speaker continues without a model call."""
        from agent.config import load_persona_name

        self.last_floor_outcome = None
        candidate_payload = [
            {
                "role": role,
                "speaker": load_persona_name(role),
                "intent": self._action_intent(self.actions[role]),
                "content": self.actions[role].content,
            }
            for role in candidates
            if role in self.actions and self.actions[role].decision == "interrupt"
        ]
        if not candidate_payload:
            data.runtime.gemma_client.log_text_only_event(
                "floor",
                f"floor control - interval {interval}",
                floor_state="occupied",
                decision="continue",
                selected_role=None,
                reason_code="no_candidates",
                reason="no interrupt candidates",
                candidates=[],
            )
            return None
        # Read the visible-prefix copy, not the source history that can still
        # contain the complete scripted utterance.
        public_context = self._recent_public_text(history, max_lines=12)
        prompt = (
            "You are the centralized floor controller for a talk-show discussion. "
            "The floor is occupied; decide whether to continue the current speaker or "
            "approve exactly one requested interrupt.\n\n"
            "Favor meaningful, spontaneous interaction without encouraging unnecessary "
            "interruptions. An interrupt can be worthwhile when intervening now adds "
            "conversational value—such as a timely challenge, clarification, correction, "
            "witty pushback, disagreement, or natural spontaneous exchange—that outweighs "
            "the disruption.\n\n"
            "Do not require an emergency, factual error, or proof that the contribution "
            "would become irrelevant if delayed. A speaker still holding the floor is not "
            "itself grounds for rejecting a request.\n\n"
            "Consider:\n"
            "- Contribution: Does the candidate add something specific, relevant, and worthwhile?\n"
            "- Timing: Is the intervention responsive to what has just been said?\n"
            "- Interaction value: Would intervening now improve the exchange?\n"
            "- Disruption: Would it unnecessarily interrupt an incomplete thought or "
            "important explanation?\n\n"
            "Prefer natural phrase boundaries. Avoid interrupting incomplete clauses, first "
            "words, or directed invitations unless there is a compelling conversational "
            "reason. Give developing substantive explanations reasonable protection, but "
            "not automatic immunity.\n\n"
            "Mere relevance, aggression, repeated objections, or routine demands for "
            "specificity do not suffice. A single candidate is not automatically approved.\n\n"
            "Compare candidates against the recent public conversation. Do not infer future "
            "speech. Continue if no candidate offers sufficient immediate conversational "
            "value. Do not rewrite candidate content.\n\n"
            f"Recent public conversation:\n{public_context}\n\n"
            f"Current speaker: {_speaker_name(speaker)} ({speaker})\n\n"
            f"Current speech prefix through interval {interval}:\n{visible_text}\n\n"
            f"Interrupt candidates:\n{json.dumps(candidate_payload, ensure_ascii=False)}\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "decision": "continue|interrupt",\n'
            '  "selected_role": "exact candidate role ID or null",\n'
            '  "reason_code": "no_candidates|insufficient_immediate_value|low_contribution|'
            'poor_timing|timely_intervention",\n'
            '  "reason": "brief rationale"\n'
            "}\n\n"
            "For continue, selected_role must be null.\n"
            "For interrupt, selected_role must match exactly one candidate role ID and "
            "reason_code must be timely_intervention."
        )
        raw = await data.runtime.gemma_client.complete_text(
            prompt,
            system_prompt="You arbitrate floor access neutrally, balancing natural interaction against disruption.",
            history_messages=[],
            raw=True,
            max_tokens=192,
            temperature=0.0,
            reasoning_effort="low",
        )
        result = _json_object(raw)
        decision = result.get("decision") if result else None
        selected = result.get("selected_role") if result else None
        reason = result.get("reason") if result else ""
        reason_code = result.get("reason_code") if result else None
        valid_candidate_roles = {candidate["role"] for candidate in candidate_payload}
        if decision != "interrupt" or selected not in valid_candidate_roles:
            decision = "continue"
            selected = None
        # A rejected interrupt cannot be labeled as an approved intervention.
        if decision == "interrupt":
            reason_code = "timely_intervention"
        elif reason_code not in _OCCUPIED_CONTINUE_CODES:
            reason_code = "insufficient_immediate_value"
        data.runtime.gemma_client.log_text_only_event(
            "floor",
            f"floor control - interval {interval}",
            floor_state="occupied",
            decision=decision,
            selected_role=selected,
            reason_code=reason_code,
            reason=reason if isinstance(reason, str) else "",
            candidates=candidate_payload,
        )
        if decision == "continue":
            return None
        self.last_floor_outcome = FloorOutcome(
            "interrupt",
            selected_role=selected,
            reason=reason if isinstance(reason, str) else "llm_interrupt",
            reason_code=reason_code if isinstance(reason_code, str) else "timely_intervention",
        )
        return selected

    async def arbitrate_free(
        self,
        data: Any,
        *,
        candidates: list[str],
        history: Any,
        interval: int,
    ) -> FloorOutcome:
        """Select a submitted speaker or direct the Host without inventing speech."""
        from agent.config import load_persona_name

        host_role = data.scenario.turn_control.listen_role
        candidate_payload = [
            {
                "role": role,
                "speaker": load_persona_name(role),
                "intent": self._action_intent(self.actions[role]),
                "content": self.actions[role].content,
                "actual_floor_turns": self.actual_floor_counts.get(role, 0),
            }
            for role in candidates
            if role in self.actions and self.actions[role].decision == "speak"
        ]
        candidate_roles = {item["role"] for item in candidate_payload}
        directed = self._addressee_pick(data, history, candidate_payload, interval)
        if directed is not None:
            return directed
        invited_role = self.pending_invitation_target
        recent_fallbacks = self.fallback_history[-4:]
        participant_counts = {
            role: self.actual_floor_counts.get(role, 0)
            for role in dict.fromkeys([host_role, *data.scenario.turn_control.order])
        }
        prompt = (
            "You are the centralized floor controller for a talk show. The current "
            "speech has ended and the floor is FREE. Choose a submitted speak candidate "
            "that will advance the discussion. Use host_fallback ONLY if nobody applied "
            "or ALL applicants are truly unsuitable (off topic, vacuous or merely "
            "repeating earlier lines). Do not invent candidates or write their speech. "
            "A relevant answer to a question or invitation ALREADY SPOKEN in the public "
            "transcript should normally be chosen; use directed_response only for this "
            "or an actually delivered pending fallback invitation. A host's unselected "
            "candidate invitation has NOT happened and creates no obligation to answer. "
            "A useful host facilitation turn can be valuable without new factual claims: "
            "connecting ideas, giving the audience structure, inviting a specific speaker, "
            "moving to an unexamined angle, or bringing a repeating exchange to a close. "
            "Do not systematically exclude the host in favor of another panelist's "
            "small incremental contribution; equally, do not let the host monopolize. "
            "Weigh relevance, actual progress, role fit and actual floor participation, "
            "without strict round robin. If fallback is needed, issue "
            "a short HOST INSTRUCTION, not final dialogue. Vary the approach among "
            "invite, summarize, contribute, reframe and close; avoid unanswered repeat "
            "invitations and circular questioning. Choose close if the topic is exhausted "
            "but not merely because one round was quiet. Never force a noncandidate "
            "to speak. selected_role and target MUST be exact machine role IDs.\n"
            f"Public transcript:\n{self._recent_public_text(history)}\n"
            f"Speak candidates: {json.dumps(candidate_payload, ensure_ascii=False)}\n"
            f"Pending host invitation target (role ID): {invited_role!r}\n"
            f"Actual floor grants: {json.dumps(participant_counts, ensure_ascii=False)}\n"
            f"Consecutive fallbacks: {self.host_fallback_streak}\n"
            f"Recent fallback history: {json.dumps(recent_fallbacks, ensure_ascii=False)}\n"
            'Return one JSON object ONLY. For speaker selection: '
            '{"decision":"speak", "selected_role":"exact candidate role ID", '
            '"reason_code":"directed_response|high_contribution|participation_balance", '
            '"reason":"short explanation"}. For host fallback: '
            '{"decision":"host_fallback", "selected_role":"' + host_role + '", '
            '"reason_code":"no_suitable_candidate|discussion_complete", '
            '"reason":"short explanation", "host_fallback":'
            '{"strategy":"invite|summarize|contribute|reframe|close", '
            '"target":null, "instruction":"short instruction"}}. '
            'When strategy=invite, target must be an actual panel role ID instead of null.'
        )
        raw = await data.runtime.gemma_client.complete_text(
            prompt,
            system_prompt="You allocate the floor based on public evidence and real conversational obligations.",
            history_messages=[],
            raw=True,
            max_tokens=320,
            temperature=0.0,
            reasoning_effort="low",
        )
        result = _json_object(raw) or {}
        decision = result.get("decision")
        selected = result.get("selected_role")
        reason = result.get("reason")
        reason_code = result.get("reason_code")
        proposed_fallback = result.get("host_fallback")
        valid_explicit_fallback = (
            decision == "host_fallback"
            and reason_code in {"no_suitable_candidate", "discussion_complete"}
            and isinstance(proposed_fallback, dict)
            and proposed_fallback.get("strategy") in HOST_STRATEGIES
            and isinstance(proposed_fallback.get("instruction"), str)
            and bool(proposed_fallback["instruction"].strip())
        )
        if decision == "speak" and selected in candidate_roles:
            outcome = FloorOutcome(
                "speak",
                selected_role=selected,
                reason=reason if isinstance(reason, str) else "Selected a submitted candidate.",
                reason_code=reason_code if isinstance(reason_code, str) else "high_contribution",
            )
            self.host_fallback_streak = 0
            self.host_fallback_instruction = None
            self.host_fallback_strategy = None
            self.host_fallback_target = None
        elif candidate_roles and not valid_explicit_fallback:
            # Malformed model output is not evidence that every candidate is unsuitable.
            # Prefer the invited speaker, then the least-served candidate.
            order = {item["role"]: i for i, item in enumerate(candidate_payload)}
            selected = min(
                candidate_roles,
                key=lambda role: (
                    role != invited_role,
                    self.actual_floor_counts.get(role, 0),
                    order[role],
                ),
            )
            outcome = FloorOutcome(
                "speak",
                selected_role=selected,
                reason="Invalid floor selection; chose an existing applicant deterministically.",
                reason_code="invalid_output_fallback",
            )
            self.host_fallback_streak = 0
            self.host_fallback_instruction = None
            self.host_fallback_strategy = None
            self.host_fallback_target = None
        else:
            fallback = result.get("host_fallback")
            if not isinstance(fallback, dict):
                fallback = {}
            strategy = fallback.get("strategy")
            instruction = fallback.get("instruction")
            target = fallback.get("target")
            panel_roles = {
                role for role in data.scenario.turn_control.order
                if role not in {host_role, "human"}
            }
            if strategy not in HOST_STRATEGIES or not isinstance(instruction, str) or not instruction.strip():
                strategy = "reframe"
                instruction = "Offer a concrete unresolved issue or example to move the conversation forward."
                target = None
            if target not in panel_roles:
                target = None
            if strategy == "invite" and target is None:
                strategy = "reframe"
                instruction = "Introduce a new concrete angle rather than repeating an unanswered invitation."
            outcome = FloorOutcome(
                "host_fallback",
                selected_role=host_role,
                reason=reason if isinstance(reason, str) else "No suitable applicant.",
                reason_code=reason_code if isinstance(reason_code, str) else "no_suitable_candidate",
                host_strategy=strategy,
                host_target=target,
                host_instruction=instruction.strip(),
            )
            self.host_fallback_streak += 1
            # Hard deadlock guard; normally the controller decides when to conclude.
            if self.host_fallback_streak >= 4:
                outcome = FloorOutcome(
                    "host_fallback",
                    selected_role=host_role,
                    reason="Repeated host fallbacks without any voluntary speaker.",
                    reason_code="discussion_complete",
                    host_strategy="close",
                    host_target=None,
                    host_instruction=(
                        "Conclude the discussion now: briefly synthesize the positions "
                        "and unresolved tension, then deliver a natural closing statement."
                    ),
                )
            self.host_fallback_instruction = outcome.host_instruction
            self.host_fallback_strategy = outcome.host_strategy
            self.host_fallback_target = outcome.host_target
            self.fallback_history.append(
                {
                    "strategy": outcome.host_strategy,
                    "target": outcome.host_target,
                    "instruction": outcome.host_instruction,
                }
            )
        # A targeted invitation is a preference for the next choice, never a
        # permission to force a silent guest onto the floor.
        self.pending_invitation_target = None
        self.last_floor_outcome = outcome
        data.runtime.gemma_client.log_text_only_event(
            "floor",
            f"floor control - interval {interval}",
            floor_state="free",
            decision=outcome.decision,
            selected_role=outcome.selected_role,
            reason=outcome.reason,
            reason_code=outcome.reason_code,
            host_fallback=(
                {
                    "strategy": outcome.host_strategy,
                    "target": outcome.host_target,
                    "instruction": outcome.host_instruction,
                }
                if outcome.decision == "host_fallback" else None
            ),
            candidates=candidate_payload,
        )
        return outcome

    def _addressee_pick(
        self,
        data: Any,
        history: Any,
        candidate_payload: list[dict[str, Any]],
        interval: int,
    ) -> FloorOutcome | None:
        """TALKSHOW_ADDRESSEE_FLOOR (default off): the panelist the human just named takes the
        free floor without a controller call, if their own decision was "speak". A named
        panelist who chose silent is not forced to speak: the controller decides as usual."""
        from agent.floor.addressee import addressed_after_human, addressee_floor_enabled

        if not addressee_floor_enabled() or history is None:
            return None
        host_role = data.scenario.turn_control.listen_role
        panel_roles = [role for role in self._floor_roles(data) if role not in {host_role, "human"}]
        addressed = addressed_after_human(history.lines, panel_roles, host_role=host_role)
        if addressed is None:
            return None
        action = self.actions.get(addressed)
        picked = any(item["role"] == addressed for item in candidate_payload)
        data.runtime.gemma_client.log_text_only_event(
            "floor",
            "addressee",
            addressed_role=addressed,
            decision=action.decision if action is not None else None,
            picked=picked,
        )
        if not picked:
            logger.info("addressee floor: %s was named but did not choose speak", addressed)
            return None
        outcome = FloorOutcome(
            "speak",
            selected_role=addressed,
            reason="The human guest addressed this panelist by name.",
            reason_code="directed_response",
        )
        self.host_fallback_streak = 0
        self.host_fallback_instruction = None
        self.host_fallback_strategy = None
        self.host_fallback_target = None
        self.pending_invitation_target = None
        self.last_floor_outcome = outcome
        data.runtime.gemma_client.log_text_only_event(
            "floor",
            f"floor control - interval {interval}",
            floor_state="free",
            decision=outcome.decision,
            selected_role=outcome.selected_role,
            reason=outcome.reason,
            reason_code=outcome.reason_code,
            host_fallback=None,
            candidates=candidate_payload,
        )
        return outcome

    @staticmethod
    def _action_intent(action: Action) -> dict[str, str | None] | None:
        if action.intent is None:
            return None
        return {
            "type": action.intent.type,
            "goal": action.intent.goal,
            "target": action.intent.target,
        }

    @staticmethod
    def _recent_public_text(history: Any, max_lines: int = 16) -> str:
        if history is None:
            return ""
        return "\n".join(
            f"{line.speaker}: {line.text}"
            for line in history.lines[-max_lines:]
            if not line.procedural
        )

    @staticmethod
    def _backchannel_segment_at(text: str, offset: int) -> int | None:
        """Index of the TTS sentence that ``offset`` falls in (its end counts as inside)."""
        from agent.session.speak_chunks import split_speak_sentences

        search_from = 0
        index = None
        for index, segment in enumerate(split_speak_sentences(text)):
            start = text.find(segment, search_from)
            if start < 0:
                return None
            if offset <= start + len(segment.rstrip()):
                return index
            search_from = start + len(segment)
        return index


    @staticmethod
    def _floor_roles(data: Any) -> list[str]:
        from agent.floor.turn_controller import TurnController

        return [
            data.scenario.turn_control.listen_role,
            *TurnController(data.scenario, data).panel_speaker_roles(),
        ]

    # --- delivery: generator → consumer ----------------------------------
    #
    # The generator never waits for playback: each line is decided interval by
    # interval at model speed and queued as a Cue. Checkpoints around every cue
    # let a human barge-in discard whatever was written past what was heard.

    async def play_line(
        self,
        data: Any,
        role: str,
        text: str,
        *,
        step: str,
        texts: dict[str, str] | None = None,
        canned: bool = False,
        on_start: list[Callable[[], Awaitable[None]]] | None = None,
    ) -> None:
        """A whole line nobody reacts to interval by interval (canned host lines)."""
        await self.wait_for_lookahead()
        if self.human_floor_pending:
            return
        cue = self._new_cue(
            data, role, text, step, texts=texts, whole=True, canned=canned, on_start=on_start
        )
        await self.consumer.submit(data, cue)
        self.consumer.close(cue, len(text))
        self.checkpoint_line_end(data)

    async def disclose_panel_line(
        self,
        data: Any,
        line: dict[str, Any],
        *,
        allow_interruptions: bool = True,
        on_start: list[Callable[[], Awaitable[None]]] | None = None,
    ) -> None:
        from agent.floor.floor_parser import strip_speech_control_tags

        role = line["role"]
        text = strip_speech_control_tags(line.get("text", "")).strip()
        await self.wait_for_lookahead()
        if not text or self.human_floor_pending:
            return
        boundaries = self._chunk_ends(text)
        step = line.get("step", "final")
        floor_roles = self._floor_roles(data)
        cue = self._new_cue(data, role, text, step, on_start=on_start)
        self._speech_line = cue.mark.line
        self.log_speech_start(data, role, text)
        await self.consumer.submit(data, cue)

        start = 0
        for index, end in enumerate(boundaries):
            chunk = text[start:end]
            self._final_interval = index + 1 == len(boundaries)
            if self.human_floor_pending:
                return  # the barge-in rollback discards this line past what was heard
            if self.host_fallback_close_spoken and role == data.scenario.turn_control.listen_role:
                data.runtime.gemma_client.log_text_only_event(
                    "interval",
                    f"closing interval {index + 1}",
                    content=chunk.strip(),
                    complete_content=text[:end].strip(),
                )
                self.consumer.disclose(cue, end, chunk, [])
                start = end
                continue
            boundary_result = await self._update_listeners(
                data,
                cue=cue,
                speaker=role,
                interval_text=chunk,
                visible_text=text[:end],
                panel_roles=floor_roles[1:],
                step=step,
                checkpoint=index + 1,
            )
            if self.human_floor_pending:
                return
            self._emit_interval(data, cue, chunk, start, end)
            start = end

            if isinstance(boundary_result, str):
                # Approved interrupt: the script cuts the line here.
                self._truncate_speech_history(data, text[:end])
                cue.interrupted_by = boundary_result
                self.consumer.close(cue, end)
                self.interrupted_role = boundary_result
                candidates = [
                    candidate
                    for candidate in floor_roles
                    if candidate != role
                    and (action := self.actions.get(candidate)) is not None
                    and action.decision == "interrupt"
                ]
                self.log_selection(
                    data,
                    selected_role=boundary_result,
                    reason="llm_interrupt",
                    floor_state="occupied",
                    candidates=candidates,
                )
                self._speech_line = None
                self.checkpoint_line_end(data)
                return
            if isinstance(boundary_result, FloorOutcome):
                candidates = [
                    candidate
                    for candidate in floor_roles
                    if (action := self.actions.get(candidate)) is not None
                    and action.decision == "speak"
                ]
                self.log_selection(
                    data,
                    selected_role=boundary_result.selected_role,
                    reason=boundary_result.decision,
                    floor_state="free",
                    candidates=candidates,
                    defer=True,
                )

        self.consumer.close(cue, len(text))
        self.flush_pending_selection(data)
        self._speech_line = None
        self.checkpoint_line_end(data)

    async def wait_for_lookahead(self) -> None:
        """Hold the generator while ``max_lookahead_turns`` lines already wait for playback
        (returns at once when a human takes the floor)."""
        limit = max_lookahead_turns()
        while limit and self.consumer.ahead() >= limit and not self.human_floor_pending:
            await asyncio.sleep(0.05)

    def _new_cue(
        self,
        data: Any,
        role: str,
        text: str,
        step: str,
        *,
        texts: dict[str, str] | None = None,
        whole: bool = False,
        canned: bool = False,
        on_start: list[Callable[[], Awaitable[None]]] | None = None,
    ) -> Cue:
        from agent.floor.floor_parser import strip_speech_control_tags

        lines = data.show_history.lines
        last = lines[-1] if lines else None
        own = (
            last
            if last is not None
            and last.role_id == role
            and strip_speech_control_tags(last.text).strip() == text.strip()
            else None
        )
        cue = Cue(
            role, text, step, texts=texts, whole=whole, canned=canned,
            on_start=list(on_start or []),
        )
        cue.mark = _Mark(line=own, anchor=last, prev=self._last_cue, start=self._checkpoint(data))
        self._last_cue = cue
        older = cue
        for _ in range(_CUE_HISTORY):
            if older.mark is None or older.mark.prev is None:
                break
            older = older.mark.prev
        else:
            older.mark.prev = None  # bound the chain: nothing rolls back that far
        return cue

    def _emit_interval(self, data: Any, cue: Cue, chunk: str, start: int, end: int) -> None:
        events = []
        for listener_role, event in self._interval_backchannels:
            offset = start + _char_end_after_words(chunk, event.after_word_index)
            if not event.content.strip():
                continue
            data.show_history.append(listener_role, event.content)
            cue.mark.backchannels.append((offset, data.show_history.lines[-1]))
            events.append((listener_role, event, offset))
            data.runtime.gemma_client.log_text_only_event(
                listener_role,
                "backchannel",
                intent={
                    "type": event.intent.type,
                    "goal": event.intent.goal,
                    "target": event.intent.target,
                },
                content=event.content,
                after_word_index=event.after_word_index,
            )
        self.consumer.disclose(cue, end, chunk, events)

    # --- rollback -----------------------------------------------------------

    def _checkpoint(self, data: Any) -> dict[str, Any]:
        def copied(value: Any) -> Any:
            return value.copy() if isinstance(value, (dict, list, set)) else value

        return {
            "interaction": {
                key: copied(value)
                for key, value in vars(self).items()
                if key not in _NOT_CHECKPOINTED
            },
            "data": {
                name: copied(getattr(data, name))
                for name in (*_EMOTION_FIELDS, "floor_next_speaker")
                if hasattr(data, name)
            },
        }

    def _restore(self, data: Any, checkpoint: dict[str, Any]) -> None:
        def copied(value: Any) -> Any:
            return value.copy() if isinstance(value, (dict, list, set)) else value

        for key, value in checkpoint["interaction"].items():
            setattr(self, key, copied(value))
        for name, value in checkpoint["data"].items():
            setattr(data, name, copied(value))

    def latest_line_text(self, role: str) -> str | None:
        """What the script kept of ``role``'s latest line (it may have been cut)."""
        cue = self._last_cue
        if cue is None or cue.role != role or cue.mark is None or cue.mark.line is None:
            return None
        return cue.mark.line.text

    @property
    def latest_cue(self) -> Cue | None:
        return self._last_cue

    def checkpoint_line_end(self, data: Any, *, cue: Cue | None = None) -> None:
        """Re-take the state after the latest cue (e.g. once listeners appraised it).

        With ``cue``, only while that cue is still the latest (a background appraisal
        must not overwrite the checkpoint of a line queued after it)."""
        if cue is not None and cue is not self._last_cue:
            return
        if self._last_cue is not None and self._last_cue.mark is not None:
            self._last_cue.mark.end = self._checkpoint(data)

    def apply_pending_rollback(self, data: Any) -> None:
        """Discard script written past the barge-in position. Call only while the
        generator is idle (between show beats), so nothing writes concurrently."""
        position, self._rollback_to = self._rollback_to, None
        if position is None:
            return
        cue, heard = position
        while heard <= 0 and cue.mark is not None and cue.mark.prev is not None:
            # Nothing of this cue was heard: the audience is at the end of the previous one.
            cue = cue.mark.prev
            heard = cue.end if cue.end is not None else len(cue.text)
        mark = cue.mark
        if mark is None:
            return
        end = cue.end if cue.end is not None else len(cue.text)
        full = heard >= end
        prefix = cue.text[:heard].strip()
        lines = data.show_history.lines
        before = len(lines)
        own = _index_of(lines, mark.line) if mark.line is not None else None
        if own is not None:
            kept = lines[:own]
            if full or prefix:
                if not full:
                    mark.line.text = prefix
                kept.append(mark.line)
            kept += [line for offset, line in mark.backchannels if full or offset <= heard]
        elif mark.anchor is None:
            kept = []
        else:
            anchor = _index_of(lines, mark.anchor)
            kept = lines if anchor is None else lines[: anchor + 1]
        data.show_history.lines = kept

        self._restore(data, mark.end if full and mark.end is not None else mark.start)
        self._last_cue = cue if (full or prefix) else mark.prev
        decided = full and mark.end is not None
        # Resuming re-thinks on a line that was cut, or heard before its listeners decided.
        self._rethink = None if decided or cue.whole or not prefix else (cue.role, prefix)
        data.runtime.gemma_client.log_text_only_event(
            "rollback",
            "human barge-in",
            speaker=cue.role,
            heard=prefix if not full else cue.text[:end].strip(),
            complete=full,
            dropped_lines=before - len(kept),
        )

    # --- floor: what the show graph asks between lines -------------------

    async def select_free_floor_outcome(self, data: Any, *, interval: int) -> FloorOutcome:
        """Who speaks when the floor is free (reuses the end-of-line arbitration)."""
        if self._rethink is not None:
            # Resuming after a barge-in that cut a line: think on what was heard of it.
            (speaker, heard), self._rethink = self._rethink, None
            return await self._react_free(data, speaker, heard)
        interrupted = self.consume_interruption()
        if interrupted is not None:
            # Resuming right after a line the script cut for an approved interrupt.
            self.last_floor_outcome = None
            return FloorOutcome("interrupt", selected_role=interrupted)
        floor_roles = self._floor_roles(data)
        candidates = [
            role
            for role in floor_roles
            if (action := self.actions.get(role)) is not None and action.decision == "speak"
        ]
        outcome = self.last_floor_outcome
        if outcome is None:
            outcome = await self.arbitrate_free(
                data, candidates=candidates, history=data.show_history, interval=interval
            )
        self.log_selection(
            data,
            selected_role=outcome.selected_role,
            reason=outcome.decision,
            floor_state="free",
            candidates=candidates,
        )
        return outcome

    def next_role_after_line(self, data: Any, line: dict[str, Any]) -> str:
        """Floor routing once a disclosed line has been delivered (or cut)."""
        listen_role = data.scenario.turn_control.listen_role
        interrupted = self.consume_interruption()
        self.last_line_interrupted = interrupted is not None
        if interrupted is not None:
            next_role = interrupted
        elif self.host_fallback_close_spoken and line.get("role") == listen_role:
            # Only close AFTER the host's final speech has been delivered.
            next_role = "close"
            self.discussion_complete = True
        else:
            outcome = self.last_floor_outcome
            if outcome is not None:
                next_role = outcome.selected_role or listen_role
            else:
                next_role = line.get("next_tag") or "host"
        self.last_floor_outcome = None
        return next_role

    async def host_fallback_line(self, data: Any, role: str) -> str | None:
        """The host's own turn when the controller directed it (None: not directed)."""
        from agent.config import load_persona_instructions, load_persona_name
        from agent.floor.floor_parser import spoken_reply_text

        if role != data.scenario.turn_control.listen_role or not self.host_fallback_instruction:
            return None
        instruction = self.host_fallback_instruction
        strategy = self.host_fallback_strategy or "reframe"
        target = self.host_fallback_target
        target_context = f"Invite role {target} if appropriate.\n" if target else ""
        prompt = (
            f"Host fallback strategy: {strategy}.\n"
            f"Controller instruction: {instruction}\n"
            f"{target_context}"
            "Use the public conversation to deliver a natural host turn that follows "
            "this instruction. Do not mention the controller or fallback. Output only "
            "the spoken words, with no labels or control tags."
        )
        raw = await data.runtime.gemma_client.complete_text(
            prompt,
            system_prompt=load_persona_instructions("host", panel_mode=True),
            history_messages=data.show_history.prior_messages(perspective=role),
            raw=True,
            max_tokens=384,
            temperature=0.45,
            reasoning_effort="none",
        )
        text = spoken_reply_text(raw)
        text = re.sub(
            r"\bHost fallback strategy:\s*[a-z_]+\b[.:]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if not text:
            if strategy == "invite" and target:
                text = f"{load_persona_name(target)}, would you like to share your perspective?"
            elif strategy == "close":
                text = "We've covered some important tensions today. Thanks to everyone for joining the discussion."
            else:
                text = "Let's look at this from another angle and consider what matters most in practice."
        if strategy == "invite" and target in self._floor_roles(data)[1:]:
            self.pending_invitation_target = target
        elif strategy == "close":
            self.host_fallback_close_spoken = True
        self.host_fallback_instruction = None
        self.host_fallback_strategy = None
        self.host_fallback_target = None
        return text

    def take_close_line_override(self) -> bool:
        """True when the host's closing speech already ended the show (skip canned close)."""
        if self.discussion_complete and self.host_fallback_close_spoken:
            self.host_fallback_close_spoken = False
            return True
        return False

    # --- human ------------------------------------------------------------

    def human_started_speaking(self, data: Any) -> None:
        """VAD: the human started talking — they take the floor now."""
        self._human_speaking = True
        self._cancel_release()
        self._take_floor(data)

    def human_stopped_speaking(self, data: Any) -> None:
        """VAD end of speech. If no transcript follows (the STT dropped the
        audio), release the floor after a while so the show cannot stall."""
        self._human_speaking = False
        if self.human_floor_pending and not self._human_turns_waiting:
            self._arm_release(data)

    def human_turn_submitted(self, data: Any) -> None:
        """A human transcript is on its way to the show; it holds the floor until
        reacted to (even if VAD never reported the speech)."""
        self._human_turns_waiting += 1
        self._cancel_release()
        self._take_floor(data)

    def human_turn_taken(self) -> None:
        """The show starts handling one submitted transcript."""
        self._human_turns_waiting = max(0, self._human_turns_waiting - 1)

    def _take_floor(self, data: Any) -> None:
        """Delivery stops at once; the script written past what the audience heard
        is rolled back once the generator has stopped (``apply_pending_rollback``)."""
        if self.human_floor_pending:
            return
        self.human_floor_pending = True
        position, cut = self.consumer.stop()
        if self._rollback_to is None:
            self._rollback_to = position
        self._barge_in_cut_beat = cut or bool(getattr(data, "panel_chain_running", False))
        data.runtime.gemma_client.log_text_only_event(
            "human",
            "barge-in",
            speaker=position[0].role if position else None,
            heard=position[0].text[: position[1]].strip() if position else None,
        )

    def _arm_release(self, data: Any) -> None:
        self._cancel_release()
        try:
            delay = float(
                os.environ.get("TALKSHOW_BARGE_IN_RELEASE_S", DEFAULT_BARGE_IN_RELEASE_S)
            )
            loop = asyncio.get_running_loop()
        except (ValueError, RuntimeError):
            return
        if delay > 0:
            self._release_handle = loop.call_later(delay, self._release_if_silent, data)

    def _cancel_release(self) -> None:
        if self._release_handle is not None:
            self._release_handle.cancel()
            self._release_handle = None

    def _release_if_silent(self, data: Any) -> None:
        self._release_handle = None
        if self.human_floor_pending and not self._human_speaking and not self._human_turns_waiting:
            from agent.show_graph.runner import abandon_human_turn

            logger.info("barge-in released: no transcript arrived")
            abandon_human_turn(data)

    def human_turn_abandoned(self) -> bool:
        """No usable transcript (noise). Returns True if a cut show should resume."""
        self._cancel_release()
        if self._human_speaking or self._human_turns_waiting:
            return False  # the human is still talking or another transcript is coming
        resume = self.human_floor_pending and self._barge_in_cut_beat
        self.human_floor_pending = False
        self._barge_in_cut_beat = False
        return resume

    async def react_to_human(self, data: Any, heard: str) -> FloorOutcome:
        """The human finished: every agent thinks on the transcript, then the floor is arbitrated.

        Call ``apply_pending_rollback`` before the human line is recorded."""
        self._cancel_release()
        self.human_floor_pending = False
        self._barge_in_cut_beat = False
        self._rethink = None
        self.discussion_complete = False
        self.host_fallback_close_spoken = False
        self.last_floor_outcome = None
        self.interrupted_role = None
        self.host_fallback_streak = 0
        self.interrupts_allowed = True

        outcome = await self._react_free(data, "human", heard)
        # The selection is consumed here; the first line after the human routes on it.
        self.last_floor_outcome = None
        if self._human_speaking or self._human_turns_waiting:
            # The human went on talking: keep the floor for their next transcript.
            self.human_floor_pending = True
            self._barge_in_cut_beat = True
        return outcome

    async def _react_free(self, data: Any, speaker: str, heard: str) -> FloorOutcome:
        """Everyone but ``speaker`` thinks on ``heard`` as a finished turn; arbitrate the free floor."""
        from agent.show.show_history import ShowHistory

        history = ShowHistory(lines=list(data.show_history.lines))
        listeners = [role for role in self._floor_roles(data) if role != speaker]
        data.runtime.gemma_client.log_text_only_event(
            "interval",
            "human transcript" if speaker == "human" else f"cut {speaker} line",
            content=heard.strip(),
            complete_content=heard.strip(),
        )
        self._final_interval = True
        await asyncio.gather(
            *(
                self._think_and_act(
                    data,
                    role,
                    history=history,
                    speaker=speaker,
                    interval_text=heard,
                    visible_text=heard,
                    checkpoint=1,
                    floor_state="free",
                )
                for role in listeners
            )
        )
        candidates = [
            role
            for role in listeners
            if (action := self.actions.get(role)) is not None and action.decision == "speak"
        ]
        outcome = await self.arbitrate_free(
            data, candidates=candidates, history=history, interval=1
        )
        self.log_selection(
            data,
            selected_role=outcome.selected_role,
            reason=outcome.decision,
            floor_state="free",
            candidates=candidates,
        )
        return outcome

    async def _update_listeners(
        self,
        data: Any,
        *,
        cue: Cue | None = None,
        speaker: str,
        interval_text: str,
        visible_text: str,
        panel_roles: list[str],
        step: str,
        checkpoint: int,
    ) -> str | FloorOutcome | None:
        from agent.show.show_history import ShowHistory

        cue = cue or Cue(speaker, visible_text, step)
        history = ShowHistory(lines=list(data.show_history.lines))
        own = _index_of(history.lines, self._speech_line) if self._speech_line is not None else None
        if own is not None:
            history.lines[own] = type(history.lines[own])(
                role_id=speaker,
                speaker=history.lines[own].speaker,
                text=visible_text,
                procedural=False,
            )
        else:
            history.append(speaker, visible_text)

        data.runtime.gemma_client.log_text_only_event(
            "interval",
            f"interval {checkpoint}",
            content=interval_text.strip(),
            complete_content=visible_text.strip(),
        )
        self._interval_backchannels = []
        listener_roles = list(dict.fromkeys([data.scenario.turn_control.listen_role, *panel_roles]))
        listeners = [candidate for candidate in listener_roles if candidate != speaker]
        results = await asyncio.gather(
            *(
                self._think_and_act(
                    data,
                    listener,
                    history=history,
                    speaker=speaker,
                    interval_text=interval_text,
                    visible_text=visible_text,
                    checkpoint=checkpoint,
                    floor_state="free" if self._final_interval else "occupied",
                )
                for listener in listeners
            )
        )
        for listener, result in zip(listeners, results):
            if result is not None:
                self._interval_backchannels.extend(
                    (listener, event) for event in result.backchannels
                )

        interval_start = len(visible_text) - len(interval_text)
        eligible: list[BackchannelCandidate] = []
        for listener, event in self._interval_backchannels:
            offset = interval_start + _char_end_after_words(interval_text, event.after_word_index)
            segment_index = self._backchannel_segment_at(cue.text, offset)
            data.runtime.gemma_client.log_text_only_event(
                listener,
                "backchannel_candidate",
                candidate=event.content,
                requested_word_index=event.after_word_index,
                target_char_offset=offset,
                target_segment_index=segment_index,
                decision="eligible",
                reason="placed_after_requested_word",
                target_audio_position_s=None,
            )
            eligible.append(BackchannelCandidate(listener, event, offset, segment_index))

        selected, decisions = self.backchannel_controller.arbitrate(id(cue), eligible)
        for candidate, reason in decisions:
            data.runtime.gemma_client.log_text_only_event(
                candidate.role,
                "backchannel_arbitration",
                candidate=candidate.event.content,
                requested_word_index=candidate.event.after_word_index,
                target_char_offset=candidate.offset,
                target_segment_index=candidate.target_segment_index,
                decision="approved" if candidate is selected else "rejected",
                reason=reason,
                target_audio_position_s=None,
            )
        self._interval_backchannels = (
            [
                (
                    selected.role,
                    Backchannel(
                        selected.event.intent,
                        selected.event.content,
                        selected.event.after_word_index,
                        selected.target_segment_index,
                    ),
                )
            ]
            if selected is not None
            else []
        )
        if self._final_interval:
            candidates = [
                role
                for role in listeners
                if (action := self.actions.get(role)) is not None
                and action.decision == "speak"
            ]
            outcome = await self.arbitrate_free(
                data,
                candidates=candidates,
                history=history,
                interval=checkpoint,
            )
            return outcome

        interrupt_candidates = [
            role
            for role in listeners
            if (action := self.actions.get(role)) is not None
            and action.decision == "interrupt"
        ]
        if not self.interrupts_allowed:
            return None
        return await self.arbitrate_occupied(
            data,
            speaker=speaker,
            visible_text=visible_text,
            candidates=interrupt_candidates,
            interval=checkpoint,
            history=history,
        )

    async def _think_and_act(
        self,
        data: Any,
        role: str,
        *,
        history: Any,
        speaker: str,
        interval_text: str,
        visible_text: str,
        checkpoint: int,
        floor_state: Literal["occupied", "free"],
    ) -> Action | None:
        from agent.config import load_persona_name
        from agent.panel.panel_prompts import panelist_personality_block
        from agent.panel.think_prompt import think_prompt_additions

        prompt = (
            f"You are {load_persona_name(role)} ({role}).\n"
            f"Current floor state: {floor_state}.\n"
            f"Current floor holder: {_speaker_name(speaker)} ({speaker}).\n"
            f"Latest public interval: {interval_text}\n"
            f"Complete visible speech prefix so far: {visible_text}\n\n"
            "Reflect privately on the actual point and possible next moves, then return "
            "one JSON object matching the required schema. Think is open-ended private "
            "understanding, not necessarily a speaking wish; do not reduce it to a rule "
            "like 'the speaker has not finished, so I must wait.' "
            "Decision controls only the main floor; a backchannel never requests the floor. "
            "When occupied, consider whether you have a specific, relevant contribution "
            "that would make the conversation more engaging, responsive, or substantive "
            "if expressed now rather than after the speaker finishes. "
            "You MAY REQUEST interrupt for a timely challenge, clarification, correction, "
            "disagreement, witty reaction, or meaningful extension of the current point. "
            "Interruption does not require urgency, an error, or a contribution that would "
            "become irrelevant if delayed. "
            "Balance the conversational value of intervening now against the disruption it "
            "would cause. Do not interrupt merely to gain airtime, repeat an existing point, "
            "or derail the conversation. "
            "You do not need to wait for the speaker to finish their full turn, but prefer "
            "natural phrase boundaries when possible. The central floor controller decides "
            "whether and when the interruption is actually granted. "
            "When free, choose speak only if you can respond, add substance, or perform "
            "a useful conversational role (including host facilitation). "
            "Backchannels are OPTIONAL and independent of your main-floor decision: "
            "speaking or requesting an interrupt does not imply a backchannel, and vice versa. "
            "Prefer backchannels: [] unless a short listener response adds clear conversational "
            "value. Return at most ONE candidate in this interval, using only meaningful "
            "acknowledgement, support, surprise, or encouragement. Keep it natural and "
            "typically 1-3 words. Avoid generic or repetitive 'Right', 'Exactly', and "
            "'Mm-hmm', especially when another agent recently gave similar feedback. "
            "Place it right after the word or phrase it responds to; it may fall mid-sentence, "
            "but do not split tightly bound words (e.g. 'small [..] changes'). "
            "The 1-based after_word_index counts whitespace-delimited words in the latest "
            "interval. Backchannels never take the floor and must not be arguments, "
            "questions requiring an answer, or attempts to interrupt. Do not put agreement "
            "before the words it agrees with. Silence is preferable to artificial feedback. "
            "Return no markdown or text outside the JSON object.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "think": "private reflection",\n'
            '  "action": {\n'
            '    "decision": "silent | interrupt | speak",\n'
            '    "intent": {"type": "acknowledge | support | challenge | clarify | correct | contribute | answer | facilitate", '
            '"goal": "specific outcome", "target": "specific target or null"} | null,\n'
            '    "content": "candidate utterance or null",\n'
            '    "backchannels": [{"intent": {"type": "acknowledge | support | surprise | encourage", '
            '"goal": "specific listener feedback", "target": "specific target or null"}, '
            '"content": "1-3 words", "after_word_index": 1}]\n'
            "  }\n"
            "}\n"
            "For silent: intent and content must be null. For speak/interrupt: both are required. "
            "Interrupt is valid only while occupied; speak is valid only when free."
        )
        # Dialogue quality fixes for panelists (flags default off: nothing is added).
        extra, extra_info = await think_prompt_additions(data, role, history)
        prompt += extra
        if extra_info:
            data.runtime.gemma_client.log_text_only_event(
                role, f"think additions - interval {checkpoint}", **extra_info
            )
        raw = await data.runtime.gemma_client.complete_text(
            prompt,
            system_prompt=panelist_personality_block(role),
            history_messages=history.prior_messages(perspective=role),
            raw=True,
            max_tokens=768,
            temperature=0.2,
            reasoning_effort="low",
        )
        result = parse_think_action(
            raw,
            floor_state=floor_state,
            interval_text=interval_text,
            allow_backchannels=True,
        )
        action = result.action
        self.actions[role] = action
        self.thoughts[role] = result.think
        self._think_sequence += 1
        self.intents[role] = self._compat_intent(action, self._think_sequence)
        self._pending_thinks.append((role, self.intents[role]))
        data.runtime.gemma_client.log_text_only_event(
            role,
            f"think - interval {checkpoint}",
            think=result.think or None,
        )
        data.runtime.gemma_client.log_text_only_event(
            role,
            f"action - interval {checkpoint}",
            decision=action.decision,
            intent=(
                {
                    "type": action.intent.type,
                    "goal": action.intent.goal,
                    "target": action.intent.target,
                }
                if action.intent
                else None
            ),
            content=action.content,
            backchannels=[
                {
                    "intent": {
                        "type": event.intent.type,
                        "goal": event.intent.goal,
                        "target": event.intent.target,
                    },
                    "content": event.content,
                    "after_word_index": event.after_word_index,
                }
                for event in action.backchannels
            ],
        )
        return action

    @staticmethod
    def _compat_intent(action: Action, think_id: int):
        from agent.session.text_only_interaction import SpeakingIntent

        requests_floor = action.decision in {"speak", "interrupt"}
        return SpeakingIntent(
            want_to_speak=requests_floor,
            time_cost_s=0 if requests_floor else 9,
            urgency=9 if action.decision == "interrupt" else 0,
            proposed_reply=action.content or "",
            think_id=think_id,
        )

    def _truncate_speech_history(self, data: Any, visible_text: str) -> None:
        own = _index_of(data.show_history.lines, self._speech_line)
        if own is None:
            return
        if visible_text.strip():
            data.show_history.lines[own].text = visible_text.strip()
        else:
            del data.show_history.lines[own]


__all__ = ["ThinkActionInteraction", "human_barge_in_enabled"]
