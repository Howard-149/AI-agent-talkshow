"""Sentence-aware speech, concurrent intent requests, and exclusive floor handoff.

Opt-in laptop path. Voice input and avatars are intentionally out of scope.
Only completely played sentences enter shared history. An interrupted in-flight
sentence can have been partly audible, but is conservatively omitted. Models
never receive the unplayed tail. Every speech/poll belongs to a monotonic epoch.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import os
import time
from typing import Any

from agent.config import load_persona_name
from agent.floor.reactive_policy import Decision, FloorRequest, ReactivePolicy, parse_request
from agent.session.speak_chunks import split_speak_sentences


@dataclass
class ActiveTurn:
    epoch: int
    role: str
    step: str
    started: float = 0.0
    heard: list[str] = field(default_factory=list)
    handle: Any = None
    winner: FloorRequest | None = None
    granted_at: float = 0.0
    poll: asyncio.Task | None = None
    polls: int = 0
    queued: list[tuple[FloorRequest, Decision]] = field(default_factory=list)


class ReactiveSession:
    def __init__(self, session: Any, data: Any, controller: Any):
        self.session, self.data, self.controller = session, data, controller
        self.policy = ReactivePolicy(
            min_speech_s=float(os.environ.get("TALKSHOW_REACTIVE_MIN_SPEECH_S", "3")),
            cooldown_s=float(os.environ.get("TALKSHOW_REACTIVE_COOLDOWN_S", "15")),
        )
        self.epoch = 0
        self.active: ActiveTurn | None = None
        self.mode = "live"
        self.demo_active = False
        self.demo_opening = False
        self.pending_human = False
        self.demo_generation: asyncio.Task | None = None
        self.ai_only = False
        self.demo_stop = False
        # "every": listeners re-decide after each fully played sentence (one
        # poll in flight at a time). "first": the original single poll.
        self.listen = os.environ.get("TALKSHOW_REACTIVE_LISTEN", "every").strip().lower()
        self.moderator = controller.listen_role()
        self.host_cut = os.environ.get("TALKSHOW_REACTIVE_HOST_CUT", "1").lower() not in {"0", "false", "no"}
        self.topic = ""

    def log(self, event: str, **fields: Any) -> None:
        if self.data.turn_log is not None:
            self.data.turn_log.log(event, room=self.data.room_name, **fields)

    async def emit(self, event: str, **fields: Any) -> None:
        from agent.ui.ui_events import publish_ui_event
        await publish_ui_event(event, **fields)

    async def decision(self, req: FloorRequest, decision: Decision) -> None:
        fields = {**asdict(req), **asdict(decision), "name": load_persona_name(req.role)
                  if req.role != "human" else "You"}
        # Do not broadcast unspoken draft content into the transcript.
        fields.pop("speech")
        self.log("reactive_decision", **fields)
        await self.emit("reactive_decision", **fields)

    async def queue_request(self, turn: ActiveTurn, req: FloorRequest, decision: Decision) -> None:
        from agent.ui.ui_events import emit_queue_state
        # Re-polls replace a listener's earlier request instead of stacking it.
        turn.queued = [pair for pair in turn.queued if pair[0].role != req.role]
        turn.queued.append((req, decision))
        self.data.hand_raise_queue.enqueue(req.role, reason=req.reason, topic=req.kind)
        await emit_queue_state(self.data, phase="reactive_wait")

    async def request_human(self) -> None:
        if self.ai_only:
            return
        turn = self.active
        if turn is None:
            if not self.demo_active:
                await self.emit("reactive_error", message="Wait until a character starts speaking.")
                return
            # Also accept Stop in the gap while the next live reply is drafted.
            # This flag is checked before any subsequent sentence can start.
            self.pending_human = True
            if self.demo_generation is not None:
                self.demo_generation.cancel()
        epoch = turn.epoch if turn else self.epoch
        req = FloorRequest("human", epoch, "interrupt", reason="Human requested the floor",
                           source="human")
        decision = self.policy.decide(
            req, epoch=epoch, speaker=turn.role if turn else self.data.active_role,
            heard=" ".join(turn.heard) if turn else "",
            elapsed_s=time.monotonic() - turn.started if turn else 0, now=time.monotonic(),
            eligible_roles=self.controller.panel_speaker_roles(),
        )
        # Commit the exclusive winner before awaiting any UI/network operations.
        previous = turn.winner if turn else None
        self.pending_human = True
        if turn is not None:
            turn.winner, turn.granted_at = req, time.monotonic()
            if turn.handle is not None:
                turn.handle.interrupt(force=True)
        if previous is not None and previous.source == "llm":
            await self.decision(previous, Decision("rejected", "superseded_by_human"))
        await self.decision(req, decision)

    async def stop_demo(self) -> None:
        """Operator stop, not an in-world human interruption or floor grant."""
        if not self.demo_active:
            return
        self.demo_stop = True
        if self.demo_generation is not None:
            self.demo_generation.cancel()
        if self.active is not None and self.active.handle is not None:
            self.active.handle.interrupt(force=True)
        self.log("reactive_demo_stop", source="operator", epoch=self.epoch)

    async def _demo_line(self, act: Any, beat) -> dict | None:
        if self.pending_human or self.demo_stop:
            return None
        await self.emit("reactive_generating", role=beat.role,
                        name=load_persona_name(beat.role), step=beat.step)
        if self.pending_human or self.demo_stop:
            return None
        task = asyncio.create_task(act.generate_demo_line(beat), name=beat.step)
        self.demo_generation = task
        try:
            return await asyncio.wait_for(task, timeout=20)
        except asyncio.CancelledError:
            if self.pending_human or self.demo_stop:
                return None
            raise
        finally:
            if self.demo_generation is task:
                self.demo_generation = None

    async def _request_role(self, role: str, turn: ActiveTurn, heard: str) -> FloorRequest | None:
        from agent.floor.intent_prompts import (
            DEMO_STANCES, intent_prompt, moderator_intent_system, panelist_intent_system,
        )

        hist = self.data.show_history.prior_messages(perspective=role)
        moderator = role == self.moderator
        if moderator:
            system = moderator_intent_system(role, topic=self.topic)
        else:
            system = panelist_intent_system(
                role, stance=DEMO_STANCES.get(role, "") if self.demo_active else "")
        prompt = intent_prompt(load_persona_name(turn.role), heard, moderator=moderator)
        t0 = time.monotonic()
        self.log("reactive_model_start", role=role, epoch=turn.epoch, heard=heard,
                 poll=turn.polls, sentences=len(turn.heard))
        try:
            raw = await asyncio.wait_for(self.data.runtime.gemma_client.complete_text(
                prompt, system_prompt=system, history_messages=hist, raw=True,
                temperature=0.4, max_tokens=400,
            ), timeout=float(os.environ.get("TALKSHOW_REACTIVE_POLL_TIMEOUT_S", "12")))
            req = parse_request(raw, role=role, epoch=turn.epoch, moderator=self.moderator)
        except asyncio.CancelledError:
            self.log("reactive_model_cancelled", role=role, epoch=turn.epoch)
            raise
        except Exception as exc:
            # Network/JSON errors never interrupt the current speaker.
            self.log("reactive_model_failed", role=role, epoch=turn.epoch, error=type(exc).__name__)
            await self.emit("reactive_decision", role=role, name=load_persona_name(role),
                            status="rejected", rule="model_error", reason="No valid request", source="llm")
            return None
        self.log("reactive_model_done", role=role, epoch=turn.epoch, poll=turn.polls,
                 latency_ms=round((time.monotonic() - t0) * 1000), raw=raw)
        return req

    def listeners(self, speaker: str) -> list[str]:
        roles = [r for r in self.controller.panel_speaker_roles() if r != speaker]
        if self.host_cut and speaker != self.moderator:
            roles.append(self.moderator)
        return roles

    async def _poll(self, turn: ActiveTurn, *, allow_ai_cut: bool) -> None:
        roles = self.listeners(turn.role)
        heard = " ".join(turn.heard)
        # Independent persona requests actually run concurrently with current TTS.
        requests = await asyncio.gather(*(self._request_role(r, turn, heard) for r in roles))
        candidates = []
        for req in requests:
            if req is None:
                continue
            now = time.monotonic()
            current_epoch = self.active.epoch if self.active else self.epoch + 1
            decision = self.policy.decide(
                req, epoch=current_epoch, speaker=turn.role, heard=" ".join(turn.heard),
                elapsed_s=now - turn.started, now=now, eligible_roles=roles,
                allow_ai_cut=allow_ai_cut, human_pending=self.pending_human or self.demo_stop,
            )
            if decision.status == "granted":
                candidates.append((req, decision))
            else:
                if decision.status == "queued":
                    await self.queue_request(turn, req, decision)
                await self.decision(req, decision)
        # Human is checked again after async UI work: it can supersede any AI intent.
        if candidates and self.active is turn and not self.pending_human and not self.demo_stop and turn.winner is None:
            candidates.sort(key=lambda pair: (-pair[1].priority, roles.index(pair[0].role)))
            req, decision = candidates[0]
            turn.winner, turn.granted_at = req, time.monotonic()
            if turn.handle is not None:
                turn.handle.interrupt(force=True)
            await self.decision(req, decision)
            for losing, _ in candidates[1:]:
                queued = Decision("queued", "another_request_won", 50)
                await self.queue_request(turn, losing, queued)
                await self.decision(losing, queued)

    async def _one_line(self, act: Any, line: dict, *, allow_ai_cut: bool) -> tuple[ActiveTurn, str]:
        from agent.session.session_handoff import switch_to_role
        from agent.show.show_history import append_role
        from agent.ui.ui_events import emit_role_active, emit_transcript

        self.epoch += 1
        turn = ActiveTurn(self.epoch, line["role"], line.get("step", "reactive"))
        self.active = turn
        full_text = line["text"]
        interrupted_fragment = ""
        try:
            await switch_to_role(self.session, self.data, turn.role, reason="reactive_floor")
            turn.started = time.monotonic()
            await self.emit("reactive_state", active=True, role=turn.role, name=load_persona_name(turn.role),
                            epoch=turn.epoch, step=turn.step, mode=self.mode, heard="")
            await emit_role_active(turn.role)
            self.data.speak_line_busy = True
            self.log("reactive_line_start", role=turn.role, epoch=turn.epoch, text=full_text)
            sentences = split_speak_sentences(full_text)
            for index, sentence in enumerate(sentences):
                if turn.winner or self.demo_stop or self.data.shutdown_event.is_set():
                    break
                # No ahead-of-time audio queue: at most one sentence is in flight.
                handle = self.session.say(sentence, allow_interruptions=True, add_to_chat_ctx=False)
                turn.handle = handle
                self.log("reactive_sentence_start", role=turn.role, epoch=turn.epoch, text=sentence)
                await handle.wait_for_playout()
                if hasattr(handle, "exception") and handle.exception() is not None:
                    raise RuntimeError("sentence speech failed")
                turn.handle = None
                if handle.interrupted or turn.winner or self.demo_stop:
                    interrupted_fragment = sentence
                    break
                turn.heard.append(sentence)
                self.log("reactive_sentence_done", role=turn.role, epoch=turn.epoch, text=sentence)
                await self.emit("reactive_state", active=True, role=turn.role, name=load_persona_name(turn.role),
                                epoch=turn.epoch, step=turn.step, mode=self.mode, heard=" ".join(turn.heard))
                # Polls run at sentence boundaries, never per token or audio frame,
                # with at most one in flight. A sentence that ends while a poll is
                # still running is covered by the next boundary's poll.
                can_poll = turn.poll is None or (self.listen == "every" and turn.poll.done())
                if (can_poll and index + 1 < len(sentences) and self.mode != "baseline"
                        and line.get("kind") == "panelist"):
                    turn.polls += 1
                    turn.poll = asyncio.create_task(self._poll(turn, allow_ai_cut=allow_ai_cut),
                                                    name=f"reactive_poll:{turn.epoch}:{turn.polls}")
            return turn, " ".join(turn.heard)
        finally:
            if turn.handle is not None and not turn.handle.done():
                turn.handle.interrupt(force=True)
                await turn.handle.wait_for_playout()
            # Invalidate first. Cancellation cannot race a late grant for this turn.
            if self.active is turn:
                self.active = None
            if turn.poll is not None and not turn.poll.done():
                turn.poll.cancel()
                await asyncio.gather(turn.poll, return_exceptions=True)
            heard = " ".join(turn.heard)
            if heard:
                append_role(self.data, turn.role, heard, appraise=False,
                            procedural=bool(line.get("procedural")))
                await emit_transcript(
                    turn.role, heard, step=turn.step, utterance_id=f"reactive-{turn.epoch}",
                    interrupted=turn.winner is not None or self.demo_stop,
                )
            self.data.speak_line_busy = False
            stop_ms = round((time.monotonic() - turn.granted_at) * 1000, 1) if turn.winner else None
            self.log("reactive_line_end", role=turn.role, epoch=turn.epoch, heard=heard,
                     interrupted=turn.winner is not None or self.demo_stop, discarded_chars=max(0, len(full_text) - len(heard)),
                     operator_stopped=self.demo_stop,
                     omitted_fragment=interrupted_fragment, stop_ack_ms=stop_ms)
            await self.emit("reactive_state", active=False, role=turn.role, name=load_persona_name(turn.role),
                            epoch=turn.epoch, step=turn.step, mode=self.mode, heard=heard,
                            interrupted=turn.winner is not None or self.demo_stop, stop_ack_ms=stop_ms)

    async def speak(self, act: Any, line: dict, *, extra_utterances: list) -> dict:
        from agent.emotion.appraisal import Utterance, snapshot_context
        from agent.floor.floor_control import apply_floor_next
        from agent.ui.ui_events import emit_floor_grant

        current = dict(line)
        roles, steps, appraisals = [], [], []
        extra_jobs = [asyncio.create_task(act.appraise(u, r))
                      for u in extra_utterances for r in act.listeners(u.speaker)]
        cuts = 0
        followup_idx = 0
        if not self.demo_active:
            self.pending_human = False
            self.demo_stop = False

        async def human_result():
            self.data.hand_raise_queue.clear_roles(self.controller.panel_speaker_roles())
            apply_floor_next(self.data, "human")
            await emit_floor_grant("human", reason="human_interrupt")
            await self.emit("reactive_complete", mode=self.mode, outcome="human_interrupt")
            return {"line": None, "human": True, "roles": roles, "steps": steps,
                    "appraisals": appraisals, "extra_turns": max(0, len(roles) - 1)}

        def stopped_result():
            self.data.hand_raise_queue.clear_roles(self.controller.panel_speaker_roles())
            apply_floor_next(self.data, "host")
            return {"line": None, "human": False, "stopped": True, "roles": roles,
                    "steps": steps, "appraisals": appraisals,
                    "extra_turns": max(0, len(roles) - 1)}

        try:
            while True:
                if self.demo_stop:
                    return stopped_result()
                if self.pending_human:
                    return await human_result()
                turn, heard = await self._one_line(
                    act, current, allow_ai_cut=(self.mode != "queue" and cuts == 0
                                               and (not self.demo_active or not roles)),
                )
                roles.append(turn.role)
                steps.append(turn.step)
                if heard:
                    act.relax(turn.role)
                    utt = Utterance(speaker=turn.role, text=heard, seq=self.epoch,
                                    context=snapshot_context(self.data, heard))
                    appraisals.extend(await asyncio.gather(*(act.appraise(utt, r) for r in act.listeners(turn.role))))
                winner = turn.winner
                if self.demo_stop:
                    return stopped_result()
                if self.pending_human or (winner and winner.role == "human"):
                    return await human_result()
                if winner:
                    self.policy.commit_cut(winner, now=time.monotonic())
                    cuts += 1
                    await act.grant_panelist(
                        winner.role, "moderator_cut" if winner.kind == "moderation" else "ai_interrupt")
                    self.log("reactive_takeover", role=winner.role, previous_role=turn.role,
                             epoch=turn.epoch, source=winner.source, draft_reused=True)
                    moderation = winner.kind == "moderation"
                    # A moderator cut steers back and returns the floor to the
                    # speaker it stopped; its own line is not open to AI cuts.
                    current = {"role": winner.role, "text": winner.speech,
                               "step": f"reactive_takeover_{self.epoch}",
                               "kind": "host_reply" if moderation else "panelist",
                               "procedural": False,
                               "next_tag": turn.role if moderation else "host"}
                    continue
                if self.demo_active:
                    from agent.multi_agent.reactive_demo_plan import DEMO_FOLLOWUPS, RYAN_FIRST_REPLY

                    # Intent drafts belong to their old listening snapshot. They
                    # can influence the queue, never be replayed as a later reply.
                    for req, _ in turn.queued:
                        self.log("reactive_draft_discarded", role=req.role, epoch=req.epoch,
                                 reason="queued_context_advanced")
                    if len(roles) == 1:
                        beat = RYAN_FIRST_REPLY
                        reason = ("sequential_baseline" if self.mode == "baseline"
                                  else "queued_after_turn" if turn.queued else "demo_normal_turn")
                    elif followup_idx < len(DEMO_FOLLOWUPS):
                        beat = DEMO_FOLLOWUPS[followup_idx]
                        followup_idx += 1
                        reason = "demo_followup"
                    else:
                        current["text"] = heard
                        self.data.hand_raise_queue.clear_roles(self.controller.panel_speaker_roles())
                        return {"line": current, "human": False, "closed": True,
                                "roles": roles, "steps": steps, "appraisals": appraisals,
                                "extra_turns": max(0, len(roles) - 1)}
                    if any(req.role == beat.role for req, _ in turn.queued):
                        await self.emit("reactive_decision", role=beat.role,
                                        name=load_persona_name(beat.role), status="queued",
                                        rule="fresh_draft_at_turn_end", source="rule",
                                        reason="Context changed. Regenerating before speaking.")
                    current = await self._demo_line(act, beat)
                    if self.demo_stop:
                        return stopped_result()
                    if current is None or self.pending_human:
                        return await human_result()
                    await act.grant_panelist(beat.role, reason)
                    self.log("reactive_demo_advance", role=beat.role, step=beat.step,
                             previous_role=turn.role, reason=reason, draft_reused=False,
                             completed_speeches=len(roles), max_speeches=7)
                    continue
                # Queued speeches are NOT replayed: the speaker finished additional
                # sentences since the intent snapshot. Route a fresh normal draft.
                if turn.queued:
                    req, decision = sorted(turn.queued, key=lambda pair: -pair[1].priority)[0]
                    self.log("reactive_draft_discarded", role=req.role, epoch=req.epoch,
                             reason="queued_context_advanced")
                    await self.emit("reactive_decision", role=req.role, name=load_persona_name(req.role),
                                    status="queued", rule="fresh_draft_at_turn_end",
                                    reason="Context changed. Regenerating before speaking.", source="rule")
                    current["next_tag"] = req.role
                current["text"] = heard
                return {"line": current, "human": False, "roles": roles, "steps": steps,
                        "appraisals": appraisals, "extra_turns": max(0, len(roles) - 1)}
        finally:
            extra_results = await asyncio.gather(*extra_jobs, return_exceptions=True)
            appraisals.extend(r for r in extra_results if isinstance(r, dict))
            self.pending_human = False


def reactive_enabled() -> bool:
    return (
        os.environ.get("TALKSHOW_REACTIVE", "0").lower() in {"1", "true", "yes"}
        and os.environ.get("TALKSHOW_TEXT_INPUT", "0") == "1"
        and os.environ.get("TALKSHOW_AVATAR_ENABLED", "1") == "0"
    )
