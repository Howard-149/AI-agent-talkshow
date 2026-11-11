"""Actuators — everything the show graph does to the outside world.

Graph nodes stay free of LiveKit / Gemma / UI details: they call these methods.
``LiveActuators`` is the production implementation; tests pass a fake with the
same methods. Who speaks and what they say comes from the Think & Action floor
interaction (the generator), which runs ahead of playback; how and when it is
delivered — console or TTS + avatar — is the interaction's speech consumer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from agent.data import TalkShowData
    from agent.emotion.appraisal import Utterance
    from agent.floor import TurnController
    from agent.show_graph.state import FloorPick, HumanEvent, LineSpec, PanelistLine

logger = logging.getLogger(__name__)

DEFAULT_APPRAISE_TIMEOUT_S = 8.0


def think_action_mood_enabled() -> bool:
    """TALKSHOW_TA_MOOD (default on): the speaker's PAD mood goes into Think & Action prompts."""
    return os.environ.get("TALKSHOW_TA_MOOD", "1").strip().lower() not in ("0", "false", "no", "off")


class Actuators(Protocol):
    # --- session / logging
    def should_stop(self) -> bool: ...
    def log(self, event: str, **fields: Any) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...

    # --- human turn
    async def commit_human_turn(self, event: HumanEvent) -> tuple[str, Utterance | None]: ...

    # --- lines
    def record_line(self, line: LineSpec) -> Utterance | None: ...
    def relax(self, role: str) -> None: ...
    def listeners(self, speaker: str) -> list[str]: ...
    async def speak(self, line: LineSpec) -> None: ...
    async def appraise(self, utt: Utterance, listener: str) -> dict[str, Any]: ...
    async def after_line(self, line: LineSpec) -> None: ...

    # --- floor
    def consume_floor_next(self) -> str: ...

    def grant_interrupt_grace_turn(self) -> bool: ...
    def queue_roles(self) -> list[str]: ...
    async def return_to_host(self, trigger: str) -> None: ...
    async def select_next_speaker(self, turn_idx: int, spoken_roles: list[str]) -> FloorPick: ...
    async def grant_panelist(self, role: str, reason: str) -> None: ...
    async def generate_panelist_line(self, role: str, step: str) -> PanelistLine: ...

    # --- canned host lines
    def open_floor_line(self) -> str: ...
    def intro_line(self, role: str) -> str: ...
    def human_floor_line(self) -> str: ...
    async def close_line(self) -> str: ...


class LiveActuators:
    """LiveKit session (or console when ``session`` is None) + TalkShowData."""

    def __init__(
        self,
        session: AgentSession | None,
        data: TalkShowData,
        controller: TurnController,
    ) -> None:
        from agent.interaction import ensure_floor_interaction

        self.session = session
        self.data = data
        # Appraisals of interrupted lines, finishing while the interrupter speaks.
        self._background: set[asyncio.Task] = set()
        self.controller = controller
        self.interaction = ensure_floor_interaction(data, session)
        # Any agent, host included, may be selected to speak; no open-floor call-outs.
        self.interval_floor_mode = True
        # Disclosed lines are appraised once decided (the script may cut them short).
        self._deferred_utterance = None
        # Floor-grant UI event, sent when the granted line actually starts playing.
        self._pending_grant: tuple[str, str] | None = None
        from agent.emotion import speaking_emotion

        # Mood shapes what an agent writes; the TTS label is fixed when its line is queued;
        # listeners appraise a finished line before thinking on its end.
        self.interaction.mood_block = self._mood_block
        self.interaction.cue_emotion = speaking_emotion
        self.interaction.on_line_complete = self._appraise_finished_line

    # --- session / logging -------------------------------------------------

    def _human_pending(self) -> bool:
        return bool(getattr(self.interaction, "human_floor_pending", False))

    def should_stop(self) -> bool:
        from agent.session.session_lifecycle import should_stop_session_work

        return should_stop_session_work(self.data, self.session) or self._human_pending()

    def log(self, event: str, **fields: Any) -> None:
        if self.data.turn_log is not None:
            self.data.turn_log.log(event, room=self.data.room_name, **fields)

    def snapshot(self) -> dict[str, Any]:
        return {
            "role_emotion": dict(getattr(self.data, "role_emotion", {}) or {}),
            "role_pad": dict(getattr(self.data, "role_pad", {}) or {}),
        }

    # --- human turn --------------------------------------------------------

    async def commit_human_turn(self, event: HumanEvent) -> tuple[str, Utterance | None]:
        """Record the human line, then every agent thinks on it and the floor is
        arbitrated. No host reply is drafted: the selected speaker answers."""
        from agent.emotion.appraisal import Utterance, snapshot_context
        from agent.floor.floor_control import apply_floor_next
        from agent.session.human_turn import record_human_line

        heard = (event.get("heard") or "").strip()
        # Uptake planner (no-op unless TALKSHOW_UPTAKE_PLANNER): starts now so it runs while the
        # line is recorded; the panelists' think calls in react_to_human wait for it (bounded).
        from agent.panel.uptake import start_uptake_plan

        start_uptake_plan(self.data, heard)
        self.interaction.human_turn_taken()
        # Drop script generated past the barge-in before the human line goes on record.
        self.interaction.apply_pending_rollback(self.data)
        if event.get("done_event") == "ghost_human_done":
            # Ghost replay (no-op unless TALKSHOW_GHOST_REPLAY): swap in the prefix and log
            # the state this turn is answered from. The generator is idle here.
            from agent.session.ghost_replay import replay_before_human_line

            replay_before_human_line(self.data, self.interaction, event.get("prefix"))
        await record_human_line(
            self.data,
            heard=heard,
            model_latency_s=float(event.get("model_latency_s") or 0.0),
            done_event=event.get("done_event") or "gemma_stt_done",
        )
        if heard:
            utt = Utterance(
                speaker="human",
                text=heard,
                seq=len(self.data.show_history.lines),
                context=snapshot_context(self.data, heard),
            )
            # Appraised before anyone reacts, so the reply is written in the new mood.
            await asyncio.gather(*(self.appraise(utt, role) for role in self.listeners("human")))
        outcome = await self.interaction.react_to_human(self.data, heard)
        apply_floor_next(self.data, outcome.selected_role or self.controller.listen_role())
        return "", None

    # --- lines -------------------------------------------------------------

    def record_line(self, line: LineSpec) -> Utterance | None:
        from agent.emotion.appraisal import Utterance, snapshot_context
        from agent.show.show_history import append_role

        text = (line.get("text") or "").strip()
        if not text or self._human_pending():
            return None
        procedural = bool(line.get("procedural"))
        append_role(self.data, line["role"], text, appraise=False, procedural=procedural)
        if procedural:
            return None
        utterance = Utterance(
            speaker=line["role"],
            text=text,
            seq=len(self.data.show_history.lines),
            context=snapshot_context(self.data, text),
        )
        if line.get("kind") == "panelist":
            self._deferred_utterance = utterance
        return utterance

    def relax(self, role: str) -> None:
        from agent.emotion.appraisal import relax_after_expression

        relax_after_expression(self.data, role)

    def _mood_block(self, data: TalkShowData, role: str) -> str:
        from agent.emotion import emotion_source, pad_prompt_block

        if emotion_source() == "llm" or not think_action_mood_enabled():
            return ""  # the legacy llm mood block asks for an [emotion] tag, not JSON
        return pad_prompt_block(data, role)

    def listeners(self, speaker: str) -> list[str]:
        from agent.emotion.appraisal import appraisal_enabled, appraisal_listeners

        if self._deferred_utterance is not None and self._deferred_utterance.speaker == speaker:
            return []
        if not appraisal_enabled():
            return []
        return appraisal_listeners(self.data, speaker)

    def _on_start(self) -> list[Any]:
        if self._pending_grant is None:
            return []
        from agent.ui.ui_events import emit_floor_grant

        (role, reason), self._pending_grant = self._pending_grant, None
        return [lambda: emit_floor_grant(role, reason=reason)]

    async def speak(self, line: LineSpec) -> None:
        """Queue the line for delivery; returns once it is decided, not played."""
        if line.get("kind") != "panelist":
            from agent.session.canned_cache import CANNED_KINDS

            if line.get("kind") == "topic" or line.get("topic"):
                from agent.floor.host_floor import log_show_topic

                log_show_topic(self.data, line.get("topic") or line["text"])
            await self.interaction.play_line(
                self.data,
                line["role"],
                line["text"],
                step=line.get("step") or line.get("kind") or "line",
                texts=line.get("texts"),
                canned=line.get("kind") in CANNED_KINDS,
                on_start=self._on_start(),
            )
            return

        await self.interaction.disclose_panel_line(self.data, line, on_start=self._on_start())
        # Still deferred: the line was cut (a finished line was appraised during disclosure).
        utterance, self._deferred_utterance = self._deferred_utterance, None
        public_text = self.interaction.latest_line_text(line["role"])
        if utterance is None or self._human_pending() or not public_text:
            return
        # Appraise what the script kept (an approved interrupt cut the line).
        utterance = self._kept(utterance, public_text)
        appraisals = asyncio.gather(
            *(self.appraise(utterance, role) for role in self.listeners(line["role"]))
        )
        if self.interaction.interrupted_role is not None:
            # Approved interrupt: the interrupter takes over at once (no pause); the
            # listeners appraise the cut line in the background.
            task = asyncio.ensure_future(
                self._finish_cut_line_appraisals(appraisals, self.interaction.latest_cue)
            )
            self._background.add(task)
            task.add_done_callback(self._background.discard)
            return
        await appraisals
        self.interaction.checkpoint_line_end(self.data)

    def _kept(self, utterance: Utterance, text: str) -> Utterance:
        from dataclasses import replace

        from agent.emotion.appraisal import snapshot_context

        return replace(utterance, text=text, context=snapshot_context(self.data, text))

    async def _appraise_finished_line(self, data: TalkShowData, speaker: str, text: str) -> None:
        """Disclosure reached the last interval uncut: listeners appraise the whole line
        before they think on it (and write the next line)."""
        utterance = self._deferred_utterance
        if utterance is None or utterance.speaker != speaker or self._human_pending():
            return
        self._deferred_utterance = None
        utterance = self._kept(utterance, text)
        await asyncio.gather(*(self.appraise(utterance, role) for role in self.listeners(speaker)))

    async def _finish_cut_line_appraisals(self, appraisals: Any, cue: Any) -> None:
        try:
            await appraisals
        except Exception:
            logger.exception("background appraisal of an interrupted line failed")
            return
        self.interaction.checkpoint_line_end(self.data, cue=cue)

    async def appraise(self, utt: Utterance, listener: str) -> dict[str, Any]:
        from agent.emotion.appraisal import (
            _env_float,
            apply_appraisal,
            run_appraisal,
        )

        timeout = _env_float("TALKSHOW_PAD_APPRAISAL_TIMEOUT_S", DEFAULT_APPRAISE_TIMEOUT_S)
        try:
            result = await asyncio.wait_for(
                run_appraisal(self.data, listener, utt), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("appraisal timed out listener=%s after %.1fs", listener, timeout)
            self.log("pad_appraisal_timeout", role=listener, speaker=utt.speaker, seq=utt.seq)
            return {"role": listener, "ok": False, "timeout": True}
        if self._human_pending() and utt.speaker != "human":
            # Script past the barge-in is rolled back; the human's own words still count.
            return {"role": listener, "ok": False, "barge_in": True}
        apply_appraisal(self.data, utt, result)
        return {
            "role": listener,
            "speaker": utt.speaker,
            "seq": utt.seq,
            "ok": result.delta is not None,
            "delta": result.delta,
            "word": result.word,
        }

    async def after_line(self, line: LineSpec) -> None:
        from agent.floor.floor_control import apply_floor_next
        from agent.floor.hand_raise_ui import dequeue_hand_raise
        from agent.ui.ui_events import emit_floor_grant

        kind = line.get("kind")
        if kind == "welcome":
            apply_floor_next(self.data, "host")
        elif kind == "panelist":
            apply_floor_next(self.data, self.interaction.next_role_after_line(self.data, line))
        elif kind == "grant_human":
            reason = line.get("reason") or "queue_fifo"
            apply_floor_next(self.data, "human")
            await emit_floor_grant("human", reason=reason)
            await dequeue_hand_raise(self.data, "human")
            logger.info("granted floor to human")
            self.log("floor_grant_human", reason=reason)
        self.data.touch_activity()

    # --- floor -------------------------------------------------------------

    def consume_floor_next(self) -> str:
        from agent.floor.floor_control import consume_floor_next

        return consume_floor_next(self.data)

    def grant_interrupt_grace_turn(self) -> bool:
        """At the turn cap: True when the last line was cut by an approved interrupt.

        The interrupter then speaks one more turn, which cannot be interrupted, so the
        show never closes on a half-finished line with the approved answer dropped."""
        if not self.interaction.last_line_interrupted:
            return False
        self.interaction.interrupts_allowed = False
        return True

    def queue_roles(self) -> list[str]:
        return []

    async def return_to_host(self, trigger: str) -> None:
        from agent.floor.host_floor import return_floor_to_host
        from agent.interaction import playout_idle

        if not playout_idle(self.data):
            # Script is still playing: the speech consumer owns the session's role.
            return
        await return_floor_to_host(self.session, self.data, self.controller, trigger=trigger)

    async def select_next_speaker(self, turn_idx: int, spoken_roles: list[str]) -> FloorPick:
        """Free floor: a submitted speak candidate, or the host directed by the controller."""
        from agent.show_graph.state import FloorPick

        outcome = await self.interaction.select_free_floor_outcome(self.data, interval=turn_idx)
        invited, self.interaction.opening_invite_role = self.interaction.opening_invite_role, None
        if outcome.decision == "host_fallback" and invited:
            # The host just opened the show and handed it to ``invited``: no second host turn.
            self.interaction.host_fallback_instruction = None
            self.interaction.host_fallback_strategy = None
            self.interaction.host_fallback_target = None
            return FloorPick(invited, "opening_invite")
        if outcome.decision == "host_fallback":
            return FloorPick(self.controller.listen_role(), "host_fallback")
        if invited and outcome.selected_role == invited:
            return FloorPick(invited, "opening_invite")
        return FloorPick(outcome.selected_role, outcome.decision)

    async def grant_panelist(self, role: str, reason: str) -> None:
        self._pending_grant = (role, reason)
        self.interaction.record_floor_grant(role)

    async def generate_panelist_line(self, role: str, step: str) -> PanelistLine:
        from agent.emotion import get_role_emotion
        from agent.show_graph.state import PanelistLine

        # Don't spend a model call on a line the script is not allowed to queue yet.
        await self.interaction.wait_for_lookahead()
        text = await self.interaction.host_fallback_line(self.data, role)
        reason = "text_host_fallback"
        if text is None:
            text = self.interaction.consume_proposed_reply(role)
            reason = "text_proposed_reply_selected"
        if text:
            self.log(reason, role=role, step=step, text=text)
            return PanelistLine(
                role=role,
                text=text,
                next_role="host",
                emotion=get_role_emotion(self.data, role),
            )

        from agent.panel.panel_speech import generate_panelist_line

        return await generate_panelist_line(self.session, self.data, speak_role=role, step=step)

    # --- canned host lines ---------------------------------------------------

    def open_floor_line(self) -> str:
        from agent.floor.host_lines import host_open_floor_line

        return host_open_floor_line()

    def intro_line(self, role: str) -> str:
        from agent.config import load_persona_name
        from agent.floor.host_lines import host_intro_speaker_line

        return host_intro_speaker_line(load_persona_name(role))

    def human_floor_line(self) -> str:
        from agent.floor.host_lines import host_human_floor_line

        return host_human_floor_line(topic=self.data.human_hand_topic or None)

    async def close_line(self) -> str:
        from agent.floor.host_floor import host_closing_line

        if self.interaction.take_close_line_override():
            return ""
        text = await host_closing_line(self.data)
        # The show is over: the idle loop waits for the human instead of restarting.
        self.interaction.discussion_complete = True
        return text
