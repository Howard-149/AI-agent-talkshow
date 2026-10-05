"""Actuators — everything the show graph does to the outside world.

Graph nodes stay free of LiveKit / Gemma / UI details: they call these methods.
``LiveActuators`` is the production implementation; tests pass a fake with the
same methods.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from agent.data import TalkShowData
    from agent.emotion.appraisal import Utterance
    from agent.floor import TurnController
    from agent.show_graph.state import FloorPick, HumanEvent, LineSpec, PanelistLine

logger = logging.getLogger(__name__)

DEFAULT_APPRAISE_TIMEOUT_S = 8.0


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
    def queue_roles(self) -> list[str]: ...
    async def return_to_host(self, trigger: str) -> None: ...
    def prefetch_hand_raise_poll(self) -> None: ...
    async def hand_raise_round(self, turn_idx: int, spoken_roles: list[str]) -> FloorPick: ...
    async def grant_panelist(self, role: str, reason: str) -> None: ...
    async def generate_panelist_line(self, role: str, step: str) -> PanelistLine: ...

    # --- canned host lines
    def open_floor_line(self) -> str: ...
    def intro_line(self, role: str) -> str: ...
    def human_floor_line(self) -> str: ...
    def close_line(self) -> str: ...


class LiveActuators:
    """LiveKit session + TalkShowData implementation."""

    def __init__(
        self,
        session: AgentSession,
        data: TalkShowData,
        controller: TurnController,
    ) -> None:
        self.session = session
        self.data = data
        self.controller = controller
        self._prefetched_poll: asyncio.Task | None = None

    # --- session / logging -------------------------------------------------

    def should_stop(self) -> bool:
        from agent.session.session_lifecycle import should_stop_session_work

        return should_stop_session_work(self.data, self.session)

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
        from agent.emotion.appraisal import Utterance, snapshot_context
        from agent.session.human_turn import commit_human_turn

        heard = (event.get("heard") or "").strip()
        reply = await commit_human_turn(
            self.data,
            heard=heard,
            parsed=event["parsed"],
            panel_turn=True,
            model_latency_s=float(event.get("model_latency_s") or 0.0),
            done_event=event.get("done_event") or "gemma_stt_done",
            raw=event.get("raw"),
            graph=True,
        )
        # The host reply is spoken by the graph; drop the StoredReplyLLM copy.
        self.data.runtime.turn_store.consume_turn()
        utt = None
        if heard:
            utt = Utterance(
                speaker="human",
                text=heard,
                seq=len(self.data.show_history.lines),
                context=snapshot_context(self.data, heard),
            )
        return reply, utt

    # --- lines -------------------------------------------------------------

    def record_line(self, line: LineSpec) -> Utterance | None:
        from agent.emotion.appraisal import Utterance, snapshot_context
        from agent.show.show_history import append_role

        text = (line.get("text") or "").strip()
        if not text:
            return None
        procedural = bool(line.get("procedural"))
        append_role(self.data, line["role"], text, appraise=False, procedural=procedural)
        if procedural:
            return None
        return Utterance(
            speaker=line["role"],
            text=text,
            seq=len(self.data.show_history.lines),
            context=snapshot_context(self.data, text),
        )

    def relax(self, role: str) -> None:
        from agent.emotion.appraisal import relax_after_expression

        relax_after_expression(self.data, role)

    def listeners(self, speaker: str) -> list[str]:
        from agent.emotion.appraisal import appraisal_enabled, appraisal_listeners

        if not appraisal_enabled():
            return []
        return appraisal_listeners(self.data, speaker)

    async def speak(self, line: LineSpec) -> None:
        from agent.session.canned_cache import CANNED_KINDS
        from agent.session.session_handoff import speak_panel_line

        await speak_panel_line(
            self.session,
            self.data,
            speak_role=line["role"],
            text=line["text"],
            step=line.get("step") or line.get("kind") or "line",
            texts=line.get("texts"),
            canned=line.get("kind") in CANNED_KINDS,
        )

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
            apply_floor_next(self.data, line.get("next_tag") or "host")
        elif kind == "direct_call" and line.get("target"):
            await dequeue_hand_raise(self.data, line["target"])
        elif kind == "grant_human":
            reason = line.get("reason") or "queue_fifo"
            apply_floor_next(self.data, "human")
            await emit_floor_grant("human", reason=reason)
            await dequeue_hand_raise(self.data, "human")
            logger.info("granted floor to human queue=%s", self.data.hand_raise_queue.roles())
            self.log(
                "floor_grant_human",
                reason=reason,
                queue_remaining=self.data.hand_raise_queue.roles(),
            )
        self.data.touch_activity()

    # --- floor -------------------------------------------------------------

    def consume_floor_next(self) -> str:
        from agent.floor.floor_control import consume_floor_next

        return consume_floor_next(self.data)

    def queue_roles(self) -> list[str]:
        return self.data.hand_raise_queue.roles()

    async def return_to_host(self, trigger: str) -> None:
        from agent.floor.host_floor import return_floor_to_host

        await return_floor_to_host(self.session, self.data, self.controller, trigger=trigger)

    def prefetch_hand_raise_poll(self) -> None:
        """Start the hand-raise poll now, while the open-floor line plays (opt-in)."""
        from agent.floor.host_floor import poll_during_open_floor_enabled, poll_panel_hand_raises

        if not poll_during_open_floor_enabled() or self.data.hand_raise_queue.roles():
            return
        if self._prefetched_poll is not None:
            self._prefetched_poll.cancel()
        self._prefetched_poll = asyncio.create_task(
            poll_panel_hand_raises(self.data, self.controller.panel_speaker_roles()),
            name="hand_raise_poll:prefetch",
        )

    async def hand_raise_round(self, turn_idx: int, spoken_roles: list[str]) -> FloorPick:
        from agent.floor.host_floor import run_hand_raise_round

        prefetched, self._prefetched_poll = self._prefetched_poll, None
        return await run_hand_raise_round(
            self.session,
            self.data,
            self.controller,
            panel_roles=self.controller.panel_speaker_roles(),
            spoken_roles=set(spoken_roles),
            turn_idx=turn_idx,
            prefetched_poll=prefetched,
        )

    async def grant_panelist(self, role: str, reason: str) -> None:
        from agent.floor.hand_raise_ui import dequeue_hand_raise
        from agent.ui.ui_events import emit_floor_grant

        await emit_floor_grant(role, reason=reason)
        await dequeue_hand_raise(self.data, role)

    async def generate_panelist_line(self, role: str, step: str) -> PanelistLine:
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

    def close_line(self) -> str:
        from agent.panel.panel_speech import PANEL_HOST_CLOSE

        return PANEL_HOST_CLOSE
