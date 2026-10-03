"""Entry points that run one show beat through the LangGraph show graph."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from agent.show_graph.actuators import Actuators, LiveActuators
from agent.show_graph.graph import show_app
from agent.show_graph.state import Entry, HumanEvent, base_state

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from agent.data import TalkShowData
    from agent.floor import TurnController

logger = logging.getLogger(__name__)

# Each spoken line takes ~5 supersteps; a 12-turn round plus moderation needs ~100.
RECURSION_LIMIT = 1000


async def run_beat(
    act: Actuators,
    *,
    entry: Entry,
    trigger: str,
    panel_roles: list[str],
    listen_role: str,
    human_event: HumanEvent | None = None,
    skip_open_floor: bool = False,
    spoken_roles: list[str] | None = None,
) -> dict[str, Any]:
    """Run the graph once; returns the final ShowState. Works with any Actuators."""
    state = base_state(
        entry=entry,
        trigger=trigger,
        panel_roles=panel_roles,
        listen_role=listen_role,
        human_event=human_event,
        skip_open_floor=skip_open_floor,
        max_turns=int(os.environ.get("TALKSHOW_PANEL_MAX_TURNS", "12")),
        max_moderations=int(os.environ.get("TALKSHOW_HOST_MODERATE_MAX_DEPTH", "8")),
        spoken_roles=spoken_roles,
        **act.snapshot(),
    )
    config = {"configurable": {"act": act}, "recursion_limit": RECURSION_LIMIT}
    return await show_app().ainvoke(state, config)


async def run_show_beat(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    entry: Entry,
    trigger: str,
    human_event: HumanEvent | None = None,
    skip_open_floor: bool = False,
    spoken_roles: set[str] | None = None,
    claimed: bool = False,
) -> str | None:
    """Run one beat on the live session. Returns ``human`` / ``close`` / None.

    ``panel_chain_running`` marks a beat in progress; ``claimed=True`` means the
    caller already set it (and this call releases it).
    """
    from agent.session.session_lifecycle import should_stop_session_work

    act = LiveActuators(session, data, controller)
    owns_busy = claimed or not data.panel_chain_running
    data.panel_chain_running = True
    if should_stop_session_work(data, session):
        if owns_busy:
            data.panel_chain_running = False
        return None
    act.log("show_beat_start", entry=entry, trigger=trigger, active_role=data.active_role)
    try:
        final = await run_beat(
            act,
            entry=entry,
            trigger=trigger,
            panel_roles=controller.panel_speaker_roles(),
            listen_role=controller.listen_role(),
            human_event=human_event,
            skip_open_floor=skip_open_floor,
            spoken_roles=sorted(spoken_roles or ()),
        )
    except RuntimeError as exc:
        if "isn't running" in str(exc):
            logger.info("show beat stopped — session no longer running")
            return None
        raise
    finally:
        if owns_busy:
            data.panel_chain_running = False
    outcome = final.get("outcome") or None
    act.log(
        "show_beat_done",
        entry=entry,
        trigger=trigger,
        outcome=outcome,
        turns=final.get("turn_idx", 0),
        moderations=final.get("moderations", 0),
        lines=len(final.get("spoken") or []),
        appraisals=len(final.get("appraisals") or []),
        stopped=bool(final.get("stopped")),
    )
    return outcome


async def submit_human_turn(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    event: HumanEvent,
    *,
    background: bool,
) -> bool:
    """Hand a finished human turn to the graph (host reply → panel round).

    ``background=True`` (mic path) returns immediately so the STT can hand its
    transcript back to LiveKit; the ghost path awaits the whole beat.
    """
    if data.panel_chain_running:
        logger.info("human turn ignored — show beat already running")
        if data.turn_log is not None:
            data.turn_log.log("human_turn_skip", reason="beat_running", room=data.room_name)
        return False
    data.panel_chain_running = True  # claim before scheduling: no double start

    show = _live_session(data)
    if show is not None:
        fut = show.post("human", event=event)
        if not background:
            await fut
        return True

    async def _run() -> None:
        listen = controller.listen_role()
        try:
            if data.turn_log is not None:
                data.turn_log.log(
                    "panel_start",
                    floor_next=data.floor_next_speaker or "host",
                    room=data.room_name,
                    active_role=data.active_role,
                )
            await run_show_beat(
                session, data, controller, entry="human", trigger="after_human",
                human_event=event, claimed=True,
            )
            if data.active_role != listen:
                from agent.session.session_handoff import switch_to_role

                await switch_to_role(session, data, listen, reason="panel_wait_human")
            if data.turn_log is not None:
                data.turn_log.log("panel_done", room=data.room_name, active_role=data.active_role)
        except Exception:
            logger.exception("show beat (human turn) failed")
        finally:
            data.panel_chain_running = False
            data.panel_followup_pending = False
            data.touch_activity()

    if background:
        asyncio.create_task(_run(), name="show_beat:human")
    else:
        await _run()
    return True


# --- compatibility entry points (callers outside the graph) ----------------


async def run_host_moderation_from_queue(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    trigger: str = "host_moderate",
    skip_open_floor: bool = False,
    depth: int = 0,
    spoken_roles: set[str] | None = None,
) -> str | None:
    """Host takes the floor and moderates until the human gets it back (or close)."""
    show = _live_session(data)
    if show is not None:
        result = await show.post("moderate", trigger=trigger, skip_open_floor=skip_open_floor)
        return (result or {}).get("last_outcome") or None
    return await run_show_beat(
        session,
        data,
        controller,
        entry="moderate",
        trigger=trigger,
        skip_open_floor=skip_open_floor,
        spoken_roles=spoken_roles,
    )


async def run_host_moderated_panel(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Panel round routed by the current [next] floor tag."""
    await run_show_beat(session, data, controller, entry="panel", trigger="after_human")


# --- long-lived session graph (phase 3) --------------------------------------


def session_graph_enabled() -> bool:
    """``TALKSHOW_SHOW_SESSION_GRAPH=0`` falls back to one graph invocation per beat."""
    return os.environ.get("TALKSHOW_SHOW_SESSION_GRAPH", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _live_session(data: TalkShowData) -> Any:
    show = getattr(data, "show_session", None)
    return show if show is not None and show.running else None


class LiveSessionHooks:
    """LiveKit-side bookkeeping around each session-graph beat."""

    def __init__(self, session: AgentSession, data: TalkShowData, controller: TurnController) -> None:
        self.session = session
        self.data = data
        self.controller = controller

    def _log(self, event: str, **fields: Any) -> None:
        if self.data.turn_log is not None:
            self.data.turn_log.log(event, room=self.data.room_name, **fields)

    async def before_beat(self, event: dict[str, Any]) -> None:
        self.data.panel_chain_running = True
        if event["kind"] == "human":
            self._log(
                "panel_start",
                floor_next=self.data.floor_next_speaker or "host",
                active_role=self.data.active_role,
            )
        self._log(
            "show_beat_start",
            entry=event["kind"],
            trigger=event.get("trigger") or event["kind"],
            active_role=self.data.active_role,
            graph="session",
        )

    async def after_beat(self, event: dict[str, Any], result: dict[str, Any] | None) -> None:
        r = result or {}
        try:
            if event["kind"] == "human":
                listen = self.controller.listen_role()
                if self.data.active_role != listen:
                    from agent.session.session_handoff import switch_to_role

                    await switch_to_role(self.session, self.data, listen, reason="panel_wait_human")
                self._log("panel_done", active_role=self.data.active_role)
        except Exception:
            logger.exception("show session after_beat failed")
        finally:
            self._log(
                "show_beat_done",
                entry=event["kind"],
                trigger=event.get("trigger") or event["kind"],
                outcome=r.get("last_outcome") or None,
                turns=r.get("turn_idx", 0),
                moderations=r.get("moderations", 0),
                lines=len(r.get("spoken") or []),
                appraisals=len(r.get("appraisals") or []),
                stopped=bool(r.get("stopped")),
                beat=r.get("beats", 0),
                ok=result is not None,
                graph="session",
            )
            self.data.panel_chain_running = False
            self.data.panel_followup_pending = False
            self.data.touch_activity()


def start_show_session(
    session: AgentSession, data: TalkShowData, controller: TurnController
) -> Any:
    """Start the session graph for host-moderated shows (idempotent)."""
    from agent.show_graph.session import ShowSession

    existing = getattr(data, "show_session", None)
    if existing is not None and existing.running:
        return existing
    show = ShowSession(
        LiveActuators(session, data, controller),
        panel_roles=controller.panel_speaker_roles(),
        listen_role=controller.listen_role(),
        session_id=f"{data.room_name or 'room'}:{int(time.time())}",
        hooks=LiveSessionHooks(session, data, controller),
        shutdown=data.shutdown_event,
        max_turns=int(os.environ.get("TALKSHOW_PANEL_MAX_TURNS", "12")),
        max_moderations=int(os.environ.get("TALKSHOW_HOST_MODERATE_MAX_DEPTH", "8")),
    )
    data.show_session = show
    show.start()
    logger.info("show session graph started id=%s", show._session_id)
    return show


async def open_show_session(
    session: AgentSession, data: TalkShowData, controller: TurnController, *, room: Any = None
) -> bool:
    """Start the session graph and run its opening beat (welcome → moderation).

    Returns False when the session graph is off or not applicable (caller falls back).
    """
    if not (session_graph_enabled() and controller.is_host_moderated_mode()):
        return False
    show = start_show_session(session, data, controller)
    if data.silent_handoff or os.environ.get("TALKSHOW_SKIP_GREETING", "").lower() in (
        "1", "true", "yes",
    ):
        return True
    if room is not None:
        from agent.locale.viewer_locales import recompute_needed_locales, wait_for_remote_humans

        await wait_for_remote_humans(room)
        needed = recompute_needed_locales(data, room)
        logger.info("session opening locales=%s", sorted(needed))
    from agent.floor.host_floor import session_welcome_line

    text, texts = session_welcome_line(data)
    await show.post("opening", text=text, texts=texts)
    return True
