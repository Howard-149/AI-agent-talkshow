"""ShowSession — one long-lived show graph per talk-show session.

The graph starts once, then pauses in ``wait_event`` (LangGraph ``interrupt``).
Event producers (session opening, human turns, the idle loop) ``post()`` events;
the driver resumes the graph with ``Command(resume=event)``, the graph runs one
beat and pauses again, and the driver resolves the event's future with how the
beat ended. Events are handled one at a time, in order.

Kept free of LiveKit: side effects go through the graph's ``Actuators``, and the
session-level bookkeeping around each beat through ``SessionHooks``.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Protocol

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from agent.show_graph.actuators import Actuators
from agent.show_graph.graph import build_show_graph
from agent.show_graph.state import base_state

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 1000
# Types that live in checkpointed state (beat payloads); everything else is plain data.
CHECKPOINT_TYPES = [
    ("agent.emotion.appraisal", "Utterance"),
    ("agent.adapters.response_parser", "ParsedTurn"),
]


class SessionHooks(Protocol):
    async def before_beat(self, event: dict[str, Any]) -> None: ...
    async def after_beat(self, event: dict[str, Any], result: dict[str, Any] | None) -> None: ...


class _NoHooks:
    async def before_beat(self, event: dict[str, Any]) -> None:
        return None

    async def after_beat(self, event: dict[str, Any], result: dict[str, Any] | None) -> None:
        return None


class ShowSession:
    def __init__(
        self,
        act: Actuators,
        *,
        panel_roles: list[str],
        listen_role: str,
        session_id: str,
        hooks: SessionHooks | None = None,
        shutdown: asyncio.Event | None = None,
        max_turns: int = 12,
        max_moderations: int = 8,
    ) -> None:
        self._act = act
        self._panel_roles = list(panel_roles)
        self._listen_role = listen_role
        self._session_id = session_id
        self._hooks = hooks or _NoHooks()
        self._shutdown = shutdown
        self._max_turns = max_turns
        self._max_moderations = max_moderations
        self._queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Future]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._generation = itertools.count()
        self._app: Any = None
        self._config: dict[str, Any] = {}
        self.running = False
        self.beats = 0

    # --- public ------------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self.running = True
            self._task = asyncio.create_task(self._run(), name=f"show_session:{self._session_id}")

    def post(self, kind: str, **payload: Any) -> asyncio.Future:
        """Queue an event; the future resolves to the beat result (a dict) or None."""
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        if not self.running:
            logger.info("show session not running — dropped event kind=%s", kind)
            fut.set_result(None)
            return fut
        self._queue.put_nowait(({"kind": kind, **payload}, fut))
        return fut

    async def stop(self, timeout: float = 10.0) -> None:
        if self._task is None:
            return
        if self.running:
            self.post("stop")
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("show session did not stop in %.0fs — cancelling", timeout)
            self._task.cancel()

    # --- driver ------------------------------------------------------------

    async def _boot(self) -> None:
        """Fresh graph + checkpointer, run up to the first wait_event pause."""
        gen = next(self._generation)
        saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
        self._app = build_show_graph(checkpointer=saver)
        self._config = {
            "configurable": {"act": self._act, "thread_id": f"{self._session_id}:{gen}"},
            "recursion_limit": RECURSION_LIMIT,
        }
        state = base_state(
            entry="session",
            trigger="session",
            panel_roles=self._panel_roles,
            listen_role=self._listen_role,
            max_turns=self._max_turns,
            max_moderations=self._max_moderations,
            **self._act.snapshot(),
        )
        await self._app.ainvoke(state, self._config, durability="exit")

    async def _next(self) -> tuple[dict[str, Any], asyncio.Future | None]:
        """Next queued event — a session shutdown always wins over queued work."""
        stop: tuple[dict[str, Any], asyncio.Future | None] = (
            {"kind": "stop", "reason": "shutdown"},
            None,
        )
        if self._shutdown is None:
            return await self._queue.get()
        if self._shutdown.is_set():
            return stop
        get = asyncio.ensure_future(self._queue.get())
        down = asyncio.ensure_future(self._shutdown.wait())
        done, _ = await asyncio.wait({get, down}, return_when=asyncio.FIRST_COMPLETED)
        if down in done:
            if get in done:
                self._queue.put_nowait(get.result())  # released by the drain on exit
            else:
                get.cancel()
            return stop
        down.cancel()
        return get.result()

    async def _run(self) -> None:
        try:
            await self._boot()
            while True:
                event, fut = await self._next()
                if event["kind"] == "stop":
                    logger.info(
                        "show session stopping id=%s beats=%d reason=%s",
                        self._session_id, self.beats, event.get("reason") or "stop",
                    )
                    await self._resume(event)
                    if fut is not None and not fut.done():
                        fut.set_result(None)
                    break
                result: dict[str, Any] | None = None
                await self._hooks.before_beat(event)
                try:
                    result = await self._resume(event)
                    if result is None or "__interrupt__" not in result:
                        logger.warning("show session graph ended unexpectedly — restarting")
                        await self._boot()
                    else:
                        self.beats += 1
                except Exception:
                    logger.exception("show session beat failed kind=%s — restarting graph", event["kind"])
                    await self._boot()
                finally:
                    await self._hooks.after_beat(event, result)
                    if fut is not None and not fut.done():
                        fut.set_result(result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("show session driver crashed")
        finally:
            self.running = False
            # Events that never ran still get after_beat, so producers' busy claims
            # (e.g. a human turn) are released.
            while not self._queue.empty():
                event, fut = self._queue.get_nowait()
                if event.get("kind") != "stop":
                    try:
                        await self._hooks.after_beat(event, None)
                    except Exception:
                        logger.exception("after_beat for dropped event failed")
                if fut is not None and not fut.done():
                    fut.set_result(None)

    async def _resume(self, event: dict[str, Any]) -> dict[str, Any] | None:
        return await self._app.ainvoke(Command(resume=event), self._config, durability="exit")
