"""Ending a ghost session cleanly instead of leaving mid-sentence.

1. The guest says a short goodbye (LLM-written; 15–30 words) at the next natural gap:
   after a line has ended, nothing playing, and not right after a line that handed the
   floor to someone else. It is written and synthesized up front.
2. As his goodbye starts playing, the ghost sends ``{"type": "wrap_up"}`` on
   ``talkshow/control``: the worker skips the panel's reaction to the goodbye and goes
   to the host's closing line (``close_round``) once he is done.
3. The ghost leaves once that line has ended and no agent audio has played for ~3 s.
   Fallback: no ``close_round`` within 90 s of wrap_up → leave once nothing is playing
   and nothing has been announced for 5 s. Never while a line is playing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable

from eval.ghost_session.addressing import addressing

logger = logging.getLogger("ghost_session")

GAP_PAUSE_S = 0.6
GAP_MAX_WAIT_S = 60.0  # after this, any moment with nothing playing will do
CLOSE_QUIET_S = 3.0
CLOSE_FALLBACK_S = 90.0
FALLBACK_QUIET_S = 5.0
END_MAX_S = 240.0  # after wrap_up: leave at the next quiet moment regardless


def in_gap(client: Any, *, pause: float = GAP_PAUSE_S, strict: bool = True) -> bool:
    """A natural moment for the guest to speak without cutting anyone in."""
    now = time.monotonic()
    if client.playing or now - client.idle_since < pause:
        return False
    if not strict:
        return True
    started = list(client.line_started_at)
    if started and max(started) not in client.line_ended_at:
        return False  # the line on air is only between two of its sentences
    if client.line_ended_at:
        last = client.agent_line(max(client.line_ended_at))
        if last is not None and client.guest_name:
            if addressing(last.text, client.guest_name, client.known_names()).startswith("other:"):
                return False  # that gap belongs to whoever was just asked
    return True


async def wait_for_gap(client: Any, *, pause: float = GAP_PAUSE_S, max_wait: float = GAP_MAX_WAIT_S) -> None:
    t0 = time.monotonic()
    if await client._until(lambda: in_gap(client, pause=pause), t0 + max_wait):
        return
    logger.info("no natural gap in %.0fs — taking the next quiet moment", max_wait)
    await client._until(lambda: in_gap(client, pause=pause, strict=False), t0 + max_wait * 3)


def show_over(client: Any, wrap_t: float) -> bool:
    now = time.monotonic()
    if client.playing:
        return False
    quiet_since = client.idle_since
    if client.close_round_ended_at is not None:
        return now - max(quiet_since, client.close_round_ended_at) >= CLOSE_QUIET_S
    last_announced = max(client.line_announced_at.values(), default=0.0)
    if now - wrap_t >= CLOSE_FALLBACK_S:
        return now - max(quiet_since, last_announced) >= FALLBACK_QUIET_S
    return now - wrap_t >= END_MAX_S


async def wait_for_show_end(client: Any, wrap_t: float) -> str:
    await client._until(lambda: show_over(client, wrap_t), wrap_t + END_MAX_S + 120)
    if client.close_round_ended_at is not None:
        why = "close_round ended"
    else:
        why = "no close_round (fallback)"
    logger.info("show over (%s) %.1fs after wrap_up — leaving", why, time.monotonic() - wrap_t)
    return why


async def goodbye_and_wrap_up(
    client: Any,
    *,
    prepare: Callable[[], Awaitable[tuple[str, Any]]],
    speak: Callable[..., Awaitable[Any]],
) -> None:
    """Goodbye at the next gap (wrap_up as it starts) → wait for the host's close."""
    import asyncio

    t0 = time.monotonic()
    task = asyncio.create_task(prepare(), name="ghost:goodbye")
    await wait_for_gap(client)
    text, speech = await task
    logger.info("goodbye (ready in %.1fs): %r", time.monotonic() - t0, text)
    sent: list[asyncio.Task[None]] = []

    def on_start() -> asyncio.Task[None]:
        logger.info("goodbye started — sending wrap_up")
        sent.append(asyncio.create_task(client.send_wrap_up(), name="ghost:wrap_up"))
        return sent[-1]

    await speak(text, speech, on_start=on_start)
    if not sent:  # speech never reported a start: send it now
        on_start()
    await sent[0]
    wrap_t = time.monotonic()
    await wait_for_show_end(client, wrap_t)
