"""Reactive guest: Howard takes part like a panelist instead of only at scripted cues.

On every agent line the worker announces (its ``transcript`` event comes before its
audio, with the full text) the guest decides what to do about it:

* **Floor handed to him** (``addressing.py``: a vocative of his name, or "our guest",
  tied to a question / invitation in the same or next sentence, at the end of the line):
  he must answer, right after the line ends (a 0.4–0.8 s pause), even if the next line
  has already started (a barge-in, like a person who was just asked).
* **Floor handed to someone else** ("…Amy, how do you see it?"): he stays silent, no LLM
  call, and does not take the gap after it either.
* **Asked in passing** (a question tied to his name mid-line): the LLM decides; the
  cooldown does not apply.
* **Otherwise** (a bare "Howard, …" vocative, a mention, or nothing): skipped without an
  LLM call while in cooldown (``min_lines_between_turns`` lines since his last turn);
  else one LLM call decides ``silent`` (default) / ``respond`` (after the line) /
  ``cut_in`` (rare, a few seconds into the line) and writes the words in the same call.
  The prompt pushes him to speak up when he has an experience, a disagreement or a
  question, and to usually respond when mentioned.
* **Nudge**: once he has been quiet for ``nudge_after_lines`` agent lines, the next
  eligible line (not handed to someone else, outside the cooldown) must be answered,
  after the line ends (never a cut-in).

The words are written and synthesized as soon as the line is announced, so the audio is
usually ready by its end; he speaks in the gap after it (a next line that is only queued
is rolled back by his barge-in on the worker side). If the next line is already playing
by then, an unaddressed reply is dropped and the newest line is considered instead (the
re-check); an answer to a direct question is given anyway. A reply goes stale, and is
dropped, when more than one newer line has been announced, when the next line has been
playing for 2 s while this line's end was never seen, or 15 s after its line ended. He
never speaks twice without an agent line announced in between, and never while his
previous turn is still playing.

Stops after ``max_turns`` turns (seed turns included) or ``max_minutes`` from the first
agent line; the caller then leaves after ``--stop-after-panel`` lines as before.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from eval.ghost_session.addressing import addressing
from eval.ghost_session.live import TranscriptLine

logger = logging.getLogger("ghost_session")

PAUSE_RANGE_S = (0.4, 0.8)
CUT_IN_AFTER_S = 2.5  # how far into a line a cut-in starts (from its announcement)
MAX_LINE_WAIT_S = 60.0  # a line whose end never arrives
# A reply goes stale: the next line has been playing this long while this line's end was
# never seen, more than one newer line has been announced, or its line ended this long ago.
STALE_AFTER_NEXT_PLAYS_S = 2.0
MAX_NEWER_LINES = 1
MAX_REPLY_AGE_S = 15.0
MAX_CUT_INS = 2


@dataclass
class ReactiveLimits:
    max_turns: int = 12
    min_lines_between_turns: int = 3
    max_minutes: float = 9.5
    # The nudge: after this many agent lines without him speaking, the next eligible line
    # (not handed to someone else, outside the cooldown) gets an answer (0 = off).
    nudge_after_lines: int = 5


@dataclass
class PreparedReply:
    action: str  # silent / respond / cut_in
    emotion: str
    text: str
    speech: tuple[Any, Any] | None  # (voice.Speech, wav) or None for a text turn
    gen_s: float
    tts_s: float


# --- the controller ------------------------------------------------------------------


class ReactiveGuest:
    def __init__(
        self,
        client: Any,
        *,
        name: str,
        limits: ReactiveLimits,
        prepare: Callable[[list[dict[str, str]], bool], Awaitable[PreparedReply]],
        speak: Callable[[str, Any], Awaitable[float | None]],
        build_messages: Callable[..., list[dict[str, str]]],
        turns_done: int = 0,
        show_started_t: float | None = None,
        pause_range: tuple[float, float] = PAUSE_RANGE_S,
        cut_in_after_s: float = CUT_IN_AFTER_S,
        max_line_wait_s: float = MAX_LINE_WAIT_S,
    ) -> None:
        self.client = client
        self.name = name
        self.limits = limits
        self._prepare = prepare  # (messages, must_answer) → PreparedReply
        self._speak = speak  # (text, speech) → monotonic start of the audio, or None
        self._build = build_messages
        self.turns_done = turns_done
        self.show_started_t = show_started_t if show_started_t is not None else time.monotonic()
        self.pause_range = pause_range
        self.cut_in_after_s = cut_in_after_s
        self.max_line_wait_s = max_line_wait_s
        self.cut_ins = 0
        self.decisions: list[dict[str, Any]] = []  # one per agent line considered (tests, logs)
        self._stopped = False

    def stop(self) -> None:
        """Stop reacting (after the line being handled)."""
        self._stopped = True
        self.client._changed.set()

    def _deadline(self) -> float:
        return self.show_started_t + self.limits.max_minutes * 60.0

    def _budget_left(self) -> bool:
        return (
            not self._stopped
            and self.turns_done < self.limits.max_turns
            and time.monotonic() < self._deadline()
        )

    async def run(self, *, last_turn_ordinal: int | None = None) -> int:
        """React until the budget runs out; returns how many turns he spoke here."""
        c = self.client
        # Lines announced since his last turn started (a reply to it, say) are still open
        # to react to; without one, everything up to now is past.
        processed = c.agent_lines if last_turn_ordinal is None else last_turn_ordinal
        last_turn = processed
        spoken_here = 0
        while self._budget_left():
            if not await c._until(lambda: c.agent_lines > processed or self._stopped, self._deadline()):
                break
            if self._stopped:
                break
            n = processed = c.agent_lines  # the newest line; older unhandled ones are past
            line = c.agent_line(n)
            if line is None or n in c.line_dropped or not line.text:
                continue
            addr = addressing(line.text, self.name, c.known_names())
            since = n - last_turn
            record: dict[str, Any] = {"line": n, "speaker": line.speaker or line.role, "addressed": addr or "no"}
            self.decisions.append(record)
            if addr.startswith("other:"):
                # The floor goes to someone else: not his to take (nor the gap after it).
                record["decision"] = "silent (floor to %s)" % addr[6:]
                self._log(record, line)
                continue
            if addr not in ("direct", "asked") and since < self.limits.min_lines_between_turns:
                record["decision"] = "silent (cooldown)"
                self._log(record, line)
                continue
            nudge = (
                addr != "direct"
                and self.limits.nudge_after_lines > 0
                and since >= max(self.limits.nudge_after_lines, self.limits.min_lines_between_turns)
            )
            if nudge:
                record["nudge"] = True
            kwargs: dict[str, Any] = dict(
                transcript=c.transcript_through(n)[:-1],
                line=line,
                addressed=addr,
                turns_spoken=self.turns_done,
                lines_since_turn=since,
                allow_cut_in=not nudge and addr in ("", "mentioned") and self.cut_ins < MAX_CUT_INS,
                cut_after_sec=self.cut_in_after_s,
            )
            if nudge:
                kwargs["must_respond"] = True
            messages = self._build(**kwargs)
            must = addr == "direct" or nudge
            task = asyncio.create_task(self._prepare(messages, must), name=f"ghost:react-{n}")
            reply, why = await self._wait_to_speak(n, addr == "direct", task)
            if reply is not None:
                record.update(gen_s=round(reply.gen_s, 1), tts_s=round(reply.tts_s, 1), text=reply.text)
            if reply is None or why != "speak":
                record["decision"] = why if reply is None or reply.action == "silent" else f"{reply.action} → dropped ({why})"
                self._log(record, line)
                continue
            # Speak. Lines announced from here on are new to him (an answer to him, say).
            ended = c.line_ended_at.get(n)
            speak_ordinal = c.agent_lines
            started = await self._speak(reply.text, reply.speech)
            if reply.action == "cut_in":
                self.cut_ins += 1
            self.turns_done += 1
            spoken_here += 1
            last_turn = processed = speak_ordinal
            record["decision"] = reply.action
            if started is not None:
                ref = ended if ended is not None and reply.action != "cut_in" else c.line_announced_at.get(n, started)
                record["latency_s"] = round(started - ref, 2)
            self._log(record, line)
        logger.info(
            "reactive: done (%d turns in all, %.1f min since the first line)",
            self.turns_done, (time.monotonic() - self.show_started_t) / 60,
        )
        return spoken_here

    async def _wait_to_speak(
        self, n: int, direct: bool, task: asyncio.Task[PreparedReply]
    ) -> tuple[PreparedReply | None, str]:
        """(reply, "speak") when it is time to say it; else (reply or None, why not)."""
        c = self.client
        pause = random.uniform(*self.pause_range)
        announced = c.line_announced_at.get(n, time.monotonic())
        reply: PreparedReply | None = None
        try:
            while True:
                now = time.monotonic()
                if reply is None and task.done():
                    try:
                        reply = task.result()
                    except Exception:
                        logger.exception("reactive: preparing a reply to line %d failed", n)
                        return None, "error"
                    if reply.action == "silent":
                        return reply, "silent"
                ended_t = c.line_ended_at.get(n)
                started_t = c.line_started_at.get(n)
                if n in c.line_dropped:  # never played: nobody heard it, him included
                    return reply, "line dropped"
                stale = self._stale(n, ended_t, now)
                if stale:
                    return reply, stale
                # Another line went on air after this one ended: he would talk over it.
                next_on_air = ended_t is not None and c.playing and c.playing_since > ended_t
                if reply is not None and reply.action == "cut_in" and ended_t is None:
                    if started_t is not None and now - started_t >= self.cut_in_after_s:
                        return reply, "speak"
                elif reply is not None and ended_t is not None and now - ended_t >= pause:
                    # The gap after the line (a queued next line is rolled back by his
                    # barge-in). Addressed: answer even over the next line — he was asked.
                    if direct or not next_on_air:
                        return reply, "speak"
                if next_on_air and not direct:
                    return reply, "next line on air"
                if now - announced > self.max_line_wait_s:
                    return reply, "line never ended"
                c._changed.clear()
                try:
                    await asyncio.wait_for(c._changed.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
        finally:
            if not task.done():
                task.cancel()

    def _stale(self, n: int, ended_t: float | None, now: float) -> str:
        """Why a reply to line ``n`` is too late to say now ("" = still fresh)."""
        c = self.client
        if c.agent_lines - n > MAX_NEWER_LINES:
            return "stale (newer lines announced)"
        if ended_t is None:
            later = [t for o, t in c.line_started_at.items() if o > n]
            if later and now - min(later) > STALE_AFTER_NEXT_PLAYS_S:
                return "stale (next line playing, end never seen)"
        elif now - ended_t > MAX_REPLY_AGE_S:
            return "stale (line ended too long ago)"
        return ""

    def _log(self, record: dict[str, Any], line: TranscriptLine) -> None:
        logger.info(
            "react line %d %s addressed=%s%s → %s%s%s%s %r",
            record["line"], record["speaker"], record["addressed"],
            " (nudge)" if record.get("nudge") else "", record.get("decision"),
            f" gen={record['gen_s']}s" if "gen_s" in record else "",
            f" tts={record['tts_s']}s" if "tts_s" in record else "",
            f" latency={record['latency_s']}s" if "latency_s" in record else "",
            line.text[:70],
        )
