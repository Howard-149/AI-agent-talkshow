"""Show graph runtime tests with fake actuators (no LiveKit / Gemma)."""

from __future__ import annotations

import asyncio
import time
import unittest
from dataclasses import dataclass
from typing import Any

from agent.emotion.appraisal import Utterance
from agent.show_graph.runner import run_beat
from agent.show_graph.state import FloorPick, PanelistLine

PANEL = ["commentator", "guest"]
AI = ["host", *PANEL]


@dataclass
class Event:
    t: float
    kind: str
    detail: Any


class FakeActuators:
    """Scripted world: floor tags, hand-raise picks, and panelist [next] tags."""

    def __init__(
        self,
        *,
        floor: list[str] | None = None,
        picks: list[FloorPick] | None = None,
        tags: dict[str, list[str]] | None = None,
        queue: list[str] | None = None,
        speak_s: float = 0.0,
        appraise_s: float = 0.0,
        stop_after_lines: int | None = None,
    ) -> None:
        self.floor = list(floor or [])
        self.picks = list(picks or [])
        self.tags = {k: list(v) for k, v in (tags or {}).items()}
        self.queue = list(queue or [])
        self.speak_s = speak_s
        self.appraise_s = appraise_s
        self.stop_after_lines = stop_after_lines
        self.events: list[Event] = []
        self.history: list[tuple[str, str, bool]] = []
        self.floor_next = ""

    def _ev(self, kind: str, detail: Any = None) -> None:
        self.events.append(Event(time.monotonic(), kind, detail))

    def kinds(self, kind: str) -> list[Any]:
        return [e.detail for e in self.events if e.kind == kind]

    # session / logging
    def should_stop(self) -> bool:
        return self.stop_after_lines is not None and len(self.kinds("speak")) >= self.stop_after_lines

    def log(self, event: str, **fields: Any) -> None:
        self._ev("log", event)

    def snapshot(self) -> dict[str, Any]:
        return {}

    # human turn
    async def commit_human_turn(self, event):
        # Like the real commit: after a human turn the host moderates next.
        self.floor_next = "host"
        self.history.append(("human", event["heard"], False))
        return event.get("reply", "Great question — panel?"), Utterance("human", event["heard"])

    # lines
    def record_line(self, line):
        self.history.append((line["role"], line["text"], bool(line.get("procedural"))))
        if line.get("procedural"):
            return None
        return Utterance(line["role"], line["text"])

    def relax(self, role: str) -> None:
        self._ev("relax", role)

    def listeners(self, speaker: str) -> list[str]:
        return [r for r in AI if r != speaker]

    async def speak(self, line) -> None:
        self._ev("speak", (line["kind"], line["role"]))
        await asyncio.sleep(self.speak_s)
        self._ev("speak_done", line["kind"])

    async def appraise(self, utt, listener):
        self._ev("appraise", (utt.speaker, listener))
        await asyncio.sleep(self.appraise_s)
        self._ev("appraise_done", (utt.speaker, listener))
        return {"role": listener, "ok": True}

    async def after_line(self, line) -> None:
        self._ev("after_line", line["kind"])
        if line["kind"] == "panelist":
            self.floor_next = line.get("next_tag") or "host"
        elif line["kind"] == "grant_human":
            self.floor_next = "human"

    # floor
    def consume_floor_next(self) -> str:
        if self.floor:
            return self.floor.pop(0)
        nxt, self.floor_next = (self.floor_next or "host"), ""
        return nxt

    def queue_roles(self) -> list[str]:
        return list(self.queue)

    async def return_to_host(self, trigger: str) -> None:
        self._ev("return_to_host", trigger)

    async def hand_raise_round(self, turn_idx, spoken_roles) -> FloorPick:
        self._ev("hand_raise", turn_idx)
        self.queue = []
        return self.picks.pop(0) if self.picks else FloorPick(None, "")

    async def grant_panelist(self, role: str, reason: str) -> None:
        self._ev("grant", (role, reason))

    async def generate_panelist_line(self, role: str, step: str) -> PanelistLine:
        self._ev("generate", role)
        tag = self.tags.get(role, ["host"]).pop(0) if self.tags.get(role) else "host"
        return PanelistLine(role=role, text=f"{role} says something", next_role=tag, emotion="Neutral")

    # canned
    def open_floor_line(self) -> str:
        return "Floor's open."

    def intro_line(self, role: str) -> str:
        return f"{role}, you're up."

    def human_floor_line(self) -> str:
        return "Back to you."

    def close_line(self) -> str:
        return "That wraps it."


def run(act: FakeActuators, **kw: Any) -> dict[str, Any]:
    kw.setdefault("entry", "human")
    kw.setdefault("trigger", "after_human")
    return asyncio.run(run_beat(act, panel_roles=PANEL, listen_role="host", **kw))


HUMAN = {"heard": "Should we freeze senior hiring?", "parsed": None}


class HumanTurnFlowTests(unittest.TestCase):
    def test_full_beat_order_and_outcome(self) -> None:
        act = FakeActuators(
            picks=[FloorPick("guest", "queue_fifo"), FloorPick("human", "queue_fifo")],
            tags={"guest": ["commentator"], "commentator": ["host"]},
        )
        final = run(act, human_event=HUMAN)
        self.assertEqual(
            act.kinds("speak"),
            [
                ("host_reply", "host"),
                ("open_floor", "host"),
                ("intro", "host"),
                ("panelist", "guest"),
                ("panelist", "commentator"),  # chained via [next:commentator], no intro
                ("open_floor", "host"),
                ("grant_human", "host"),
            ],
        )
        self.assertEqual(final["outcome"], "human")
        self.assertEqual(final["turn_idx"], 2)
        self.assertEqual(final["spoken_roles"], ["guest", "commentator"])
        self.assertEqual(act.kinds("grant"), [("guest", "queue_fifo"), ("commentator", "next_tag")])

    def test_only_substantive_lines_are_appraised(self) -> None:
        act = FakeActuators(
            picks=[FloorPick("guest", "queue_fifo"), FloorPick("human", "queue_fifo")],
        )
        run(act, human_event=HUMAN)
        appraised = act.kinds("appraise")
        # Human line → all three AI roles; host reply → panelists; guest line → others.
        self.assertEqual(
            sorted(appraised),
            sorted(
                [("human", r) for r in AI]
                + [("host", r) for r in PANEL]
                + [("guest", "host"), ("guest", "commentator")]
            ),
        )
        # Canned lines (open floor, intro, grant) never reach appraisal; speakers relax.
        self.assertEqual(act.kinds("relax"), ["host", "guest"])

    def test_join_waits_for_appraisals_before_next_speaker(self) -> None:
        act = FakeActuators(
            picks=[FloorPick("guest", "queue_fifo"), FloorPick("human", "x")],
            speak_s=0.01,
            appraise_s=0.08,
        )
        run(act, human_event=HUMAN)
        last_human_appraisal = max(
            e.t for e in act.events if e.kind == "appraise_done" and e.detail[0] in ("human", "host")
        )
        guest_generate = next(e.t for e in act.events if e.kind == "generate")
        self.assertGreater(guest_generate, last_human_appraisal)

    def test_appraisals_run_in_parallel_with_tts(self) -> None:
        act = FakeActuators(picks=[FloorPick("human", "x")], speak_s=0.1, appraise_s=0.1)
        t0 = time.monotonic()
        run(act, human_event=HUMAN)
        # host reply: 1 speak + 5 appraisals (human×3, host×2) of 0.1 s each — in parallel.
        first_line = [e for e in act.events if e.kind in ("speak_done", "appraise_done")][:6]
        self.assertLess(max(e.t for e in first_line) - t0, 0.35)


class ModerationTests(unittest.TestCase):
    def test_direct_call_line_replaces_intro_and_is_appraised(self) -> None:
        act = FakeActuators(
            floor=["host"],
            picks=[
                FloorPick("guest", "host_direct_no_raises", host_line="Amy, what do you think?"),
                FloorPick("human", "x"),
            ],
        )
        run(act, entry="panel")
        self.assertEqual(
            [k for k, _ in act.kinds("speak")],
            ["open_floor", "direct_call", "panelist", "open_floor", "grant_human"],
        )
        self.assertIn(("host", "guest"), act.kinds("appraise"))

    def test_queue_nonempty_skips_open_floor(self) -> None:
        act = FakeActuators(queue=["guest"], picks=[FloorPick("human", "queue_fifo")])
        run(act, entry="moderate", trigger="idle_wait")
        self.assertEqual([k for k, _ in act.kinds("speak")], ["grant_human"])
        self.assertIn("open_floor_skip", act.kinds("log"))

    def test_session_start_skips_open_floor_line(self) -> None:
        act = FakeActuators(picks=[FloorPick("human", "queue_fifo")])
        final = run(act, entry="moderate", trigger="session_start", skip_open_floor=True)
        self.assertEqual([k for k, _ in act.kinds("speak")], ["grant_human"])
        self.assertEqual(final["outcome"], "human")

    def test_nobody_picked_ends_beat(self) -> None:
        act = FakeActuators(picks=[FloorPick(None, "")])
        final = run(act, entry="moderate", trigger="idle_wait")
        self.assertEqual([k for k, _ in act.kinds("speak")], ["open_floor"])
        self.assertEqual(final["outcome"], "")

    def test_close_tag_closes_round(self) -> None:
        act = FakeActuators(floor=["close"])
        final = run(act, entry="panel")
        self.assertEqual([k for k, _ in act.kinds("speak")], ["close"])
        self.assertEqual(final["outcome"], "close")


class LimitTests(unittest.TestCase):
    def test_max_turns_closes_a_chain(self) -> None:
        import os
        from unittest import mock

        act = FakeActuators(
            floor=["guest"],
            tags={"guest": ["commentator"] * 10, "commentator": ["guest"] * 10},
        )
        with mock.patch.dict(os.environ, {"TALKSHOW_PANEL_MAX_TURNS": "4"}):
            final = run(act, entry="panel")
        kinds = [k for k, _ in act.kinds("speak")]
        self.assertEqual(kinds, ["panelist"] * 4 + ["close"])
        self.assertEqual(final["outcome"], "close")

    def test_moderation_cap(self) -> None:
        import os
        from unittest import mock

        # Every pick goes to a panelist who hands back to the host → loops.
        act = FakeActuators(picks=[FloorPick("guest", "queue_fifo")] * 20)
        with mock.patch.dict(os.environ, {"TALKSHOW_HOST_MODERATE_MAX_DEPTH": "3"}):
            final = run(act, entry="moderate", trigger="idle_wait")
        self.assertEqual(final["moderations"], 3)
        self.assertEqual(len(act.kinds("hand_raise")), 3)

    def test_session_stop_ends_without_more_speech(self) -> None:
        act = FakeActuators(
            picks=[FloorPick("guest", "queue_fifo"), FloorPick("human", "x")],
            stop_after_lines=2,
        )
        final = run(act, human_event=HUMAN)
        self.assertEqual(len(act.kinds("speak")), 2)
        self.assertTrue(final["stopped"])


if __name__ == "__main__":
    unittest.main()
