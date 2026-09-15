#!/usr/bin/env python3
"""
Compare floor-related JSONL event *types* (order) against a reference session.

Usage:
  PYTHONPATH=. python tests/show_graph/check_floor_jsonl_order.py \\
    logs/session-REF.jsonl logs/session-NEW.jsonl

Exit 0 if the filtered event-type sequences match; 1 otherwise.
Prints a unified diff on mismatch.

Babel smoke (manual):
  1. pip install -r requirements.txt  # includes langgraph
  2. Restart agent worker; run welcome → human → poll → panel → human hand
  3. Compare new log to a pre-migration reference with this script
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FLOOR_EVENTS = frozenset(
    {
        "hand_raise_poll",
        "hand_raise_poll_skip",
        "hand_raise_wait_skip",
        "hand_raise_grant_pause",
        "queue_fifo_pick",
        "open_floor_skip",
        "floor_grant_human",
        "floor_return_host",
        "floor_grant",  # if logged as event name
    }
)


def floor_event_types(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        name = ev.get("event") or ev.get("type") or ""
        if name in FLOOR_EVENTS:
            out.append(name)
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    ref = Path(sys.argv[1])
    new = Path(sys.argv[2])
    a = floor_event_types(ref)
    b = floor_event_types(new)
    if a == b:
        print(f"OK — {len(a)} floor events match")
        return 0
    import difflib

    diff = difflib.unified_diff(
        a, b, fromfile=str(ref), tofile=str(new), lineterm=""
    )
    print("\n".join(diff))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
