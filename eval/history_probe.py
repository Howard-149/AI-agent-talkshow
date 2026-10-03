#!/usr/bin/env python3
"""
History-format probe: do panelists answer as *someone else* or copy speaker labels?

Rebuilds the transcript from real session logs and, at every panelist turn,
regenerates that panelist's line with the live ``PersonaWorker`` N times under
both chat-history formats:

  legacy       every AI line is a labelled ``assistant`` turn ("Amy (Guest): …")
  perspective  the speaker's own lines are unlabelled ``assistant`` turns,
               everyone else's are labelled ``user`` turns

Per format it reports how often the raw output starts with a speaker label,
starts with *another* speaker's label (identity bleed), contains another
speaker's label anywhere, addresses the host by name (it should answer the
room / the human, not Lessac), and carries a parsable [next] tag.

Canned host lines (open floor + "<Name>, take it away.") are re-inserted before
each panelist turn, as the live graph speaks them.

Usage (Babel compute node, vLLM running):
    python eval/history_probe.py logs/session-1790081803.jsonl logs/session-1790083242.jsonl
    python eval/history_probe.py --trials 10 --max-turns 6 logs/session-*.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agent.show.show_history import ShowHistory  # noqa: E402

RESULTS_DIR = _REPO / "eval" / "results" / "history"
LABEL = re.compile(
    r"(?:^|\n|\[reply\]\s*:\s*)\s*(Lessac|Ryan|Amy|Human guest)\s*(?:\([^)]*\))?\s*:", re.I
)
NAMES = {"host": "Lessac", "commentator": "Ryan", "guest": "Amy"}


FORMATS = {
    "legacy": {"TALKSHOW_HISTORY_FORMAT": "legacy"},
    "perspective": {"TALKSHOW_HISTORY_FORMAT": "perspective"},
}


def transcript_with_turns(path: Path) -> list[tuple[str, str]]:
    """(role, text) lines in order, from the turn events every session logs."""
    lines: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(raw)
        ev = d.get("event")
        if ev in ("ghost_human_done", "gemma_stt_done"):
            if d.get("heard"):
                lines.append(("human", d["heard"]))
            if d.get("reply"):
                lines.append(("host", d["reply"]))
        elif ev == "host_direct_call" and d.get("reply"):
            lines.append(("host", d["reply"]))
        elif ev == "panel_model_done" and d.get("reply"):
            lines.append((d["role"], d["reply"]))
    return lines


class _Log:
    def __init__(self) -> None:
        self.raw: str = ""

    def log(self, event: str, **fields: Any) -> None:
        if event == "panel_model_done":
            self.raw = fields.get("raw") or ""


JUDGE_SYSTEM = "You check who said what in a talk-show transcript. Answer with one word."


async def misattributes_host(client: Any, hist: ShowHistory, role: str, raw: str) -> bool:
    """LLM check: does the line credit the host with someone else's words?

    Answering a question or framing the host really gave is fine; crediting the host
    with the human guest's / another panelist's point is the error.
    """
    from agent.show.show_history import _line_label

    context = "\n".join(f"{_line_label(ln)}: {ln.text}" for ln in hist.lines[-8:])
    body = re.sub(r"^\s*\[reply\]\s*:\s*", "", raw.strip()).split("[next")[0].strip()
    user = (
        f"Transcript (most recent last):\n{context}\n\n"
        f'New line by {NAMES[role]} (panelist):\n"{body}"\n\n'
        "The host is Lessac. Does the new line credit Lessac with something that was "
        "actually said by the human guest or by another panelist, not by Lessac? "
        "Responding to a question or framing that Lessac really gave is fine.\n"
        "Answer yes or no."
    )
    out = await client.complete_text(
        user, system_prompt=JUDGE_SYSTEM, history_messages=None, raw=True,
        temperature=0.0, max_tokens=4,
    )
    return out.strip().lower().startswith("yes")


def classify(role: str, raw: str) -> dict[str, bool]:
    from agent.adapters.response_parser import parse_host_speech

    labels = [m.group(1).lower() for m in LABEL.finditer(raw)]
    own = NAMES[role].lower()
    starts = LABEL.match(raw.strip())
    first = starts.group(1).lower() if starts else None
    body = re.sub(r"^\s*\[reply\]\s*:\s*", "", raw.strip()).split("[next")[0]
    return {
        "addresses_host": re.search(r"\bLessac\b", body, re.I) is not None,
        "starts_label": first is not None,
        "starts_other": first is not None and first != own,
        "other_label_anywhere": any(x != own for x in labels),
        "has_next": parse_host_speech(raw).next_speaker is not None,
    }


async def run(paths: list[Path], trials: int, max_turns: int) -> dict[str, Any]:
    from agent.config import load_scenario
    from agent.multi_agent.worker import PersonaWorker
    from agent.session.talkshow_runtime import build_runtime

    runtime = build_runtime()
    scenario = load_scenario()
    rows: list[dict[str, Any]] = []
    for path in paths:
        lines = transcript_with_turns(path)
        turns = [i for i, (r, _) in enumerate(lines) if r in ("commentator", "guest")][:max_turns]
        for idx in turns:
            role = lines[idx][0]
            hist = ShowHistory()
            for j, (r, t) in enumerate(lines[:idx]):
                if r in NAMES and r != "host":  # earlier panelist turns were introduced too
                    _add_intro(hist, r)
                hist.append(r, t)
            _add_intro(hist, role)
            for fmt, env in FORMATS.items():
                os.environ.update(env)
                for _ in range(trials):
                    log = _Log()
                    data = SimpleNamespace(
                        runtime=runtime, scenario=scenario, show_history=hist,
                        dialogue_example_index=0, turn_log=log, room_name="probe",
                        role_pad={}, role_emotion={}, role_pad_neighbors={},
                        pending_appraisals={}, active_role=role,
                    )
                    await PersonaWorker(role).draft(data, step="probe")
                    c = classify(role, log.raw)
                    c["misattributes_host"] = bool(c["addresses_host"]) and await misattributes_host(
                        runtime.gemma_client, hist, role, log.raw
                    )
                    rows.append({"session": path.name, "turn": idx, "role": role,
                                 "format": fmt, **c, "raw": log.raw[:300]})
    os.environ.pop("TALKSHOW_HISTORY_FORMAT", None)
    return {"rows": rows}


def _add_intro(hist: ShowHistory, role: str) -> None:
    from agent.floor.host_lines import host_intro_speaker_line, host_open_floor_line

    hist.append("host", host_open_floor_line(), procedural=True)
    hist.append("host", host_intro_speaker_line(NAMES[role]), procedural=True)


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for fmt in FORMATS:
        rs = [r for r in rows if r["format"] == fmt]
        n = len(rs) or 1
        out[fmt] = {
            "n": len(rs),
            **{k: sum(r[k] for r in rs) / n for k in
               ("starts_label", "starts_other", "other_label_anywhere", "addresses_host",
                "misattributes_host", "has_next")},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=8, help="panelist turns per session")
    args = ap.parse_args()

    res = asyncio.run(run(args.logs, args.trials, args.max_turns))
    summary = summarize(res["rows"])
    print(f"{'format':<18}{'n':>5}{'starts label':>14}{'starts OTHER':>14}{'other label':>13}"
          f"{'says Lessac':>13}{'misattrib.':>12}{'[next] ok':>11}")
    for fmt, s in summary.items():
        print(f"{fmt:<18}{s['n']:>5}{s['starts_label']:>14.1%}{s['starts_other']:>14.1%}"
              f"{s['other_label_anywhere']:>13.1%}{s['addresses_host']:>13.1%}"
              f"{s['misattributes_host']:>12.1%}{s['has_next']:>11.1%}")
    bad = [r for r in res["rows"] if r["starts_other"] or r["misattributes_host"]]
    for r in bad[:6]:
        print(f"  [{r['format']}] {r['session']} turn {r['turn']} {r['role']}: {r['raw'][:120]!r}")
    out = RESULTS_DIR / f"history_probe_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, **res}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
