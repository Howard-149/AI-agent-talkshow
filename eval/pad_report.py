#!/usr/bin/env python3
"""
PAD compliance report for talkshow session JSONL logs.

Answers: did the model emit [pad]? did it parse? did deltas move the MSP tag?
Live path: ``pad_appraisal`` events from agent/emotion/appraisal.py. Logs from the
removed inline mode ([pad] inside dialogue output) still get the legacy table.

Usage:
    python eval/pad_report.py                      # newest logs/session-*.jsonl
    python eval/pad_report.py logs/session-*.jsonl
    python eval/pad_report.py --last 3             # newest 3 sessions
    python eval/pad_report.py --show-raw           # print raw output of failed turns
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from log_parse import load_events  # noqa: E402

_REPO = _EVAL_DIR.parent
_NEUTRAL = {"neutral"}


def _status(e: dict[str, Any]) -> str:
    delta = e.get("pad_delta")
    if delta is not None:
        return "zero" if all(abs(float(x)) < 1e-9 for x in delta) else "parsed"
    return "unparsed" if e.get("has_pad_tag") else "no_tag"


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "   - "


def _fmt3(xs: Any) -> str:
    if not xs:
        return "-"
    return "(" + ", ".join(f"{float(x):+.2f}" for x in xs) + ")"


def report(events: list[dict[str, Any]], *, show_raw: bool) -> None:
    llm_turns = [e for e in events if "has_pad_tag" in e]
    cols = ("parsed", "zero", "unparsed", "no_tag")

    appraisals = [e for e in events if e.get("event") == "pad_appraisal"]
    if appraisals:
        _appraisal_section(events, appraisals)

    if not appraisals and not llm_turns:
        print("(no pad_appraisal events — is the agent running the appraisal build?)\n")
    if llm_turns:
        # Legacy logs from the removed inline mode ([pad] inside dialogue output).
        print("== legacy inline [pad] compliance (old logs) ==")
        by_key: dict[tuple[str, str], collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for e in llm_turns:
            role = e.get("role") or e.get("active_role") or "host"
            by_key[(e.get("event", "?"), role)][_status(e)] += 1
        print(f"{'event':<22}{'role':<13}{'n':>4}  " + "  ".join(f"{c:>9}" for c in cols))
        total = collections.Counter()
        for (ev, role), cnt in sorted(by_key.items()):
            n = sum(cnt.values())
            total.update(cnt)
            print(
                f"{ev:<22}{role:<13}{n:>4}  "
                + "  ".join(f"{cnt[c]:>3} {_pct(cnt[c], n)}" for c in cols)
            )
        n = sum(total.values())
        print(
            f"{'TOTAL':<35}{n:>4}  "
            + "  ".join(f"{total[c]:>3} {_pct(total[c], n)}" for c in cols)
        )


    pad_maps = [e for e in events if e.get("event") == "pad_map"]
    deltas = [e for e in pad_maps if e.get("reason") in ("delta", "appraisal")]
    print(f"\n== PAD updates: {len(deltas)} delta / {len(pad_maps)} pad_map ==")
    print(f"{'role':<13}{'reason':<11}{'delta':<24}{'pad_after':<24}emotion")
    for e in deltas:
        was = e.get("emotion_was") or "-"
        emo = e.get("emotion") or "-"
        arrow = f"{was} → {emo}" if was.lower() != emo.lower() else emo
        print(
            f"{e.get('role', '?'):<13}{e.get('reason', ''):<11}"
            f"{_fmt3(e.get('delta')):<24}{_fmt3(e.get('pad_after')):<24}{arrow}"
        )

    print("\n== Final PAD / emotion per role ==")
    last: dict[str, dict[str, Any]] = {}
    for e in pad_maps:
        last[e.get("role", "?")] = e
    for role, e in sorted(last.items()):
        print(f"{role:<13}{_fmt3(e.get('pad_after')):<24}{e.get('emotion')}")

    labels = collections.Counter(
        (e.get("emotion") or "?") for e in deltas
    )
    non_neutral = sum(v for k, v in labels.items() if k.lower() not in _NEUTRAL)
    print(
        f"\nemotion after delta turns: {dict(labels.most_common())}  "
        f"non-neutral {non_neutral}/{len(deltas)}"
    )

    if show_raw:
        bad = [e for e in llm_turns if _status(e) in ("unparsed", "no_tag")]
        print(f"\n== Raw output of {len(bad)} turns without a usable [pad] ==")
        for e in bad:
            role = e.get("role") or e.get("active_role") or "host"
            raw = (e.get("raw") or "").strip().replace("\n", "⏎ ")
            print(f"- [{e.get('event')}/{role}/{_status(e)}] {raw[-300:]}")


def _appraisal_section(
    events: list[dict[str, Any]], appraisals: list[dict[str, Any]]
) -> None:
    print("== separate appraisal (pad_appraisal) ==")
    by_role: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in appraisals:
        by_role[e.get("role", "?")].append(e)
    print(f"{'listener':<13}{'n':>4}{'ok':>9}{'zero':>9}{'avg |Δ|':>10}{'lat avg':>9}{'lat max':>9}")
    for role, rows in sorted(by_role.items()):
        ok = [r for r in rows if r.get("ok")]
        zero = [r for r in ok if all(abs(float(x)) < 1e-9 for x in r["delta"])]
        mags = [sum(abs(float(x)) for x in r["delta"]) / 3 for r in ok]
        lats = [float(r.get("latency_s") or 0) for r in rows]
        print(
            f"{role:<13}{len(rows):>4}{len(ok):>4} {_pct(len(ok), len(rows))}"
            f"{len(zero):>4} {_pct(len(zero), len(ok))}"
            f"{(sum(mags) / len(mags) if mags else 0):>10.3f}"
            f"{(sum(lats) / len(lats) if lats else 0):>8.2f}s{max(lats or [0]):>8.2f}s"
        )

    distinct = collections.Counter(
        tuple(round(float(x), 2) for x in r["delta"]) for r in appraisals if r.get("ok")
    )
    total_ok = sum(distinct.values())
    if total_ok:
        top, top_n = distinct.most_common(1)[0]
        print(
            f"distinct deltas {len(distinct)}/{total_ok}; most common {_fmt3(top)} "
            f"×{top_n} ({_pct(top_n, total_ok).strip()}) — high share = copying examples"
        )
    words = collections.Counter(r.get("word") or "-" for r in appraisals if r.get("ok"))
    print("feeling words:", dict(words.most_common(12)))

    waits = [e for e in events if e.get("event") == "pad_appraisal_wait"]
    if waits:
        late = [w for w in waits if not w.get("ok")]
        avg = sum(float(w.get("waited_s") or 0) for w in waits) / len(waits)
        print(f"speaker waits: {len(waits)}  avg {avg:.2f}s  timed out {len(late)}")

    expr = [e for e in events if e.get("event") == "pad_expression"]
    if expr:
        drops = [float(e["arousal_before"]) - float(e["arousal_after"]) for e in expr]
        print(
            f"expression relax: {len(expr)} lines  γ={expr[0].get('gamma')}  "
            f"avg ΔA {-sum(drops) / len(drops):+.3f}"
        )
    vals = collections.Counter(r.get("valence") or "-" for r in appraisals if r.get("ok"))
    flips = [r for r in appraisals if r.get("sign_flipped")]
    print(f"valence: {dict(vals)}  ΔP sign fixed to match valence: {len(flips)}")
    for r in flips[:8]:
        print(f"   fixed {r.get('role')}: {r.get('word')!r} {r.get('valence')} → ΔP {r['delta'][0]:+.2f}")
    neg = sum(1 for r in appraisals if r.get("ok") and r["delta"][0] < -0.03)
    pos = sum(1 for r in appraisals if r.get("ok") and r["delta"][0] > 0.03)
    print(f"ΔP direction: {pos} up / {neg} down / {len(appraisals) - pos - neg} ~0")
    protos = collections.Counter(r.get("protocol") or "-" for r in appraisals)
    replies = sum(1 for r in appraisals if r.get("reply_to_self"))
    print(f"protocol: {dict(protos)}  replies-to-listener: {replies}/{len(appraisals)}")

    print("\nper line (speaker → listener: word delta):")
    by_seq: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in appraisals:
        by_seq[(r.get("seq"), r.get("speaker"), r.get("text"))].append(r)
    for (seq, speaker, text), rows in sorted(by_seq.items(), key=lambda kv: kv[0][0] or 0):
        print(f"  [{seq}] {speaker}: {(text or '')[:70]!r}")
        for r in rows:
            fix = " [sign fixed]" if r.get("sign_flipped") else ""
            got = f"{r.get('word') or '-'} {_fmt3(r.get('delta'))}{fix}" if r.get("ok") else (
                f"FAILED {r.get('error') or repr((r.get('raw') or '')[:60])}"
            )
            tag = " (reply to them)" if r.get("reply_to_self") else ""
            print(f"      → {r.get('role')}{tag}: {got}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("logs", nargs="*", type=Path, help="session JSONL files")
    ap.add_argument("--last", type=int, default=1, help="newest N sessions when no logs given")
    ap.add_argument("--show-raw", action="store_true", help="print raw model output of failed turns")
    args = ap.parse_args()

    paths: list[Path] = args.logs
    if not paths:
        found = sorted(
            (_REPO / "logs").glob("session-*.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        paths = found[-args.last :]
        if not paths:
            sys.exit("no logs/session-*.jsonl found")
    print("logs:", ", ".join(p.name for p in paths), "\n")
    report(load_events(*paths), show_raw=args.show_raw)


if __name__ == "__main__":
    main()
