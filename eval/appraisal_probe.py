#!/usr/bin/env python3
"""
Appraisal sign probes: does the live appraisal prompt move pleasure the right way?

Each probe is one line with an unambiguous expected direction of ΔP for a listener
(someone else's humiliation → down, being praised → up, a flat fact → ~0). The prompt is
built with the live ``build_appraisal_prompt`` (same personas, reply marking, parser),
starting from neutral PAD, and repeated N times at the live temperature.

Usage (Babel compute node, vLLM running):
    python eval/appraisal_probe.py                    # 20 trials per probe/listener
    python eval/appraisal_probe.py --trials 10 --tag baseline
    python eval/appraisal_probe.py --dry-run          # print one prompt, no LLM
    # JSON: eval/results/appraisal/probes_<tag>_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_EVAL = _REPO / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import yaml  # noqa: E402

from agent.emotion import appraisal as A  # noqa: E402
from agent.emotion import state as S  # noqa: E402
from agent.show.show_history import ShowHistory  # noqa: E402

DEFAULT_FIXTURE = _REPO / "eval" / "fixtures" / "appraisal_probes.yaml"
RESULTS_DIR = _REPO / "eval" / "results" / "appraisal"
SIGN_DEADZONE = 0.03  # |ΔP| below this counts as "no change"
NEUTRAL_TOL = 0.05  # expect "0": success if |ΔP| <= this


def build_prompt(probe: dict[str, Any], listener: str) -> tuple[str, str]:
    """Live prompt for ``listener`` hearing ``probe`` (neutral starting mood)."""
    hist = ShowHistory()
    reply_to = probe.get("reply_to")
    for role, text in probe.get("context") or []:
        hist.append(str(role), str(text))
    hist.append(str(probe["speaker"]), str(probe["text"]).strip())
    data = SimpleNamespace(
        show_history=hist,
        role_pad={},
        role_pad_neighbors={},
        role_emotion={},
    )
    text = str(probe["text"]).strip()
    utt = A.Utterance(str(probe["speaker"]), text, context=A.snapshot_context(data, text))
    if reply_to and listener == reply_to:
        assert A.replied_line(listener, utt, utt.context) is not None, probe["id"]
    return A.build_appraisal_prompt(data, listener, utt)  # type: ignore[arg-type]


def parse(raw: str) -> dict[str, Any]:
    """Word, delta, and — when the parser supports it — valence class / sign flip."""
    full = getattr(S, "parse_pad_appraisal_full", None)
    if full is not None:
        r = full(raw)
        return {
            "word": r.word,
            "delta": r.delta,
            "valence": r.valence,
            "raw_p": r.raw_p,
            "flipped": r.flipped,
        }
    word, delta = S.parse_pad_appraisal(raw)
    return {
        "word": word,
        "delta": delta,
        "valence": None,
        "raw_p": delta[0] if delta else None,
        "flipped": False,
    }


def hit(expect: str, dp: float) -> bool:
    if expect == "+":
        return dp > SIGN_DEADZONE
    if expect == "-":
        return dp < -SIGN_DEADZONE
    return abs(dp) <= NEUTRAL_TOL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=A.DEFAULT_APPRAISAL_TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=A.DEFAULT_APPRAISAL_MAX_TOKENS)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fx = yaml.safe_load(args.fixture.read_text(encoding="utf-8"))
    probes = fx["probes"]
    if args.dry_run:
        p = probes[0]
        listener = next(iter(p["listeners"]))
        system, user = build_prompt(p, listener)
        print(f"===== {p['id']} → {listener} — system =====\n{system}\n===== user =====\n{user}")
        return

    from appraisal_sentipolis import VllmResponder

    respond = VllmResponder(temperature=args.temperature, max_tokens=args.max_tokens)
    print(f"appraisal probes → {respond._llm.model}  T={args.temperature}  "
          f"trials={args.trials}  tag={args.tag}\n")
    rows: list[dict[str, Any]] = []
    for p in probes:
        for listener, expect in p["listeners"].items():
            system, user = build_prompt(p, listener)
            for _ in range(args.trials):
                raw = respond(system, user)
                r = parse(raw)
                rows.append({"probe": p["id"], "listener": listener, "expect": expect,
                             "raw": raw.strip()[:200], **r})

    print(f"{'probe':<20}{'listener':<13}{'exp':>4}{'acc':>7}{'mean ΔP':>9}"
          f"{'fail':>6}{'conflict':>10}{'flipped':>9}  top words")
    groups: dict[str, list[bool]] = {"+": [], "-": [], "0": []}
    total_conflict = total_flip = total_fail = 0
    for p in probes:
        for listener, expect in p["listeners"].items():
            rs = [r for r in rows if r["probe"] == p["id"] and r["listener"] == listener]
            ok = [r for r in rs if r["delta"] is not None]
            hits = [hit(expect, r["delta"][0]) for r in ok]
            groups[expect].extend(hits + [False] * (len(rs) - len(ok)))
            # Conflict = the model's own pleasant/unpleasant call disagreed with its ΔP sign.
            conflict = sum(1 for r in ok if r["flipped"])
            fail = len(rs) - len(ok)
            total_conflict += conflict
            total_flip += conflict
            total_fail += fail
            mean_dp = statistics.fmean(r["delta"][0] for r in ok) if ok else float("nan")
            words: dict[str, int] = {}
            for r in ok:
                words[r["word"] or "-"] = words.get(r["word"] or "-", 0) + 1
            top = ", ".join(w for w, _ in sorted(words.items(), key=lambda kv: -kv[1])[:3])
            print(f"{p['id']:<20}{listener:<13}{expect:>4}{sum(hits) / len(rs):>7.0%}"
                  f"{mean_dp:>+9.2f}{fail:>6}{conflict:>10}{conflict:>9}  {top}")
    all_hits = [h for g in groups.values() for h in g]
    print(
        f"\nsign accuracy: overall {sum(all_hits) / len(all_hits):.0%}  "
        + "  ".join(
            f"expect {k}: {sum(v) / len(v):.0%} (n={len(v)})" for k, v in groups.items() if v
        )
    )
    print(f"parse failures {total_fail}/{len(rows)}  sign flips applied {total_flip}")

    out = RESULTS_DIR / f"probes_{args.tag}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fixture": fx["id"], "tag": args.tag, "trials": args.trials,
        "temperature": args.temperature, "model": respond._llm.model,
        "accuracy": {k: (sum(v) / len(v) if v else None) for k, v in groups.items()},
        "overall": sum(all_hits) / len(all_hits), "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
