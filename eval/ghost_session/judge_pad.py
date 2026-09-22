"""Judge whether session ``pad_map`` deltas track the spoken beat.

Heuristic only: lexicon valence on the nearest prior utterance vs ΔP / mapped
emotion. Does not call a model. Missing ``pad_map`` rows fail closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.log_parse import load_events, role_label

_NEG = frozenset(
    {
        "fired",
        "firing",
        "humiliation",
        "humiliate",
        "cruel",
        "boxed",
        "walked",
        "security",
        "betray",
        "betrayed",
        "angry",
        "anger",
        "outraged",
        "disgust",
        "sad",
        "grief",
        "loss",
        "gag",
        "meritocracy",
        "walkout",
        "blame",
    }
)
_POS = frozenset(
    {
        "relieved",
        "relief",
        "raise",
        "hired",
        "landed",
        "grateful",
        "glad",
        "hope",
        "hopeful",
        "celebrate",
        "award",
        "loyalty",
        "begging",
        "feet",
    }
)
_HIGH_A = frozenset(
    {
        "humiliation",
        "outraged",
        "walked",
        "security",
        "walkout",
        "begging",
        "fired",
        "boxed",
    }
)

# Mapped MSP-ish tags: expected pleasure sign (None = no claim).
_EMOTION_P: dict[str, int | None] = {
    "happiness": 1,
    "joy": 1,
    "amused": 1,
    "contempt": -1,
    "anger": -1,
    "sadness": -1,
    "fear": -1,
    "disgust": -1,
    "concerned": -1,
    "surprised": None,
    "surprise": None,
    "neutral": 0,
    "vague": None,
}


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    buf: list[str] = []
    for ch in (text or "").lower():
        if ch.isalpha():
            buf.append(ch)
        else:
            if buf:
                out.add("".join(buf))
            buf = []
    if buf:
        out.add("".join(buf))
    return out


def expected_signs(text: str) -> tuple[int | None, int | None]:
    """Return (pleasure_sign, arousal_sign) in {-1, 0, 1, None}."""
    words = _tokens(text)
    n = len(words & _NEG)
    p = len(words & _POS)
    a = len(words & _HIGH_A)
    pleasure: int | None
    if n > p + 1:
        pleasure = -1
    elif p > n + 1:
        pleasure = 1
    elif n and not p:
        pleasure = -1
    elif p and not n:
        pleasure = 1
    else:
        pleasure = None
    arousal = 1 if a >= 2 else (None if a == 0 else 1)
    return pleasure, arousal


def _nearest_speech(events: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    for j in range(idx - 1, -1, -1):
        e = events[j]
        ev = e.get("event")
        if ev in ("panel_model_done", "ghost_human_done", "gemma_stt_done"):
            return e
        if ev == "assistant_reply" and (e.get("text") or "").strip():
            return e
    return {}


def _sign(x: float, *, eps: float = 0.04) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def judge_pad_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    ok = 0
    bad = 0
    for i, e in enumerate(events):
        if e.get("event") != "pad_map":
            continue
        speech = _nearest_speech(events, i)
        text = (
            (speech.get("reply") or speech.get("heard") or speech.get("text") or "")
        ).strip()
        exp_p, exp_a = expected_signs(text)
        delta = e.get("delta") or [0.0, 0.0, 0.0]
        d_p = float(delta[0]) if len(delta) > 0 else 0.0
        d_a = float(delta[1]) if len(delta) > 1 else 0.0
        emotion = str(e.get("emotion") or "")
        flags: list[str] = []
        if e.get("delta") is None:
            flags.append("no_delta")
        if exp_p is not None and _sign(d_p) not in (0, exp_p) and abs(d_p) >= 0.08:
            flags.append(f"ΔP={d_p:+.2f} vs speech valence {exp_p:+d}")
        if exp_a == 1 and d_a < -0.08:
            flags.append(f"ΔA={d_a:+.2f} but speech looks high-arousal")
        p_expect = _EMOTION_P.get(emotion.lower())
        after = e.get("pad_after") or [0.0, 0.0, 0.0]
        after_p = float(after[0]) if after else 0.0
        if p_expect == 1 and after_p < -0.15:
            flags.append(f"tag {emotion} but P={after_p:.2f}")
        if p_expect == -1 and after_p > 0.15:
            flags.append(f"tag {emotion} but P={after_p:.2f}")
        if p_expect == 0 and abs(after_p) > 0.45:
            flags.append(f"tag {emotion} but |P|={abs(after_p):.2f}")
        verdict = "ok" if not flags else "check"
        if verdict == "ok":
            ok += 1
        else:
            bad += 1
        rows.append(
            {
                "role": e.get("role"),
                "reason": e.get("reason"),
                "delta": e.get("delta"),
                "pad_before": e.get("pad_before"),
                "pad_after": e.get("pad_after"),
                "emotion_was": e.get("emotion_was"),
                "emotion": emotion,
                "neighbors": e.get("neighbors") or [],
                "speech": text[:160],
                "flags": flags,
                "verdict": verdict,
            }
        )
    return {
        "n": len(rows),
        "ok": ok,
        "check": bad,
        "rows": rows,
    }


def format_pad_judgment(report: dict[str, Any]) -> str:
    if report["n"] == 0:
        return (
            "pad_map: 0 events — worker is not logging PAD "
            "(sync pad_pipeline + resubmit the 3-GPU job)."
        )
    lines = [
        f"pad_map: {report['n']}  ok={report['ok']}  check={report['check']}",
    ]
    for row in report["rows"]:
        role = role_label(str(row.get("role") or "?"))
        was = row.get("emotion_was") or "?"
        now = row.get("emotion") or "?"
        flag_s = ("  ! " + "; ".join(row["flags"])) if row["flags"] else ""
        lines.append(
            f"  [{row['verdict']}] {role} {row.get('pad_before')} "
            f"Δ{row.get('delta')} → {row.get('pad_after')}  {was}→{now}{flag_s}"
        )
        speech = row.get("speech") or ""
        if speech:
            lines.append(f"         speech: {speech!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge pad_map vs nearby speech")
    parser.add_argument("log", type=Path, help="session-*.jsonl")
    args = parser.parse_args(argv)
    events = load_events(args.log)
    print(format_pad_judgment(judge_pad_events(events)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
