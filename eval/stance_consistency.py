#!/usr/bin/env python3
"""
Stance consistency analysis for panel replies in talkshow session JSONL logs.

Uses an LLM judge (vLLM / Gemma) to label each panel line's stance on the
human's debate topic, then scores whether each panelist stays consistent
(and optionally matches assigned sides from the prompt).

Requires VLLM_BASE_URL (and running vLLM). Loads .env from repo root.

Usage:
    python eval/stance_consistency.py logs/session-*.jsonl
    python eval/stance_consistency.py logs/session-*.jsonl \\
        --ryan-stance keep_death_penalty --amy-stance abolish_death_penalty
    # default JSON: eval/results/stance/session-<timecode>_stance.json
    # default plot: eval/figures/stance/session-<timecode>_stance.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from dotenv import load_dotenv

load_dotenv(_REPO / ".env")

if TYPE_CHECKING:
    import httpx

from agent.config import LocaleLLMConfig, load_config

from log_parse import HumanTurn, extract_human_turns, load_events, role_name, session_metadata
from session_paths import default_figure_path, default_result_path

STANCE_UNCLEAR = "unclear"
STANCE_UNMATCHED = "unmatched"
STANCE_MIXED = "mixed"
STANCE_NEUTRAL = "neutral"
STANCE_EVASIVE = "evasive"
STANCE_KEEP_DP = "keep_death_penalty"
STANCE_ABOLISH_DP = "abolish_death_penalty"

JUDGE_SYSTEM = """You judge talk-show panel lines for STANCE on the debate topic.
Output valid JSON only — no markdown, no extra text.

Rules:
- Use the topic axis given in the user message (e.g. keep_death_penalty vs abolish_death_penalty).
- For each TARGET line, pick stance_id on that axis.
- Allowed stance_id values: keep_death_penalty, abolish_death_penalty, mixed, neutral, evasive.
- unclear only if the line has no discernible position on the axis.
- Judge the speaker's argument, not filler phrases like "I hear you".
- Include seq from each TARGET in your results for matching."""


@dataclass
class StanceLabel:
    role: str
    reply: str
    stance_id: str
    stance_label: str
    confidence: float
    reason: str
    source: str
    turn_num: int
    step: str = ""
    seq: int = 0
    judge_latency_s: float = 0.0
    method: str = "unmatched"


@dataclass
class RoleConsistency:
    role: str
    speaker: str
    turn_num: int
    source: str
    reply_count: int
    dominant_stance: str
    self_consistency: float
    flip_count: int
    stances: list[str] = field(default_factory=list)
    expected_stance: str | None = None
    assignment_alignment: float | None = None


def _parse_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"No JSON object in judge response: {text[:200]!r}")


def detect_assigned_stances(heard: str) -> dict[str, str]:
    """Parse forced-debate prompts: Ryan pro / Amy anti (per-speaker clause)."""
    text = heard.lower()
    out: dict[str, str] = {}

    ryan_clause = ""
    amy_clause = ""
    if m := re.search(r"ryan\b(.*?)(?=,\s*amy\b|\s+amy\b|$)", text, re.I):
        ryan_clause = m.group(1)
    if m := re.search(r"amy\b(.*)$", text, re.I):
        amy_clause = m.group(1)

    def _side(clause: str) -> str | None:
        if not clause:
            return None
        if re.search(r"\b(keep|pro|affirmative|support)\b", clause):
            return STANCE_KEEP_DP
        if re.search(r"\b(abolish|anti|negative|oppose|repeal)\b", clause):
            return STANCE_ABOLISH_DP
        return None

    if side := _side(ryan_clause):
        out["commentator"] = side
    if side := _side(amy_clause):
        out["guest"] = side
    return out


def infer_topic_axis(heard: str, assigned: dict[str, str]) -> str:
    if re.search(r"death\s*penalty|capital\s*punishment", heard, re.I):
        return "keep_death_penalty vs abolish_death_penalty"
    if assigned:
        return "assigned_debate_sides"
    return "topic_from_human_prompt"


def _match_judge_row(
    pr: Any,
    judge_rows: list[dict[str, Any]],
    role_idx: int,
) -> dict[str, Any] | None:
    if not judge_rows:
        return None

    by_key = {
        (str(r.get("role", "")), str(r.get("step", ""))): r for r in judge_rows
    }
    row = by_key.get((pr.role, pr.step))
    if row:
        return row

    for r in judge_rows:
        try:
            if int(r.get("seq", -999)) == int(pr.seq):
                return r
        except (TypeError, ValueError):
            continue

    role_rows = [r for r in judge_rows if r.get("role") == pr.role]
    if role_idx < len(role_rows):
        return role_rows[role_idx]
    if len(role_rows) == 1:
        return role_rows[0]
    return None


def _normalize_stance(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if not key:
        return STANCE_UNCLEAR
    return key


def _stances_match(a: str, b: str) -> bool:
    if a == b:
        return True
    # Treat keep/pro and abolish/anti families as matching within side
    keep_tokens = ("keep", "pro", "support", "retain", "affirmative")
    abolish_tokens = ("abolish", "anti", "oppose", "repeal", "ban", "against")
    a_low, b_low = a.lower(), b.lower()
    if any(t in a_low for t in keep_tokens) and any(t in b_low for t in keep_tokens):
        return True
    if any(t in a_low for t in abolish_tokens) and any(t in b_low for t in abolish_tokens):
        return True
    return False


def _scorable_stances(stances: list[str]) -> list[str]:
    return [s for s in stances if s != STANCE_UNMATCHED]


def _dominant_stance(stances: list[str]) -> str:
    scored = _scorable_stances(stances)
    if not scored:
        return STANCE_UNMATCHED
    counts: dict[str, int] = {}
    for s in scored:
        counts[s] = counts.get(s, 0) + 1
    return max(counts, key=lambda k: (counts[k], k))


def _flip_count(stances: list[str]) -> int:
    scored = _scorable_stances(stances)
    if len(scored) < 2:
        return 0
    flips = 0
    prev = scored[0]
    for s in scored[1:]:
        if not _stances_match(prev, s):
            flips += 1
        prev = s
    return flips


def _self_consistency(stances: list[str]) -> float:
    scored = _scorable_stances(stances)
    if not scored:
        return 0.0
    dom = _dominant_stance(scored)
    matches = sum(1 for s in scored if _stances_match(s, dom))
    return round(matches / len(scored), 3)


def _assignment_alignment(stances: list[str], expected: str | None) -> float | None:
    scored = _scorable_stances(stances)
    if not expected or not scored:
        return None
    matches = sum(1 for s in scored if _stances_match(s, expected))
    return round(matches / len(scored), 3)


class LlmStanceJudge:
    def __init__(self, llm: LocaleLLMConfig) -> None:
        import httpx

        self._llm = llm
        self._http = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    def close(self) -> None:
        self._http.close()

    def classify_turn(
        self,
        turn: HumanTurn,
        *,
        topic_axis: str,
        assigned: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not turn.panel_replies:
            return []

        targets = [
            {
                "role": pr.role,
                "step": pr.step,
                "seq": pr.seq,
                "line": pr.reply,
            }
            for pr in turn.panel_replies
        ]
        assign_note = ""
        if assigned:
            assign_note = (
                "\nAssigned sides for this debate: "
                + ", ".join(f"{role}={side}" for role, side in assigned.items())
            )
        user_prompt = f"""Classify STANCE for each TARGET panel line.

Topic axis: {topic_axis}
Allowed stance_id: keep_death_penalty, abolish_death_penalty, mixed, neutral, evasive, unclear
{assign_note}

Human guest said:
{turn.heard or "(none)"}

Host said:
{turn.host_reply or "(none)"}

TARGET lines (return one result per line, include seq):
{json.dumps(targets, indent=2, ensure_ascii=False)}

Output exactly this JSON shape:
{{
  "topic_axis": "{topic_axis}",
  "results": [
    {{
      "role": "<role>",
      "step": "<step>",
      "seq": 0,
      "stance_id": "keep_death_penalty",
      "stance_label": "<short label>",
      "confidence": 0.9,
      "reason": "<short>"
    }}
  ]
}}
"""
        return self._call_judge(user_prompt)

    def classify_one(
        self,
        turn: HumanTurn,
        pr: Any,
        *,
        topic_axis: str,
        assigned: dict[str, str],
    ) -> dict[str, Any] | None:
        assign_note = ""
        if assigned:
            assign_note = (
                "Assigned: "
                + ", ".join(f"{role}={side}" for role, side in assigned.items())
            )
        user_prompt = f"""Classify STANCE for this ONE panel line.

Topic axis: {topic_axis}
Allowed stance_id: keep_death_penalty, abolish_death_penalty, mixed, neutral, evasive
{assign_note}

Human guest said: {turn.heard or "(none)"}

TARGET ({pr.role}, seq={pr.seq}):
{pr.reply}

Output JSON only:
{{"stance_id": "...", "stance_label": "...", "confidence": 0.9, "reason": "..."}}
"""
        try:
            rows = self._call_judge(user_prompt)
            return rows[0] if rows else None
        except Exception:
            return None

    def _call_judge(self, user_prompt: str) -> list[dict[str, Any]]:
        t0 = time.monotonic()
        payload = {
            "model": self._llm.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": min(self._llm.max_tokens, 1024),
            "temperature": 0.1,
        }
        url = f"{self._llm.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._llm.api_key}"}
        resp = self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            content = str(content)
        parsed = _parse_json_payload(content)
        latency = time.monotonic() - t0

        rows = parsed.get("results")
        if not isinstance(rows, list):
            if isinstance(parsed, dict) and parsed.get("stance_id"):
                rows = [parsed]
            else:
                raise ValueError(f"Judge JSON missing results list: {parsed!r}")

        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                row["_judge_latency_s"] = round(latency, 3)
                row["_topic_axis"] = parsed.get("topic_axis", "")
                out.append(row)
        return out


def _resolve_stance(
    pr: Any,
    turn: HumanTurn,
    row: dict[str, Any] | None,
    *,
    judge: LlmStanceJudge,
    topic_axis: str,
    assigned: dict[str, str],
) -> StanceLabel:
    if row:
        stance_id = _normalize_stance(str(row.get("stance_id", STANCE_UNCLEAR)))
        stance_label = str(row.get("stance_label", stance_id)).strip()
        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = str(row.get("reason", "")).strip()
        return StanceLabel(
            role=pr.role,
            reply=pr.reply,
            stance_id=stance_id,
            stance_label=stance_label,
            confidence=confidence,
            reason=reason,
            source=turn.source,
            turn_num=turn.turn_num,
            step=pr.step,
            seq=pr.seq,
            judge_latency_s=float(row.get("_judge_latency_s", 0.0)),
            method="llm",
        )

    one = judge.classify_one(turn, pr, topic_axis=topic_axis, assigned=assigned)
    if one:
        stance_id = _normalize_stance(str(one.get("stance_id", STANCE_UNCLEAR)))
        stance_label = str(one.get("stance_label", stance_id)).strip()
        try:
            confidence = float(one.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = str(one.get("reason", "")).strip()
        return StanceLabel(
            role=pr.role,
            reply=pr.reply,
            stance_id=stance_id,
            stance_label=stance_label,
            confidence=confidence,
            reason=reason,
            source=turn.source,
            turn_num=turn.turn_num,
            step=pr.step,
            seq=pr.seq,
            judge_latency_s=float(one.get("_judge_latency_s", 0.0)),
            method="llm_per_line",
        )

    return StanceLabel(
        role=pr.role,
        reply=pr.reply,
        stance_id=STANCE_UNMATCHED,
        stance_label="Unmatched",
        confidence=0.0,
        reason="no LLM judge result matched this line",
        source=turn.source,
        turn_num=turn.turn_num,
        step=pr.step,
        seq=pr.seq,
        judge_latency_s=0.0,
        method="unmatched",
    )


def classify_turns(
    turns: list[HumanTurn],
    llm: LocaleLLMConfig,
) -> tuple[list[StanceLabel], dict[tuple[str, int], str]]:
    judge = LlmStanceJudge(llm)
    labels: list[StanceLabel] = []
    topic_axes: dict[tuple[str, int], str] = {}

    try:
        for turn in turns:
            if not turn.panel_replies:
                continue

            assigned = detect_assigned_stances(turn.heard)
            topic_axis = infer_topic_axis(turn.heard, assigned)
            topic_axes[(turn.source, turn.turn_num)] = topic_axis

            judge_rows: list[dict[str, Any]] = []
            try:
                judge_rows = judge.classify_turn(
                    turn, topic_axis=topic_axis, assigned=assigned
                )
            except Exception as exc:
                print(
                    f"WARN turn {turn.turn_num} ({turn.source}): "
                    f"batch judge failed — {exc}",
                    file=sys.stderr,
                )

            role_counters: dict[str, int] = {}
            for pr in turn.panel_replies:
                idx = role_counters.get(pr.role, 0)
                role_counters[pr.role] = idx + 1
                row = _match_judge_row(pr, judge_rows, idx)
                labels.append(
                    _resolve_stance(
                        pr,
                        turn,
                        row,
                        judge=judge,
                        topic_axis=topic_axis,
                        assigned=assigned,
                    )
                )
    finally:
        judge.close()

    return labels, topic_axes


def compute_consistency(
    turns: list[HumanTurn],
    labels: list[StanceLabel],
    *,
    expected_by_role: dict[str, str],
) -> list[RoleConsistency]:
    label_by_key = {
        (lb.source, lb.turn_num, lb.role, lb.step): lb for lb in labels
    }
    rows: list[RoleConsistency] = []

    for turn in turns:
        assigned = detect_assigned_stances(turn.heard)
        merged_expected = {**assigned, **expected_by_role}

        by_role: dict[str, list[StanceLabel]] = {}
        for pr in turn.panel_replies:
            key = (turn.source, turn.turn_num, pr.role, pr.step)
            lb = label_by_key.get(key)
            if lb is None:
                continue
            by_role.setdefault(pr.role, []).append(lb)

        for role, role_labels in by_role.items():
            role_labels.sort(key=lambda x: x.seq)
            stances = [lb.stance_id for lb in role_labels]
            expected = merged_expected.get(role)
            rows.append(
                RoleConsistency(
                    role=role,
                    speaker=role_name(role),
                    turn_num=turn.turn_num,
                    source=turn.source,
                    reply_count=len(stances),
                    dominant_stance=_dominant_stance(stances),
                    self_consistency=_self_consistency(stances),
                    flip_count=_flip_count(stances),
                    stances=stances,
                    expected_stance=expected,
                    assignment_alignment=_assignment_alignment(stances, expected),
                )
            )

    return rows


def print_summary(
    labels: list[StanceLabel],
    consistency: list[RoleConsistency],
    topic_axes: dict[tuple[str, int], str],
) -> None:
    if not labels:
        print("No panel replies to classify.")
        return

    print(f"Stance labels: {len(labels)}")
    unmatched = sum(1 for lb in labels if lb.method == "unmatched")
    if unmatched:
        print(f"  unmatched (no LLM result): {unmatched}")
    print()
    for (source, turn_num), axis in sorted(topic_axes.items()):
        print(f"Turn {turn_num} ({source}) topic_axis: {axis}")
    print()

    if consistency:
        df = pd.DataFrame([asdict(c) for c in consistency])
        cols = [
            "speaker",
            "turn_num",
            "reply_count",
            "dominant_stance",
            "self_consistency",
            "flip_count",
            "expected_stance",
            "assignment_alignment",
        ]
        print("Consistency by role / turn:")
        print(df[cols].to_string(index=False))
        print()

    print("Per-reply stances:")
    for lb in labels:
        snippet = lb.reply[:80] + ("…" if len(lb.reply) > 80 else "")
        print(
            f"  turn {lb.turn_num} {role_name(lb.role)} [{lb.stance_id}] "
            f"({lb.confidence:.2f}, {lb.method}): {snippet}"
        )


def plot_consistency(consistency: list[RoleConsistency], out: str | None) -> None:
    panel_rows = [c for c in consistency if c.role in ("commentator", "guest")]
    if not panel_rows:
        return

    df = pd.DataFrame([asdict(c) for c in panel_rows])
    df["label"] = df["speaker"] + " · turn " + df["turn_num"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, len(df) * 0.55)))

    y_pos = list(range(len(df)))
    axes[0].barh(y_pos, df["self_consistency"], color="#059669", alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df["label"])
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Self-consistency (fraction matching dominant stance)")
    axes[0].set_title("Stance self-consistency")
    axes[0].invert_yaxis()

    align = df["assignment_alignment"].fillna(-0.05)
    colors = ["#4f46e5" if v >= 0 else "#d1d5db" for v in align]
    axes[1].barh(y_pos, align.clip(lower=0), color=colors, alpha=0.85)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(df["label"])
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Assignment alignment (if expected side known)")
    axes[1].set_title("Match assigned side")
    axes[1].invert_yaxis()

    fig.suptitle("Panel Stance Consistency", fontsize=13)
    plt.tight_layout()

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


def build_export(
    turns: list[HumanTurn],
    labels: list[StanceLabel],
    consistency: list[RoleConsistency],
    topic_axes: dict[tuple[str, int], str],
    meta: dict[str, Any],
    *,
    judge_model: str,
) -> dict[str, Any]:
    return {
        "session": meta,
        "judge": {"model": judge_model, "method": "llm_stance"},
        "topic_axes": [
            {"source": s, "turn_num": t, "topic_axis": a}
            for (s, t), a in sorted(topic_axes.items())
        ],
        "stance_labels": [asdict(lb) for lb in labels],
        "consistency": [asdict(c) for c in consistency],
        "turns": [
            {
                "turn_num": t.turn_num,
                "source": t.source,
                "heard": t.heard,
                "panel_reply_count": len(t.panel_replies),
            }
            for t in turns
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panel stance consistency (LLM judge)"
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Session JSONL log file(s)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="App config yaml (default: CONFIG_PATH or config/multimodal.yaml)",
    )
    parser.add_argument(
        "--ryan-stance",
        default="",
        help="Expected commentator stance_id (e.g. keep_death_penalty)",
    )
    parser.add_argument(
        "--amy-stance",
        default="",
        help="Expected guest stance_id (e.g. abolish_death_penalty)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="Plot output (default: eval/figures/stance/session-<timecode>_stance.png)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSON export (default: eval/results/stance/session-<timecode>_stance.json)",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()

    expected: dict[str, str] = {}
    if args.ryan_stance:
        expected["commentator"] = _normalize_stance(args.ryan_stance)
    if args.amy_stance:
        expected["guest"] = _normalize_stance(args.amy_stance)

    app_cfg = load_config(args.config)
    llm = app_cfg.locale().llm
    print(f"LLM stance judge → {llm.base_url} model={llm.model}")

    events = load_events(*args.logs)
    meta = session_metadata(events)
    turns = extract_human_turns(events)
    labels, topic_axes = classify_turns(turns, llm)
    consistency = compute_consistency(turns, labels, expected_by_role=expected)

    print_summary(labels, consistency, topic_axes)

    payload = build_export(
        turns, labels, consistency, topic_axes, meta, judge_model=llm.model
    )

    if not args.no_json:
        json_path = args.json or default_result_path(
            _EVAL_DIR, "stance", *args.logs, events=events
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote JSON → {json_path}")

    if not args.no_plot and labels:
        out = args.out or str(
            default_figure_path(_EVAL_DIR, "stance", *args.logs, events=events)
        )
        plot_consistency(consistency, out)


if __name__ == "__main__":
    main()
