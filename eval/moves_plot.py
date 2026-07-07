#!/usr/bin/env python3
"""
Dialogue-move breakdown from talkshow session JSONL logs.

Reads panel_model_done.reply lines and classifies them into moves from
config/dialogue_libraries/panel.yaml using an LLM judge (vLLM / Gemma).

Requires VLLM_BASE_URL (and running vLLM). Loads .env from repo root.

Usage:
    python eval/moves_plot.py logs/session-*.jsonl
    # default JSON: eval/results/moves/session-<timecode>_moves.json
    # default plot: eval/figures/moves/session-<timecode>_moves.png
    python eval/moves_plot.py logs/session-*.jsonl --no-plot
    python eval/moves_plot.py logs/session-*.jsonl --no-json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

load_dotenv(_REPO / ".env")

from agent.config import LocaleLLMConfig, load_config
from agent.dialogue_library import DialogueMove, load_dialogue_library

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from session_paths import default_figure_path, default_result_path

# Eval-only move ids
EVAL_OTHER_MOVE_ID = "other"  # only when LLM explicitly picks other
EVAL_UNMATCHED_MOVE_ID = "unmatched"  # no LLM result matched this line

JUDGE_SYSTEM = f"""You classify talk-show panel lines into dialogue moves.
Output valid JSON only — no markdown, no extra text.

Rules:
- Pick the single best-fitting move_id for each TARGET line from the catalog.
- Use move_id "{EVAL_OTHER_MOVE_ID}" only when the line is valid panel speech but none of the
  catalog moves clearly fit (e.g. neutral aside, greeting, topic shift).
- Judge the rhetorical move (how they speak), not topic agreement alone.
- collaborative vs pushback tone comes from the catalog entry you pick.
- Include seq from each TARGET in your results for matching."""


@dataclass
class PanelReply:
    role: str
    reply: str
    step: str = ""
    seq: int = 0
    next_tag: str = ""
    model_latency_s: float | None = None
    dialogue_pick: str | None = None
    hint_chars: int = 0


@dataclass
class HumanTurn:
    turn_num: int
    source: str
    heard: str = ""
    host_reply: str = ""
    panel_replies: list[PanelReply] = field(default_factory=list)
    session_meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Classification:
    role: str
    reply: str
    move_id: str
    move_label: str
    tone: str
    confidence: float
    reason: str
    source: str
    turn_num: int
    step: str = ""
    judge_latency_s: float = 0.0
    method: str = "unmatched"

    @property
    def score(self) -> int:
        """0–100 for JSON export compatibility."""
        return int(round(self.confidence * 100))


def load_events(*paths: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj["_source"] = p.name
                events.append(obj)
    events.sort(key=lambda e: e["ts"])
    return events


def session_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    for e in events:
        if e.get("event") != "session_start":
            continue
        return {
            k: e.get(k)
            for k in (
                "room",
                "log_file",
                "scenario_id",
                "turn_mode",
                "dialogue_library",
                "dialogue_pick",
            )
        } | {"source": e.get("_source", "")}
    return {}


def extract_human_turns(events: list[dict[str, Any]]) -> list[HumanTurn]:
    """One human turn = gemma_stt_start → next gemma_stt_start (or EOF)."""
    meta = session_metadata(events)
    stt_starts = [
        (i, e) for i, e in enumerate(events) if e["event"] == "gemma_stt_start"
    ]
    turns: list[HumanTurn] = []

    for t, (start_idx, _) in enumerate(stt_starts):
        end_idx = stt_starts[t + 1][0] if t + 1 < len(stt_starts) else len(events)
        window = events[start_idx:end_idx]
        source = window[0].get("_source", "")

        stt_done = next((e for e in window if e["event"] == "gemma_stt_done"), None)
        if stt_done is None:
            continue

        turn = HumanTurn(
            turn_num=t + 1,
            source=source,
            heard=(stt_done.get("heard") or "").strip(),
            host_reply=(stt_done.get("reply") or "").strip(),
            session_meta=dict(meta),
        )

        seq = 0
        for e in window:
            if e["event"] != "panel_model_done":
                continue
            reply = (e.get("reply") or "").strip()
            if not reply:
                continue
            turn.panel_replies.append(
                PanelReply(
                    role=e.get("role", ""),
                    reply=reply,
                    step=e.get("step", ""),
                    seq=seq,
                    next_tag=e.get("next_tag", ""),
                    model_latency_s=e.get("model_latency_s"),
                    dialogue_pick=e.get("dialogue_pick"),
                    hint_chars=int(e.get("hint_chars") or 0),
                )
            )
            seq += 1
        turns.append(turn)

    return turns


def panel_move_catalog(library_id: str) -> tuple[str, dict[str, DialogueMove]]:
    """Catalog text for the judge + move lookup (panelist moves only)."""
    library = load_dialogue_library(library_id)
    moves = [
        m
        for m in library.moves
        if not m.roles or "commentator" in m.roles or "guest" in m.roles
    ]
    move_by_id = {m.id: m for m in moves}
    move_by_id[EVAL_OTHER_MOVE_ID] = DialogueMove(
        id=EVAL_OTHER_MOVE_ID,
        label="Other",
        example=(
            "Valid panel line that does not match build_on, challenge, etc. "
            "(neutral comment, filler, topic pivot)."
        ),
        tone="other",
        roles=(),
    )
    move_by_id[EVAL_UNMATCHED_MOVE_ID] = DialogueMove(
        id=EVAL_UNMATCHED_MOVE_ID,
        label="Unmatched",
        example="No LLM judge result was matched to this line.",
        tone="unmatched",
        roles=(),
    )

    blocks: list[str] = []
    if library.preamble:
        blocks.append(f"Library guidance:\n{library.preamble.strip()}")
    blocks.append("Move catalog:")
    for m in moves:
        blocks.append(
            f"- move_id: {m.id}\n  label: {m.label}\n  tone: {m.tone}\n"
            f"  example:\n{m.example.strip()}"
        )
    other = move_by_id[EVAL_OTHER_MOVE_ID]
    blocks.append(
        f"- move_id: {other.id}\n  label: {other.label}\n  tone: {other.tone}\n"
        f"  when_to_use:\n{other.example.strip()}"
    )
    return "\n\n".join(blocks), move_by_id


def _normalize_move_id(raw: str, move_by_id: dict[str, DialogueMove]) -> str:
    key = raw.strip().lower()
    if not key:
        return EVAL_UNMATCHED_MOVE_ID
    if key == EVAL_OTHER_MOVE_ID:
        return EVAL_OTHER_MOVE_ID
    return key if key in move_by_id else EVAL_UNMATCHED_MOVE_ID


def _match_judge_row(
    pr: PanelReply,
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


def _judge_max_tokens(llm: LocaleLLMConfig, *, n_targets: int) -> int:
    """Batch judge needs more headroom when a turn has many panel volleys."""
    per_line = 140
    floor = max(llm.max_tokens, 512)
    needed = 200 + per_line * max(n_targets, 1)
    return min(2048, max(floor, needed))


class LlmMoveJudge:
    def __init__(self, llm: LocaleLLMConfig, *, catalog: str) -> None:
        self._llm = llm
        self._catalog = catalog
        self._http = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    def close(self) -> None:
        self._http.close()

    def _call_judge(
        self, user_prompt: str, *, max_tokens: int
    ) -> tuple[list[dict[str, Any]], float]:
        t0 = time.monotonic()
        payload = {
            "model": self._llm.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        url = f"{self._llm.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._llm.api_key}"}
        resp = self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            content = str(content)
        finish = choice.get("finish_reason")
        if finish == "length":
            raise ValueError(
                f"Judge response truncated (max_tokens={max_tokens}); "
                "retry with per-line fallback"
            )
        parsed = _parse_json_payload(content)
        latency = time.monotonic() - t0

        rows = parsed.get("results")
        if not isinstance(rows, list):
            if isinstance(parsed, dict) and parsed.get("move_id"):
                rows = [parsed]
            else:
                raise ValueError(f"Judge JSON missing results list: {parsed!r}")

        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                row["_judge_latency_s"] = round(latency, 3)
                out.append(row)
        return out, latency

    def classify_turn(self, turn: HumanTurn) -> list[dict[str, Any]]:
        """One LLM call per human turn; returns raw result dicts per panel line."""
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
        user_prompt = f"""Classify each TARGET panel line below.

Human guest said:
{turn.heard or "(none)"}

Host said:
{turn.host_reply or "(none)"}

Catalog:
{self._catalog}

TARGET lines (classify each — include seq in output):
{json.dumps(targets, indent=2, ensure_ascii=False)}

Output exactly this JSON shape:
{{
  "results": [
    {{"role": "<role>", "step": "<step>", "seq": 0, "move_id": "<id>", "confidence": 0.9, "reason": "<short>"}}
  ]
}}
"""
        rows, _ = self._call_judge(
            user_prompt,
            max_tokens=_judge_max_tokens(self._llm, n_targets=len(targets)),
        )
        return rows

    def classify_one(self, turn: HumanTurn, pr: PanelReply) -> dict[str, Any] | None:
        user_prompt = f"""Classify this ONE panel line into a dialogue move.

Human guest said: {turn.heard or "(none)"}
Host said: {turn.host_reply or "(none)"}

Catalog:
{self._catalog}

TARGET ({pr.role}, seq={pr.seq}, step={pr.step}):
{pr.reply}

Output JSON only:
{{"move_id": "<id>", "confidence": 0.9, "reason": "<short>"}}
"""
        try:
            rows, _ = self._call_judge(user_prompt, max_tokens=256)
            return rows[0] if rows else None
        except Exception:
            return None


def _resolve_classification(
    pr: PanelReply,
    turn: HumanTurn,
    judge_row: dict[str, Any] | None,
    move_by_id: dict[str, DialogueMove],
    *,
    method: str = "llm",
) -> Classification:
    if not judge_row:
        move = move_by_id[EVAL_UNMATCHED_MOVE_ID]
        return Classification(
            role=pr.role,
            reply=pr.reply,
            move_id=EVAL_UNMATCHED_MOVE_ID,
            move_label=move.label,
            tone=move.tone,
            confidence=0.0,
            reason="no LLM judge result matched this line",
            source=turn.source,
            turn_num=turn.turn_num,
            step=pr.step,
            judge_latency_s=0.0,
            method="unmatched",
        )

    raw_id = str(judge_row.get("move_id", ""))
    move_id = _normalize_move_id(raw_id, move_by_id)
    try:
        confidence = float(judge_row.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(judge_row.get("reason", "")).strip()
    judge_latency_s = float(judge_row.get("_judge_latency_s", 0.0))

    move = move_by_id[move_id]
    return Classification(
        role=pr.role,
        reply=pr.reply,
        move_id=move_id,
        move_label=move.label,
        tone=move.tone or move_id,
        confidence=confidence,
        reason=reason,
        source=turn.source,
        turn_num=turn.turn_num,
        step=pr.step,
        judge_latency_s=judge_latency_s,
        method=method,
    )


def classify_turns_llm(
    turns: list[HumanTurn],
    *,
    library_id: str,
    llm: LocaleLLMConfig,
) -> list[Classification]:
    catalog, move_by_id = panel_move_catalog(library_id)
    judge = LlmMoveJudge(llm, catalog=catalog)
    results: list[Classification] = []

    try:
        for turn in turns:
            if not turn.panel_replies:
                continue
            try:
                judge_rows = judge.classify_turn(turn)
            except Exception as exc:
                print(
                    f"WARN turn {turn.turn_num} ({turn.source}): judge failed — {exc}",
                    file=sys.stderr,
                )
                judge_rows = []

            role_counters: dict[str, int] = {}
            for pr in turn.panel_replies:
                idx = role_counters.get(pr.role, 0)
                role_counters[pr.role] = idx + 1
                row = _match_judge_row(pr, judge_rows, idx)
                method = "llm"
                if row is None:
                    row = judge.classify_one(turn, pr)
                    if row is not None:
                        method = "llm_per_line"
                results.append(
                    _resolve_classification(
                        pr, turn, row, move_by_id, method=method
                    )
                )
    finally:
        judge.close()

    return results


def print_summary(results: list[Classification]) -> None:
    if not results:
        print("No panel replies with text found.")
        print("Deploy agent with panel_model_done.reply logging, then re-run sessions.")
        return

    total = len(results)
    by_move = Counter(r.move_id for r in results)
    by_tone = Counter(
        r.tone
        if r.move_id not in (EVAL_OTHER_MOVE_ID, EVAL_UNMATCHED_MOVE_ID)
        else r.move_id
        for r in results
    )
    by_role = Counter(r.role for r in results)

    print(f"Panel replies classified (LLM judge): {total}")
    unmatched = sum(1 for r in results if r.method == "unmatched")
    per_line = sum(1 for r in results if r.method == "llm_per_line")
    if unmatched:
        print(f"  unmatched (no LLM result): {unmatched}")
    if per_line:
        print(f"  llm_per_line (batch miss fallback): {per_line}")
    print()
    print("By move:")
    for move_id, count in by_move.most_common():
        pct = 100.0 * count / total
        print(f"  {move_id:16} {count:3}  ({pct:5.1f}%)")
    print("\nBy tone:")
    for tone, count in by_tone.most_common():
        pct = 100.0 * count / total
        print(f"  {tone:16} {count:3}  ({pct:5.1f}%)")
    print("\nBy role:")
    for role, count in by_role.most_common():
        print(f"  {role:16} {count:3}")

    print("\nSamples (first per move):")
    seen: set[str] = set()
    for r in results:
        if r.move_id in seen:
            continue
        seen.add(r.move_id)
        snippet = r.reply[:100] + ("…" if len(r.reply) > 100 else "")
        reason_bit = f" — {r.reason}" if r.reason else ""
        print(f"  [{r.move_id}] {r.role}: {snippet}{reason_bit}")


def plot_moves(results: list[Classification], out: str | None) -> None:
    import matplotlib.pyplot as plt

    if not results:
        return

    counts = Counter(r.move_id for r in results)
    order = [m for m, _ in counts.most_common()]
    values = [counts[m] for m in order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(order) * 0.45)))
    bars = ax.barh(order, values, color="#4f46e5", alpha=0.85)
    ax.set_xlabel("Panel replies")
    ax.set_title("Dialogue Move Classification (LLM judge)")
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=10,
        )
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
    results: list[Classification],
    meta: dict[str, Any],
    *,
    library_id: str,
    judge_model: str,
) -> dict[str, Any]:
    classified_by_key = {
        (r.source, r.turn_num, r.role, r.step): r for r in results
    }
    turn_rows = []
    for turn in turns:
        replies = []
        for pr in turn.panel_replies:
            key = (turn.source, turn.turn_num, pr.role, pr.step)
            cls = classified_by_key.get(key)
            replies.append(
                {
                    **asdict(pr),
                    "move_id": cls.move_id if cls else EVAL_UNMATCHED_MOVE_ID,
                    "move_label": cls.move_label if cls else "Unmatched",
                    "tone": cls.tone if cls else "unmatched",
                    "confidence": cls.confidence if cls else 0.0,
                    "score": cls.score if cls else 0,
                    "reason": cls.reason if cls else "",
                    "judge_latency_s": cls.judge_latency_s if cls else 0.0,
                    "method": cls.method if cls else "unmatched",
                }
            )
        turn_rows.append(
            {
                "turn_num": turn.turn_num,
                "source": turn.source,
                "heard": turn.heard,
                "host_reply": turn.host_reply,
                "panel_replies": replies,
            }
        )
    return {
        "session": meta,
        "judge": {"model": judge_model, "library": library_id, "method": "llm"},
        "turn_count": len(turns),
        "panel_reply_count": len(results),
        "classifications": [asdict(r) for r in results],
        "turns": turn_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify panel replies into dialogue moves (LLM judge)"
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Session JSONL log file(s)")
    parser.add_argument(
        "--library",
        default="panel",
        help="Dialogue library id (default: panel)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="App config yaml (default: CONFIG_PATH or config/multimodal.yaml)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="Bar chart output path (default: eval/figures/moves/session-<timecode>_moves.png)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSON export path (default: eval/results/moves/session-<timecode>_moves.json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON export",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip bar chart (summary only)",
    )
    args = parser.parse_args()

    app_cfg = load_config(args.config)
    llm = app_cfg.locale().llm
    print(f"LLM judge → {llm.base_url} model={llm.model} library={args.library}")

    events = load_events(*args.logs)
    meta = session_metadata(events)
    turns = extract_human_turns(events)
    results = classify_turns_llm(turns, library_id=args.library, llm=llm)

    print_summary(results)

    payload = build_export(
        turns,
        results,
        meta,
        library_id=args.library,
        judge_model=llm.model,
    )

    if not args.no_json:
        json_path = args.json or default_result_path(
            _EVAL_DIR, "moves", *args.logs, events=events
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote JSON → {json_path}")

    if not args.no_plot and results:
        out = args.out or str(
            default_figure_path(_EVAL_DIR, "moves", *args.logs, events=events)
        )
        plot_moves(results, out)


if __name__ == "__main__":
    main()
