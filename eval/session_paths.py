"""Derive session timecodes from log paths for eval output naming."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SESSION_RE = re.compile(r"session-(\d+)")


def _code_from_name(name: str) -> str | None:
    m = _SESSION_RE.search(name)
    return m.group(1) if m else None


def session_timecodes(
    *log_paths: Path,
    events: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Unique timecodes from log stems, then session_start log_file fields."""
    codes: list[str] = []
    seen: set[str] = set()
    for p in log_paths:
        code = _code_from_name(p.stem)
        if code and code not in seen:
            codes.append(code)
            seen.add(code)
    if codes:
        return codes
    if not events:
        return []
    for e in events:
        if e.get("event") != "session_start":
            continue
        for field in ("log_file", "_source"):
            raw = e.get(field)
            if not raw:
                continue
            code = _code_from_name(str(raw))
            if code and code not in seen:
                codes.append(code)
                seen.add(code)
    return codes


def session_output_stem(
    *log_paths: Path,
    events: list[dict[str, Any]] | None = None,
) -> str:
    codes = session_timecodes(*log_paths, events=events)
    if not codes:
        return "session-unknown"
    if len(codes) == 1:
        return f"session-{codes[0]}"
    return "sessions-" + "_".join(codes)


def default_eval_filename(
    *log_paths: Path,
    suffix: str,
    ext: str,
    events: list[dict[str, Any]] | None = None,
) -> str:
    """e.g. session-1782786500_latency.png"""
    return f"{session_output_stem(*log_paths, events=events)}_{suffix}.{ext}"


def figures_subdir(eval_dir: Path, kind: str) -> Path:
    """Per-metric figure directory, e.g. eval/figures/latency/."""
    return eval_dir / "figures" / kind


def default_figure_path(
    eval_dir: Path,
    kind: str,
    *log_paths: Path,
    events: list[dict[str, Any]] | None = None,
) -> Path:
    """e.g. eval/figures/latency/session-1782786500_latency.png"""
    return figures_subdir(eval_dir, kind) / default_eval_filename(
        *log_paths, suffix=kind, ext="png", events=events
    )


def results_subdir(eval_dir: Path, kind: str) -> Path:
    """Per-metric JSON directory, e.g. eval/results/moves/."""
    return eval_dir / "results" / kind


def default_result_path(
    eval_dir: Path,
    kind: str,
    *log_paths: Path,
    events: list[dict[str, Any]] | None = None,
    ext: str = "json",
) -> Path:
    """e.g. eval/results/moves/session-1782786500_moves.json"""
    return results_subdir(eval_dir, kind) / default_eval_filename(
        *log_paths, suffix=kind, ext=ext, events=events
    )


def find_result_json(eval_dir: Path, kind: str, log: Path) -> Path | None:
    """Resolve exported JSON for a session log (new layout, then legacy runs/)."""
    code = _code_from_name(log.name)
    if not code:
        return None
    candidates = [
        results_subdir(eval_dir, kind) / f"session-{code}_{kind}.json",
        results_subdir(eval_dir, kind) / f"{code}_{kind}.json",
        eval_dir / "runs" / f"session-{code}_{kind}.json",
        eval_dir / "runs" / f"{code}_{kind}.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None
