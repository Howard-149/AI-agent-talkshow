#!/usr/bin/env python3
"""Ghost scripts for the listening test (eval/README.md).

Three stages per topic in pairs.yaml:

1. ``prefix``  — one live session per topic: turn1 + orig turn2, run with
   TALKSHOW_GHOST_REPLAY=1 so the worker logs ``history_snapshot`` at each
   ghost_human_start. The snapshot taken just before turn 2 is the shared prefix.

       python -m eval.data.minimal_pairs.make_pairs prefix
       → eval/data/minimal_pairs/prefix/pfx-<slug>.yaml

2. ``extract`` — pull that snapshot out of the prefix-run logs:

       python -m eval.data.minimal_pairs.make_pairs extract logs/ghost-pairs-prefix
       → eval/data/minimal_pairs/prefix/<slug>.json

3. ``conditions`` — one script per (topic, condition, resample k). Each replays the
   prefix and injects only the (edited) turn 2:

       python -m eval.data.minimal_pairs.make_pairs conditions -k 5 [--topics a,b]
       → eval/data/minimal_pairs/cond/<slug>__<condition>__k<i>.yaml   (git-ignored)

   ``--arm X --flags KEY=VAL,...`` writes ``<...>__vX.yaml`` scripts that set those dialogue-quality flags for their
   session only (worker needs TALKSHOW_GHOST_REPLAY=1), so two arms can run interleaved in one job.

The prefix run's room must be named ``ghost-pfx-<slug>-...`` (the ghost batch job names rooms after the script file).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# Other topic sets (e.g. the public test set in eval/data/public_set/build/, raw text, git-ignored): override all three.
PAIRS = Path(os.environ.get("TALKSHOW_PAIRS", HERE / "pairs.yaml"))
PREFIX_DIR = Path(os.environ.get("TALKSHOW_PREFIX_DIR", HERE / "prefix"))
COND_DIR = Path(os.environ.get("TALKSHOW_COND_DIR", HERE / "cond"))

TEXT_CONDITIONS = ("orig", "paraphrase", "negate", "number", "entity", "addressee")
CONDITIONS = (*TEXT_CONDITIONS, "persona_swap")


def load_topics() -> list[dict]:
    topics = yaml.safe_load(PAIRS.read_text(encoding="utf-8"))["topics"]
    for t in topics:
        missing = set(TEXT_CONDITIONS) - set(t["turn2"])
        assert not missing, f"{t['slug']}: missing {missing}"
    return topics


def _dump(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000), encoding="utf-8")


def cmd_prefix(_args: argparse.Namespace) -> None:
    topics = load_topics()
    for t in topics:
        _dump(
            PREFIX_DIR / f"pfx-{t['slug']}.yaml",
            {
                "locale": "en",
                "identity": "ghost-guest",
                "topic": t["topic"],
                "turns": [
                    {"text": t["turn1"], "topic": t["topic"]},
                    {"text": t["turn2"]["orig"], "topic": t["topic"]},
                ],
            },
        )
    print(f"wrote {len(topics)} prefix scripts → {PREFIX_DIR}")


def cmd_extract(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT))
    from eval.log_parse import load_events

    want = {t["slug"] for t in load_topics()}
    found: dict[str, tuple[float, dict]] = {}
    for path in sorted(Path(args.log_dir).glob("session-*.jsonl")):
        events = load_events(path)
        room = next((e.get("room", "") for e in events if e["event"] == "session_start"), "")
        slug = next((s for s in want if room.startswith(f"ghost-pfx-{s}-")), None)
        if slug is None:
            continue
        snaps = [e for e in events if e["event"] == "history_snapshot"]
        # Snapshot #2 = state right before the orig turn 2 (turn 1 + its full beat).
        if len(snaps) < 2:
            print(f"skip {path.name} ({slug}): {len(snaps)} snapshots", file=sys.stderr)
            continue
        snap = snaps[1]
        if not any(line.get("role_id") in ("guest", "commentator") for line in snap["lines"]):
            print(f"skip {path.name} ({slug}): prefix has no panel line", file=sys.stderr)
            continue
        if slug not in found or snap["ts"] > found[slug][0]:
            found[slug] = (snap["ts"], snap)
    PREFIX_DIR.mkdir(parents=True, exist_ok=True)
    for slug, (_ts, snap) in sorted(found.items()):
        dest = PREFIX_DIR / f"{slug}.json"
        if dest.exists() and not args.force:
            # Condition runs replay this exact file; never change it under them.
            print(f"{slug}: keep existing {dest.name} (--force to replace)")
            continue
        prefix = {k: snap[k] for k in ("lines", "role_pad", "role_emotion", "dialogue_example_index") if k in snap}
        dest.write_text(json.dumps(prefix, ensure_ascii=False, indent=1), encoding="utf-8")
        n_panel = sum(line["role_id"] in ("guest", "commentator") for line in prefix["lines"])
        print(f"{slug}: {len(prefix['lines'])} lines ({n_panel} panel)")
    print(f"extracted {len(found)}/{len(want)} prefixes → {PREFIX_DIR}; missing: {sorted(want - set(found))}")


def parse_flags(spec: str) -> dict[str, str]:
    """"KEY=VAL,KEY2=VAL2" -> dict. Values stay strings (the replay prefix passes strings through unchanged)."""
    out = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        key, _, val = item.partition("=")
        out[key.strip()] = val.strip()
    return out


def cmd_conditions(args: argparse.Namespace) -> None:
    topics = load_topics()
    if args.topics:
        keep = set(args.topics.split(","))
        topics = [t for t in topics if t["slug"] in keep]
    conds = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    flags = parse_flags(args.flags)
    if flags and not args.arm:
        raise SystemExit("--flags needs --arm (the room suffix that tells the arms apart)")
    n = 0
    for t in topics:
        prefix_json = PREFIX_DIR / f"{t['slug']}.json"
        if not prefix_json.is_file():
            print(f"skip {t['slug']}: no {prefix_json.name} (run prefix + extract)", file=sys.stderr)
            continue
        for cond in conds:
            text = t["turn2"]["orig" if cond == "persona_swap" else cond]
            for k in range(args.k):
                doc = {
                    "locale": "en",
                    "identity": "ghost-guest",
                    "topic": t["topic"],
                    "condition": cond,
                    "resample": k,
                    "prefix_file": str(prefix_json.resolve().relative_to(ROOT)),
                    "turns": [{"text": text, "topic": t["topic"]}],
                }
                if cond == "persona_swap":
                    doc["persona_swap"] = True
                suffix = ""
                if args.arm:
                    if flags:
                        doc["flags"] = dict(flags)
                    suffix = f"__v{args.arm}"
                _dump(COND_DIR / f"{t['slug']}__{cond}__k{k}{suffix}.yaml", doc)
                n += 1
    print(f"wrote {n} condition scripts → {COND_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ghost scripts for the listening test: prefix runs, then condition runs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prefix", help="one prefix-run script per topic -> prefix/pfx-<slug>.yaml").set_defaults(fn=cmd_prefix)
    p = sub.add_parser("extract", help="saved show state from prefix-run logs -> prefix/<slug>.json")
    p.add_argument("log_dir")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("conditions", help="one replay script per topic x condition x resample -> cond/")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--topics", default="")
    p.add_argument("--conditions", default="")
    p.add_argument("--arm", default="", help="arm name (letters/digits): scripts get the suffix __v<arm>")
    p.add_argument("--flags", default="", help="KEY=VAL,... dialogue-quality flags for this arm's sessions")
    p.set_defaults(fn=cmd_conditions)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
