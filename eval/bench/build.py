"""Item builders for T1-T4 (CPU, deterministic: no sampling anywhere, same inputs give byte-identical files).

    python -m eval.bench.build --split dev  [--pack FACTS.json]   # -> eval/data/bench/dev.jsonl (committed)
    python -m eval.bench.build --split test [--pack FACTS.json]   # -> eval/data/bench/build/test.jsonl (git-ignored),
                                                                  #    checked against eval/data/bench/test_manifest.jsonl

Sources. dev: the 30 hand-written topics (eval/data/minimal_pairs/pairs.yaml) and their recorded show states
(prefix/<topic>.json). test: the 42 public topics (eval/data/public_set/build/pairs_public.yaml, rebuilt locally from
the public corpus, never committed). The fact pack (T3) is a JSON file {topic_id: {"topic", "facts": [{"fact", "page",
"url", ...}]}} given with --pack or TALKSHOW_FACT_PACK; without it T3 is skipped.

History. "prefix" (dev default): the recorded show up to the audience member's new line (the panel has already
discussed the audience member's opening line). "opening" (test default): the host's two opening lines only, so the
audience member's line opens the discussion. Recorded states exist for only part of the test topics, and every test
item must be rebuildable from committed data plus the public corpus.

Tasks (README.md):
  T1 listening   the audience line, edited one element at a time; the addressed panelist answers
  T2 addressing  the same question put to each panelist by name (template below); both panelists are tested
  T3 grounding   checked facts offered; one numeric fact edited by the rule in fact_edit.py
  T4 persona     the two panelists' persona cards swapped; both panelists are tested
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.bench.items import PANEL, Item, item_hash, read_items, write_items  # noqa: E402

DEV_PAIRS = ROOT / "eval" / "data" / "minimal_pairs" / "pairs.yaml"
DEV_PREFIX = ROOT / "eval" / "data" / "minimal_pairs" / "prefix"
TEST_PAIRS = ROOT / "eval" / "data" / "public_set" / "build" / "pairs_public.yaml"
TEST_PREFIX = ROOT / "eval" / "data" / "public_set" / "build" / "prefix"
DATA = ROOT / "eval" / "data" / "bench"
BUILD = DATA / "build"
MANIFEST = DATA / "test_manifest.jsonl"

# T1: (version, key in the pairs file's turn2, is_control). "orig" is the reference of every T1 pair.
T1_VERSIONS = (
    ("orig", "orig", False),
    ("paraphrase", "paraphrase", True),
    ("number", "number", False),
    ("stance", "negate", False),
    ("entity", "entity", False),
    ("cause", "causal_flip", False),
)
# T2: the audience question, its opening vocative removed, put to one panelist by name.
T2_TEMPLATE = "{name}, {question}"
VOCATIVE = re.compile(r"^\s*(?:Ryan|Amy)\s*[,—–:\-]\s*")
T3_MAX_EDITS = 2
REFERENCE = {"T1": "orig", "T2": "addr_commentator", "T3": "fact_orig", "T4": "persona_orig"}

# The host's opening lines in the show (the first two lines of every recorded show state).
OPENING = (
    "Welcome — I'm Lessac, your host. Raise your hand if you'd like to speak or bring a topic — I'll call on you.",
    "You wanted to talk about {topic} — the floor is yours, go ahead.",
)
SHOW_STATE_KEYS = ("role_pad", "role_emotion", "role_pad_neighbors", "dialogue_example_index")


# ------------------------------------------------------------------------------------------------ shared pieces

def names() -> dict[str, str]:
    from agent.config import load_persona_name

    return {r: load_persona_name(r) for r in ("host", *PANEL)}


def persona_cards(swap: bool) -> dict[str, str]:
    """Panel persona cards as the show writes them; swap=True swaps the two panelists' traits (names stay)."""
    from agent.config import load_persona_personality, set_persona_swap

    set_persona_swap(swap)
    try:
        return {r: load_persona_personality(r) for r in PANEL}
    finally:
        set_persona_swap(False)


def opening_history(topic: str) -> list[dict[str, Any]]:
    host = names()["host"]
    return [{"speaker": host, "role": "host", "text": t.format(topic=topic.rstrip(".?")), "procedural": True}
            for t in OPENING]


def history_from_prefix(prefix: dict) -> list[dict[str, Any]]:
    out = []
    for line in prefix["lines"]:
        h = {"speaker": line["speaker"], "role": line["role_id"], "text": line["text"]}
        if line.get("procedural"):
            h["procedural"] = True
        out.append(h)
    return out


def prefix_from_history(history: list[dict[str, Any]], show_state: dict | None = None) -> dict[str, Any]:
    """The show's replay prefix for a history (agent/session/ghost_replay.apply_prefix)."""
    lines = [{"role_id": h["role"], "speaker": h["speaker"], "text": h["text"], "procedural": bool(h.get("procedural"))}
             for h in history]
    return {"lines": lines, **(show_state or {})}


def load_topics(split: str, pairs: Path, prefix_dir: Path, history: str) -> list[dict[str, Any]]:
    import yaml

    if not pairs.is_file():
        raise SystemExit(f"{pairs} not found" + (" (rebuild it: eval/data/public_set/README.md)" if split == "test"
                                                  else ""))
    out = []
    for t in yaml.safe_load(pairs.read_text())["topics"]:
        rec = {"slug": t["slug"], "topic": t["topic"], "turn2": t["turn2"]}
        if history == "prefix":
            pf = prefix_dir / f"{t['slug']}.json"
            if not pf.is_file():
                raise SystemExit(f"no recorded show state {pf}; use --history opening")
            prefix = json.loads(pf.read_text())
            rec["history"] = history_from_prefix(prefix)
            rec["show_state"] = {k: prefix[k] for k in SHOW_STATE_KEYS if k in prefix}
        else:
            rec["history"] = opening_history(t["topic"])
            rec["show_state"] = {}
        rec["history_source"] = history
        out.append(rec)
    return out


def _item(task: str, split: str, t: dict, version: str, role: str, group: str, *, audience: str, personas: dict,
          is_control: bool = False, fact_pack: list[str] | None = None, expected: dict | None = None,
          meta: dict | None = None) -> Item:
    return Item(
        item_id=f"{task}/{split}/{t['slug']}/{version}/{role}",
        task=task, split=split, topic_id=t["slug"], topic=t["topic"], pair_group=group, version=version,
        is_control=is_control, role_under_test=role, personas=personas, history=[dict(h) for h in t["history"]],
        audience_turn=audience, fact_pack=fact_pack, expected=expected or {},
        meta={"reference_version": REFERENCE[task], "names": names(), "history_source": t["history_source"],
              **({"show_state": t["show_state"]} if t["show_state"] else {}), **(meta or {})},
    )


def addressed(text: str) -> str | None:
    from agent.floor.addressee import addressed_role

    return addressed_role(text, {r: n for r, n in names().items() if r in PANEL})


# ------------------------------------------------------------------------------------------------ T1-T4

def build_t1(topics: list[dict], split: str, log: Counter) -> list[Item]:
    items, cards = [], persona_cards(False)
    for t in topics:
        role = addressed(t["turn2"]["orig"])
        if role is None:
            log["T1 skipped topic: original turn names no panelist"] += 1
            continue
        for version, key, control in T1_VERSIONS:
            if key not in t["turn2"]:
                log[f"T1 {version}: not in the {split} pairs"] += 1
                continue
            items.append(_item("T1", split, t, version, role, f"T1:{split}:{t['slug']}", audience=t["turn2"][key],
                               personas=cards, is_control=control,
                               expected={"addressee": role, "edit": None if version == "orig" else version},
                               meta={"source": f"turn2.{key}"}))
    return items


def t2_turn(question_line: str, role: str) -> str:
    q = VOCATIVE.sub("", question_line, count=1)
    return T2_TEMPLATE.format(name=names()[role], question=q)


def build_t2(topics: list[dict], split: str, log: Counter) -> list[Item]:
    items, cards = [], persona_cards(False)
    for t in topics:
        texts = {a: t2_turn(t["turn2"]["orig"], a) for a in PANEL}
        if any(addressed(texts[a]) != a for a in PANEL):
            log["T2 skipped topic: name not detected as the addressee"] += 1
            continue
        for role in PANEL:
            for a in PANEL:
                items.append(_item("T2", split, t, f"addr_{a}", role, f"T2:{split}:{t['slug']}:{role}",
                                   audience=texts[a], personas=cards,
                                   expected={"addressee": a, "should_answer": role == a},
                                   meta={"source": "turn2.orig", "template": T2_TEMPLATE}))
    return items


@contextmanager
def _env(**kv: str | None) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def offered_facts(pack_path: str, t: dict, audience: str) -> list[int]:
    """Pack indices of the facts the show offers the panelist at this turn (agent/panel/fact_pack.pack_block, default
    number of facts)."""
    from agent.panel.fact_pack import pack_block
    from eval.model_only import _data_for

    with _env(TALKSHOW_FACT_PACK=pack_path, TALKSHOW_FACT_PACK_K=None, TALKSHOW_FACT_PACK_TOPIC=None):
        data = _data_for(prefix_from_history(t["history"], t["show_state"]), audience)
        _block, info = pack_block(data, t["slug"])
    return list(info.get("fact_pack_offered") or [])


def build_t3(topics: list[dict], split: str, pack_path: str, log: Counter) -> list[Item]:
    from eval.bench.fact_edit import edit_fact

    pack = json.loads(Path(pack_path).read_text(encoding="utf-8"))
    items, cards = [], persona_cards(False)
    for t in topics:
        if t["slug"] not in pack:
            log["T3 skipped topic: not in the fact pack"] += 1
            continue
        audience = t["turn2"]["orig"]
        role = addressed(audience) or "commentator"
        facts = pack[t["slug"]]["facts"]
        idx = offered_facts(pack_path, t, audience)
        offered = [facts[i]["fact"].strip() for i in idx]
        sources = [{"pack_index": i, "page": facts[i].get("page"), "url": facts[i].get("url")} for i in idx]
        context = [t["topic"], audience, *(h["text"] for h in t["history"])]
        edits = []
        for j, fact in enumerate(offered):
            if len(edits) == T3_MAX_EDITS:
                break
            got = edit_fact(fact, context + [f for k, f in enumerate(offered) if k != j])
            if got:
                edits.append((j, *got))
        if not edits:
            log["T3 skipped topic: no offered fact with an editable number"] += 1
            continue
        log[f"T3 topics with {len(edits)} edited fact(s)"] += 1
        group = f"T3:{split}:{t['slug']}"
        common = {"source": "turn2.orig", "fact_sources": sources}
        items.append(_item("T3", split, t, "fact_orig", role, group, audience=audience, personas=cards,
                           fact_pack=offered, meta=common,
                           expected={"edits": [{"fact_index": j, "old_key": e.old_key, "new_key": e.new_key}
                                               for j, _f, e in edits]}))
        for n, (j, new_fact, e) in enumerate(edits, start=1):
            pack_v = list(offered)
            pack_v[j] = new_fact
            items.append(_item(
                "T3", split, t, f"fact_edit_{n}", role, group, audience=audience, personas=cards, fact_pack=pack_v,
                meta={**common, "fact_edit": {"fact_index": j, "start": e.start, "end": e.end}},
                expected={"fact_index": j, "old_fact": offered[j], "new_fact": new_fact, "old_value": e.old,
                          "new_value": e.new, "kind": e.kind, "old_key": e.old_key, "new_key": e.new_key}))
    return items


def build_t4(topics: list[dict], split: str, log: Counter) -> list[Item]:
    items = []
    cards = {False: persona_cards(False), True: persona_cards(True)}
    other = {"commentator": "guest", "guest": "commentator"}
    for t in topics:
        for role in PANEL:
            for swap in (False, True):
                items.append(_item("T4", split, t, "persona_swap" if swap else "persona_orig", role,
                                   f"T4:{split}:{t['slug']}:{role}", audience=t["turn2"]["orig"], personas=cards[swap],
                                   expected={"persona_of": other[role] if swap else role},
                                   meta={"source": "turn2.orig", "persona_swap": swap}))
    return items


BUILDERS = {"T1": build_t1, "T2": build_t2, "T4": build_t4}


def build(split: str, tasks: tuple[str, ...] = ("T1", "T2", "T3", "T4"), pack: str = "", history: str = "",
          pairs: Path | None = None, prefix_dir: Path | None = None, log: Counter | None = None) -> list[Item]:
    log = log if log is not None else Counter()
    history = history or ("prefix" if split == "dev" else "opening")
    pairs = pairs or (DEV_PAIRS if split == "dev" else TEST_PAIRS)
    prefix_dir = prefix_dir or (DEV_PREFIX if split == "dev" else TEST_PREFIX)
    topics = load_topics(split, Path(pairs), Path(prefix_dir), history)
    items: list[Item] = []
    for task in tasks:
        if task == "T3":
            if not pack:
                log["T3 skipped: no fact pack given"] += 1
                continue
            items += build_t3(topics, split, pack, log)
        else:
            items += BUILDERS[task](topics, split, log)
    return items


# ------------------------------------------------------------------------------------------------ test manifest

def manifest_row(it: Item) -> dict[str, Any]:
    """What is committed for a test item: ids, the edit, a content hash. No source text."""
    row = {k: getattr(it, k) for k in ("item_id", "task", "split", "topic_id", "pair_group", "version", "is_control",
                                       "role_under_test")}
    row["source"] = it.meta.get("source")
    if it.fact_pack is not None:
        row["fact_pack_indices"] = [s["pack_index"] for s in it.meta.get("fact_sources", [])]
    if "fact_edit" in it.meta:
        e = it.expected
        row["fact_edit"] = {**it.meta["fact_edit"], "old": e["old_value"], "new": e["new_value"], "kind": e["kind"]}
    row["sha"] = item_hash(it)
    return row


def check_manifest(items: list[Item], path: Path) -> list[str]:
    want = {r["item_id"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}
    got = {it.item_id: manifest_row(it) for it in items}
    problems = [f"missing {i}" for i in sorted(set(want) - set(got))]
    problems += [f"extra {i}" for i in sorted(set(got) - set(want))]
    problems += [f"differs {i}" for i in sorted(set(want) & set(got)) if want[i]["sha"] != got[i]["sha"]]
    return problems


def counts(items: list[Item]) -> dict[str, Any]:
    by: dict[str, Any] = defaultdict(lambda: {"items": 0, "topics": set(), "versions": Counter()})
    for it in items:
        c = by[it.task]
        c["items"] += 1
        c["topics"].add(it.topic_id)
        c["versions"][it.version] += 1
    return {k: {"items": v["items"], "topics": len(v["topics"]), "versions": dict(v["versions"])}
            for k, v in sorted(by.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--tasks", default="T1,T2,T3,T4")
    ap.add_argument("--pack", default=os.environ.get("TALKSHOW_FACT_PACK", ""),
                    help="fact pack json for T3 (default TALKSHOW_FACT_PACK; none = T3 skipped)")
    ap.add_argument("--history", choices=("prefix", "opening"), default="",
                    help="history before the audience line (default: prefix for dev, opening for test)")
    ap.add_argument("--pairs", default="", help="pairs file (default: the split's)")
    ap.add_argument("--prefix-dir", default="", help="recorded show states (--history prefix)")
    ap.add_argument("--out", default="", help="items jsonl (default eval/data/bench/dev.jsonl or build/test.jsonl)")
    ap.add_argument("--write-manifest", action="store_true", help="test: (re)write eval/data/bench/test_manifest.jsonl")
    args = ap.parse_args()

    log: Counter = Counter()
    items = build(args.split, tuple(args.tasks.split(",")), args.pack, args.history,
                  Path(args.pairs) if args.pairs else None, Path(args.prefix_dir) if args.prefix_dir else None, log)
    out = Path(args.out) if args.out else (DATA / "dev.jsonl" if args.split == "dev" else BUILD / "test.jsonl")
    n = write_items(out, items)
    assert len(read_items(out)) == n
    print(f"wrote {n} items -> {out}")
    print(json.dumps(counts(items), indent=1))
    for k, v in sorted(log.items()):
        print(f"  {k}: {v}")
    if args.split == "test":
        if args.write_manifest:
            MANIFEST.write_text("".join(json.dumps(manifest_row(it), ensure_ascii=False) + "\n" for it in items))
            print(f"wrote manifest -> {MANIFEST}")
        elif MANIFEST.is_file():
            problems = check_manifest(items, MANIFEST)
            if problems:
                print(f"MISMATCH with {MANIFEST}: {len(problems)} items, e.g. {problems[:5]}")
                sys.exit(1)
            print(f"all {n} items match {MANIFEST}")


if __name__ == "__main__":
    main()
