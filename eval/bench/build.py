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
  T2f floor      the same lines plus a control naming nobody; both panelists bid, the floor selection picks
  T3a grounding  the audience asks about one fact's subject; that fact (original or with one number edited by the
                 rule in fact_edit.py) is offered among 6 facts
  T3b            the T3a items plus one instruction to use the facts (meta.fact_instruction)
  T4 persona     the two panelists' persona cards swapped; both panelists are tested
"""

from __future__ import annotations

import argparse
import hashlib
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
T3_OTHER_FACTS = 5
SEED = 20261110  # seeds the T3 question template and the edited fact's position
# T3: the audience asks about the edited fact's subject (its Wikipedia page title). No number in any template.
T3_QUESTIONS = (
    "What does the record actually say about {subject}? I want the specifics.",
    "Can you walk me through the facts on {subject}?",
    "I keep hearing about {subject}. What do we actually know?",
    "Before we go further, what are the hard facts about {subject}?",
)
DROPPED: list[dict] = []  # implausible T3 edits seen by the last build (written to build/t3_dropped_<split>.jsonl)
REFERENCE = {"T1": "orig", "T2": "addr_commentator", "T2f": "addr_commentator", "T3a": "fact_orig_1", "T3b": "fact_orig_1",
             "T4": "persona_orig"}  # T3a/T3b: set per pair

# The host's opening lines in the show (the first two lines of every recorded show state).
OPENING = (
    "Welcome — I'm Lessac, your host. Raise your hand if you'd like to speak or bring a topic — I'll call on you.",
    "You wanted to talk about {topic} — the floor is yours, go ahead.",
)
SHOW_STATE_KEYS = ("role_pad", "role_emotion", "role_pad_neighbors", "dialogue_example_index")


# ------------------------------------------------------------------------------------------------ shared pieces

def names() -> dict[str, str]:
    from agent.config import load_persona_name

    with _env(**NO_LIBRARY):
        return {r: load_persona_name(r) for r in ("host", *PANEL)}


# Items carry the seats' own prose cards and names: a persona library selected for a system run
# (TALKSHOW_PERSONA_LIBRARY) is a property of that system, not of the items.
NO_LIBRARY = {"TALKSHOW_PERSONA_LIBRARY": None, "TALKSHOW_PERSONA_LIBRARY_NAMES": None}


def persona_cards(swap: bool) -> dict[str, str]:
    """Panel persona cards as the show writes them; swap=True swaps the two panelists' traits (names stay)."""
    from agent.config import load_persona_personality, set_persona_swap

    set_persona_swap(swap)
    try:
        with _env(**NO_LIBRARY):
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


def build_t2f(topics: list[dict], split: str, log: Counter) -> list[Item]:
    """T2f: the T2 audience lines (the question put to each panelist by name) plus a control with nobody named; one
    item per version, answered by the floor selection over both panelists (role "panel")."""
    items, cards = [], persona_cards(False)
    for t in topics:
        texts = {a: t2_turn(t["turn2"]["orig"], a) for a in PANEL}
        q = VOCATIVE.sub("", t["turn2"]["orig"], count=1)
        texts[None] = q[:1].upper() + q[1:]
        if any(addressed(texts[a]) != a for a in (*PANEL, None)):
            log["T2f skipped topic: addressee not detected as intended"] += 1
            continue
        for a in (*PANEL, None):
            items.append(_item("T2f", split, t, f"addr_{a or 'none'}", "panel", f"T2f:{split}:{t['slug']}",
                               audience=texts[a], personas=cards, is_control=a is None,
                               expected={"addressee": a},
                               meta={"source": "turn2.orig", "template": T2_TEMPLATE if a else "{question}"}))
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


def offered_facts(pack_path: str, t: dict, audience: str, k: int | None = None) -> list[int]:
    """Pack indices of the facts the show offers the panelist at this turn (agent/panel/fact_pack.pack_block; k facts,
    default the show's number), most relevant first."""
    from agent.panel.fact_pack import pack_block
    from eval.model_only import _data_for

    with _env(TALKSHOW_FACT_PACK=pack_path, TALKSHOW_FACT_PACK_K=str(k) if k else None, TALKSHOW_FACT_PACK_TOPIC=None):
        data = _data_for(prefix_from_history(t["history"], t["show_state"]), audience)
        _block, info = pack_block(data, t["slug"])
    return list(info.get("fact_pack_offered") or [])


def _hash(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(map(str, (SEED, *parts))).encode()).hexdigest()[:8], 16)


def page_subject(page: str) -> str:
    """A page title as the subject of a question: no parenthetical, no token with a digit ("Fair Labor Standards Act of
    1938" -> "Fair Labor Standards Act")."""
    title = re.sub(r"\s*\([^)]*\)", "", page or "").strip()
    words = [w for w in title.split() if not re.search(r"\d", w)]
    while words and words[-1].lower() in ("of", "in", "from", "to", "and", "the", "–", "-"):
        words.pop()
    return " ".join(words)


def t3_question(subject: str, role: str, group: str) -> str:
    """The audience question for a T3 pair: one of T3_QUESTIONS (by seeded hash of the pair), put to the panelist."""
    q = T3_QUESTIONS[_hash(group, "question") % len(T3_QUESTIONS)].format(subject=subject)
    if not q.startswith("I "):  # "I keep hearing" keeps its capital after the name
        q = q[0].lower() + q[1:]
    return T2_TEMPLATE.format(name=names()[role], question=q)


def build_t3(topics: list[dict], split: str, pack_path: str, log: Counter) -> list[Item]:
    """Per topic up to 2 pairs. A pair edits one numeric fact; the audience asks about that fact's page, and the panelist
    is offered the fact (original or edited) plus the 5 most relevant facts from other pages, the fact at a seeded
    position. The edited facts are taken, in the show's order, from the 6 facts it would offer for the original turn."""
    from eval.bench.fact_edit import edit_fact
    from eval.quality_ruler.specifics import extract

    pack = json.loads(Path(pack_path).read_text(encoding="utf-8"))
    items, cards = [], persona_cards(False)
    for t in topics:
        if t["slug"] not in pack:
            log["T3a skipped topic: not in the fact pack"] += 1
            continue
        role = addressed(t["turn2"]["orig"]) or "commentator"
        facts = pack[t["slug"]]["facts"]
        context = [t["topic"], *(h["text"] for h in t["history"])]
        pairs, pages_used = [], set()
        for i in offered_facts(pack_path, t, t["turn2"]["orig"]):
            if len(pairs) == T3_MAX_EDITS:
                break
            page = facts[i].get("page") or ""
            subject = page_subject(page)
            if not subject or page in pages_used:
                continue
            dropped: list = []
            got = edit_fact(facts[i]["fact"].strip(), context, dropped)
            for why, old_f, new_f in dropped:
                log[f"T3a edit dropped as implausible: {why}"] += 1
                DROPPED.append({"split": split, "topic_id": t["slug"], "reason": why, "fact": old_f, "edited": new_f})
            if not got:
                continue
            new_fact, e = got
            n = len(pairs) + 1
            group = f"T3a:{split}:{t['slug']}:{n}"
            key = f"T3:{split}:{t['slug']}:{n}"  # seeds the question and the position (the key the items were first built with)
            question = t3_question(subject, role, key)
            # the other 5: most relevant to the question first, other pages only, neither value anywhere in them
            others = []
            for k in offered_facts(pack_path, t, question, k=len(facts)):
                if len(others) == T3_OTHER_FACTS:
                    break
                if k == i or (facts[k].get("page") or "") == page:
                    continue
                keys = {s["key"] for s in extract(facts[k]["fact"])}
                if e.old_key in keys or e.new_key in keys:
                    continue
                others.append(k)
            if len(others) < T3_OTHER_FACTS:
                log["T3a edit skipped: fewer than 5 facts from other pages"] += 1
                continue
            pos = _hash(key, "position") % (T3_OTHER_FACTS + 1)
            idx = others[:pos] + [i] + others[pos:]
            pages_used.add(page)
            pairs.append((n, group, key, question, idx, pos, new_fact, e))
        if not pairs:
            log["T3a skipped topic: no usable pair (no editable number, or fewer than 5 facts from other pages)"] += 1
            continue
        log[f"T3a topics with {len(pairs)} pair(s)"] += 1
        for n, group, key, question, idx, pos, new_fact, e in pairs:
            offered = [facts[k]["fact"].strip() for k in idx]
            sources = [{"pack_index": k, "page": facts[k].get("page"), "url": facts[k].get("url")} for k in idx]
            common = {"source": "fact_page_question", "fact_sources": sources, "reference_version": f"fact_orig_{n}",
                      "question_template": _hash(key, "question") % len(T3_QUESTIONS)}
            edited = list(offered)
            edited[pos] = new_fact
            items.append(_item("T3a", split, t, f"fact_orig_{n}", role, group, audience=question, personas=cards,
                               fact_pack=offered, meta=common,
                               expected={"edits": [{"fact_index": pos, "old_key": e.old_key, "new_key": e.new_key}]}))
            items.append(_item(
                "T3a", split, t, f"fact_edit_{n}", role, group, audience=question, personas=cards, fact_pack=edited,
                meta={**common, "fact_edit": {"fact_index": pos, "start": e.start, "end": e.end}},
                expected={"fact_index": pos, "old_fact": offered[pos], "new_fact": new_fact, "old_value": e.old,
                          "new_value": e.new, "kind": e.kind, "old_key": e.old_key, "new_key": e.new_key}))
    return items


def t3b_from_t3a(items: list[Item]) -> list[Item]:
    """T3b: the T3a items unchanged except the task, ids and one instruction (meta.fact_instruction)."""
    import copy
    from dataclasses import replace

    from eval.bench.items import FACT_INSTRUCTION

    out = []
    for it in items:
        meta = copy.deepcopy(it.meta)
        meta["fact_instruction"] = FACT_INSTRUCTION
        out.append(replace(it, task="T3b", item_id="T3b/" + it.item_id.split("/", 1)[1],
                           pair_group="T3b:" + it.pair_group.split(":", 1)[1], meta=meta,
                           personas=dict(it.personas), history=[dict(h) for h in it.history],
                           fact_pack=list(it.fact_pack or []), expected=copy.deepcopy(it.expected)))
    return out


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


BUILDERS = {"T1": build_t1, "T2": build_t2, "T2f": build_t2f, "T4": build_t4}


def build(split: str, tasks: tuple[str, ...] = ("T1", "T2", "T2f", "T3a", "T3b", "T4"), pack: str = "",
          history: str = "",
          pairs: Path | None = None, prefix_dir: Path | None = None, log: Counter | None = None) -> list[Item]:
    log = log if log is not None else Counter()
    history = history or ("prefix" if split == "dev" else "opening")
    pairs = pairs or (DEV_PAIRS if split == "dev" else TEST_PAIRS)
    prefix_dir = prefix_dir or (DEV_PREFIX if split == "dev" else TEST_PREFIX)
    topics = load_topics(split, Path(pairs), Path(prefix_dir), history)
    from eval.bench.items import GROUNDING, expand_tasks

    items: list[Item] = []
    t3a: list[Item] | None = None
    for task in expand_tasks(list(tasks)):
        if task in GROUNDING:
            if not pack:
                log[f"{task} skipped: no fact pack given"] += 1
                continue
            if t3a is None:
                t3a = build_t3(topics, split, pack, log)
            items += t3a if task == "T3a" else t3b_from_t3a(t3a)
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
    ap.add_argument("--tasks", default="T1,T2,T2f,T3a,T3b,T4", help="T3 = T3a and T3b")
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
    if DROPPED:
        BUILD.mkdir(parents=True, exist_ok=True)
        (BUILD / f"t3_dropped_{args.split}.jsonl").write_text(
            "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in DROPPED))
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
