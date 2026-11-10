"""Score a run (CPU parts) and write the judge requests.

    python -m eval.bench.score RUN_DIR [--items ITEMS.jsonl] [--pack FACTS.json] [--out RUN_DIR/score]

Rules, no model:
  T2 right responder   systems that decide the floor (floor_action set): per (topic, addressee, rerun), did the
                       addressed panelist take the floor and the other one hold back?
  T3 follows the fact  does the reply state the edited value ("new"), the original value ("old"), both, or neither
                       (fact_edit.follows); on fact_orig replies, how often the original values are stated
  unsupported specifics  every reply: numbers, years, amounts and names that appear nowhere in what the panelist was
                       shown (the item's facts, the transcript, the audience line; eval/quality_ruler/specifics.py).
                       --pack adds the topic's whole fact pack as support.
Hooks for model-based measures:
  response shift R     OUT/R/<task>/lines.jsonl in eval/responsiveness.py format (one slug per pair group, the
                       reference version as "orig", reruns as samples). Embed with
                       `python -m eval.responsiveness --out OUT/R/<task> embed` (GPU); the next score run reports R.
  judge requests       OUT/judge/identification.jsonl: "which input version was this reply answering" (A/B, both
                       orders, chance 0.5; eval/judge_context.py wording; T4 asks which persona card, as the quality
                       ruler's attribution item). OUT/judge/rubric.jsonl: the quality ruler's kept yes/no items
                       (specific, persona fidelity) in its jury job format. A judge run answers them (next step).
Report: OUT/score.json and a per-task table.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.bench.items import PANEL, Item, read_items  # noqa: E402

IDENT_FACT_SYSTEM = (
    "You are a careful annotator of talk-show transcripts. A panelist was shown a list of checked facts; one fact came "
    "in one of two versions. You will see both versions and the panelist's reply. Decide which version the panelist "
    "was shown. Look for details the reply picks up (numbers, years, amounts). If you cannot tell, still pick the more "
    "likely one."
)


def load_rows(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "outputs.jsonl").read_text().splitlines() if l.strip()]


def items_for(run: Path, given: str) -> dict[str, Item]:
    if given:
        path = Path(given)
    else:
        cfg = json.loads((run / "run.json").read_text())
        path = Path(cfg["items"]) if Path(cfg["items"]).is_absolute() else ROOT / cfg["items"]
    return {it.item_id: it for it in read_items(path)}


def episode(it: Item, text: str, rid: str, reply_role: str | None = None) -> dict:
    """The item plus the reply as a quality-ruler episode (eval/quality_ruler/episodes.py format)."""
    lines = [{"role": h["role"], "speaker": h["speaker"], "text": h["text"],
              **({"procedural": True} if h.get("procedural") else {})} for h in it.history]
    if it.audience_turn:
        lines.append({"role": "human", "speaker": "Audience member", "text": it.audience_turn})
    role = reply_role or it.role_under_test
    lines.append({"role": role, "speaker": it.meta.get("names", {}).get(it.role_under_test, role), "text": text})
    return {"id": rid, "topic_id": it.topic_id, "topic": it.topic, "lines": lines, "scored": [len(lines) - 1]}


# ------------------------------------------------------------------------------------------------ rules

def right_responder(rows: list[dict], items: dict[str, Item]) -> dict:
    """T2 with floor actions: per (pair group topic, addressee, rerun), both panelists' actions."""
    cells: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r["task"] != "T2" or r.get("floor_action") is None:
            continue
        it = items[r["item_id"]]
        cells[(it.topic_id, it.expected["addressee"], r["rerun"])][it.role_under_test] = r["floor_action"]
    if not cells:
        return {"n": 0, "note": "no floor actions (the system does not decide the floor)"}
    out = Counter()
    took = defaultdict(list)
    for (_t, addressee, _k), acts in cells.items():
        if set(acts) != set(PANEL):
            out["incomplete"] += 1
            continue
        other = next(r for r in PANEL if r != addressee)
        a, o = acts[addressee] == "speak", acts[other] == "speak"
        out["right" if a and not o else "both" if a and o else "wrong" if o else "neither"] += 1
        took["addressed"].append(a)
        took["not_addressed"].append(o)
    n = sum(v for k, v in out.items() if k != "incomplete")
    res = {"n": n, **{k: round(out[k] / n, 3) if n else None for k in ("right", "both", "wrong", "neither")}}
    res["p_speak_addressed"] = round(sum(took["addressed"]) / len(took["addressed"]), 3) if took["addressed"] else None
    res["p_speak_not_addressed"] = (round(sum(took["not_addressed"]) / len(took["not_addressed"]), 3)
                                    if took["not_addressed"] else None)
    if out["incomplete"]:
        res["incomplete"] = out["incomplete"]
    return res


def follows_fact(rows: list[dict], items: dict[str, Item]) -> dict:
    from eval.bench.fact_edit import follows

    edited, orig = Counter(), Counter()
    for r in rows:
        it = items[r["item_id"]]
        if it.task != "T3" or not r["text"]:
            continue
        if it.version == "fact_orig":
            for e in it.expected.get("edits", []):
                orig[follows(r["text"], e["old_key"], e["new_key"])] += 1
        else:
            edited[follows(r["text"], it.expected["old_key"], it.expected["new_key"])] += 1

    def share(c: Counter) -> dict:
        n = sum(c.values())
        return {"n": n, **{k: round(c[k] / n, 3) if n else None for k in ("new", "old", "both", "neither")}}

    # On fact_orig replies "old" = the (unedited) value was stated; "new" should never happen.
    return {"edited": share(edited), "orig": share(orig)}


def unsupported(rows: list[dict], items: dict[str, Item], pack: dict | None) -> dict:
    from eval.quality_ruler.specifics import episode_lines, summarise

    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        if not r["text"]:
            continue
        it = items[r["item_id"]]
        if pack is not None:
            p = {it.topic_id: {"topic": it.topic, "facts": [{"fact": f} for f in (it.fact_pack or [])]
                               + list(pack.get(it.topic_id, {}).get("facts", []))}}
        else:
            p = {it.topic_id: {"topic": it.topic, "facts": [{"fact": f} for f in (it.fact_pack or [])]}}
        ep = episode(it, r["text"], f"{r['item_id']}#{r['rerun']}")
        by_task[it.task] += episode_lines(ep, p)
    keys = ("lines", "specifics", "specifics_per100w", "share_supported", "unsupported_per100w",
            "share_lines_unsupported")
    out = {}
    for task, lines in sorted(by_task.items()):
        s = summarise(lines, n_examples=5)
        out[task] = {k: (None if s[k] != s[k] else round(s[k], 3)) if isinstance(s[k], float) else s[k]
                     for k in keys}  # NaN (no specifics) -> null
        out[task]["examples_unsupported"] = s["examples_unsupported"]
    return out


# ------------------------------------------------------------------------------------------------ R hook

def r_hook(rows: list[dict], items: dict[str, Item], out: Path, boot: int = 1000) -> dict:
    """Write responsiveness-format lines per task; report R where embeddings exist."""
    from eval import responsiveness

    res = {}
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["text"]:
            by_task[r["task"]].append(r)
    for task, rs in sorted(by_task.items()):
        d = out / "R" / task
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for r in rs:
            it = items[r["item_id"]]
            cond = "orig" if it.version == it.meta.get("reference_version") else it.version
            lines.append({"slug": it.pair_group, "topic": it.topic, "cond": cond, "k": r["rerun"], "pos": 1,
                          "role": it.role_under_test, "text": r["text"], "heard": it.audience_turn or "",
                          "between": [], "log": f"{r['item_id']}#{r['rerun']}"})
        new = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
        path = d / "lines.jsonl"
        if path.is_file() and path.read_text() != new:  # the run changed: old embeddings no longer line up
            (d / "emb.npy").unlink(missing_ok=True)
        path.write_text(new)
        if (d / "emb.npy").is_file():
            responsiveness.cmd_report(Namespace(out=str(d), lp_name="lp.jsonl", seed=0, boot=boot))
            rep = json.loads((d / "responsiveness.json").read_text())
            res[task] = rep.get("positions", {}).get("pos1", {})
        else:
            res[task] = {"todo": f"python -m eval.responsiveness --out {d} embed  (GPU), then score again"}
    return res


# ------------------------------------------------------------------------------------------------ judge requests

def judge_requests(rows: list[dict], items: dict[str, Item], out: Path) -> dict:
    from eval.judge_context import SYSTEM as IDENT_SYSTEM
    from eval.judge_context import prompt as ident_prompt
    from eval.quality_ruler.items import JUDGE_SYSTEM
    from eval.quality_ruler.jury import attribution_prompt, build_jobs

    d = out / "judge"
    d.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, Item]] = defaultdict(dict)
    for it in items.values():
        groups[it.pair_group][it.version] = it
    ident, rubric = [], []
    for r in rows:
        if not r["text"]:
            continue
        it = items[r["item_id"]]
        rid = f"{r['item_id']}#{r['rerun']}"
        ref = it.meta.get("reference_version")
        others = [groups[it.pair_group][ref]] if it.version != ref else [
            v for k, v in sorted(groups[it.pair_group].items()) if k != ref]
        for other in others:
            edit = other if it.version == ref else it  # the non-reference side names the pair
            for true_is_a in (True, False):
                base = {"id": f"{rid}|{edit.version}|{'A' if true_is_a else 'B'}", "task": it.task,
                        "item_id": it.item_id, "rerun": r["rerun"], "pair_group": it.pair_group,
                        "pair": f"{ref} vs {edit.version}", "line_version": it.version,
                        "is_control": edit.is_control, "true": "A" if true_is_a else "B"}
                if it.task in ("T1", "T2"):
                    a, b = (it.audience_turn, other.audience_turn)
                    a, b = (a, b) if true_is_a else (b, a)
                    msgs = [{"role": "system", "content": IDENT_SYSTEM},
                            {"role": "user", "content": ident_prompt(it.topic, a, b, r["text"])}]
                elif it.task == "T3":
                    j = edit.expected["fact_index"]
                    a, b = it.fact_pack[j], other.fact_pack[j]
                    a, b = (a, b) if true_is_a else (b, a)
                    msgs = [{"role": "system", "content": IDENT_FACT_SYSTEM},
                            {"role": "user", "content": (
                                f"Topic: {it.topic}\n\nVersion A of the fact:\n\"{a}\"\n\nVersion B of the fact:\n\"{b}\""
                                f"\n\nPanelist's reply:\n\"{r['text']}\"\n\nWhich version was the panelist shown? "
                                "Answer with a single letter: A or B.")}]
                elif it.task == "T4":
                    # Which persona card fits the hidden speaker; the truth is the card this version gave the role.
                    owner = it.expected["persona_of"]
                    a_first = (owner == "commentator") == true_is_a
                    ep = episode(it, r["text"], rid)
                    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": attribution_prompt(ep, len(ep["lines"]) - 1, a_first)}]
                else:
                    continue
                ident.append({**base, "messages": msgs})
        # rubric: the kept yes/no items, persona items for the card the role carried
        ep = episode(it, r["text"], rid, reply_role=it.expected.get("persona_of"))
        for job in build_jobs([ep], [], 0.0, 0, dims=("specific", "fidelity")):
            rubric.append({"id": f"{rid}|{job['dim']}|{job['item']}", "task": it.task, "item_id": it.item_id,
                           "rerun": r["rerun"], "dim": job["dim"], "rubric_item": job["item"],
                           "reverse": job["reverse"], "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                                                                   {"role": "user", "content": job["prompt"]}]})
    for name, rows_ in (("identification", ident), ("rubric", rubric)):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows_))
    return {"identification": len(ident), "rubric": len(rubric), "dir": str(d)}


# ------------------------------------------------------------------------------------------------ main

def score(run: Path, items_path: str = "", pack_path: str = "", out: Path | None = None, boot: int = 1000) -> dict:
    rows = load_rows(run)
    items = items_for(run, items_path)
    missing = {r["item_id"] for r in rows} - set(items)
    if missing:
        raise SystemExit(f"{len(missing)} output rows have no item, e.g. {sorted(missing)[:3]}")
    out = out or run / "score"
    out.mkdir(parents=True, exist_ok=True)
    pack = json.loads(Path(pack_path).read_text()) if pack_path else None
    per_task: dict[str, Any] = {}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task]
        per_task[task] = {"replies": len(rs), "empty": sum(not r["text"] for r in rs),
                          "items": len({r["item_id"] for r in rs}), "reruns": len({r["rerun"] for r in rs}),
                          "floor_actions": dict(Counter(r.get("floor_action") or "none" for r in rs))}
    spec = unsupported(rows, items, pack)
    for task, v in spec.items():
        per_task[task]["unsupported_specifics"] = v
    if "T2" in per_task:
        per_task["T2"]["right_responder"] = right_responder(rows, items)
    if "T3" in per_task:
        per_task["T3"]["follows_fact"] = follows_fact(rows, items)
    for task, v in r_hook(rows, items, out, boot).items():
        per_task[task]["response_shift_R"] = v
    report = {"run": str(run), "config": json.loads((run / "run.json").read_text()) if (run / "run.json").is_file()
              else {}, "tasks": per_task, "judge_requests": judge_requests(rows, items, out)}
    (out / "score.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def print_report(rep: dict) -> None:
    for task, v in rep["tasks"].items():
        print(f"{task}: {v['replies']} replies ({v['empty']} empty), {v['items']} items x {v['reruns']} reruns; "
              f"floor actions {v['floor_actions']}")
        s = v.get("unsupported_specifics")
        if s:
            print(f"  unsupported specifics: {s['specifics']} specifics, share supported {s['share_supported']}, "
                  f"lines with one unsupported {s['share_lines_unsupported']}")
        if "right_responder" in v:
            print(f"  right responder: {json.dumps(v['right_responder'])}")
        if "follows_fact" in v:
            print(f"  follows edited fact: {json.dumps(v['follows_fact'])}")
        r = v.get("response_shift_R", {})
        print(f"  response shift R: {json.dumps(r)}")
    print(f"judge requests: {rep['judge_requests']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="run directory (outputs.jsonl, run.json)")
    ap.add_argument("--items", default="", help="items jsonl (default: the one named in run.json)")
    ap.add_argument("--pack", default="", help="fact pack json: count its facts as support for every reply")
    ap.add_argument("--out", type=Path, default=None, help="default RUN/score")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    print_report(score(args.run, args.items, args.pack, args.out, args.boot))


if __name__ == "__main__":
    main()
