"""Score a run (CPU parts) and write the judge requests.

    python -m eval.bench.score RUN_DIR [--items ITEMS.jsonl] [--pack FACTS.json] [--out RUN_DIR/score]

Rules, no model:
  T2 right floor       systems that decide the floor (floor_action set): per panelist per rerun, speak when
                       addressed, yield when not; accuracy for each case and balanced (always-speak = 0.5)
  T2f floor            both panelists bid, the system's floor selection picks: the named panelist got the floor
                       (balanced over the two named versions), the control's distribution (floor_score)
  T3a/T3b follows     does the reply state the edited value ("new"), the original value ("old"), both, or neither
                       (fact_edit.follows); on fact_orig_<n> replies, how often the original values are stated
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
Judge answers (eval/bench/judge.py writes OUT/judge/answers_<tag>.jsonl) are merged on the next score run:
  identification accuracy J per task and pair type with a 95% CI from resampling topics, the paraphrase control
  separately, rubric pass rates per dimension; per judge and pooled. task_scores() turns them into 0-100 task scores.
Rubric requests cover rerun 0 only by default (--rubric-reruns, "all" for every rerun); identification uses every rerun.
Report: OUT/score.json, OUT/report.md (one page), and `--leaderboard RUN1/score RUN2/score ...` for a table across
systems.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.bench.items import GROUNDING, PANEL, Item, read_items  # noqa: E402

NO_REPLY_TASKS = ("T2f",)  # scored on who got the floor, not on a reply
OVERALL_TASKS = ("T1", "T2", "T3a", "T3b", "T4")  # T2f is reported separately for now

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

SPEAKS = ("speak", "interrupt")


def addressed_row(it: Item) -> bool:
    return it.role_under_test == it.expected.get("addressee")


def right_floor(rows: list[dict], items: dict[str, Item]) -> dict:
    """T2 floor decision, per panelist per rerun: right = speak when addressed, yield when not (yield, wait and
    backchannel all leave the floor). Accuracy is reported for the two cases separately and balanced (their mean), so
    a system that always speaks or always yields gets 0.5. Systems that make no floor decision: not applicable."""
    from eval.bench.systems import YIELDS

    hits: dict[str, list[int]] = {"addressed": [], "not_addressed": []}
    for r in rows:
        if r["task"] != "T2" or r.get("floor_action") is None:
            continue
        it = items[r["item_id"]]
        spoke = r["floor_action"] in SPEAKS
        if addressed_row(it):
            hits["addressed"].append(int(spoke))
        else:
            hits["not_addressed"].append(int(r["floor_action"] in YIELDS))
    if not hits["addressed"] and not hits["not_addressed"]:
        return {"n": 0, "applicable": False, "note": "the system makes no floor decision; only the reply is scored"}
    acc = {k: round(sum(v) / len(v), 3) if v else None for k, v in hits.items()}
    parts = [x for x in acc.values() if x is not None]
    return {"n": sum(len(v) for v in hits.values()), "applicable": True,
            "acc_addressed": acc["addressed"], "acc_not_addressed": acc["not_addressed"],
            "balanced": round(sum(parts) / len(parts), 3)}


FLOOR_OUTCOMES = ("commentator", "guest", "host", "none")


def floor_score(rows: list[dict], items: dict[str, Item]) -> dict:
    """T2f: who got the floor (selected_role) per version. Right = the named panelist got it; balanced accuracy is
    the mean over the two named versions (always picking one panelist gives 0.5). The control (nobody named) reports
    the distribution. floor_mode says how it was measured: "show" (the show's floor selection), "question" (a plain
    system's answer to the floor question) or "not_applicable" (no floor selection)."""
    rs = [r for r in rows if r["task"] == "T2f"]
    modes = Counter(r.get("floor_mode") or "unknown" for r in rs)
    rs = [r for r in rs if r.get("floor_mode") != "not_applicable"]
    if not rs:
        return {"n": 0, "applicable": False, "modes": dict(modes)}
    dist: dict[str, Counter] = defaultdict(Counter)
    hits: dict[str, list[int]] = defaultdict(list)
    for r in rs:
        it = items[r["item_id"]]
        sel = r.get("selected_role")
        sel = sel if sel in FLOOR_OUTCOMES else "none"
        dist[it.version][sel] += 1
        if it.expected.get("addressee"):
            hits[it.expected["addressee"]].append(int(sel == it.expected["addressee"]))
    acc = {a: round(sum(v) / len(v), 3) for a, v in sorted(hits.items())}
    res = {"n": len(rs), "applicable": True, "modes": dict(modes),
           "acc_by_addressee": acc, "balanced": round(sum(acc.values()) / len(acc), 3) if acc else None,
           "distribution": {v: {k: round(c[k] / sum(c.values()), 3) for k in FLOOR_OUTCOMES}
                            for v, c in sorted(dist.items())}}
    ctrl = [r for r in rs if r.get("controller_called") is not None]
    if ctrl:
        res["controller_called"] = round(sum(bool(r["controller_called"]) for r in ctrl) / len(ctrl), 3)
    return res


def follows_fact(rows: list[dict], items: dict[str, Item], task: str = "T3a") -> dict:
    from eval.bench.fact_edit import follows

    edited, orig = Counter(), Counter()
    for r in rows:
        it = items[r["item_id"]]
        if it.task != task or not r["text"]:
            continue
        if it.version.startswith("fact_orig"):
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
        if not r["text"] or r["task"] in NO_REPLY_TASKS:
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
        if r["text"] and r["task"] not in NO_REPLY_TASKS:
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

def judge_requests(rows: list[dict], items: dict[str, Item], out: Path, rubric_reruns: set[int] | None = None) -> dict:
    """Write the identification requests (every rerun) and the rubric requests (reruns in rubric_reruns; None = all)."""
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
        if not r["text"] or r["task"] in NO_REPLY_TASKS:
            continue
        it = items[r["item_id"]]
        if it.task == "T2" and not (addressed_row(it) and r.get("floor_action") in (None, *SPEAKS)):
            continue  # T2 replies are judged only when the panelist was addressed and took the floor
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
                elif it.task in GROUNDING:
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
        if rubric_reruns is not None and r["rerun"] not in rubric_reruns:
            continue
        ep = episode(it, r["text"], rid, reply_role=it.expected.get("persona_of"))
        for job in build_jobs([ep], [], 0.0, 0, dims=("specific", "fidelity")):
            rubric.append({"id": f"{rid}|{job['dim']}|{job['item']}", "task": it.task, "item_id": it.item_id,
                           "rerun": r["rerun"], "dim": job["dim"], "rubric_item": job["item"],
                           "reverse": job["reverse"], "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                                                                   {"role": "user", "content": job["prompt"]}]})
    for name, rows_ in (("identification", ident), ("rubric", rubric)):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows_))
    return {"identification": len(ident), "rubric": len(rubric), "dir": str(d),
            "rubric_reruns": "all" if rubric_reruns is None else sorted(rubric_reruns)}


# ------------------------------------------------------------------------------------------------ judge answers

def _boot_ci(by_topic: dict[str, list[float]], boot: int, seed: int = 0) -> tuple[float | None, list]:
    """Mean over all values and a 95% CI from resampling topics (all values of a topic stay together)."""
    topics = sorted(t for t, v in by_topic.items() if v)
    if not topics:
        return None, [None, None]

    def mean(ts):
        n = sum(len(by_topic[t]) for t in ts)
        return sum(sum(by_topic[t]) for t in ts) / n if n else float("nan")

    rng = random.Random(seed)
    boots = sorted(mean([rng.choice(topics) for _ in topics]) for _ in range(boot))
    lo, hi = boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))]
    return round(mean(topics), 3), [round(lo, 3), round(hi, 3)]


def load_answers(out: Path) -> dict[str, list[dict]]:
    """{judge tag: answer rows} from OUT/judge/answers_<tag>.jsonl (eval/bench/judge.py)."""
    res = {}
    for p in sorted((out / "judge").glob("answers_*.jsonl")):
        if p.name.endswith(".partial.jsonl"):
            continue
        res[p.stem[len("answers_"):]] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return res


def judge_scores(answers: dict[str, list[dict]], items: dict[str, Item], out: Path, boot: int = 1000) -> dict:
    """Per task: identification accuracy J (per pair type, the non-control edits pooled, the control), and rubric pass
    rates per dimension; per judge and over all judges pooled. Answers whose request no longer exists are ignored."""
    current = set()
    for name in ("identification", "rubric"):
        p = out / "judge" / f"{name}.jsonl"
        if p.is_file():
            current |= {json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip()}
    pooled = [dict(a, judge=tag) for tag, rows in answers.items() for a in rows if a["id"] in current]
    if not pooled:
        return {}
    res: dict[str, Any] = {"judges": sorted(answers), "answered": len(pooled), "tasks": {}}
    for tag in ["all", *sorted(answers)]:
        rows = [a for a in pooled if tag == "all" or a["judge"] == tag]
        for task in sorted({a["task"] for a in rows}):
            t = res["tasks"].setdefault(task, {}).setdefault(tag, {})
            ident = [a for a in rows if a["task"] == task and a["request"] == "identification"]
            if ident:
                def j_of(sel):
                    by: dict[str, list[float]] = defaultdict(list)
                    for a in sel:
                        by[items[a["item_id"]].topic_id].append(a["hit"])
                    j, ci = _boot_ci(by, boot)
                    return {"J": j, "ci95": ci, "n": len(sel)}
                t["J"] = j_of([a for a in ident if not a["is_control"]])
                if any(a["is_control"] for a in ident):
                    t["J_control"] = j_of([a for a in ident if a["is_control"]])
                t["J_by_pair"] = {p: j_of([a for a in ident if a["pair"] == p])
                                  for p in sorted({a["pair"] for a in ident})}
            rub = [a for a in rows if a["task"] == task and a["request"] == "rubric"]
            if rub:
                dims: dict[str, list[int]] = defaultdict(list)
                for a in rub:
                    dims[a["dim"]].append(int((a["p_yes"] > 0.5) != bool(a["reverse"])))
                t["rubric_pass"] = {d: round(sum(v) / len(v), 3) for d, v in sorted(dims.items())}
                t["rubric_n"] = len(rub)
    return res


def _clip(x: float) -> float:
    return round(max(0.0, min(100.0, x)), 1)


def task_scores(per_task: dict, judged: dict) -> dict:
    """0-100 per task (higher is better), each the mean of the components that exist for this run:
    T1  200*(J-0.5) on the non-control edits
    T2  200*(balanced right-floor accuracy - 0.5) (systems that decide the floor); 200*(J-0.5) on the addressee
        pair, judged on the replies given when addressed
    T3a, T3b (each)  200*(J-0.5) on the fact pair; 100*share of edited-fact replies stating the new value only;
        100*(1 - share of replies with an unsupported specific)
    T4  200*(J-0.5) on persona attribution; 100*persona item pass rate
    T2f 200*(balanced floor accuracy - 0.5): the named panelist got the floor (reported, not in the overall yet)
    J is pooled over all judges. Components without data are left out; the overall score is the mean over the tasks
    scored (T1, T2, T3a, T3b, T4 on a full run)."""
    jt = {k: v.get("all", {}) for k, v in judged.get("tasks", {}).items()}
    comps: dict[str, dict[str, float]] = defaultdict(dict)
    for task in per_task:
        j = jt.get(task, {}).get("J", {}).get("J")
        if j is not None:
            comps[task]["identification"] = _clip(200 * (j - 0.5))
    rf = per_task.get("T2", {}).get("right_floor", {})
    if rf.get("applicable"):
        comps["T2"]["right_floor"] = _clip(200 * (rf["balanced"] - 0.5))
    for task in GROUNDING:
        ff = per_task.get(task, {}).get("follows_fact", {}).get("edited", {})
        if ff.get("new") is not None:
            comps[task]["follows_fact"] = _clip(100 * ff["new"])
        us = per_task.get(task, {}).get("unsupported_specifics", {})
        if us.get("share_lines_unsupported") is not None:
            comps[task]["supported_specifics"] = _clip(100 * (1 - us["share_lines_unsupported"]))
    fid = jt.get("T4", {}).get("rubric_pass", {}).get("fidelity")
    if fid is not None:
        comps["T4"]["persona_items"] = _clip(100 * fid)
    fl = per_task.get("T2f", {}).get("floor", {})
    if fl.get("applicable") and fl.get("balanced") is not None:
        comps["T2f"]["right_floor"] = _clip(200 * (fl["balanced"] - 0.5))
    scores = {t: {"score": round(sum(c.values()) / len(c), 1), "components": c} for t, c in sorted(comps.items()) if c}
    in_overall = [v["score"] for t, v in scores.items() if t in OVERALL_TASKS]
    if in_overall:
        scores["overall"] = round(sum(in_overall) / len(in_overall), 1)
    return scores


def markdown_report(rep: dict) -> str:
    cfg = rep.get("config", {})
    sysd = cfg.get("system", {})
    name = system_name(rep)
    lines = [f"# {name}", "",
             f"Split {cfg.get('split', '?')}, {cfg.get('n_items', '?')} items x {cfg.get('reruns', '?')} reruns, "
             f"model `{cfg.get('model', '?')}`, system `{json.dumps(sysd)}`, commit {cfg.get('commit', '?')}.",
             f"Judges: {', '.join(rep.get('judged', {}).get('judges', [])) or 'none yet'}.", "",
             "| task | score (0-100) | components |", "|---|---|---|"]
    sc = rep.get("task_scores", {})
    for t, v in sc.items():
        if t != "overall":
            lines.append(f"| {t} | {v['score']} | " + ", ".join(f"{k} {x}" for k, x in v["components"].items()) + " |")
    lines.append(f"| overall | {sc.get('overall', '-')} | mean over {', '.join(OVERALL_TASKS)} |")
    jt = rep.get("judged", {}).get("tasks", {})
    for task, v in rep["tasks"].items():
        lines += ["", f"## {task}", "",
                  f"{v['replies']} replies ({v['empty']} empty), floor actions {v['floor_actions']}."]
        fl = v.get("floor")
        if fl:
            if fl.get("applicable"):
                lines.append(f"Floor ({', '.join(fl['modes'])}): balanced {fl['balanced']}, named panelist got the floor "
                             f"{fl['acc_by_addressee']}; distribution by version {fl['distribution']}"
                             + (f"; controller called on {fl['controller_called']} of items." if "controller_called" in fl
                                else "."))
            else:
                lines.append("Floor: not applicable (this system has no floor selection).")
        a = jt.get(task, {}).get("all", {})
        if a.get("J"):
            lines.append(f"Identification J {a['J']['J']} (95% CI {a['J']['ci95']}, n {a['J']['n']}); chance 0.5.")
            if a.get("J_control"):
                lines.append(f"Control (paraphrase) J {a['J_control']['J']} (95% CI {a['J_control']['ci95']}); "
                             "should stay near 0.5.")
            lines.append("By pair: " + "; ".join(f"{p} {x['J']} {x['ci95']}" for p, x in a["J_by_pair"].items()) + ".")
        if a.get("rubric_pass"):
            lines.append(f"Rubric pass rates: {a['rubric_pass']} (n {a['rubric_n']}).")
        rf = v.get("right_floor")
        if rf and rf.get("applicable"):
            lines.append(f"Right floor: balanced {rf['balanced']} (speaks when addressed {rf['acc_addressed']}, "
                         f"yields when not {rf['acc_not_addressed']}; n {rf['n']}).")
        elif rf:
            lines.append("Right floor: not applicable (no floor decision); only the reply is scored.")
        if v.get("follows_fact"):
            e = v["follows_fact"]["edited"]
            lines.append(f"Edited fact: new {e['new']}, old {e['old']}, both {e['both']}, neither {e['neither']} "
                         f"(n {e['n']}).")
        u = v.get("unsupported_specifics")
        if u:
            lines.append(f"Unsupported specifics: share supported {u['share_supported']}, lines with one unsupported "
                         f"{u['share_lines_unsupported']}.")
        rs = v.get("response_shift_R", {})
        if rs and "todo" not in rs:
            lines.append("Response shift R: " + "; ".join(f"{k} {x['R']} {x['ci95']}" for k, x in rs.items()) + ".")
    return "\n".join(lines) + "\n"


def system_name(rep: dict) -> str:
    cfg = rep.get("config", {})
    sysd = cfg.get("system", {})
    parts = str(cfg.get("model", "?")).rstrip("/").split("/")
    snap = [p for p in parts if p.startswith("models--")]  # a local snapshot path: models--org--name/snapshots/<hash>
    model = snap[-1].split("--")[-1] if snap else parts[-1]
    arm = f" ({sysd['arm']})" if sysd.get("arm") else ""
    return f"{sysd.get('system', '?')}{arm} · {model}"


def leaderboard(score_dirs: list[Path]) -> str:
    reps = [json.loads((d / "score.json").read_text()) for d in score_dirs]
    tasks = sorted({t for r in reps for t in r.get("task_scores", {}) if t != "overall"})
    reps.sort(key=lambda r: -(r.get("task_scores", {}).get("overall") or -1))
    lines = ["| system | split | overall | " + " | ".join(tasks) + " | judges |",
             "|---|---|---|" + "---|" * len(tasks) + "---|"]
    for r in reps:
        sc = r.get("task_scores", {})
        cells = [str(sc.get(t, {}).get("score", "-")) for t in tasks]
        lines.append(f"| {system_name(r)} | {r.get('config', {}).get('split', '?')} | {sc.get('overall', '-')} | "
                     + " | ".join(cells) + f" | {', '.join(r.get('judged', {}).get('judges', [])) or '-'} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ main

def score(run: Path, items_path: str = "", pack_path: str = "", out: Path | None = None, boot: int = 1000,
          rubric_reruns: set[int] | None = frozenset({0})) -> dict:
    rows = load_rows(run)
    n_na = sum(r.get("fact_instruction") == "not_applicable" for r in rows)
    rows = [r for r in rows if r.get("fact_instruction") != "not_applicable"]  # T3b the system could not be given
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
        per_task["T2"]["right_floor"] = right_floor(rows, items)
    if "T2f" in per_task:
        per_task["T2f"]["floor"] = floor_score(rows, items)
    for task in GROUNDING:
        if task in per_task:
            per_task[task]["follows_fact"] = follows_fact(rows, items, task)
            modes = Counter(r.get("fact_instruction") or "none" for r in rows if r["task"] == task)
            per_task[task]["fact_instruction"] = dict(modes)
    for task, v in r_hook(rows, items, out, boot).items():
        per_task[task]["response_shift_R"] = v
    report = {"run": str(run), "config": json.loads((run / "run.json").read_text()) if (run / "run.json").is_file()
              else {}, "tasks": per_task, "t3b_not_applicable_rows": n_na,
              "judge_requests": judge_requests(rows, items, out, None if rubric_reruns is None else set(rubric_reruns))}
    report["judged"] = judge_scores(load_answers(out), items, out, boot)
    report["task_scores"] = task_scores(per_task, report["judged"])
    (out / "score.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(markdown_report(report))
    return report


def print_report(rep: dict) -> None:
    for task, v in rep["tasks"].items():
        print(f"{task}: {v['replies']} replies ({v['empty']} empty), {v['items']} items x {v['reruns']} reruns; "
              f"floor actions {v['floor_actions']}")
        s = v.get("unsupported_specifics")
        if s:
            print(f"  unsupported specifics: {s['specifics']} specifics, share supported {s['share_supported']}, "
                  f"lines with one unsupported {s['share_lines_unsupported']}")
        if "right_floor" in v:
            print(f"  right floor: {json.dumps(v['right_floor'])}")
        if "follows_fact" in v:
            print(f"  follows edited fact: {json.dumps(v['follows_fact'])}")
        r = v.get("response_shift_R", {})
        print(f"  response shift R: {json.dumps(r)}")
    print(f"judge requests: {rep['judge_requests']}")
    if rep.get("task_scores"):
        print(f"task scores (0-100): {json.dumps(rep['task_scores'])}")


def main(argv: list[str] | None = None) -> None:
    from eval.bench.judge import parse_reruns

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, nargs="?", help="run directory (outputs.jsonl, run.json)")
    ap.add_argument("--items", default="", help="items jsonl (default: the one named in run.json)")
    ap.add_argument("--pack", default="", help="fact pack json: count its facts as support for every reply")
    ap.add_argument("--out", type=Path, default=None, help="default RUN/score (with --leaderboard: the markdown file)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--rubric-reruns", default="0", help='rubric requests for these reruns only (comma list or "all")')
    ap.add_argument("--leaderboard", nargs="+", type=Path, default=None,
                    help="score dirs (RUN/score) of scored runs: write one table across systems")
    args = ap.parse_args(argv)
    if args.leaderboard:
        table = leaderboard(args.leaderboard)
        if args.out:
            args.out.write_text(table)
        print(table)
        return
    if args.run is None:
        ap.error("give a run directory or --leaderboard")
    print_report(score(args.run, args.items, args.pack, args.out, args.boot, parse_reruns(args.rubric_reruns)))


if __name__ == "__main__":
    main()
