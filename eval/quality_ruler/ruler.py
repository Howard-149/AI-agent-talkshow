#!/usr/bin/env python3
"""Quality ruler: score panel episodes on validated measures, and validate the ruler itself.

Score a version of the show (R = --root, default eval/results/ruler):
  generate  (GPU)  model-only 2-beat episodes with the production panel prompts (+ optional knobs) -> episodes jsonl
  live      (CPU)  episodes from live ghost replay session logs -> episodes jsonl
  embed     (GPU)  Qwen3-Embedding for every line of the given episode files -> emb.npy (for the counted metrics)
  jury      (GPU)  one judge model per run: yes/no items, speaker attribution -> jury_<tag>.jsonl
  score     (CPU)  kept measures + counted metrics per episode file -> scores_<name>.json + a printed table

Validate the ruler (final measure set: --v2):
  build2    (CPU)  units = system beats + human anchors + length-matched anchors + live episodes; deterministic
                   perturbations -> R/episodes.jsonl
  rewrite   (GPU)  LLM-rewrite perturbations (strip reasons, verbose padding, persona inversion) -> R/rewrites.jsonl
                   (re-run build2 after)
  embed, jury      as above, on R/episodes.jsonl
  validate  (CPU)  checks a-g -> R/VALIDATION.md, R/scores.json
The first measure set (no --v2) uses build / invert instead of build2 / rewrite.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.quality_ruler import auto, episodes as E, jury, stats  # noqa: E402
from eval.quality_ruler.items import LINE_ITEMS  # noqa: E402
from eval.quality_ruler import perturb as P  # noqa: E402
from eval.quality_ruler.perturb import EXPECT, INVERT_PROMPT, perturb  # noqa: E402

R = ROOT / "eval" / "results" / "ruler"
SEED = 20261006
DIMS = ["specific", "novelty", "interaction", "argument", "distinct", "fidelity", "repetition", "flow"]
PERSONA_DIMS = {"distinct", "fidelity"}
CONTENT_DIMS = ["specific", "novelty", "interaction", "argument", "repetition", "flow"]


def use_v2() -> None:
    """Final measure set: verified interaction + numbered repetition replace the failed checklists; host craft on real
    host lines; the matching perturbations."""
    global DIMS, CONTENT_DIMS, EXPECT
    DIMS = ["specific", "interaction2", "argument", "distinct", "fidelity", "repetition2", "flow", "host"]
    CONTENT_DIMS = ["specific", "interaction2", "argument", "repetition2", "flow"]
    EXPECT = P.EXPECT_V2


# --------------------------------------------------------------------------------------------- build

def cmd_build(args) -> None:
    system = E.load(R / "system.jsonl") if (R / "system.jsonl").is_file() else []
    anchors = E.anchors()
    units = [E.beat_episode(e, 0, "") for e in system if e.get("k", 0) == 0] + anchors + [shorten(a) for a in anchors]
    for u in units:
        u.setdefault("base_id", u["id"])
        u["unit"] = True
    rng = random.Random(SEED)
    pert = []
    for u in units:
        if u["source"] == "anchor_short":
            continue
        for kind in EXPECT:
            if kind in ("label_swap", "persona_inversion") and u["source"] == "anchor":
                continue  # anchors are not our personas
            p = perturb(u, kind, rng)
            if p is not None:
                pert.append(p)
    inv = E.load(R / "inverted.jsonl") if (R / "inverted.jsonl").is_file() else []
    baseline = [{**e, "baseline": True} for e in system]  # full 2-beat episodes, all k
    eps = units + pert + inv + baseline
    pairs = [(p["base_id"], p["id"]) for p in pert + inv]
    sys_unit = {u["topic_id"]: u["id"] for u in units if u["source"] == "system"}
    pairs += [(u["id"], sys_unit[u["topic_id"]]) for u in units
              if u["source"] in ("anchor", "anchor_short") and u["topic_id"] in sys_unit]
    E.save(R / "episodes.jsonl", eps)
    (R / "pairs.json").write_text(json.dumps(pairs))
    n_lines = sum(len(e["scored"]) for e in eps)
    print(f"units {len(units)} (system {sum(u['source'] == 'system' for u in units)}, anchors "
          f"{sum(u['source'] == 'anchor' for u in units)} + length-matched {sum(u['source'] == 'anchor_short' for u in units)}), perturbed {len(pert)} + inverted {len(inv)}, baseline "
          f"{len(baseline)} → {len(eps)} episodes, {n_lines} scored lines, {len(pairs)} pairs → {R}")


SHORT_W = 45  # length-matched anchors: whole sentences up to ~ the panel's line length


def shorten(a: dict) -> dict:
    """Length-matched human anchor: each anchor panel line cut to whole sentences totalling ≤ SHORT_W words
    (at least the first sentence). Same real text, panel-like length — the direct length control for gate (b)."""
    import re as _re

    e = json.loads(json.dumps(a))
    e.update({"id": a["id"].replace("anchor:", "anchor_short:"), "source": "anchor_short", "base_id": None})
    for i in e["scored"]:
        sents = _re.split(r"(?<=[.?!])\s+", e["lines"][i]["text"].strip())
        keep, w = [], 0
        for snt in sents:
            n = len(snt.split())
            if keep and w + n > SHORT_W:
                break
            keep.append(snt)
            w += n
        e["lines"][i]["text"] = " ".join(keep)
    return e


def cmd_build2(args) -> None:
    """Final episode set: system beats (k=0, beat 1) + the most interactive human anchors + their length-matched copies +
    live production episodes (real host replies); deterministic perturbations; rewrites merged if present."""
    system = E.load(Path(args.system) if args.system else R / "system.jsonl")
    sysu = [E.beat_episode(e, 0, "") for e in system if e.get("k", 0) == 0]
    anc = E.anchors("_v2")
    live = (E.live_episodes(Path(args.live_logs), args.live_arm, limit=args.live, seed=SEED) if args.live_logs
            else [])
    units = sysu + anc + [shorten(a) for a in anc] + live
    for u in units:
        u.setdefault("base_id", u["id"])
        u["unit"] = True
    # off-thread pool: same-role panel lines from OTHER motions
    pool: dict = defaultdict(lambda: defaultdict(list))
    for u in sysu + live:
        for i in u["scored"]:
            pool[u["topic_id"]][u["lines"][i]["role"]].append(u["lines"][i]["text"])
    topics = sorted(pool)
    P.OTHER_POOL.clear()
    for t in topics:
        P.OTHER_POOL[t] = {r: [x for t2 in topics if t2 != t for x in pool[t2][r]] for r in ("commentator", "guest")}
    rng = random.Random(SEED)
    pert = []
    for u in units:
        if u["source"] == "anchor_short":
            continue
        for kind in P.EXPECT_V2:
            if kind in ("persona_inversion", "strip_reasons", "verbose_pad"):
                continue  # LLM rewrites (`rewrite` stage)
            if kind in ("label_swap",) and u["source"] == "anchor":
                continue
            if kind == "host_template" and u["source"] != "live":
                continue
            p = perturb(u, kind, rng)
            if p is not None:
                pert.append(p)
    rew = E.load(R / "rewrites.jsonl") if (R / "rewrites.jsonl").is_file() else []
    eps = units + pert + rew
    E.save(R / "episodes.jsonl", eps)
    (R / "pairs.json").write_text(json.dumps([]))  # no pairwise: it failed validation in the first measure set
    print(f"v2 units {len(units)} (system {len(sysu)}, anchors {len(anc)} + short, live {len(live)}), perturbed "
          f"{len(pert)} + rewrites {len(rew)} → {len(eps)} episodes, {sum(len(e['scored']) for e in eps)} scored lines")


def cmd_rewrite(args) -> None:
    """LLM rewrites for v2 perturbations: strip_reasons + verbose_pad on every unit (not short anchors), persona_inversion
    on system + live units (our personas). One model, temperature 0."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    eps = E.load(R / "episodes.jsonl")
    units = [e for e in eps if e.get("unit") and e["source"] in ("system", "anchor", "live")]
    jobs = []
    for u in units:
        for kind in ("strip_reasons", "verbose_pad", "persona_inversion"):
            if kind == "persona_inversion" and u["source"] == "anchor":
                continue
            for i in u["scored"]:
                prompt = (INVERT_PROMPT[u["lines"][i]["role"]] if kind == "persona_inversion"
                          else P.REWRITE_PROMPT[kind]).format(text=u["lines"][i]["text"])
                jobs.append((u, kind, i, prompt))
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p_}], tokenize=False, add_generation_prompt=True)
               for *_, p_ in jobs]
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.9,
              disable_custom_all_reduce=args.tp > 1)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=240))
    new: dict = {}
    for (u, kind, i, _), o in zip(jobs, outs):
        key = f"{u['id']}~{kind}"
        n = new.setdefault(key, json.loads(json.dumps(u)) | {"id": key, "base_id": u["id"], "source": "perturbed",
                                                             "perturbation": kind, "base_source": u["source"],
                                                             "unit": False})
        n["lines"][i]["text"] = o.outputs[0].text.strip().strip('"')
    E.save(R / "rewrites.jsonl", list(new.values()))
    print(f"rewrote {len(new)} episodes ({len(jobs)} lines) → {R / 'rewrites.jsonl'}")


def cmd_invert(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    eps = E.load(R / "episodes.jsonl")
    units = [e for e in eps if e.get("unit") and e["source"] == "system"]
    tok = AutoTokenizer.from_pretrained(args.model)
    jobs = [(u, i) for u in units for i in u["scored"]]
    prompts = [tok.apply_chat_template([{"role": "user", "content": INVERT_PROMPT[u["lines"][i]["role"]].format(
        text=u["lines"][i]["text"])}], tokenize=False, add_generation_prompt=True) for u, i in jobs]
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.9,
              disable_custom_all_reduce=args.tp > 1)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=160))
    new = {}
    for (u, i), o in zip(jobs, outs):
        n = new.setdefault(u["id"], json.loads(json.dumps(u)) | {"id": f"{u['id']}~persona_inversion",
                                                                   "base_id": u["id"], "source": "perturbed",
                                                                   "perturbation": "persona_inversion",
                                                                   "base_source": "system", "unit": False})
        n["lines"][i]["text"] = o.outputs[0].text.strip().strip('"')
    E.save(R / "inverted.jsonl", list(new.values()))
    print(f"inverted {len(new)} units → {R / 'inverted.jsonl'}")


# --------------------------------------------------------------------------------------------- scoring

def load_jury() -> dict[str, list[dict]]:
    return {p.stem[len("jury_"):]: E.load(p) for p in sorted(R.glob("jury_*.jsonl")) if not p.stem.endswith(".partial")}


def item_pass(r: dict) -> float:
    return float((r["p_yes"] > 0.5) != r["reverse"])


def judge_scores(rows: list[dict]) -> dict[str, dict[str, float]]:
    """episode → dim → score in [0, 1] (checklist share passed; distinct = attribution accuracy); + persuasion (1–5)."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    v2_inter: dict = {}
    for r in sorted(rows, key=lambda x: x["kind"] == "ic"):  # iq rows before their ic follow-ups
        if r.get("retest"):
            continue
        if r["kind"] == "item":
            acc[r["ep"]][r["dim"]].append(item_pass(r))
            acc[r["ep"]][r["dim"] + "~soft"].append(1 - r["p_yes"] if r["reverse"] else r["p_yes"])
            if r["dim"] == "host":  # the kept host measure is H1 (follows up) alone; H2 failed validation
                acc[r["ep"]]["host." + r["item"]].append(item_pass(r))
        elif r["kind"] == "attr":
            a_is = ("commentator", "for") if r["a_first"] else ("guest", "against")
            p_true = r["p_a"] if r["true"] in a_is else 1 - r["p_a"]
            acc[r["ep"]]["distinct"].append(p_true)
        elif r["kind"] == "persuasion":
            acc[r["ep"]]["persuasion"].append(r["score"])
        elif r["kind"] == "rq":  # v2 repetition: passes when the judge names no repeated earlier line
            acc[r["ep"]]["repetition2"].append(float(r.get("num") is None))
        elif r["kind"] == "iq":  # v2 interaction: provisional fail; a verified content follow-up overrides it
            v2_inter[(r["ep"], r["line"])] = 0.0
        elif r["kind"] == "ic":
            v2_inter[(r["ep"], r["line"])] = float(r["p_yes"] > 0.5)
    for (ep, _), v in v2_inter.items():
        acc[ep]["interaction2"].append(v)
    out = {}
    for ep, dims in acc.items():
        out[ep] = {d: float(np.mean(v)) for d, v in dims.items()}
        if "distinct" in out[ep]:  # order-averaged p_true per line → accuracy = share of lines identified
            out[ep]["distinct"] = float(np.mean(dims["distinct"]))
    return out


def combine(js: dict[str, dict], am: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Jury mean across judges per (episode, dim), plus automatic metrics merged into 'flow' etc. (kept separate)."""
    eps = set(am) | {e for s in js.values() for e in s}
    out = {}
    for e in eps:
        d = {}
        for dim in ([x for x in DIMS if x != "flow"] + ["persuasion", "host"]
                    + [x + "~soft" for x in DIMS if x not in ("flow", "distinct")] + ["host~soft"]):
            vals = [s[e][dim] for s in js.values() if e in s and dim in s[e]]
            if vals:
                d[dim] = float(np.mean(vals))
        for k, v in am.get(e, {}).items():
            d[f"auto:{k}"] = v
        # flow = the automatic flow metrics for every source (model-only system episodes have template host lines, so
        # the judged host items are reported separately as 'host' and only where host lines are real)
        if e in am:
            fl = [am[e][k] for k in ("flow.alternation", "flow.human_answered") if not np.isnan(am[e].get(k, np.nan))]
            if fl:
                d["flow"] = float(np.mean(fl))
        out[e] = d
    return out


# --------------------------------------------------------------------------------------------- score

# The measures kept after validation (judged, mean over the jury) and the counted metrics (no judge).
KEPT = ["specific", "distinct", "fidelity", "host.H1"]
GUARDRAIL = ["argument"]
COUNTED = ["auto:specific.figs_names_per100w", "auto:specific.1-generic_sim", "auto:novelty.share_novel",
           "auto:repetition.distinct2", "auto:repetition.1-repeated_opener", "auto:length.words_per_line"]


def score_sets(sets: dict[str, list[dict]], jrows: dict[str, list[dict]], am: dict[str, dict]) -> dict[str, dict]:
    """Per episode set: mean over episodes of each kept measure (jury mean; specific also per judge), the argument
    guardrail and the counted metrics, plus the set-level reuse metrics. NaN where a measure has no data."""
    js = {t: judge_scores(rows) for t, rows in jrows.items()}

    def ep_score(eid: str, k: str) -> float:
        if k.startswith("auto:"):
            return am.get(eid, {}).get(k[5:], np.nan)
        if "@" in k:
            dim, t = k.split("@")
            return js.get(t, {}).get(eid, {}).get(dim, np.nan)
        v = [s[eid][k] for s in js.values() if eid in s and k in s[eid]]
        return float(np.mean(v)) if v else np.nan

    keys = KEPT + [f"specific@{t}" for t in js] + GUARDRAIL + COUNTED
    out = {}
    for name, eps in sets.items():
        row = {"episodes": len(eps), "lines": sum(len(e["scored"]) for e in eps)}
        for k in keys:
            vals = [ep_score(e["id"], k) for e in eps]
            vals = [v for v in vals if not np.isnan(v)]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        # reuse is sampled from the line list: order it by topic (then file order) so the numbers do not depend on
        # how the file happens to be sorted
        by_topic: dict = defaultdict(list)
        for e in eps:
            by_topic[e["topic_id"]].append(e)
        row.update(auto.reuse_metrics([e for tid in sorted(by_topic) for e in by_topic[tid]]))
        out[name] = row
    return out


def cmd_score(args) -> None:
    files = [Path(f) for f in args.episodes]
    d = Path(args.dir) if args.dir else files[0].parent
    sets = {f.stem: E.load(f) for f in files}
    jrows = {p.stem[len("jury_"):]: E.load(p) for p in sorted(d.glob("jury_*.jsonl")) if "partial" not in p.name}
    am = auto.metrics([e for v in sets.values() for e in v], d)
    res = score_sets(sets, jrows, am)
    keys = [k for k in next(iter(res.values())) if k not in ("episodes", "lines")]
    L = ["| measure | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for k in ["episodes", "lines"] + keys:
        L.append(f"| {k.replace('auto:', '')} | " + " | ".join(
            f"{v[k]:.3f}" if isinstance(v[k], float) else str(v[k]) for v in res.values()) + " |")
    out = Path(args.out) if args.out else d / "scores.json"
    out.write_text(json.dumps({"judges": sorted(jrows), "sets": res}, indent=1, default=float))
    print(f"judges: {', '.join(sorted(jrows)) or 'none'}  (embeddings: {'yes' if (d / 'emb.npy').is_file() else 'no'})")
    print("\n".join(L))
    print(f"→ {out}")


def cmd_live(args) -> None:
    eps = E.live_episodes(Path(args.logs), args.arm, limit=args.limit, seed=args.seed)
    E.save(Path(args.out), eps)
    print(f"{len(eps)} live episodes (arm {args.arm or 'none'}) → {args.out}")


# --------------------------------------------------------------------------------------------- validate

def cmd_validate(args) -> None:
    eps = {e["id"]: e for e in E.load(R / "episodes.jsonl")}
    pairs = [tuple(p) for p in json.loads((R / "pairs.json").read_text())]
    jrows = load_jury()
    am = auto.metrics(list(eps.values()), R)
    js = {t: judge_scores(rows) for t, rows in jrows.items()}
    S = combine(js, am)
    units = [e for e in eps.values() if e.get("unit")]
    pert = [e for e in eps.values() if e.get("perturbation")]
    L = [f"# Quality ruler ({'final' if EXPECT is P.EXPECT_V2 else 'first'} measure set) — validation", ""]
    summary: list[str] = []
    detail: list[str] = []
    gates = {}

    # (a) perturbation × dimension matrix
    keys = DIMS + ["auto:specific.1-generic_sim", "auto:novelty.share_novel", "auto:interaction.lexical_link",
                   "auto:repetition.distinct2", "auto:repetition.1-repeated_opener", "auto:flow.alternation"]
    detail += ["## (a) Known-bad perturbations — share of cases where the perturbed beat scores LOWER than its original",
               "", "Own dimensions in **bold** (gate: ≥ 0.90). Off-dimension cells should stay low/near 0.5 (ties count as not lower).",
               "", "| perturbation | n | " + " | ".join(k.replace("auto:", "a:") for k in keys) + " |",
               "|---|---|" + "---|" * len(keys)]
    own_ok = []
    for kind, own in EXPECT.items():
        ps = [p for p in pert if p["perturbation"] == kind and p["base_id"] in S and p["id"] in S]
        if not ps:
            continue
        cells = []
        for k in keys:
            lo = [S[p["id"]][k] < S[p["base_id"]][k] for p in ps if k in S[p["id"]] and k in S[p["base_id"]]
                  and not (np.isnan(S[p["id"]][k]) or np.isnan(S[p["base_id"]][k]))]
            v = float(np.mean(lo)) if lo else float("nan")
            bold = k in own
            if bold:
                own_ok.append((kind, k, v))
            cells.append(f"**{v:.2f}**" if bold else f"{v:.2f}")
        detail.append(f"| {kind} | {len(ps)} | " + " | ".join(cells) + " |")
    # Diagnostics: soft scores (mean P(pass) instead of binarized pass/fail → no ties) and floor rate (share of originals
    # already at 0 on the dimension, where "lower" is impossible).
    soft = [d + "~soft" for d in DIMS if d not in ("flow", "distinct")]
    detail += ["", "Diagnostics — soft checklist scores (mean P(pass); own dimensions in bold) and floor rate of the originals:", "",
               "| perturbation | " + " | ".join(soft) + " | floor rate (own dims) |", "|---|" + "---|" * (len(soft) + 1)]
    for kind, own in EXPECT.items():
        ps = [p for p in pert if p["perturbation"] == kind and p["base_id"] in S and p["id"] in S]
        if not ps:
            continue
        cells = []
        for k in soft:
            lo = [S[p["id"]][k] < S[p["base_id"]][k] for p in ps if k in S[p["id"]] and k in S[p["base_id"]]]
            v = float(np.mean(lo)) if lo else float("nan")
            cells.append(f"**{v:.2f}**" if k[:-5] in own else f"{v:.2f}")
        floors = [f"{d}:{np.mean([S[p['base_id']].get(d, 1) == 0 for p in ps]):.2f}" for d in own if not d.startswith("auto:")]
        detail.append(f"| {kind} | " + " | ".join(cells) + f" | {', '.join(floors) or '–'} |")
    # Off-dimension drift as an effect size: mean (perturbed − original) in units of the originals' SD.
    detail += ["", "Drift — mean (perturbed − original) / SD(originals) per measure (own measures in bold; |d| < 0.3 is small):", "",
               "| perturbation | " + " | ".join(k.replace("auto:", "a:") for k in keys) + " |", "|---|" + "---|" * len(keys)]
    orig_ids = [u["id"] for u in units if u["source"] in ("system", "anchor", "live")]
    for kind, own in EXPECT.items():
        ps = [p for p in pert if p["perturbation"] == kind and p["base_id"] in S and p["id"] in S]
        if not ps:
            continue
        cells = []
        for k in keys:
            sd = np.nanstd([S[e].get(k, np.nan) for e in orig_ids if e in S])
            dl = [S[p["id"]][k] - S[p["base_id"]][k] for p in ps if k in S[p["id"]] and k in S[p["base_id"]]]
            v = float(np.nanmean(dl)) / sd if dl and sd > 0 else float("nan")
            cells.append(f"**{v:+.2f}**" if k in own else f"{v:+.2f}")
        detail.append(f"| {kind} | " + " | ".join(cells) + " |")
    gates["a"] = all(v >= 0.9 for _, _, v in own_ok) if own_ok else None
    fails = [f"{k}/{d} {v:.2f}" for k, d, v in own_ok if not v >= 0.9]
    summary.append(f"- (a) known-bad detection: {'PASS' if gates['a'] else 'FAIL'}"
                   + (f" — below 0.90: {', '.join(fails)}" if fails else ""))
    detail.append("")

    # (b) human anchor ranks top on content dims vs system and perturbed versions
    detail += ["## (b) Human anchors (IQ2) vs system vs perturbed — same motions", "",
               "| dimension | anchor > system (share of motions) | anchor mean | system mean | perturbed-system mean |",
               "|---|---|---|---|---|"]
    bwins = {}
    anc = {u["topic_id"]: u["id"] for u in units if u["source"] == "anchor"}
    sysu = {u["topic_id"]: u["id"] for u in units if u["source"] == "system"}
    common = sorted(set(anc) & set(sysu))
    for d in CONTENT_DIMS + ["persuasion"]:
        w = [S[anc[t]].get(d, np.nan) > S[sysu[t]].get(d, np.nan) for t in common if d in S[anc[t]] and d in S[sysu[t]]]
        am_ = np.nanmean([S[anc[t]].get(d, np.nan) for t in common]) if common else np.nan
        sm_ = np.nanmean([S[sysu[t]].get(d, np.nan) for t in common]) if common else np.nan
        pm_ = np.nanmean([S[p["id"]].get(d, np.nan) for p in pert if p.get("base_source") == "system" and p["id"] in S])
        bwins[d] = float(np.mean(w)) if w else float("nan")
        detail.append(f"| {d} | {bwins[d]:.2f} (n={len(w)}) | {am_:.2f} | {sm_:.2f} | {pm_:.2f} |")
    ash = {u["topic_id"]: u["id"] for u in units if u["source"] == "anchor_short"}
    common_s = sorted(set(ash) & set(sysu))
    detail += ["", "Length-matched human anchors (each anchor line cut to whole sentences ≤ 45 words) vs system:", "",
               "| dimension | short anchor > system (share) | short-anchor mean | system mean |", "|---|---|---|---|"]
    swins = {}
    for d in CONTENT_DIMS + ["persuasion"]:
        w = [S[ash[t]].get(d, np.nan) > S[sysu[t]].get(d, np.nan) for t in common_s if d in S[ash[t]] and d in S[sysu[t]]]
        swins[d] = float(np.mean(w)) if w else float("nan")
        detail.append(f"| {d} | {swins[d]:.2f} (n={len(w)}) | "
                      f"{np.nanmean([S[ash[t]].get(d, np.nan) for t in common_s]) if common_s else float('nan'):.2f} | "
                      f"{np.nanmean([S[sysu[t]].get(d, np.nan) for t in common_s]) if common_s else float('nan'):.2f} |")
    gates["b_short_anchor"] = all(v > 0.5 for v in swins.values() if not np.isnan(v)) if swins else None
    detail.append("")
    # Length control (mandatory: IQ2 anchor lines are longer than panel lines). Per dimension:
    #  (i) regression on the anchor+system units: score = a + b·log(words/line) + g·[anchor]; g = length-controlled
    #      anchor advantage, 95% topic-bootstrap CI;  (ii) length-matched subset: motions where the two units' words/line
    #      differ by < 30%, share of motions where the anchor scores higher.
    wpl = {e: am[e]["length.words_per_line"] for e in am}
    detail += ["### (b) with length control", "",
               f"Words per panel line: anchors median {np.median([wpl[anc[t]] for t in common]) if common else float('nan'):.0f}, "
               f"system median {np.median([wpl[sysu[t]] for t in common]) if common else float('nan'):.0f}.", "",
               "| dimension | raw: anchor − system (mean) | length-controlled anchor effect g [95% CI] | "
               "length-matched: anchor > system (n) |", "|---|---|---|---|"]
    rng_b = random.Random(SEED)
    bctrl = {}
    for d in CONTENT_DIMS + ["persuasion"]:
        rows_ = [(t, S[anc[t]].get(d, np.nan), S[sysu[t]].get(d, np.nan)) for t in common]
        rows_ = [r_ for r_ in rows_ if not (np.isnan(r_[1]) or np.isnan(r_[2]))]
        if len(rows_) < 5:
            continue

        def fit(ts):
            y, X = [], []
            for t in ts:
                for eid, is_a in ((anc[t], 1.0), (sysu[t], 0.0)):
                    v = S[eid].get(d, np.nan)
                    if not np.isnan(v):
                        y.append(v)
                        X.append([1.0, np.log(max(wpl[eid], 1.0)), is_a])
            X, y = np.array(X), np.array(y)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return float(beta[2])

        ts = [r_[0] for r_ in rows_]
        g = fit(ts)
        boots = sorted(fit([rng_b.choice(ts) for _ in ts]) for _ in range(200))
        lo, hi = boots[5], boots[194]
        matched = [r_ for r_ in rows_ if abs(wpl[anc[r_[0]]] - wpl[sysu[r_[0]]]) / max(wpl[sysu[r_[0]]], 1) < 0.30]
        mw = float(np.mean([r_[1] > r_[2] for r_ in matched])) if matched else float("nan")
        bctrl[d] = (g, lo, hi)
        detail.append(f"| {d} | {np.mean([r_[1] - r_[2] for r_ in rows_]):+.2f} | {g:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                      f"{mw:.2f} (n={len(matched)}) |")
    gates["b"] = all(v > 0.5 for v in bwins.values() if not np.isnan(v)) if bwins else None
    gates["b_length_controlled"] = all(lo > 0 for _, lo, _ in bctrl.values()) if bctrl else None
    summary.append(f"- (b) human anchors rank above the system — raw: {'PASS' if gates['b'] else 'FAIL'} "
                   f"(anchor beats system on " + ", ".join(f"{d} {v:.2f}" for d, v in bwins.items()) + "); "
                   f"length-controlled: {'PASS' if gates['b_length_controlled'] else 'FAIL'} "
                   f"(anchor effect g: " + ", ".join(f"{d} {g_:+.2f}" for d, (g_, _, _) in bctrl.items()) + "); "
                   f"length-matched anchors: {'PASS' if gates.get('b_short_anchor') else 'FAIL'} (short anchor beats system on "
                   + ", ".join(f"{d} {v:.2f}" for d, v in swins.items()) + ")")
    detail.append("")

    # (c) Krippendorff alpha per dimension across judges (binary item decisions; attribution correct/incorrect)
    detail += ["## (c) Jury agreement — Krippendorff's α per dimension (gate ≥ 0.4)", "", "| dimension | α | units |", "|---|---|---|"]
    alphas = {}
    if len(jrows) >= 2:
        for d in dict.fromkeys(DIMS + ["persuasion", "host"]):  # v2 DIMS already include host
            units_a: dict = defaultdict(lambda: [None] * len(jrows))
            for k, (t, rows) in enumerate(jrows.items()):
                for r in rows:
                    if r.get("retest"):
                        continue
                    if r["kind"] == "item" and r["dim"] == d:
                        units_a[(r["ep"], r["line"], r["item"])][k] = item_pass(r)
                    elif d == "distinct" and r["kind"] == "attr":
                        a_is = ("commentator", "for") if r["a_first"] else ("guest", "against")
                        pt = r["p_a"] if r["true"] in a_is else 1 - r["p_a"]
                        units_a[(r["ep"], r["line"], r["a_first"])][k] = float(pt > 0.5)
                    elif d == "persuasion" and r["kind"] == "persuasion":
                        units_a[r["ep"]][k] = r["score"]
                    elif d == "repetition2" and r["kind"] == "rq":
                        units_a[(r["ep"], r["line"])][k] = float(r.get("num") is None)
                    elif d == "interaction2" and r["kind"] in ("iq", "ic"):
                        cur = units_a[(r["ep"], r["line"])][k]
                        units_a[(r["ep"], r["line"])][k] = (float(r["p_yes"] > 0.5) if r["kind"] == "ic"
                                                            else (0.0 if cur is None else cur))
            if units_a:
                alphas[d] = stats.krippendorff_alpha(dict(units_a), "interval" if d == "persuasion" else "nominal")
                detail.append(f"| {d} | {alphas[d]:.2f} | {len(units_a)} |")
    low = [d for d, a in alphas.items() if not a >= 0.4]
    gates["c"] = (not low) if alphas else None
    summary.append(f"- (c) jury agreement α ≥ 0.4: {'PASS' if gates['c'] else 'FAIL' if alphas else 'n/a (< 2 judges)'}"
                   + (f" — below: {', '.join(f'{d} {alphas[d]:.2f}' for d in low)}" if low else ""))
    detail.append("")

    # (d) judge sanity: speaker identification on human (IQ2) turns (debate side), per judge
    detail += ["## (d) Judge sanity — identifying real speakers (IQ2 anchors: which side said the line), per judge", "",
               "| judge | accuracy on anchors | accuracy on system (persona) |", "|---|---|---|"]
    dacc = {}
    for t, s in js.items():
        a = [s[u["id"]]["distinct"] for u in units if u["source"] == "anchor" and u["id"] in s and "distinct" in s[u["id"]]]
        b = [s[u["id"]]["distinct"] for u in units if u["source"] == "system" and u["id"] in s and "distinct" in s[u["id"]]]
        dacc[t] = float(np.mean(a)) if a else float("nan")
        detail.append(f"| {t} | {dacc[t]:.2f} | {np.mean(b) if b else float('nan'):.2f} |")
    summary.append("- (d) judge speaker-ID on human turns: " + ", ".join(f"{t} {v:.2f}" for t, v in dacc.items())
                   + " (chance 0.50; persona scores trusted only for judges well above chance)")
    detail.append("")

    # (e) length: pairwise with and without length control; dimension detection vs length-padding
    detail += ["## (e) Length control — pairwise preference for the ORIGINAL over each perturbation", "",
               "β0 = length-controlled log-odds (logistic fit on Δ words); raw = mean share. Padding should NOT win.", "",
               "| judge | perturbation | raw share original wins | β0 (length-controlled) | β1 (per Δ word) |", "|---|---|---|---|---|"]
    for t, rows in jrows.items():
        by = defaultdict(list)
        for r in rows:
            if r["kind"] == "pair" and not r.get("retest"):
                by[(r["a"], r["b"])].append(r["p_a"] if r["a_first"] else 1 - r["p_a"])
        groups = defaultdict(list)
        for (a, b), v in by.items():
            kind = eps[b].get("perturbation") or "anchor_vs_system"
            dl = (sum(len(eps[a]["lines"][i]["text"].split()) for i in eps[a]["scored"]) -
                  sum(len(eps[b]["lines"][i]["text"].split()) for i in eps[b]["scored"]))
            groups[kind].append((float(np.mean(v)), dl))
        for kind, xs in sorted(groups.items()):
            b0, b1 = stats.logistic_length([x for x, _ in xs], [d / 100 for _, d in xs])
            detail.append(f"| {t} | {kind} | {np.mean([x for x, _ in xs]):.2f} (n={len(xs)}) | {b0:+.2f} | {b1:+.3f} |")
    summary.append("- (e) length: every pairwise result reported raw and length-controlled (table e)")
    detail.append("")

    # (f) non-redundancy: correlation of dimension scores across original episodes
    origs = [e["id"] for e in eps.values() if not e.get("perturbation")]
    detail += ["## (f) Non-redundancy — Pearson r between dimensions (original episodes; gate |r| < 0.7)", "",
               "| | " + " | ".join(DIMS) + " |", "|---|" + "---|" * len(DIMS)]
    high = []
    for a in DIMS:
        row = []
        for b in DIMS:
            r_ = stats.pearson([S[e].get(a, np.nan) for e in origs if e in S], [S[e].get(b, np.nan) for e in origs if e in S])
            row.append(f"{r_:.2f}")
            if a < b and abs(r_) >= 0.7:
                high.append(f"{a}–{b} {r_:.2f}")
        detail.append(f"| {a} | " + " | ".join(row) + " |")
    gates["f"] = not high
    summary.append(f"- (f) non-redundancy |r| < 0.7: {'PASS' if gates['f'] else 'FAIL — ' + ', '.join(high)}")
    detail.append("")

    # (g) test–retest: agreement of the re-scored subset with the main pass, per judge
    detail += ["## (g) Test–retest (temperature 0; seeded 10% subset re-scored in a separate pass)", "",
               "| judge | items: same decision | attribution/pairs: same decision | persuasion: mean |Δ| |", "|---|---|---|---|"]
    rt = {}
    RT_OUT = ("p_yes", "p_a", "score", "retest", "k", "answer", "num")  # judge outputs, not part of the match key
    for t, rows in jrows.items():
        main = {}
        for r in rows:
            if not r.get("retest"):
                main[json.dumps({k: v for k, v in r.items() if k not in RT_OUT}, sort_keys=True)] = r
        same_i, same_p, dps = [], [], []
        for r in rows:
            if not r.get("retest"):
                continue
            m = main.get(json.dumps({k: v for k, v in r.items() if k not in RT_OUT}, sort_keys=True))
            if m is None:
                continue
            if r["kind"] in ("item", "ic"):
                same_i.append((r["p_yes"] > 0.5) == (m["p_yes"] > 0.5))
            elif r["kind"] in ("iq", "rq"):  # v2 numbered-line answers: same number (or both NONE)
                same_i.append(r.get("num") == m.get("num"))
            elif r["kind"] in ("attr", "pair"):
                same_p.append((r["p_a"] > 0.5) == (m["p_a"] > 0.5))
            elif r["kind"] == "persuasion":
                dps.append(abs(r["score"] - m["score"]))
        rt[t] = float(np.mean(same_i)) if same_i else float("nan")
        detail.append(f"| {t} | {rt[t]:.3f} (n={len(same_i)}) | {np.mean(same_p) if same_p else float('nan'):.3f} | "
                      f"{np.mean(dps) if dps else float('nan'):.3f} |")
    summary.append("- (g) test–retest item agreement: " + ", ".join(f"{t} {v:.3f}" for t, v in rt.items()))
    detail.append("")

    # Overall: Bradley–Terry over sources (anchor, system, perturbations) from the pairwise jury
    bt_rows = []
    for t, rows in jrows.items():
        by = defaultdict(list)
        for r in rows:
            if r["kind"] == "pair" and not r.get("retest"):
                by[(r["a"], r["b"])].append(r["p_a"] if r["a_first"] else 1 - r["p_a"])
        for (a, b), v in by.items():
            la = eps[a].get("perturbation") or eps[a]["source"]
            lb = eps[b].get("perturbation") or eps[b]["source"]
            bt_rows.append({"a": la, "b": lb, "s": float(np.mean(v)), "topic": eps[a]["topic_id"]})
    if bt_rows:
        point, ci = stats.bt_bootstrap(bt_rows, "a", "b", "topic", boot=args.boot)
        detail += ["## Overall — Bradley–Terry log-strength by source (all judges; 95% topic-bootstrap CI)", "",
                   "| source | log-strength | 95% CI |", "|---|---|---|"]
        for k, v in sorted(point.items(), key=lambda x: -x[1]):
            detail.append(f"| {k} | {v:+.2f} | [{ci[k][0]:+.2f}, {ci[k][1]:+.2f}] |")
        detail.append("")

    # Baseline: production show (full 2-beat system episodes) vs human anchors — weakest dimensions
    base = [e["id"] for e in eps.values() if e.get("baseline")]
    anc_ids = [u["id"] for u in units if u["source"] == "anchor"]
    detail += ["## Baseline — full 2-beat system episodes vs human anchors", "",
               "| dimension | production | human anchors | gap (production − anchors) |", "|---|---|---|---|"]
    gaps = {}
    if not base:
        detail.append("| (none: this episode set has no full 2-beat production episodes — see the like-for-like table) | | | |")
    for d in (DIMS + ["persuasion"]) if base else []:
        pv = np.nanmean([S[e].get(d, np.nan) for e in base if e in S]) if base else np.nan
        av = np.nanmean([S[e].get(d, np.nan) for e in anc_ids if e in S]) if anc_ids else np.nan
        gaps[d] = pv - av
        detail.append(f"| {d} | {pv:.2f} | {av:.2f} | {gaps[d]:+.2f} |")
    # Like-for-like baseline: 4-line system beats vs human anchors vs length-matched anchors, on the measures that passed
    # validation, plus cross-episode opener reuse (share of lines starting with one of the source's 10 commonest openers).
    import collections as _c
    import re as _re

    def _opener(t: str) -> str:
        return " ".join(_re.findall(r"[A-Za-z']+", t.lower())[:3])

    grp = {g: [u for u in units if u["source"] == g] for g in ("system", "anchor", "anchor_short")}
    like = ["specific", "auto:specific.1-generic_sim", "auto:specific.figs_names_per100w", "argument",
            "auto:novelty.share_novel", "auto:repetition.distinct2", "auto:repetition.1-repeated_opener",
            "auto:length.words_per_line"]
    detail += ["", "Like-for-like (4-line beats, same motions) on the validated measures:", "",
               "| measure | system | human anchor | length-matched anchor |", "|---|---|---|---|"]
    for k in like:
        vals = [np.nanmean([S[u["id"]].get(k, np.nan) for u in grp[g] if u["id"] in S]) for g in grp]
        detail.append(f"| {k.replace('auto:', 'a:')} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    reuse = {}
    for g in ("system", "anchor"):
        ops = [_opener(u["lines"][i]["text"]) for u in grp[g] for i in u["scored"]]
        c = _c.Counter(ops)
        reuse[g] = sum(n for _, n in c.most_common(10)) / max(len(ops), 1)
        detail.append(f"| cross-episode opener reuse ({g}): top-10 openers cover | {reuse[g]:.2f} | "
                      f"{', '.join(repr(o) for o, _ in c.most_common(3))} | |")
    weakest = [d for d, g in sorted(((d, g) for d, g in gaps.items() if d not in PERSONA_DIMS and not np.isnan(g)),
                                    key=lambda x: x[1])[:3]]
    summary.append("- Baseline (only meaningful if the gates pass): weakest dimensions vs human anchors: "
                   + ", ".join(f"{d} ({gaps[d]:+.2f})" for d in weakest))

    notes = (R / "NOTES.md").read_text().strip() + "\n" if (R / "NOTES.md").is_file() else ""
    L += (["## Verdict (written after reading the numbers below)", "", notes] if notes else [])
    L += ["## Plain summary (automatic)", ""] + summary + ["", "Details below; method in eval/quality_ruler/README.md.", ""] + detail
    (R / "VALIDATION.md").write_text("\n".join(L) + "\n")
    (R / "scores.json").write_text(json.dumps({"gates": gates, "alphas": alphas, "gaps": gaps}, indent=1, default=str))
    print("\n".join(L[:2 + len(summary) + 2]))


def main() -> None:
    global R
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="", help="results dir R (default eval/results/ruler)")
    ap.add_argument("--v2", action="store_true", help="final measure set (validation stages: build2, rewrite, validate)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score", help="kept measures + counted metrics per episode file (CPU)")
    p.add_argument("episodes", nargs="+", help="episode jsonl files, one per version of the show")
    p.add_argument("--dir", default="", help="dir with jury_*.jsonl and emb.npy (default: the first file's dir)")
    p.add_argument("--out", default="", help="scores json (default <dir>/scores.json)")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("live", help="episodes from live ghost replay logs (CPU)")
    p.add_argument("logs", help="dir with session-*.jsonl")
    p.add_argument("--arm", default="", help="room suffix __v<arm> to keep (default: sessions without a suffix)")
    p.add_argument("--limit", type=int, default=0, help="sample this many sessions (0 = all)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_live)
    p = sub.add_parser("build2", help="validation episode set, final measures (CPU)")
    p.add_argument("--system", default="", help="model-only production episodes (default R/system.jsonl)")
    p.add_argument("--live-logs", default="", help="dir with live replay session logs (default: no live episodes)")
    p.add_argument("--live-arm", default="A", help="room suffix of the production arm in --live-logs")
    p.add_argument("--live", type=int, default=60, help="live episodes sampled")
    p.set_defaults(fn=cmd_build2)
    p = sub.add_parser("rewrite", help="LLM-rewrite perturbations for build2 (GPU)")
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.set_defaults(fn=cmd_rewrite)
    p = sub.add_parser("generate", help="model-only episodes (GPU)")
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("-k", type=int, default=2)
    p.add_argument("--lines-per-beat", type=int, default=4)
    p.add_argument("--topics", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(R / "system.jsonl"))
    p.add_argument("--arm", default="production", help="knob preset from eval/quality_ruler/arms.py ARMS")
    p.add_argument("--arm-spec", default="", help="JSON knobs added to the preset")
    p.add_argument("--pairs", default=str(E.PUBLIC / "build" / "pairs_public.yaml"))
    p.set_defaults(fn=lambda a: E.generate(a))
    sub.add_parser("build", help="validation episode set, first measures (CPU)").set_defaults(fn=cmd_build)
    p = sub.add_parser("invert", help="persona-inversion rewrites for build (GPU)")
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.set_defaults(fn=cmd_invert)
    p = sub.add_parser("embed", help="line embeddings for the counted metrics (GPU)")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--episodes", default="", help="comma list of episode files (default R/episodes.jsonl)")
    p.add_argument("--out-dir", default="")
    p.set_defaults(fn=_embed)
    p = sub.add_parser("jury", help="one judge model over an episode set (GPU)")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--repeat", type=float, default=0.10, help="test–retest share re-scored in a separate pass")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dims", default="", help="restrict to these dims (comma list); default all")
    p.add_argument("--episodes", default="", help="episodes jsonl (default R/episodes.jsonl); pairs only with the default")
    p.add_argument("--out-dir", default="", help="where jury_<tag>.jsonl goes (default R)")
    p.set_defaults(fn=_jury)
    p = sub.add_parser("validate", help="run the validation checks -> R/VALIDATION.md (CPU)")
    p.add_argument("--boot", type=int, default=1000)
    p.set_defaults(fn=cmd_validate)
    args = ap.parse_args()
    if args.root:
        R = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    if args.v2:
        use_v2()
    if args.cmd not in ("score", "live"):
        R.mkdir(parents=True, exist_ok=True)
    args.fn(args)


def _embed(a) -> None:
    if a.episodes:
        eps = [e for f in a.episodes.split(",") for e in E.load(Path(f))]
        out = Path(a.out_dir or Path(a.episodes.split(",")[0]).parent)
    else:
        eps, out = E.load(R / "episodes.jsonl"), R
    auto.embed_stage(a, eps, out)


def _jury(a) -> None:
    if a.episodes:  # an arbitrary episode file (no pairwise plan)
        eps = []
        for f in a.episodes.split(","):
            eps += E.load(Path(f))
        a.out_dir = a.out_dir or str(Path(a.episodes.split(",")[0]).parent)
        jury.run(a, eps, [])
        return
    a.out_dir = a.out_dir or str(R)
    jury.run(a, E.load(R / "episodes.jsonl"), [tuple(p) for p in json.loads((R / "pairs.json").read_text())])


if __name__ == "__main__":
    main()
