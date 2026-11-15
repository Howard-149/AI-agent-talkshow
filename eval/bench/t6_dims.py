"""T6 pairwise judging by dimension, length-neutral, with reference judges.

The single "which is the better talk show" question mixed six criteria, never asked about humour, and preferred longer
episodes. Here every pair (same topic and rerun, both orders) is judged on each dimension separately, the judges are
told that length is not a criterion, and a sample of pairs also goes to strong reference judges (an API model and a
second one run separately) to check how well the open judge agrees.

    python -m eval.bench.t6_dims build --runs DIR,DIR,... --out OUT [--ref-pairs 48]
        writes OUT/judge/pairwise.jsonl (open judge, one request per pair, order and dimension; answer with eval.bench.judge)
        and OUT/ref_requests.jsonl (all dimensions in one prompt, for reference judges)
    python -m eval.bench.t6_dims ref-openai --out OUT --model gpt-5.5     (needs OPENAI_API_KEY)
    python -m eval.bench.t6_dims score --out OUT
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from eval.bench.show import episode_text

SYSTEM = ("You are an expert producer of live talk shows, judging transcripts. Use only the transcripts you are shown.")
LENGTH_RULE = ("Length is not a criterion: do not prefer an episode or a line for being longer or shorter; judge what is "
               "said and how it would play live, not how much is said.")
DIMS = {
    "audience": "Which episode's panel better answers what the audience member actually asked or claimed?",
    "interplay": "In which episode do the speakers better react to each other (pick up, push back, build on), "
                 "rather than talk past each other?",
    "novelty": "In which episode do the speakers keep adding new points, rather than repeating ideas they already made?",
    "persona": "In which episode do the speakers sound more like distinct, consistent characters?",
    "entertainment": "Which episode is more entertaining to watch: wit, humour, surprise, memorable lines?",
    "naturalness": "Which episode sounds more like natural live conversation rather than written essays?",
    "overall": "Overall, which episode would make the better live talk show segment?",
}
HEADER = ("Both episodes are the same live talk show on the same topic, with the same audience member saying the same "
          "things at the same points.")


def load_runs(dirs: list[str]) -> dict[str, list[dict]]:
    runs = {}
    for d in dirs:
        lab = json.loads((Path(d) / "run.json").read_text())["label"]
        runs[lab] = [json.loads(l) for l in (Path(d) / "episodes.jsonl").read_text().splitlines() if l.strip()]
    return runs


def pairs(runs: dict[str, list[dict]]) -> list[tuple[str, str, dict, dict]]:
    labels = sorted(runs)
    by = {lab: {(e["topic_id"], e["k"]): e for e in runs[lab]} for lab in labels}
    out = []
    for x in range(len(labels)):
        for y in range(x + 1, len(labels)):
            sa, sb = labels[x], labels[y]
            for key in sorted(set(by[sa]) & set(by[sb])):
                out.append((sa, sb, by[sa][key], by[sb][key]))
    return out


def body(topic: str, a: dict, b: dict) -> str:
    return f"Topic: {topic}\n\nEpisode A:\n{episode_text(a)}\n\nEpisode B:\n{episode_text(b)}\n\n{HEADER} {LENGTH_RULE}"


def words(ep: dict) -> int:
    return sum(len(re.findall(r"[A-Za-z0-9$%']+", l["text"])) for l in ep["lines"]
               if l["role"] != "human" and not l.get("procedural"))


def cmd_build(args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / "judge").mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs.split(","))
    ps = pairs(runs)
    reqs, meta = [], []
    for sa, sb, ea, eb in ps:
        pid = f"{ea['topic_id']}|k{ea['k']}|{sa}|{sb}"
        meta.append({"pair": pid, "sys_a": sa, "sys_b": sb, "words_a": words(ea), "words_b": words(eb)})
        for order, (a, b) in (("ab", (ea, eb)), ("ba", (eb, ea))):
            for dim, q in DIMS.items():
                reqs.append({"id": f"{pid}|{order}|{dim}", "task": "T6", "pair": pid, "order": order, "dim": dim,
                             "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": body(ea["topic"], a, b) +
                                           f"\n\n{q} Answer with a single letter: A or B."}]})
    (out / "judge" / "pairwise.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (out / "pairs.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    rng = random.Random(args.seed)
    sample = rng.sample(ps, min(args.ref_pairs, len(ps)))
    refs = []
    qs = "\n".join(f'- "{d}": {q}' for d, q in DIMS.items())
    for sa, sb, ea, eb in sample:
        pid = f"{ea['topic_id']}|k{ea['k']}|{sa}|{sb}"
        for order, (a, b) in (("ab", (ea, eb)), ("ba", (eb, ea))):
            prompt = (body(ea["topic"], a, b) + "\n\nAnswer each question with \"A\" or \"B\" (pick one even if close):\n"
                      + qs + "\n\nReturn only a JSON object mapping each key to \"A\" or \"B\", plus \"why\": one "
                      "sentence on the overall choice.")
            refs.append({"id": f"{pid}|{order}", "pair": pid, "order": order, "system": SYSTEM, "prompt": prompt})
    (out / "ref_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in refs))
    print(f"{len(ps)} pairs, {len(reqs)} open-judge requests, {len(refs)} reference requests ({len(sample)} pairs)")


def parse_ref(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ans = {k: str(d.get(k, "")).strip().upper()[:1] for k in DIMS}
    return ans if all(v in ("A", "B") for v in ans.values()) else None


def cmd_ref_openai(args: argparse.Namespace) -> None:
    out = Path(args.out)
    key = os.environ["OPENAI_API_KEY"]
    path = out / f"ref_{args.tag}.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.is_file() else set()
    reqs = [json.loads(l) for l in (out / "ref_requests.jsonl").read_text().splitlines()]
    with path.open("a") as fh:
        for r in reqs:
            if r["id"] in done:
                continue
            payload = {"model": args.model, "messages": [{"role": "system", "content": r["system"]},
                                                         {"role": "user", "content": r["prompt"]}]}
            for attempt in range(4):
                try:
                    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                                 data=json.dumps(payload).encode(), method="POST",
                                                 headers={"Content-Type": "application/json",
                                                          "Authorization": f"Bearer {key}"})
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        text = json.loads(resp.read())["choices"][0]["message"]["content"]
                    break
                except Exception as e:  # noqa: BLE001
                    text = ""
                    print("retry", r["id"], type(e).__name__, file=sys.stderr)
                    time.sleep(5 * (attempt + 1))
            fh.write(json.dumps({"id": r["id"], "pair": r["pair"], "order": r["order"], "answer": parse_ref(text),
                                 "raw": text}) + "\n")
            fh.flush()
    print("done", path)


def _pref(p_ab: float, p_ba: float) -> float:
    """Preference for the pair's first system, averaged over both orders (p = P(first-shown episode))."""
    return (p_ab + (1.0 - p_ba)) / 2.0


def cmd_score(args: argparse.Namespace) -> None:
    out = Path(args.out)
    meta = {m["pair"]: m for m in map(json.loads, (out / "pairs.jsonl").read_text().splitlines())}
    open_p: dict[tuple, float] = {}
    for tagf in sorted((out / "judge").glob("answers_*.jsonl")):
        if tagf.name.endswith(".partial.jsonl"):
            continue
        tag = tagf.stem.replace("answers_", "")
        for r in map(json.loads, tagf.read_text().splitlines()):
            pair, order, dim = r["id"].rsplit("|", 2)
            open_p[(tag, pair, order, dim)] = r["p_a"]
    tags = sorted({k[0] for k in open_p})
    # per judge, per dimension: system win rates (order-averaged), order consistency, length correlation
    lines = ["# T6 by dimension (length-neutral prompts)", ""]
    prefs: dict[tuple, float] = {}
    for tag in tags:
        lines.append(f"## Open judge: {tag}")
        lines.append("")
        lines.append("| dimension | order consistency | " + " | ".join(sorted({m['sys_a'] for m in meta.values()} |
                                                                         {m['sys_b'] for m in meta.values()})) + " | longer wins |")
        systems = sorted({m['sys_a'] for m in meta.values()} | {m['sys_b'] for m in meta.values()})
        lines[-1] = "| dimension | order consistency | " + " | ".join(s.split("@")[0] for s in systems) + " | longer wins |"
        lines.append("|" + "---|" * (len(systems) + 3))
        for dim in DIMS:
            wins, cnt, cons, n, longer, nl = defaultdict(float), defaultdict(int), 0, 0, 0, 0
            for pid, m in meta.items():
                if (tag, pid, "ab", dim) not in open_p or (tag, pid, "ba", dim) not in open_p:
                    continue
                pab, pba = open_p[(tag, pid, "ab", dim)], open_p[(tag, pid, "ba", dim)]
                p = _pref(pab, pba)
                prefs[(tag, pid, dim)] = p
                n += 1
                cons += (pab > 0.5) != (pba > 0.5)
                wins[m["sys_a"]] += p
                wins[m["sys_b"]] += 1 - p
                cnt[m["sys_a"]] += 1
                cnt[m["sys_b"]] += 1
                if m["words_a"] != m["words_b"] and abs(p - 0.5) > 1e-9:
                    nl += 1
                    longer += (p > 0.5) == (m["words_a"] > m["words_b"])
            row = [dim, f"{cons / n:.2f}" if n else "-"] + [f"{wins[s] / cnt[s]:.2f}" if cnt[s] else "-" for s in systems]
            row.append(f"{longer / nl:.2f}" if nl else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # reference judges
    for rf in sorted(out.glob("ref_*.jsonl")):
        if rf.name == "ref_requests.jsonl":
            continue
        tag = rf.stem.replace("ref_", "")
        rows = [json.loads(l) for l in rf.read_text().splitlines() if l.strip()]
        by: dict[str, dict] = defaultdict(dict)
        for r in rows:
            if r.get("answer"):
                by[r["pair"]][r["order"]] = r["answer"]
        lines.append(f"## Reference judge: {tag} ({len(by)} pairs)")
        lines.append("")
        lines.append("| dimension | order consistency | " + " | ".join(f"agree with {t}" for t in tags) +
                     " | longer wins | first system wins |")
        lines.append("|" + "---|" * (len(tags) + 4))
        for dim in DIMS:
            cons = n = longer = nl = firstw = 0
            agree = defaultdict(lambda: [0, 0])
            for pid, o in by.items():
                if "ab" not in o or "ba" not in o:
                    continue
                n += 1
                ab_first = o["ab"][dim] == "A"
                ba_first = o["ba"][dim] == "B"
                consistent = ab_first == ba_first
                cons += consistent
                if not consistent:
                    continue
                firstw += ab_first
                m = meta[pid]
                if m["words_a"] != m["words_b"]:
                    nl += 1
                    longer += ab_first == (m["words_a"] > m["words_b"])
                for t in tags:
                    if (t, pid, dim) in prefs:
                        agree[t][0] += (prefs[(t, pid, dim)] > 0.5) == ab_first
                        agree[t][1] += 1
            row = [dim, f"{cons / n:.2f}" if n else "-"]
            row += [f"{agree[t][0] / agree[t][1]:.2f} (n {agree[t][1]})" if agree[t][1] else "-" for t in tags]
            row += [f"{longer / nl:.2f}" if nl else "-", f"{firstw}/{cons}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--runs", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--ref-pairs", type=int, default=48)
    b.add_argument("--seed", type=int, default=0)
    o = sub.add_parser("ref-openai")
    o.add_argument("--out", required=True)
    o.add_argument("--model", default="gpt-5.5")
    o.add_argument("--tag", default="gpt55")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"build": cmd_build, "ref-openai": cmd_ref_openai, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
