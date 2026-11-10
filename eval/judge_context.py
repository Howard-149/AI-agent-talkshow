#!/usr/bin/env python3
"""Listening test, judge accuracy J: can an LLM judge tell which version of the human turn a reply answered?

For every panel line right after the edit (pos 1) from condition c (paraphrase, negate, number, entity, addressee) and
from orig, an open judge model sees the topic, the two candidate human turns (orig text and the condition-c text) and
the reply, and answers which turn the reply responds to. Both presentation orders are asked; the choice is read from
the A/B next-token logprobs (no sampling).

    accuracy(c) = mean over (lines of c and of orig) x both orders of [judge picks the true turn]

0.5 = coin flip: the reply carries no trace of the edit (paraphrase should sit near 0.5). Averaging both orders cancels a
constant A/B position bias. 95% CI: bootstrap over topics. Needs a GPU (deploy/slurm-eval-listen.sh, stage judge).

--validity asks a different question with no replies: should a thoughtful panelist reply differently to the two
versions? It checks that each edit is one that ought to matter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "eval" / "data" / "minimal_pairs" / "pairs.yaml"
JUDGED = ("paraphrase", "negate", "number", "entity", "addressee")

SYSTEM = (
    "You are a careful annotator of talk-show transcripts. You will see two slightly different "
    "versions of something a guest said, and one reply from a panelist. Decide which version the "
    "panelist was actually responding to. Look for details the reply picks up (numbers, names, stance, "
    "who was addressed). If you cannot tell, still pick the more likely one."
)


def prompt(topic: str, a: str, b: str, reply: str) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"Version A of the guest's line:\n\"{a}\"\n\n"
        f"Version B of the guest's line:\n\"{b}\"\n\n"
        f"Panelist's reply:\n\"{reply}\"\n\n"
        "Which version was the panelist responding to? Answer with a single letter: A or B."
    )


def p_answer_a(out) -> float:
    """P(A) / (P(A) + P(B)) from the first generated token's top logprobs (0.5 if neither shows up)."""
    lp = out.outputs[0].logprobs[0]
    sa = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().upper() == "A")
    sb = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().upper() == "B")
    return sa / (sa + sb) if sa + sb > 0 else 0.5


def p_answer_yes(out) -> float:
    """P(yes) / (P(yes) + P(no)) from the first generated token's top logprobs (0.5 if neither shows up)."""
    lp = out.outputs[0].logprobs[0]
    y = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == "yes")
    n = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == "no")
    return y / (y + n) if y + n > 0 else 0.5


VALIDITY_SYSTEM = (
    "You are an experienced talk-show producer. You judge whether a change to what a guest said "
    "should change how a thoughtful, attentive panelist replies."
)


def _validity_items(t: dict):
    """(cond, ref, kind, user prompt, orig_first) for one topic."""
    for c, ref in tuple((c, "orig") for c in JUDGED):
        for first_ref in (True, False):
            a, b = (t["turn2"][ref], t["turn2"][c]) if first_ref else (t["turn2"][c], t["turn2"][ref])
            user = (
                f"Topic: {t['topic']}\n\nVersion A of the guest's line:\n\"{a}\"\n\n"
                f"Version B of the guest's line:\n\"{b}\"\n\n"
                "Should a thoughtful, attentive panelist's reply be different in substance for A than for B "
                "(e.g. engage a different number, name, stance, example, reason, question, or person)? Answer Yes or No."
            )
            yield c, ref, "differ", user, first_ref


def run_validity(args, pairs: dict) -> None:
    """Is each edit one that SHOULD change a good reply? P(yes), both orders, per condition."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, meta = [], []
    for slug, t in pairs.items():
        for c, ref, kind, user, first_ref in _validity_items(t):
            msgs = [{"role": "system", "content": VALIDITY_SYSTEM}, {"role": "user", "content": user}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            meta.append((slug, c, ref, kind, first_ref))
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.9,
              disable_custom_all_reduce=args.tp > 1)
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    per: dict = defaultdict(list)
    rows = []
    for (slug, c, ref, kind, first_ref), o in zip(meta, outs):
        p_yes = p_answer_yes(o)
        key = f"{c} vs {ref}"
        per[key].append(p_yes)
        rows.append({"slug": slug, "cond": c, "ref": ref, "kind": kind, "orig_first": first_ref, "p_yes": round(p_yes, 4)})
    out = args.resp / f"validity{args.tag}.json"
    summary = {c: {"p_yes": round(sum(v) / len(v), 3), "share_yes": round(sum(x > 0.5 for x in v) / len(v), 3),
                   "n": len(v)} for c, v in per.items()}
    # Position bias: how often the two orders of the same pair disagree on yes/no.
    by_pair: dict = defaultdict(dict)
    for r in rows:
        if r["kind"] == "differ":
            by_pair[(r["slug"], r["cond"], r["ref"])][r["orig_first"]] = r["p_yes"] > 0.5
    flips = [v[True] != v[False] for v in by_pair.values() if len(v) == 2]
    out.write_text(json.dumps({"model": args.model, "summary": summary,
                               "order_disagreement": round(sum(flips) / max(len(flips), 1), 3), "rows": rows}, indent=1))
    print(f"validity judge={args.model}  order disagreement {sum(flips)}/{len(flips)}")
    for c, v in summary.items():
        print(f"  {c:32s} P(yes)={v['p_yes']:.3f} share_yes={v['share_yes']:.3f} n={v['n']}")


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resp", type=Path, required=True, help="responsiveness results dir with lines.jsonl (output too)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel size (GPUs)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", default="", help="output suffix: judge<tag>.jsonl / judge<tag>.json")
    ap.add_argument("--conds", default="", help="comma list of conditions to judge vs orig (default: paraphrase,negate,number,entity,addressee)")
    ap.add_argument("--pairs", type=Path, default=PAIRS, help="pairs file (default: the 30 hand-written topics)")
    ap.add_argument("--validity", action="store_true",
                    help="no replies: rate whether each edit SHOULD change a thoughtful reply")
    args = ap.parse_args()

    pairs = {t["slug"]: t for t in yaml.safe_load(args.pairs.read_text())["topics"]}
    global JUDGED
    if args.conds:
        JUDGED = tuple(args.conds.split(","))
    if args.validity:
        run_validity(args, pairs)
        return
    rows = [json.loads(l) for l in (args.resp / "lines.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["pos"] == 1]

    # (row, cond-pair, true text, other text) — orig lines are judged against every edit.
    items = []
    for r in rows:
        t2 = pairs[r["slug"]]["turn2"]
        if r["cond"] in JUDGED:
            items.append((r, r["cond"], t2[r["cond"]], t2["orig"]))
        elif r["cond"] == "orig":
            for c in JUDGED:
                items.append((r, c, t2["orig"], t2[c]))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, meta = [], []
    for r, c, true, other in items:
        for true_is_a in (True, False):
            a, b = (true, other) if true_is_a else (other, true)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(r["topic"], a, b, r["text"])}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            meta.append((r, c, true_is_a))
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.9)
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))

    per = []
    for (r, c, true_is_a), o in zip(meta, outs):
        pa = p_answer_a(o)
        correct = pa if true_is_a else 1.0 - pa  # probability mass on the true version
        per.append({"slug": r["slug"], "pair": c, "line_cond": r["cond"], "role": r["role"],
                    "true_is_a": true_is_a, "p_true": round(correct, 4), "hit": int(correct > 0.5), "log": r["log"]})
    (args.resp / f"judge{args.tag}.jsonl").write_text("\n".join(json.dumps(x) for x in per) + "\n")

    rng = random.Random(0)
    report = {"model": args.model, "pairs": {}}
    by = defaultdict(lambda: defaultdict(list))
    for x in per:
        by[x["pair"]][x["slug"]].append(x["hit"])
    for c, topics in by.items():
        slugs = sorted(topics)
        acc = lambda ss: sum(sum(topics[s]) for s in ss) / max(1, sum(len(topics[s]) for s in ss))  # noqa: E731
        boots = sorted(acc([rng.choice(slugs) for _ in slugs]) for _ in range(args.boot))
        bias_a = sum(1 for x in per if x["pair"] == c and ((x["hit"] == 1) == x["true_is_a"])) / max(
            1, sum(1 for x in per if x["pair"] == c))
        report["pairs"][c] = {
            "accuracy": round(acc(slugs), 3),
            "ci95": [round(boots[int(0.025 * (len(boots) - 1))], 3), round(boots[int(0.975 * (len(boots) - 1))], 3)],
            "n": sum(len(v) for v in topics.values()),
            "topics": len(slugs),
            "share_answer_A": round(bias_a, 3),
        }
    (args.resp / f"judge{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"judge={args.model}")
    for c, v in report["pairs"].items():
        print(f"  orig vs {c:11s} acc={v['accuracy']:.3f} 95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] "
              f"n={v['n']} topics={v['topics']} A-share={v['share_answer_A']}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
