"""LLM jury (GPU): checklist items, speaker attribution, pairwise preference, 1–5 persuasiveness — one judge model per
run, all answers read from next-token logprobs (no sampling → test–retest = re-running gives the same scores up to
numerical noise; `--repeat` re-scores a seeded 10% subset to measure that noise).

Prompts put the transcript first and the question last, so vLLM prefix caching shares the context across the ~16 items
asked about the same line.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from eval.quality_ruler.items import (HOST_ITEMS, JUDGE_SYSTEM, LINE_ITEMS, PAIRWISE_QUESTION, PERSONA_CARDS,
                                      PERSONA_ITEMS, PERSUASION_ANCHORS)

CTX = 8  # previous lines shown as context


def _fmt(lines: list[dict], hide_panel: bool = False) -> str:
    out = []
    for l in lines:
        who = "Panelist" if (hide_panel and l["role"] in ("commentator", "guest")) else l["speaker"]
        out.append(f"{who}: {l['text']}")
    return "\n".join(out)


def line_prompt(ep: dict, i: int, question: str) -> str:
    L = ep["lines"]
    ctx = _fmt(L[max(0, i - CTX): i])
    return (f"Topic: {ep['topic']}\n\nTranscript so far:\n{ctx}\n\nLine to judge — {L[i]['speaker']}: {L[i]['text']}\n\n"
            f"Question: {question} Answer Yes or No.")


def attribution_prompt(ep: dict, i: int, a_first: bool) -> str:
    """Persona attribution (system/perturbed): which persona card fits the hidden speaker. On human anchors the cards
    are the debate sides instead (gate d: can the judge identify real speakers at all?)."""
    L = ep["lines"]
    if L[i].get("side"):
        a, b = ("for", "against") if a_first else ("against", "for")
        cards = (f"Speaker A argues {a.upper()} the motion \"{ep['topic']}\".\n"
                 f"Speaker B argues {b.upper()} the motion \"{ep['topic']}\".")
        return (f"Topic: {ep['topic']}\n\nTwo debaters, names hidden:\n{cards}\n\nTranscript so far (names hidden):\n"
                f"{_fmt(L[max(0, i - CTX): i], hide_panel=True)}\n\nNext line: {L[i]['text']}\n\n"
                "Which speaker said the next line? Answer with a single letter: A or B.")
    a, b = ("commentator", "guest") if a_first else ("guest", "commentator")
    body = {r: c.split(": ", 1)[1] for r, c in PERSONA_CARDS.items()}  # drop the card's own "Speaker X:" label
    cards = f"Speaker A: {body[a]}\nSpeaker B: {body[b]}"
    return (f"Topic: {ep['topic']}\n\nTwo panelists, names hidden:\n{cards}\n\nTranscript so far (panelist names hidden):\n"
            f"{_fmt(L[max(0, i - CTX): i], hide_panel=True)}\n\nNext line: {L[i]['text']}\n\n"
            "Which speaker said the next line? Answer with a single letter: A or B.")


def beat_text(ep: dict) -> str:
    L = ep["lines"]
    first = ep["scored"][0]
    start = max(0, first - 3)
    return _fmt(L[start: ep["scored"][-1] + 1])


def pairwise_prompt(topic: str, a: str, b: str) -> str:
    return f"Topic: {topic}\n\nExchange A:\n{a}\n\nExchange B:\n{b}\n\n{PAIRWISE_QUESTION}"


def persuasion_prompt(ep: dict) -> str:
    return (f"Topic: {ep['topic']}\n\nPanel exchange:\n{beat_text(ep)}\n\nRate how persuasive this exchange is as a debate, "
            f"on this scale:\n{PERSUASION_ANCHORS}\n\nAnswer with a single digit from 1 to 5.")


ALL_DIMS = ("specific", "novelty", "interaction", "argument", "repetition", "fidelity", "distinct", "host", "persuasion")


# ---- final measure set (ruler.py --v2): numbered-reference questions, verified in code

def numbered_ctx(ep: dict, i: int) -> tuple[str, dict[int, int]]:
    """Earlier lines (within CTX), non-procedural only, numbered [1..n]; returns text and number → line index."""
    L = ep["lines"]
    idx = [j for j in range(max(0, i - CTX), i) if not L[j].get("procedural")]
    num = {n + 1: j for n, j in enumerate(idx)}
    return "\n".join(f"[{n}] {L[j]['speaker']}: {L[j]['text']}" for n, j in num.items()), num


def interaction_q(ep: dict, i: int) -> str:
    ctx, _ = numbered_ctx(ep, i)
    L = ep["lines"]
    return (f"Topic: {ep['topic']}\n\nEarlier lines:\n{ctx}\n\nTarget line — {L[i]['speaker']}: {L[i]['text']}\n\n"
            "Which numbered earlier line does the target line respond to most directly? Answer with that line's number "
            "only, or NONE if it does not respond to any specific earlier line.")


def interaction_content(ep: dict, i: int, j: int) -> str:
    L = ep["lines"]
    return (f"Topic: {ep['topic']}\n\nEarlier line — {L[j]['speaker']}: {L[j]['text']}\n\n"
            f"Target line — {L[i]['speaker']}: {L[i]['text']}\n\n"
            "Does the target line engage the CONTENT of the earlier line — agreeing with, disputing, or extending its "
            "specific claim — rather than only naming the speaker or restating the general topic? Answer Yes or No.")


def repeat_q(ep: dict, i: int) -> str:
    ctx, _ = numbered_ctx(ep, i)
    L = ep["lines"]
    return (f"Topic: {ep['topic']}\n\nEarlier lines:\n{ctx}\n\nTarget line — {L[i]['speaker']}: {L[i]['text']}\n\n"
            "Does the target line repeat the main point of any numbered earlier line? Answer with that line's number "
            "only, or NONE.")


def parse_num(text: str, valid: dict[int, int]) -> int | None:
    import re as _re

    m = _re.search(r"\d+", text or "")
    if not m or "none" in (text or "").lower()[:6]:
        return None
    n = int(m.group(0))
    return n if n in valid else None


def build_jobs(eps: list[dict], pairs: list[tuple[str, str]], repeat: float, seed: int,
               dims: tuple[str, ...] = ALL_DIMS) -> list[dict]:
    """Every judge question as a job dict (kind, ids, prompt). pairs = (episode id A, episode id B) to compare.
    dims restricts the checklist dimensions / attribution / host / persuasion asked (e.g. only the kept measures)."""
    jobs = []
    byid = {e["id"]: e for e in eps}
    for e in eps:
        for i in e["scored"]:
            role = e["lines"][i]["role"]
            for dim, (_, items) in LINE_ITEMS.items():
                if dim not in dims:
                    continue
                for iid, q, rev in items:
                    jobs.append({"kind": "item", "ep": e["id"], "line": i, "dim": dim, "item": iid, "reverse": rev,
                                 "prompt": line_prompt(e, i, q)})
            if "interaction2" in dims:
                jobs.append({"kind": "iq", "ep": e["id"], "line": i, "dim": "interaction2", "prompt": interaction_q(e, i)})
            if "repetition2" in dims:
                jobs.append({"kind": "rq", "ep": e["id"], "line": i, "dim": "repetition2", "prompt": repeat_q(e, i)})
            for iid, q, rev in (PERSONA_ITEMS[role] if "fidelity" in dims else []):
                jobs.append({"kind": "item", "ep": e["id"], "line": i, "dim": "fidelity", "item": iid, "reverse": rev,
                             "prompt": line_prompt(e, i, q)})
            for a_first in ((True, False) if "distinct" in dims else ()):
                truth = e["lines"][i].get("side") or role  # anchors: the debate side; otherwise the persona
                jobs.append({"kind": "attr", "ep": e["id"], "line": i, "a_first": a_first, "true": truth,
                             "prompt": attribution_prompt(e, i, a_first)})
        # host lines: live episodes list their real host replies in host_scored; otherwise the non-procedural host
        # lines inside the scored beat (production tee-up templates are procedural → skipped)
        lo, hi = e["scored"][0], e["scored"][-1]
        host_idx = e.get("host_scored", [x for x in range(lo, hi + 1) if e["lines"][x]["role"] == "host"])
        for i in host_idx:
            l = e["lines"][i]
            if l["role"] == "host" and not l.get("procedural") and "host" in dims:
                for iid, q, rev in HOST_ITEMS:
                    jobs.append({"kind": "item", "ep": e["id"], "line": i, "dim": "host", "item": iid, "reverse": rev,
                                 "prompt": line_prompt(e, i, q)})
        if "persuasion" in dims:
            jobs.append({"kind": "persuasion", "ep": e["id"], "prompt": persuasion_prompt(e)})
    for a, b in pairs:
        ea, eb = byid[a], byid[b]
        for a_first in (True, False):
            x, y = (ea, eb) if a_first else (eb, ea)
            jobs.append({"kind": "pair", "a": a, "b": b, "a_first": a_first,
                         "prompt": pairwise_prompt(ea["topic"], beat_text(x), beat_text(y))})
    if repeat > 0:  # test–retest subset: the same prompts again, scored in a separate pass
        rng = random.Random(seed)
        jobs += [{**j, "retest": True} for j in rng.sample(jobs, int(repeat * len(jobs)))]
    return jobs


def _probs(o, tokens: list[str]) -> list[float]:
    lp = o.outputs[0].logprobs[0]
    mass = [sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == t.lower()) for t in tokens]
    s = sum(mass)
    return [m / s for m in mass] if s > 0 else [1 / len(tokens)] * len(tokens)


def run(args, eps: list[dict], pairs: list[tuple[str, str]]) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    dims = tuple(args.dims.split(",")) if getattr(args, "dims", "") else ALL_DIMS
    jobs = build_jobs(eps, pairs, args.repeat, args.seed, dims)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": j["prompt"]}],
                                       tokenize=False, add_generation_prompt=True) for j in jobs]
    print(f"{len(jobs)} judge prompts ({args.model})", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=6144, gpu_memory_utilization=0.9,
              enable_prefix_caching=True, disable_custom_all_reduce=args.tp > 1)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def record(k: int, o) -> dict:
        j = jobs[k]
        rec = {key: v for key, v in j.items() if key != "prompt"}
        rec["k"] = k
        if j["kind"] in ("item", "ic"):
            rec["p_yes"] = round(_probs(o, ["yes", "no"])[0], 4)
        elif j["kind"] in ("attr", "pair"):
            rec["p_a"] = round(_probs(o, ["a", "b"])[0], 4)
        else:
            ps = _probs(o, ["1", "2", "3", "4", "5"])
            rec["score"] = round(sum((n + 1) * p for n, p in enumerate(ps)), 4)
        return rec

    # Checkpointing (a preempted job is requeued): results are appended per chunk to a partial file keyed by
    # the job index k (build_jobs is deterministic), and a restarted run skips every k already done.
    out = Path(args.out_dir) / f"jury_{args.tag}.jsonl"
    part = Path(args.out_dir) / f"jury_{args.tag}.partial.jsonl"
    done = {}
    if part.is_file():
        for line in part.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["k"]] = r
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut by the preemption
    print(f"resume: {len(done)} of {len(jobs)} already done", flush=True)
    byid = {e["id"]: e for e in eps}
    gen_sp = SamplingParams(max_tokens=6, temperature=0.0)
    chunk = 6000

    def run_batch(ks: list[int], f) -> None:
        for c in range(0, len(ks), chunk):
            part_ks = ks[c: c + chunk]
            gen = [k for k in part_ks if jobs[k]["kind"] in ("iq", "rq")]
            lp = [k for k in part_ks if jobs[k]["kind"] not in ("iq", "rq")]
            if gen:
                for k, o in zip(gen, llm.generate([prompts[k] for k in gen], gen_sp)):
                    j = jobs[k]
                    _, num = numbered_ctx(byid[j["ep"]], j["line"])
                    n = parse_num(o.outputs[0].text, num)
                    rec = {key: v for key, v in j.items() if key != "prompt"}
                    rec.update({"k": k, "answer": o.outputs[0].text.strip()[:20], "num": n,
                                "ref_line": num.get(n) if n else None})
                    done[k] = rec
                    f.write(json.dumps(rec) + "\n")
            if lp:
                for k, o in zip(lp, llm.generate([prompts[k] for k in lp], sp)):
                    done[k] = record(k, o)
                    f.write(json.dumps(done[k]) + "\n")
            f.flush()
            print(f"checkpoint: {len(done)}/{len(jobs)}", flush=True)

    main = [k for k, j in enumerate(jobs) if not j.get("retest") and k not in done]
    rest = [k for k, j in enumerate(jobs) if j.get("retest") and k not in done]
    with part.open("a") as f:
        for batch in (main, rest):  # the retest pass runs after the main pass, as separate generate calls
            run_batch(batch, f)
        # v2 interaction, step 2: for every verified reference (other speaker, real line), ask about CONTENT.
        followups = []
        for k in range(len(jobs)):
            r = done.get(k)
            if r and r["kind"] == "iq" and r.get("ref_line") is not None:
                e = byid[r["ep"]]
                tgt, ref = e["lines"][r["line"]], e["lines"][r["ref_line"]]
                if ref["role"] != tgt["role"] and not ref.get("procedural"):
                    followups.append({"kind": "ic", "ep": r["ep"], "line": r["line"], "dim": "interaction2",
                                      "ref_line": r["ref_line"], "retest": r.get("retest", False), "parent": k,
                                      "prompt": interaction_content(e, r["line"], r["ref_line"])})
        base = len(jobs)
        jobs.extend(followups)
        prompts.extend(tok.apply_chat_template([{"role": "system", "content": JUDGE_SYSTEM},
                                                {"role": "user", "content": x["prompt"]}],
                                               tokenize=False, add_generation_prompt=True) for x in followups)
        run_batch([k for k in range(base, len(jobs)) if k not in done], f)
    with out.open("w") as f:
        for k in range(len(jobs)):
            f.write(json.dumps(done[k]) + "\n")
    print(f"wrote {out}")
