"""Automatic metrics (no judge). Embeddings come from a GPU stage (`embed`: Qwen3-Embedding-0.6B via vLLM pooling,
the same model as the listening benchmark's R); everything else is CPU.

Per episode, over its scored panel lines (higher = better unless noted):
  specific     1 − max cosine to the generic-filler pool;  figures+names per 100 words (digits, number words,
               capitalised non-initial tokens — a spaCy-free proxy)
  novelty      share of lines whose max cosine to every EARLIER line of the episode is < NOVEL_THR
  interaction  share of lines (after the first) that share a content bigram, a figure or a name with an earlier line by a
               different speaker
  repetition   distinct-2, distinct-3 over the beat's panel words; 1 − repeated-opener share (first 3 words, within the
               episode); cross-episode opener reuse is reported per source in validate (not per episode)
  flow         alternation share (consecutive panel lines by different speakers); turn-share entropy over the 2
               panelists (normalised to [0, 1]); human turn answered (the first panel line after a human turn shares a
               content word with it)

Per set of episodes (reuse_metrics, used by `ruler.py score`): cross-episode template reuse, rarefied to 100-line subsets.
"""

from __future__ import annotations

import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval.quality_ruler.items import GENERIC_POOL

NOVEL_THR = 0.80
STOP = set("""a an the and or but so to of in on at for with by from as is are was were be been being it its this that these those
i you he she we they me him her us them my your his our their not no do does did have has had will would can could should
just very really more most much many some any all about than then there here what which who whom when where why how if
because also only even still too into out up down over again like well yes yeah okay ok think know mean say said going
get got make one way thing things people right""".split())
NUM = re.compile(r"\d|\b(?:percent|million|billion|trillion|hundred|thousand|half|third|quarter|dozen|two|three|four|five|"
                 r"six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b", re.IGNORECASE)


def words(t: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$%']+", t)


def content(t: str) -> list[str]:
    return [w.lower() for w in words(t) if w.lower() not in STOP and len(w) > 2]


def names_figs(t: str) -> set[str]:
    out = set(m.group(0).lower() for m in re.finditer(r"\d[\d,.]*%?", t))
    toks = t.split()
    for k, w in enumerate(toks):
        if k > 0 and not toks[k - 1].endswith((".", "?", "!")) and re.match(r"^[A-Z][a-z]{2,}", w):
            out.add(re.sub(r"\W", "", w).lower())
    return out


def embed(args, texts: list[str]) -> np.ndarray:
    from vllm import LLM

    from eval.hf_local import local_snapshot

    # offline: the cached snapshot dir (a copied snapshot can fail the hub check by name)
    llm = LLM(model=local_snapshot(args.embed_model), runner="pooling", gpu_memory_utilization=0.5)
    e = np.array([r.outputs.embedding for r in llm.embed(texts)], dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def embed_stage(args, eps: list[dict], out_dir: Path) -> None:
    keys, texts = [], []
    for e in eps:
        for i, l in enumerate(e["lines"]):
            keys.append([e["id"], i])
            texts.append(l["text"])
    keys += [["__generic__", k] for k in range(len(GENERIC_POOL))]
    texts += GENERIC_POOL
    np.save(out_dir / "emb.npy", embed(args, texts))
    (out_dir / "emb_keys.json").write_text(json.dumps(keys))
    print(f"embedded {len(texts)} lines → {out_dir}/emb.npy")


def metrics(eps: list[dict], out_dir: Path) -> dict[str, dict]:
    emb = np.load(out_dir / "emb.npy") if (out_dir / "emb.npy").is_file() else None
    idx = {}
    if emb is not None:
        for k, (eid, i) in enumerate(json.loads((out_dir / "emb_keys.json").read_text())):
            idx[(eid, i)] = k
        gen = emb[[idx[("__generic__", k)] for k in range(len(GENERIC_POOL))]]
    res = {}
    for e in eps:
        L, sc = e["lines"], e["scored"]
        m: dict[str, float] = {}
        txt = [L[i]["text"] for i in sc]
        nw = sum(len(words(t)) for t in txt) or 1
        m["specific.figs_names_per100w"] = 100 * sum(len(NUM.findall(t)) + len(names_figs(t)) for t in txt) / nw
        if emb is not None and all((e["id"], i) in idx for i in range(len(L))):
            V = {i: emb[idx[(e["id"], i)]] for i in range(len(L))}
            m["specific.1-generic_sim"] = float(np.mean([1 - float((gen @ V[i]).max()) for i in sc]))
            nov = [float(max((V[i] @ V[j] for j in range(i)), default=0.0)) < NOVEL_THR for i in sc]
            m["novelty.share_novel"] = float(np.mean(nov))
        # interaction: lexical link to an earlier line by another (non-host) speaker
        links = []
        for i in sc[1:]:
            mine = content(L[i]["text"])
            my_bi = set(zip(mine, mine[1:]))
            my_nf = names_figs(L[i]["text"])
            hit = False
            for j in range(i):
                if L[j]["role"] in ("host",) or L[j]["role"] == L[i]["role"]:
                    continue
                other = content(L[j]["text"])
                if my_bi & set(zip(other, other[1:])) or my_nf & names_figs(L[j]["text"]):
                    hit = True
                    break
            links.append(hit)
        m["interaction.lexical_link"] = float(np.mean(links)) if links else float("nan")
        # repetition
        toks = [w.lower() for t in txt for w in words(t)]
        for n in (2, 3):
            grams = list(zip(*[toks[k:] for k in range(n)]))
            m[f"repetition.distinct{n}"] = len(set(grams)) / len(grams) if grams else 1.0
        openers = [" ".join(w.lower() for w in words(t)[:3]) for t in txt]
        m["repetition.1-repeated_opener"] = 1 - (len(openers) - len(set(openers))) / max(len(openers), 1)
        # flow
        roles = [L[i]["role"] for i in sc]
        m["flow.alternation"] = float(np.mean([a != b for a, b in zip(roles, roles[1:])])) if len(roles) > 1 else float("nan")
        p = np.array([roles.count(r) for r in ("commentator", "guest")], dtype=float) / max(len(roles), 1)
        m["flow.turn_entropy"] = float(-sum(x * math.log2(x) for x in p if x > 0))
        answered = []
        for h, l in enumerate(L):
            if l["role"] == "human":
                nxt = next((k for k in range(h + 1, len(L)) if L[k]["role"] in ("commentator", "guest")), None)
                if nxt is not None and nxt in sc:
                    answered.append(bool(set(content(l["text"])) & set(content(L[nxt]["text"]))))
        m["flow.human_answered"] = float(np.mean(answered)) if answered else float("nan")
        m["length.words_per_line"] = nw / max(len(sc), 1)
        res[e["id"]] = m
    return res


# --------------------------------------------------------------------------------------------- cross-episode reuse

def opener(t: str, n: int = 3) -> str:
    return " ".join(re.findall(r"[A-Za-z']+", t.lower())[:n])


# Acknowledgment-formula openers ("I (really/do) hear…", "Ryan is right/hitting…", "That's a fair point…") as one template
# family: 3-word opener classes miss paraphrased templates ("I really hear" replacing "I hear").
ACK = re.compile(r"^\W*(?:(?:ryan|amy|lessac)\W+)?(?:i\s+(?:\w+\s+)?(?:hear|see|understand|agree|appreciate|get)\b|"
                 r"(?:ryan|amy|lessac)(?:'s)?\s+(?:is\s+)?(?:right|hitting|point|raises|makes|brings)|"
                 r"(?:that's|that\s+is|you're)\s+(?:a\s+)?(?:great|good|fair|valid|right|true))", re.I)

RARE_N, RARE_DRAWS = 100, 30  # rarefaction: every reuse metric is computed on random 100-line subsets (size-independent)
REUSE_KEYS = ("reuse.top10_opener_share", "reuse.opener_classes_per100", "reuse.corpus_distinct2",
              "reuse.corpus_distinct3", "reuse.ack_template_share")


def reuse_metrics(eps: list[dict]) -> dict[str, float]:
    """Cross-episode template reuse, rarefied to RARE_N lines (mean over RARE_DRAWS seeded subsets), so sets with
    different numbers of lines are comparable: top-10-opener share, distinct opener classes per 100 lines,
    distinct-2/3 over the subset's words, share of acknowledgment-formula openers."""
    lines = [e["lines"][i]["text"] for e in eps for i in e["scored"]]
    if not lines:
        return {k: float("nan") for k in REUSE_KEYS}
    rng = random.Random(7)
    acc = defaultdict(list)
    for _ in range(RARE_DRAWS):
        sub = rng.sample(lines, min(RARE_N, len(lines)))
        ops = [opener(t) for t in sub]
        c = Counter(ops)
        acc["reuse.top10_opener_share"].append(sum(n for _, n in c.most_common(10)) / len(ops))
        acc["reuse.opener_classes_per100"].append(100 * len(c) / len(ops))
        acc["reuse.ack_template_share"].append(sum(bool(ACK.match(t)) for t in sub) / len(sub))
        toks = [w.lower() for t in sub for w in re.findall(r"[A-Za-z0-9$%']+", t)]
        for n in (2, 3):
            g = list(zip(*[toks[k:] for k in range(n)]))
            acc[f"reuse.corpus_distinct{n}"].append(len(set(g)) / max(len(g), 1))
    return {k: float(np.mean(v)) for k, v in acc.items()}
