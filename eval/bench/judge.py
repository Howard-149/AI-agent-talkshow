"""Answer the judge requests written by score.py with one open judge model (offline vLLM, GPU).

    python -m eval.bench.judge --requests RUN/score[,RUN2/score,...] --model Qwen/Qwen2.5-72B-Instruct-AWQ --tag qwen72 --tp 2

Reads <dir>/judge/identification.jsonl and <dir>/judge/rubric.jsonl, writes <dir>/judge/answers_<tag>.jsonl (one row per
request id). Every answer is read from the first generated token's top-20 logprobs, no sampling, as the existing judges
do: identification A/B with eval/judge_context.p_answer_a (T4 persona attribution with the quality ruler's A/B reader),
rubric yes/no with the quality ruler's reader (eval/quality_ruler/jury._probs). Prompts are the request messages as
written (the existing judge prompts), rendered with the judge's chat template.

Checkpointing: answers are appended per chunk to answers_<tag>.partial.jsonl; a restarted (requeued) job skips every id
already answered. --rubric-reruns (default 0): rubric requests of these reruns only ("all" = every rerun);
identification requests always use every rerun. --fake: no model, a deterministic stand-in (tests, plumbing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHUNK = 6000
MAX_MODEL_LEN = 6144  # as the quality ruler's jury; the longest request (dev transcript + question) is ~2.5k tokens


def parse_reruns(spec: str) -> set[int] | None:
    """"0" -> {0}; "0,1" -> {0, 1}; "all" -> None (every rerun)."""
    spec = str(spec).strip().lower()
    return None if spec in ("all", "") else {int(x) for x in spec.split(",")}


def load_requests(dirs: list[Path], rubric_reruns: set[int] | None) -> list[tuple[Path, dict]]:
    out = []
    for d in dirs:
        for name in ("identification", "rubric"):
            path = d / "judge" / f"{name}.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["request"] = name
                if name == "rubric" and rubric_reruns is not None and r["rerun"] not in rubric_reruns:
                    continue
                out.append((d, r))
    return out


def read_answer(req: dict, out: Any) -> dict:
    """The answer fields for one request from a vLLM output (first-token logprobs)."""
    from eval.judge_context import p_answer_a
    from eval.quality_ruler.jury import _probs

    if req["request"] == "rubric":
        return {"p_yes": round(_probs(out, ["yes", "no"])[0], 4)}
    p_a = _probs(out, ["a", "b"])[0] if req["task"] == "T4" else p_answer_a(out)
    p_true = p_a if req["true"] == "A" else 1.0 - p_a
    return {"p_a": round(p_a, 4), "p_true": round(p_true, 4), "hit": int(p_true > 0.5)}


def fake_output(req: dict) -> Any:
    """A stand-in vLLM output: a fixed pseudo-random first-token distribution per request id."""
    h = int(hashlib.sha256(req["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    toks = ("Yes", "No") if req["request"] == "rubric" else ("A", "B")
    lp = {0: SimpleNamespace(decoded_token=toks[0], logprob=math.log(max(h, 1e-6))),
          1: SimpleNamespace(decoded_token=toks[1], logprob=math.log(max(1 - h, 1e-6)))}
    return SimpleNamespace(outputs=[SimpleNamespace(logprobs=[lp])])


def run(dirs: list[Path], tag: str, generate: Callable[[list[dict]], list[Any]], rubric_reruns: set[int] | None,
        chunk: int = CHUNK) -> dict[Path, int]:
    """Answer every pending request; ``generate(requests) -> outputs`` (vLLM or fake). Returns answers per dir."""
    reqs = load_requests(dirs, rubric_reruns)
    done: dict[Path, dict[str, dict]] = {d: {} for d in dirs}
    for d in dirs:
        part = d / "judge" / f"answers_{tag}.partial.jsonl"
        if part.is_file():
            for line in part.read_text().splitlines():
                try:
                    r = json.loads(line)
                    done[d][r["id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass  # a line cut by the preemption
    todo = [(d, r) for d, r in reqs if r["id"] not in done[d]]
    print(f"{len(reqs)} requests, {len(reqs) - len(todo)} already answered", flush=True)
    files = {d: (d / "judge" / f"answers_{tag}.partial.jsonl").open("a") for d in dirs}
    try:
        for c in range(0, len(todo), chunk):
            part = todo[c: c + chunk]
            for (d, r), o in zip(part, generate([r for _d, r in part])):
                rec = {k: v for k, v in r.items() if k != "messages"}
                rec.update(read_answer(r, o))
                done[d][r["id"]] = rec
                files[d].write(json.dumps(rec, ensure_ascii=False) + "\n")
            for f in files.values():
                f.flush()
            print(f"checkpoint: {min(c + chunk, len(todo))}/{len(todo)}", flush=True)
    finally:
        for f in files.values():
            f.close()
    counts = {}
    for d in dirs:
        ids = [r["id"] for dd, r in reqs if dd == d]
        with (d / "judge" / f"answers_{tag}.jsonl").open("w") as f:
            for i in ids:
                f.write(json.dumps(done[d][i], ensure_ascii=False) + "\n")
        counts[d] = len(ids)
    return counts


def vllm_generate(model: str, tp: int) -> Callable[[list[dict]], list[Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9,
              enable_prefix_caching=True, disable_custom_all_reduce=tp > 1)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def generate(reqs: list[dict]) -> list[Any]:
        prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in reqs]
        return llm.generate(prompts, sp, use_tqdm=False)

    return generate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--requests", required=True, help="comma list of score dirs (each holds judge/*.jsonl)")
    ap.add_argument("--model", default="", help="judge model (HF id or path)")
    ap.add_argument("--tag", required=True, help="judge name in the answers file, e.g. qwen72")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--rubric-reruns", default="0", help='rubric items for these reruns only (comma list or "all")')
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--fake", action="store_true", help="no model: deterministic stand-in answers")
    args = ap.parse_args(argv)
    dirs = [Path(x) for x in args.requests.split(",") if x]
    if args.fake:
        gen = lambda reqs: [fake_output(r) for r in reqs]  # noqa: E731
    else:
        if not args.model:
            sys.exit("--model is required (or --fake)")
        gen = vllm_generate(args.model, args.tp)
    for d, n in run(dirs, args.tag, gen, parse_reruns(args.rubric_reruns), args.chunk).items():
        print(f"{n} answers -> {d}/judge/answers_{args.tag}.jsonl")


if __name__ == "__main__":
    main()
