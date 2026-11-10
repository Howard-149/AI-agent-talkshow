"""Run one system on the benchmark items.

    python -m eval.bench.run --system plain --model Qwen/Qwen2.5-7B-Instruct --tasks T1,T2,T3,T4 --split test \
        --reruns 3 --out DIR
    python -m eval.bench.run --system think_action --arm fast --model google/gemma-4-E4B-it ... --out DIR
    python -m eval.bench.run ... --dry-run   # no model: writes every exact prompt to DIR/prompts.jsonl (+ DIR/prompts/*.txt)
    python -m eval.bench.run ... --fake      # no model: placeholder replies, the full output format (tests, plumbing)

Systems: plain | worker | think_action (systems.py). --arm (show systems): production | fast | concrete_fact_pack.
Output DIR/outputs.jsonl: one row per (item, rerun) with the reply, floor_action, latency, prompt hash; DIR/run.json:
the configuration. Paired items share sampling seeds: the seed depends on (pair group, role, rerun, --seed), not on the
version, so the two sides of a counterfactual pair differ only in the edited input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.bench.items import Item, expand_tasks, observation, read_items  # noqa: E402

DEV_ITEMS = ROOT / "eval" / "data" / "bench" / "dev.jsonl"
TEST_ITEMS = ROOT / "eval" / "data" / "bench" / "build" / "test.jsonl"


def seed_for(item: Item, rerun: int, base: int = 0) -> int:
    key = f"{base}:{item.pair_group}:{item.role_under_test}:{rerun}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def select(items: list[Item], tasks: list[str], topics: str = "", limit_topics: int = 0) -> list[Item]:
    out = [it for it in items if it.task in tasks]
    if topics:
        keep = set(topics.split(","))
        out = [it for it in out if it.topic_id in keep]
    if limit_topics:
        first = list(dict.fromkeys(it.topic_id for it in out))[:limit_topics]
        out = [it for it in out if it.topic_id in set(first)]
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_prompts(out: Path, records: list, items: list[Item]) -> None:
    """Dry run: every request as JSONL, and one readable file per item (rerun 0)."""
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    with (out / "prompts.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps({"tag": r.tag, "kind": r.kind, "temperature": r.temperature, "max_tokens": r.max_tokens,
                                "seed": r.seed, "messages": r.messages}, ensure_ascii=False) + "\n")
    by_item: dict[str, list] = {}
    for r in records:
        by_item.setdefault(r.tag, []).append(r)
    for it in items:
        with (out / "prompts" / (it.item_id.replace("/", "__") + ".txt")).open("w") as f:
            f.write(f"# item={it.item_id} version={it.version} role={it.role_under_test}\n")
            for r in by_item.get(it.item_id, []):
                f.write(f"\n######## {r.kind} (temperature {r.temperature}, max_tokens {r.max_tokens})\n")
                for m in r.messages:
                    f.write(f"\n===== {m['role']} =====\n{m['content']}\n")


def main(argv: list[str] | None = None) -> None:
    from eval.bench.systems import DryRunBackend, FakeBackend, make_system

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", choices=("plain", "worker", "think_action"), required=True)
    ap.add_argument("--model", default="", help="HF model id or path (not needed with --dry-run / --fake)")
    ap.add_argument("--arm", default="production", help="show systems: production | fast | concrete_fact_pack")
    ap.add_argument("--tasks", default="T1,T2,T3a,T3b,T4", help="comma list; T3 = T3a and T3b")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--items", default="", help="items jsonl (default: eval/data/bench/dev.jsonl or build/test.jsonl)")
    ap.add_argument("--topics", default="", help="comma list of topic ids (default all)")
    ap.add_argument("--limit-topics", type=int, default=0, help="only the first N topics (smoke runs)")
    ap.add_argument("--reruns", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-fallback", action="store_true",
                    help="think_action: a panelist who does not choose to speak gets no line (default: the worker line)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="no model: write the exact prompts")
    mode.add_argument("--fake", action="store_true", help="no model: placeholder replies in the real output format")
    args = ap.parse_args(argv)

    items_path = Path(args.items) if args.items else (DEV_ITEMS if args.split == "dev" else TEST_ITEMS)
    if not items_path.is_file():
        sys.exit(f"{items_path} not found (test items: python -m eval.bench.build --split test)")
    items = select(read_items(items_path), expand_tasks(args.tasks), args.topics, args.limit_topics)
    items = [it for it in items if it.split == args.split]
    if not items:
        sys.exit("no items selected")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run or args.fake:
        # The think call splits panel lines with a tokenizer; without a model, split on words (no download).
        os.environ.setdefault("TALKSHOW_TEXT_TOKENIZER", "word")
        backend = DryRunBackend() if args.dry_run else FakeBackend()
        model = args.model or "none"
    else:
        if not args.model:
            sys.exit("--model is required (or --dry-run / --fake)")
        # The show splits panel lines with the generating model's tokenizer (as the live show).
        if not (os.environ.get("VLLM_MODEL_PATH") or os.environ.get("VLLM_MODEL")):
            os.environ["VLLM_MODEL_PATH"] = args.model
        from eval.bench.systems import VLLMBackend

        backend = VLLMBackend(args.model, tp=args.tp, max_model_len=args.max_model_len, seed=args.seed)
        model = args.model
    kw = {"fallback": not args.no_fallback} if args.system == "think_action" else {}
    system = make_system(args.system, backend, args.arm, **kw)

    reruns = 1 if args.dry_run else args.reruns
    jobs = [(it, k) for k in range(reruns) for it in items]
    t0 = time.time()
    results = system.respond_batch([observation(it) for it, _k in jobs], [seed_for(it, k, args.seed) for it, k in jobs])
    wall = time.time() - t0
    config = {"system": system.describe(), "model": model, "split": args.split, "tasks": args.tasks,
              "items": str(items_path.relative_to(ROOT) if items_path.is_relative_to(ROOT) else items_path),
              "items_sha": hashlib.sha256(items_path.read_bytes()).hexdigest()[:16], "n_items": len(items),
              "reruns": reruns, "seed": args.seed, "commit": git_commit(), "wall_s": round(wall, 1),
              "mode": "dry_run" if args.dry_run else "fake" if args.fake else "model"}
    (out / "run.json").write_text(json.dumps(config, indent=1))
    if args.dry_run:
        write_prompts(out, backend.records, items)
        print(f"dry run: {len(backend.records)} requests for {len(items)} items -> {out}/prompts.jsonl")
        return
    n_empty = 0
    with (out / "outputs.jsonl").open("w") as f:
        for (it, k), r in zip(jobs, results):
            n_empty += not r["text"]
            row = {"item_id": it.item_id, "task": it.task, "split": it.split, "topic_id": it.topic_id,
                   "pair_group": it.pair_group, "version": it.version, "role": it.role_under_test, "rerun": k,
                   "seed": seed_for(it, k, args.seed), "system": args.system, "arm": args.arm if args.system != "plain"
                   else None, "model": model, **{k2: v for k2, v in r.items() if k2 != "calls"},
                   "calls": [{c: v for c, v in call.items() if c != "raw"} for call in r["calls"]]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(jobs)} replies ({len(items)} items x {reruns} reruns, {n_empty} empty) in {wall:.0f}s -> "
          f"{out}/outputs.jsonl")


if __name__ == "__main__":
    main()
