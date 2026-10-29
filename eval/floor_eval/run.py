"""Query listener intents at every played prefix of every test line.

Uses the production client (VLLM_* settings), production prompts and the same
call parameters as ReactiveSession._request_role. Results are cached as JSONL;
rerunning resumes and only fills in missing calls.

    python -m eval.floor_eval.run --reps 3 --out logs/floor-eval/<name>
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

from dotenv import load_dotenv

VARIANTS = ("clean", "overclaim", "factual_error", "off_topic")
LISTENERS = ("commentator", "host")
SPEAKER = "guest"


def variant_sentences(item: dict, variant: str) -> list[str]:
    sentences = list(item["clean"])
    if variant != "clean":
        sentences[item["plant_at"] - 1] = item[variant]
    return sentences


def opening_history(motion: str):
    from agent.show.show_history import ShowHistory

    history = ShowHistory()
    history.append("host", f"Tonight's motion: {motion}. Amy, you go first.")
    return history


def listener_request(role: str, motion: str, heard: str) -> tuple[str, str]:
    from agent.config import load_persona_name
    from agent.floor.intent_prompts import intent_prompt, moderator_intent_system, panelist_intent_system

    moderator = role == "host"
    system = moderator_intent_system(role, topic=motion) if moderator else panelist_intent_system(role)
    return system, intent_prompt(load_persona_name(SPEAKER), heard, moderator=moderator)


async def main(stimuli: Path, out: Path, reps: int, concurrency: int, limit: int | None,
               model: str | None = None) -> None:
    from dataclasses import replace

    from agent.adapters.gemma_mm_client import GemmaMMClient
    from agent.config import load_config

    items = json.loads(stimuli.read_text())["items"][:limit]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "requests.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("raw") is not None:
                done.add((r["item"], r["variant"], r["prefix"], r["listener"], r["rep"]))
    llm = load_config().locale().llm
    if model:
        llm = replace(llm, model=model)  # e.g. a small model as a stand-in for on-device Gemma
    client = GemmaMMClient(llm, system_prompt="")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    counts = {"ok": 0, "error": 0}

    async def one(item, variant, prefix, listener, rep):
        sentences = variant_sentences(item, variant)
        heard = " ".join(sentences[:prefix])
        system, prompt = listener_request(listener, item["motion"], heard)
        hist = opening_history(item["motion"]).prior_messages(perspective=listener)
        rec = {"item": item["id"], "variant": variant, "prefix": prefix, "listener": listener, "rep": rep,
               "model": llm.model, "heard": heard}
        async with sem:
            t0 = time.monotonic()
            try:
                rec["raw"] = await asyncio.wait_for(client.complete_text(
                    prompt, system_prompt=system, history_messages=hist, raw=True,
                    temperature=0.4, max_tokens=400), timeout=45)
                counts["ok"] += 1
            except Exception as exc:  # recorded; the analysis treats it as a failed request
                rec["raw"], rec["error"] = None, f"{type(exc).__name__}: {exc}"[:200]
                counts["error"] += 1
            rec["latency_ms"] = round((time.monotonic() - t0) * 1000)
        async with lock:
            with path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    jobs = [one(item, v, p, ls, rep)
            for item in items for v in VARIANTS
            for p in range(1, len(item["clean"]))  # production never polls after the last sentence
            for ls in LISTENERS for rep in range(reps)
            if (item["id"], v, p, ls, rep) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} cached) with {llm.model}", flush=True)
    t0 = time.monotonic()
    try:
        await asyncio.gather(*jobs)
    finally:
        await client.aclose()
    (out / "run_meta.json").write_text(json.dumps({
        "model": llm.model, "base_url": llm.base_url, "reps": reps, "items": len(items),
        "stimuli": str(stimuli), "seconds": round(time.monotonic() - t0, 1), **counts}, indent=2) + "\n")
    print(json.dumps(counts), f"in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", type=Path, default=Path("eval/floor_eval/data/stimuli.json"))
    ap.add_argument("--out", type=Path, default=Path("logs/floor-eval/main"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", help="override VLLM_MODEL for this run")
    args = ap.parse_args()
    asyncio.run(main(args.stimuli, args.out, args.reps, args.concurrency, args.limit, args.model))
