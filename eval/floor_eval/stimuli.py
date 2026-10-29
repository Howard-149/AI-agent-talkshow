"""Generate and validate planted-problem test lines for Amy (guest).

Each motion gets a 5-sentence measured opening plus three replacement
sentences: an overclaim (worth a rebuttal), a factual error (worth a
correction) and an off-topic drift (worth a moderator cut). A second model
checks every sentence; motions with any failed check are dropped, so the
planted labels are not just the generator's say-so.

    python -m eval.floor_eval.stimuli --out eval/floor_eval/data/stimuli.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
import httpx

from eval.floor_eval.llm import chat_json
from eval.floor_eval.motions import MOTIONS

PLANTS = ("overclaim", "factual_error", "off_topic")
# Rotated so off-topic drifts do not all sound alike.
DRIFTS = ("a hiking trip last weekend", "a TV series she is binge-watching", "her new kitten",
          "a concert she went to", "learning to bake sourdough", "her favourite football team",
          "a home renovation disaster", "a video game she is hooked on")

GEN_SYSTEM = "You write test material for a talk-show floor-control experiment. Output JSON only."
GEN_USER = """Motion: {motion}

Amy is a warm, thoughtful panelist who speaks first. Write natural spoken English, first person as Amy.
Return a JSON object with:
- "clean": a list of exactly 5 sentences Amy says about the motion. Measured and balanced, acknowledging both
  sides. No extreme claims, no statistics or specific facts that could be wrong, no personal attacks, always on
  the motion. Each sentence 12 to 25 words, one sentence per item, ending with a period.
- "overclaim": one sentence (12 to 25 words) in Amy's voice making an extreme, sweeping claim about the motion
  that a sharp opposing panelist would want to rebut at once. An overreaching opinion, not a factual error.
- "factual_error": one sentence (12 to 25 words) in Amy's voice about the motion containing one clear, widely
  known factual error (for example a wrong famous date, number or scientific fact). Otherwise natural.
- "factual_truth": the correct fact, briefly.
- "off_topic": one sentence (12 to 25 words) in Amy's voice that abruptly drifts to {drift},
  with no connection to the motion.
Avoid abbreviations with periods (e.g., U.S.) and decimal numbers so sentence splitting stays exact."""

VAL_SYSTEM = "You are a strict annotator checking test sentences for a debate experiment. Output JSON only."
VAL_USER = """Motion: {motion}

Answer each question with true or false.
"clean": for EACH of these sentences, is it on the motion, measured (no extreme or sweeping claim) and free of
factual errors? {clean}
"overclaim": is this an extreme, overreaching opinion about the motion that an opposing panelist would want to
rebut, WITHOUT a factual error? {overclaim}
"factual_error": does this sentence contain a clear factual error? {factual_error}
"off_topic": is this sentence unrelated to the motion? {off_topic}
Return {{"clean": [5 booleans], "overclaim": bool, "factual_error": bool, "off_topic": bool}}."""


def _well_formed(sentences: list[str]) -> bool:
    from agent.session.speak_chunks import split_speak_sentences

    return all(len(split_speak_sentences(s)) == 1 for s in sentences) and \
        len(split_speak_sentences(" ".join(sentences))) == len(sentences)


async def build_one(http, sem, index: int, domain: str, motion: str, gen_model: str, val_model: str) -> dict:
    async with sem:
        gen = await chat_json(http, gen_model, GEN_SYSTEM, GEN_USER.format(motion=motion, drift=DRIFTS[index % len(DRIFTS)]), temperature=0.7)
        clean = [str(s).strip() for s in gen.get("clean", [])]
        item = {"id": f"m{index:02d}", "domain": domain, "motion": motion, "clean": clean,
                **{k: str(gen.get(k, "")).strip() for k in (*PLANTS, "factual_truth")},
                # Planted position cycles 1..4 (1-based), never the last sentence.
                "plant_at": index % 4 + 1}
        problems = []
        if len(clean) != 5:
            problems.append("clean_not_5")
        if not _well_formed(clean + [item[k] for k in PLANTS]):
            problems.append("sentence_split")
        if not problems:
            val = await chat_json(http, val_model, VAL_SYSTEM, VAL_USER.format(
                motion=motion, clean=json.dumps(clean), **{k: json.dumps(item[k]) for k in PLANTS}),
                temperature=0, max_tokens=400)
            item["validation"] = val
            if not (isinstance(val.get("clean"), list) and len(val["clean"]) == 5 and all(val["clean"])):
                problems.append("clean_failed_check")
            problems += [f"{k}_failed_check" for k in PLANTS if val.get(k) is not True]
        item["problems"] = problems
        return item


async def main(out: Path, gen_model: str, val_model: str, limit: int | None) -> None:
    motions = MOTIONS[:limit] if limit else MOTIONS
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as http:
        items = await asyncio.gather(*(build_one(http, sem, i, d, m, gen_model, val_model)
                                       for i, (d, m) in enumerate(motions)))
    kept = [x for x in items if not x["problems"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generator": gen_model, "validator": val_model,
                               "items": kept, "dropped": [x for x in items if x["problems"]]},
                              indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"motions": len(items), "kept": len(kept),
                      "dropped": {x["id"]: x["problems"] for x in items if x["problems"]}}, indent=2))


if __name__ == "__main__":
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("eval/floor_eval/data/stimuli.json"))
    ap.add_argument("--gen-model", default="gpt-4.1")
    ap.add_argument("--val-model", default="gpt-5.5")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    asyncio.run(main(args.out, args.gen_model, args.val_model, args.limit))
