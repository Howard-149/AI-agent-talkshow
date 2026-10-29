"""Simulate floor-control arms from cached listener answers and score them.

Every arm replays the same model answers; arms differ only in the poll
schedule (first sentence vs every sentence), whether the deterministic policy
runs, and whether the moderator listens. The first granted cut ends the line.

    python -m eval.floor_eval.analyze --run logs/floor-eval/main \
        --out eval/floor_eval/results/main
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics

import numpy as np

from agent.floor.reactive_policy import ReactivePolicy, parse_request
from eval.floor_eval.run import VARIANTS, variant_sentences

# Fish voices played Amy's live lines at about 2.6 words per second.
WORDS_PER_SECOND = 2.6
EXPECTED = {"overclaim": "commentator", "factual_error": "commentator", "off_topic": "host"}


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    schedule: str  # first | every
    rules: bool
    listeners: tuple[str, ...]


ARMS = (
    Arm("first_rules", "Before: first sentence only, rules", "first", True, ("commentator",)),
    Arm("every_rules", "Every sentence, rules", "every", True, ("commentator",)),
    Arm("every_rules_mod", "Every sentence, rules + moderator (new)", "every", True, ("commentator", "host")),
    Arm("every_llm_only", "Every sentence + moderator, LLM only (no rules)", "every", False, ("commentator", "host")),
)


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def targets(evidence: str, sentence: str) -> bool:
    """Does a quote point at this sentence (substring, or most of its words)?"""
    ev, se = norm(evidence), norm(sentence)
    if not ev:
        return False
    if ev in se or se in ev:
        return True
    words = ev.split()
    return sum(w in set(se.split()) for w in words) / len(words) >= 0.6


def lenient(raw: str | None) -> dict | None:
    if raw is None:
        return None
    text = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def simulate(item: dict, variant: str, rep: int, arm: Arm, answers: dict) -> dict:
    sentences = variant_sentences(item, variant)
    prefixes = [1] if arm.schedule == "first" else range(1, len(sentences))
    policy = ReactivePolicy(min_speech_s=3.0, cooldown_s=15.0)
    for prefix in prefixes:
        heard = " ".join(sentences[:prefix])
        elapsed = len(heard.split()) / WORDS_PER_SECOND
        granted = []
        for listener in arm.listeners:
            raw = answers.get((item["id"], variant, prefix, listener, rep))
            if arm.rules:
                try:
                    req = parse_request(raw, role=listener, epoch=1)
                except (ValueError, TypeError, AttributeError):
                    continue  # production fails closed: no request
                decision = policy.decide(req, epoch=1, speaker="guest", heard=heard, elapsed_s=elapsed,
                                         now=1000.0, eligible_roles=list(arm.listeners))
                if decision.status == "granted":
                    granted.append((decision.priority, listener, req.kind, req.evidence))
            else:
                obj = lenient(raw)
                if obj and obj.get("action") == "interrupt":
                    granted.append((80 if listener == "host" else 50, listener,
                                    str(obj.get("kind", "")), str(obj.get("evidence", ""))))
        if granted:
            granted.sort(key=lambda g: (-g[0], arm.listeners.index(g[1])))
            _, listener, kind, evidence = granted[0]
            return {"cut": True, "prefix": prefix, "listener": listener, "kind": kind,
                    "grounded": bool(norm(evidence)) and norm(evidence) in norm(heard),
                    "on_plant": variant != "clean" and targets(evidence, item[variant])}
    return {"cut": False}


def classify(item: dict, variant: str, out: dict) -> str:
    if variant == "clean":
        return "false_cut" if out["cut"] else "no_cut"
    if not out["cut"]:
        return "missed"
    if out["prefix"] < item["plant_at"]:
        return "premature"
    if not out["on_plant"]:
        return "wrong_target"
    return "caught" if out["listener"] == EXPECTED[variant] else "caught_by_other"


def bootstrap_ci(per_item: dict[str, list[float]], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean, resampling motions (reps stay together)."""
    rng = np.random.default_rng(seed)
    groups = list(per_item.values())
    means = [np.mean([v for g in rng.choice(len(groups), len(groups)) for v in groups[g]]) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main(run: Path, stimuli: Path, out: Path) -> None:
    items = {x["id"]: x for x in json.loads(stimuli.read_text())["items"]}
    rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
    answers = {(r["item"], r["variant"], r["prefix"], r["listener"], r["rep"]): r["raw"] for r in rows}
    items = {k: v for k, v in items.items() if k in {r["item"] for r in rows}}
    reps = sorted({r["rep"] for r in rows})
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}

    arms = {}
    for arm in ARMS:
        outcomes = defaultdict(Counter)
        per_item = defaultdict(lambda: defaultdict(list))  # variant -> item -> [success 0/1]
        lags, ungrounded, cuts = defaultdict(list), 0, 0
        by_position = defaultdict(lambda: [0, 0])  # plant_at -> [caught, total]
        for item in items.values():
            for variant in VARIANTS:
                for rep in reps:
                    sim = simulate(item, variant, rep, arm, answers)
                    label = classify(item, variant, sim)
                    outcomes[variant][label] += 1
                    good = label == "no_cut" if variant == "clean" else label == "caught"
                    per_item[variant][item["id"]].append(float(good))
                    if sim["cut"]:
                        cuts += 1
                        ungrounded += not sim["grounded"]
                    if label == "caught":
                        lags[variant].append(sim["prefix"] - item["plant_at"])
                    if variant != "clean":
                        by_position[item["plant_at"]][0] += label == "caught"
                        by_position[item["plant_at"]][1] += 1
        summary = {"label": arm.label}
        for variant in VARIANTS:
            total = sum(outcomes[variant].values())
            rate = {k: v / total for k, v in outcomes[variant].items()}
            key = "no_cut" if variant == "clean" else "caught"
            summary[variant] = {"n": total, "rates": rate, "success": rate.get(key, 0.0),
                                "success_ci95": bootstrap_ci(per_item[variant]),
                                "lag_sentences": statistics.mean(lags[variant]) if lags[variant] else None}
        planted = {i: [v for var in EXPECTED for v in per_item[var][i]] for i in items}
        summary["planted_detection"] = float(np.mean([v for g in planted.values() for v in g]))
        summary["planted_detection_ci95"] = bootstrap_ci(planted)
        summary["ungrounded_cuts"] = ungrounded
        summary["cuts"] = cuts
        summary["detection_by_plant_position"] = {str(k): v[0] / v[1] for k, v in sorted(by_position.items())}
        arms[arm.name] = summary

    # Request-level behaviour of the production listeners (every prefix, every rep).
    req = {"calls": len(rows), "errors": sum(r["raw"] is None for r in rows)}
    latencies = [r["latency_ms"] for r in rows if r["raw"] is not None]
    req["latency_ms_median"] = statistics.median(latencies)
    req["latency_ms_p90"] = float(np.percentile(latencies, 90))
    strict_fail = 0
    actions = defaultdict(Counter)
    interrupt_quotes = Counter()
    for r in rows:
        item = items[r["item"]]
        try:
            parsed = parse_request(r["raw"], role=r["listener"], epoch=1)
        except (ValueError, TypeError, AttributeError):
            strict_fail += 1
            continue
        heard_plant = r["variant"] != "clean" and r["prefix"] >= item["plant_at"]
        state = r["variant"] if heard_plant else ("clean" if r["variant"] == "clean" else "before_plant")
        actions[f"{r['listener']}:{state}"][parsed.action] += 1
        if parsed.action == "interrupt":
            interrupt_quotes["grounded" if parsed.evidence and norm(parsed.evidence) in norm(r["heard"])
                             else "ungrounded"] += 1
            if parsed.kind == "addition":
                interrupt_quotes["addition_asked_to_interrupt"] += 1
    req["strict_parse_failures"] = strict_fail
    req["actions"] = {k: dict(v) for k, v in sorted(actions.items())}
    req["interrupt_requests"] = dict(interrupt_quotes)

    out.mkdir(parents=True, exist_ok=True)
    result = {"model": meta.get("model"), "items": len(items), "reps": len(reps),
              "words_per_second": WORDS_PER_SECOND, "arms": arms, "requests": req, "run_meta": meta}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "summary.md").write_text(render(result) + "\n")
    print(render(result))


def render(r: dict) -> str:
    a = r["arms"]
    lines = [f"# Floor-control offline evaluation ({r['model']}, {r['items']} motions x {r['reps']} reps)", "",
             "Success = caught: cut by the right listener, after the planted sentence was played, quoting it.",
             "Clean = no cut on a measured, on-topic line. 95% CI: bootstrap over motions.", "",
             "| Arm | Overclaim caught | Factual error caught | Off-topic caught | All planted | Clean: no cut | Ungrounded cuts |",
             "|---|---|---|---|---|---|---|"]
    for name, s in a.items():
        cells = [pct(s[v]["success"]) for v in ("overclaim", "factual_error", "off_topic")]
        lo, hi = s["planted_detection_ci95"]
        lines.append(f"| {s['label']} | " + " | ".join(cells) +
                     f" | {pct(s['planted_detection'])} ({pct(lo)}-{pct(hi)}) | {pct(s['clean']['success'])} "
                     f"| {s['ungrounded_cuts']}/{s['cuts']} |")
    lines += ["", "## Outcome breakdown per planted type", ""]
    for name, s in a.items():
        lines.append(f"**{s['label']}**")
        for v in VARIANTS:
            rates = ", ".join(f"{k} {pct(x)}" for k, x in sorted(s[v]["rates"].items(), key=lambda kv: -kv[1]))
            lag = s[v]["lag_sentences"]
            lines.append(f"- {v}: {rates}" + (f"; mean lag {lag:.2f} sentences" if lag is not None else ""))
        lines.append(f"- detection by planted sentence position: " +
                     ", ".join(f"s{k} {pct(x)}" for k, x in s["detection_by_plant_position"].items()))
        lines.append("")
    q = r["requests"]
    lines += ["## Listener requests (all prefixes)", "",
              f"- calls {q['calls']}, errors {q['errors']}, strict parse failures {q['strict_parse_failures']}",
              f"- latency median {q['latency_ms_median']:.0f} ms, p90 {q['latency_ms_p90']:.0f} ms",
              f"- interrupt requests: {q['interrupt_requests']}"]
    for k, v in q["actions"].items():
        total = sum(v.values())
        lines.append(f"- {k}: " + ", ".join(f"{a_} {pct(n / total)}" for a_, n in sorted(v.items())) + f" (n={total})")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("logs/floor-eval/main"))
    ap.add_argument("--stimuli", type=Path, default=Path("eval/floor_eval/data/stimuli.json"))
    ap.add_argument("--out", type=Path, default=Path("eval/floor_eval/results/main"))
    args = ap.parse_args()
    main(args.run, args.stimuli, args.out)
