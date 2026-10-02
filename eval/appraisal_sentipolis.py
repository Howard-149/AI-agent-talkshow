#!/usr/bin/env python3
"""
Offline check of the PAD appraisal step against Sentipolis Fig. 4 (Tom ↔ John).

Runs only the appraisal model (no show, TTS, or avatar): replays the paper's
4-round conversation, asks our appraisal prompt for each agent's Δ, and compares
with the paper's reported Δ per round, cumulative Σ Δ, final PAD and kNN label.

Units:
  round (default)  paper protocol — after each round both agents appraise both lines
  line             each agent appraises only the other's line; the two line deltas
                   in a round are summed for comparison
  exchange         like line, but a reply to the listener is appraised together
                   with the listener's own preceding line (one pair)
  reply            like line, but a reply to the listener is flagged as such; the
                   listener's own line stays in context (pairs well with --self)
  --self           the speaker also appraises their own line right after saying it

Usage (Babel, vLLM running; loads .env for VLLM_BASE_URL):
    python eval/appraisal_sentipolis.py
    python eval/appraisal_sentipolis.py --trials 10 --unit exchange
    python eval/appraisal_sentipolis.py --update soft      # our diminishing-returns update
    python eval/appraisal_sentipolis.py --dry-run          # print prompts, no LLM
    # JSON: eval/results/appraisal/sentipolis_fig4_<unit>_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml  # noqa: E402

from agent.emotion.appraisal import render_appraisal_prompt  # noqa: E402
from agent.emotion.pad_state import soft_bounded_add  # noqa: E402
from agent.emotion.state import affect_descriptor, parse_pad_appraisal  # noqa: E402

DEFAULT_FIXTURE = _REPO / "eval" / "fixtures" / "sentipolis_fig4.yaml"
RESULTS_DIR = _REPO / "eval" / "results" / "appraisal"
AXES = ("P", "A", "D")
# |Δ| below this counts as "no change" when comparing signs.
SIGN_DEADZONE = 0.03

Vec = tuple[float, float, float]
# (system, user) → raw model text
Responder = Callable[[str, str], str]


def _sign(x: float) -> int:
    return 0 if abs(x) < SIGN_DEADZONE else (1 if x > 0 else -1)


def _add(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


def apply_update(pad: Vec, delta: Vec, mode: str) -> Vec:
    if mode == "soft":
        return tuple(soft_bounded_add(v, d) for v, d in zip(pad, delta))  # type: ignore[return-value]
    return tuple(_clip(v + d) for v, d in zip(pad, delta))  # type: ignore[return-value]


@dataclass
class TrialResult:
    deltas: dict[str, list[Vec]] = field(default_factory=dict)  # agent → per-round Δ
    words: dict[str, list[list[str]]] = field(default_factory=dict)
    final_pad: dict[str, Vec] = field(default_factory=dict)
    final_label: dict[str, str] = field(default_factory=dict)
    parse_failures: int = 0
    calls: int = 0


def _knn_labels(pad: Vec, knn: Any) -> list[str]:
    if knn is None:
        return ["Neutral"]
    return list(knn.query(pad).neighbor_labels)


def run_trial(
    fx: dict[str, Any],
    respond: Responder,
    *,
    unit: str,
    update: str,
    self_appraise: bool = False,
    knn: Any = None,
    log_prompt: Callable[[str, str, str], None] | None = None,
) -> TrialResult:
    agents = fx["agents"]
    pad: dict[str, Vec] = {a: tuple(agents[a]["initial_pad"]) for a in agents}  # type: ignore[misc]
    out = TrialResult(
        deltas={a: [] for a in agents}, words={a: [] for a in agents}
    )
    history: list[tuple[str, str]] = []  # (agent id, text)

    def label(aid: str) -> str:
        return agents[aid]["name"]

    def appraise(
        aid: str,
        new: list[tuple[str, str]],
        context: list[tuple[str, str]] | None = None,
        reply_to_self: bool = False,
        self_spoken: bool = False,
    ) -> tuple[Vec, str]:
        labels = _knn_labels(pad[aid], knn)
        ctx = history if context is None else context
        system, user = render_appraisal_prompt(
            name=agents[aid]["name"],
            persona=" ".join(str(agents[aid]["profile"]).split()),
            mood=affect_descriptor(pad[aid], labels),
            pad=pad[aid],
            context_lines=[f"{label(s)}: {t}" for s, t in ctx],
            new_lines=[(label(s), t) for s, t in new],
            addressed=agents[aid]["name"].split()[0].lower()
            in " ".join(t for s, t in new if s != aid).lower(),
            reply_to_self=reply_to_self,
            self_spoken=self_spoken,
        )
        if log_prompt:
            log_prompt(aid, system, user)
        raw = respond(system, user)
        out.calls += 1
        word, delta = parse_pad_appraisal(raw)
        if delta is None:
            out.parse_failures += 1
            return (0.0, 0.0, 0.0), "(unparsed)"
        return delta, word or "-"

    for rnd in fx["rounds"]:
        lines = [(str(s), str(t)) for s, t in rnd["lines"]]
        round_delta: dict[str, Vec] = {a: (0.0, 0.0, 0.0) for a in agents}
        round_words: dict[str, list[str]] = {a: [] for a in agents}
        if unit == "round":
            for aid in agents:
                d, w = appraise(aid, lines)
                round_delta[aid], round_words[aid] = d, [w]
            history.extend(lines)
        else:  # line / exchange: each listener hears the other's line as it is spoken
            for speaker, text in lines:
                for aid in agents:
                    if aid == speaker:
                        if self_appraise:
                            d, w = appraise(aid, [(speaker, text)], self_spoken=True)
                            round_delta[aid] = _add(round_delta[aid], d)
                            round_words[aid].append(f"self:{w}")
                            pad[aid] = apply_update(pad[aid], d, update)
                        continue
                    if unit == "reply" and history and history[-1][0] == aid:
                        d, w = appraise(aid, [(speaker, text)], reply_to_self=True)
                    elif unit == "exchange" and history and history[-1][0] == aid:
                        # A reply to the listener: appraise their own line + the reply.
                        d, w = appraise(
                            aid, [history[-1], (speaker, text)],
                            context=history[:-1], reply_to_self=True,
                        )
                    else:
                        d, w = appraise(aid, [(speaker, text)])
                    round_delta[aid] = _add(round_delta[aid], d)
                    round_words[aid].append(w)
                    pad[aid] = apply_update(pad[aid], d, update)
                history.append((speaker, text))
        for aid in agents:
            if unit == "round":
                pad[aid] = apply_update(pad[aid], round_delta[aid], update)
            out.deltas[aid].append(round_delta[aid])
            out.words[aid].append(round_words[aid])
    for aid in agents:
        out.final_pad[aid] = pad[aid]
        labels = _knn_labels(pad[aid], knn)
        out.final_label[aid] = "/".join(dict.fromkeys(x.lower() for x in labels))
    return out


def score(fx: dict[str, Any], trials: list[TrialResult]) -> dict[str, Any]:
    """Agreement with the paper: per-axis sign match, MAE, cumulative Σ, final PAD."""
    agents = fx["agents"]
    report: dict[str, Any] = {"agents": {}}
    for aid, meta in agents.items():
        paper = [tuple(r["paper_delta"][aid]) for r in fx["rounds"]]
        paper_sum = [round(sum(d[i] for d in paper), 3) for i in range(3)]
        sign_hits = [0, 0, 0]
        abs_err: list[list[float]] = [[], [], []]
        sums: list[list[float]] = [[], [], []]
        finals: list[list[float]] = [[], [], []]
        labels: dict[str, int] = {}
        n = 0
        for tr in trials:
            for got, want in zip(tr.deltas[aid], paper):
                n += 1
                for i in range(3):
                    sign_hits[i] += _sign(got[i]) == _sign(want[i])
                    abs_err[i].append(abs(got[i] - want[i]))
            for i in range(3):
                sums[i].append(sum(d[i] for d in tr.deltas[aid]))
                finals[i].append(tr.final_pad[aid][i])
            lab = tr.final_label.get(aid, "?")
            labels[lab] = labels.get(lab, 0) + 1

        def ms(xs: list[float]) -> list[float]:
            return [round(statistics.fmean(xs), 3), round(statistics.pstdev(xs), 3)]

        report["agents"][aid] = {
            "name": meta["name"],
            "sign_agreement": {AXES[i]: round(sign_hits[i] / n, 3) if n else None for i in range(3)},
            "mae": {AXES[i]: round(statistics.fmean(abs_err[i]), 3) for i in range(3)},
            "sum_delta_mean_std": {AXES[i]: ms(sums[i]) for i in range(3)},
            "paper_sum_delta": dict(zip(AXES, paper_sum)),
            "final_pad_mean_std": {AXES[i]: ms(finals[i]) for i in range(3)},
            "paper_final_pad": dict(zip(AXES, meta["final_pad"])),
            "final_labels": labels,
            "paper_final_label": meta["final_label"],
        }
    # Headline qualitative check from the paper: Tom's pleasure rises, John's falls.
    report["pleasure_divergence_rate"] = round(
        statistics.fmean(
            1.0
            if sum(d[0] for d in tr.deltas.get("tom", [])) > 0
            and sum(d[0] for d in tr.deltas.get("john", [])) < 0
            else 0.0
            for tr in trials
        ),
        3,
    ) if {"tom", "john"} <= set(agents) else None
    report["parse_failures"] = sum(t.parse_failures for t in trials)
    report["calls"] = sum(t.calls for t in trials)
    return report


def _fmt(v: Sequence[float]) -> str:
    return "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"


def print_report(fx: dict[str, Any], trials: list[TrialResult], rep: dict[str, Any]) -> None:
    print(f"== per-round Δ (trial 1) vs paper ==")
    t0 = trials[0]
    for aid, meta in fx["agents"].items():
        print(f"{meta['name']}")
        for i, rnd in enumerate(fx["rounds"]):
            words = ", ".join(t0.words[aid][i])
            print(
                f"  R{i + 1}  ours {_fmt(t0.deltas[aid][i])}  paper {_fmt(rnd['paper_delta'][aid])}"
                f"   {words}"
            )
    print(f"\n== agreement over {len(trials)} trial(s) ==")
    for aid, a in rep["agents"].items():
        sa = a["sign_agreement"]
        mae = a["mae"]
        s = a["sum_delta_mean_std"]
        f = a["final_pad_mean_std"]
        print(f"{a['name']}")
        print(f"  sign agreement  P {sa['P']:.0%}  A {sa['A']:.0%}  D {sa['D']:.0%}")
        print(f"  MAE             P {mae['P']:.3f}  A {mae['A']:.3f}  D {mae['D']:.3f}")
        print(
            "  Σ Δ  ours " + _fmt([s[k][0] for k in AXES])
            + " ± " + _fmt([s[k][1] for k in AXES])
            + "   paper " + _fmt([a["paper_sum_delta"][k] for k in AXES])
        )
        print(
            "  final ours " + _fmt([f[k][0] for k in AXES])
            + "   paper " + _fmt([a["paper_final_pad"][k] for k in AXES])
        )
        print(f"  final label ours {a['final_labels']}   paper {a['paper_final_label']}")
    if rep.get("pleasure_divergence_rate") is not None:
        print(
            f"\nTom P↑ and John P↓ (paper's headline pattern): "
            f"{rep['pleasure_divergence_rate']:.0%} of trials"
        )
    print(f"parse failures: {rep['parse_failures']}/{rep['calls']} calls")


class VllmResponder:
    def __init__(self, *, temperature: float, max_tokens: int) -> None:
        import httpx
        from dotenv import load_dotenv

        from agent.config import load_config

        load_dotenv(_REPO / ".env")
        self._llm = load_config().locale().llm
        self._http = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.latencies: list[float] = []

    def __call__(self, system: str, user: str) -> str:
        t0 = time.monotonic()
        resp = self._http.post(
            f"{self._llm.base_url.rstrip('/')}/chat/completions",
            json={
                "model": self._llm.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
            headers={"Authorization": f"Bearer {self._llm.api_key}"},
        )
        resp.raise_for_status()
        self.latencies.append(time.monotonic() - t0)
        content = resp.json()["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else str(content)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--unit", choices=("round", "line", "exchange", "reply"), default="round")
    ap.add_argument("--self", dest="self_appraise", action="store_true",
                    help="speaker also appraises their own line (line/exchange/reply units)")
    ap.add_argument("--update", choices=("plain", "soft"), default="plain",
                    help="plain = paper's add+clip; soft = our diminishing returns")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--no-knn", action="store_true", help="skip MSP anchors (mood = Neutral)")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompts; no LLM")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    fx = yaml.safe_load(args.fixture.read_text(encoding="utf-8"))
    knn = None
    if not args.no_knn:
        from agent.emotion.pad_pipeline import get_pad_knn

        knn = get_pad_knn()
        if knn is None:
            print("note: MSP anchors missing — mood descriptors fall back to Neutral")

    if args.dry_run:
        shown: list[str] = []

        def log_prompt(aid: str, system: str, user: str) -> None:
            if aid not in shown:
                shown.append(aid)
                print(f"===== {aid} — system =====\n{system}\n===== user =====\n{user}\n")

        run_trial(fx, lambda s, u: "[pad]: unmoved 0 0 0", unit=args.unit,
                  update=args.update, knn=knn, log_prompt=log_prompt)
        return

    responder = VllmResponder(temperature=args.temperature, max_tokens=args.max_tokens)
    print(f"appraisal model → {responder._llm.base_url} model={responder._llm.model} "
          f"unit={args.unit} update={args.update} T={args.temperature}\n")
    trials = [
        run_trial(fx, responder, unit=args.unit, update=args.update, knn=knn,
                  self_appraise=args.self_appraise)
        for _ in range(max(1, args.trials))
    ]
    rep = score(fx, trials)
    rep["config"] = {
        "fixture": str(args.fixture.relative_to(_REPO)) if args.fixture.is_relative_to(_REPO) else str(args.fixture),
        "unit": args.unit,
        "self_appraise": args.self_appraise,
        "update": args.update,
        "temperature": args.temperature,
        "trials": len(trials),
        "model": responder._llm.model,
        "latency_mean_s": round(statistics.fmean(responder.latencies), 3) if responder.latencies else None,
    }
    rep["trials"] = [
        {"deltas": t.deltas, "words": t.words, "final_pad": t.final_pad, "final_label": t.final_label}
        for t in trials
    ]
    print_report(fx, trials, rep)

    tag = f"{args.unit}{'+self' if args.self_appraise else ''}"
    out = args.out or RESULTS_DIR / f"{fx['id']}_{tag}_{args.update}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
