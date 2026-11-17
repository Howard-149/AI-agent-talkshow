"""T6 full show: a system runs a whole multi-beat episode on a topic, around a fixed audience script.

One T6 item = a topic, the two panel personas and a fixed audience script of 3 turns (eval/data/bench/
t6_dev_scripts.yaml). The episode opens with the host's two opening lines (as the test items), then: audience turn 1,
SLOTS lines by the show, audience turn 2, SLOTS lines, audience turn 3, SLOTS lines. The audience turns are fixed text,
so every system gets the same input at the same points; everything between them is the system's own (who speaks, what
they say, whether the host steps in).

How each system runs an episode (offline, batched over episodes; `generate`):
- think_action: the show's own free floor, slot by slot. Every floor role except the one who spoke last (host, Ryan,
  Amy) makes its production think call on the finished turn (eval/think_action_prompt.think_request, as
  ThinkActionInteraction._react_free does live); the production selection picks (eval/bench/floor.arbitrate: addressee
  priority, else the LLM floor controller, with the floor bookkeeping carried from slot to slot). The selected role
  speaks its think content; a selected panelist with no content of its own (forced by addressee priority) gets the
  worker draft, as live. When the controller hands the floor back to the host (host_fallback), the host speaks the
  production host_fallback_line for the controller's instruction (eval/bench/floor.host_fallback). Uptake planner and
  arm flags as in the item benchmark (arms in eval/quality_ruler/arms.py).
- plain: no floor of its own. Each slot asks the floor question of T2f ("Who should speak next: Ryan or Amy?"), then
  the chosen panelist answers with the public PlainLLM prompt. It has no host.

Differences from the live show, by design: no interrupts or backchannels (each think is on a finished turn), no
speech timing, the human never takes the floor between script turns, the episode has a fixed number of slots.

Scores (`compare`):
- counted, no judge: opener repetition (a panelist's line starting with the same first word as its own previous
  line; per panelist the share of its most frequent first word), self-repetition (content-bigram overlap with the
  speaker's own earlier lines), the quality ruler's counted metrics (eval/quality_ruler/auto.py, without embeddings),
  words per line, unsupported specifics (eval/quality_ruler/specifics.py, against the fact pack), floor statistics;
- audience uptake: a yes/no judge item on the first panel line after each audience turn;
- show quality: a pairwise judge (two episodes of the same topic and rerun, both orders), Bradley-Terry over systems
  with a bootstrap over topics; gate: known-bad perturbations of episodes (eval/quality_ruler/perturb.py) must lose
  to their originals; judge agreement (Krippendorff's alpha) between the two judges.

    python -m eval.bench.show generate --system think_action --arm fast_v2_lib_force --model google/gemma-4-E4B-it \
        --topics tipping,ai-jobs --reruns 3 --out DIR        (--fake / --dry-run: no model)
    python -m eval.bench.show compare --runs DIR1,DIR2,... --base DIR1 --out CMP --pack <fact pack json>
    python -m eval.bench.judge --requests CMP --model <judge> --tag <name> --tp 2 --rubric-reruns all --max-model-len 8192
    python -m eval.bench.show compare ... (again: merges every CMP/judge/answers_<tag>.jsonl)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEV_SCRIPTS = ROOT / "eval" / "data" / "bench" / "t6_dev_scripts.yaml"
SLOTS = 3  # show lines after each audience turn
PANEL = ("commentator", "guest")
HOST = "host"
FLOOR_ROLES = (HOST, *PANEL)
AUDIENCE = "Audience member"
OPENER_MARK = "open this one differently"  # agent/panel/think_prompt.opener_block (TALKSHOW_OPENER_VARIETY)
PERTURBATIONS = ("generic_swap", "duplicate_argument", "same_opener", "turn_shuffle")

PAIRWISE_SYSTEM = ("You are a careful annotator of talk-show transcripts. Answer exactly what is asked, using only the "
                   "transcripts you are shown.")
PAIRWISE_QUESTION = (
    "Both episodes are the same live talk show on the same topic, with the same audience member saying the same "
    "things at the same points. Which episode is more like a good talk show panel: the panel answers the audience "
    "member, the speakers respond to each other, each line adds something new instead of repeating earlier points, "
    "the points are concrete, the speakers sound like distinct people, and it is lively and natural to listen to? "
    "Answer with a single letter: A or B.")
UPTAKE_QUESTION = ("Does this line respond to what the audience member just said (answer their question or take up "
                   "their point), rather than carry on with something else? Answer Yes or No.")


# ------------------------------------------------------------------------------------------------ scripts, episodes

def load_scripts(path: Path = DEV_SCRIPTS, topics: str = "") -> list[dict]:
    import yaml

    out = yaml.safe_load(Path(path).read_text())["topics"]
    for t in out:
        if not 1 <= len(t.get("audience") or []) <= 3 or not all(isinstance(x, str) and x.strip() for x in t["audience"]):
            raise ValueError(f"{t.get('slug')}: a script needs 1 to 3 audience turns")
    if topics:
        keep = topics.split(",")
        missing = set(keep) - {t["slug"] for t in out}
        if missing:
            raise SystemExit(f"no script for {sorted(missing)}")
        out = [t for t in out if t["slug"] in keep]
    return out


def seed_of(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def new_episode(script: dict, label: str, k: int) -> dict:
    from eval.bench.build import opening_history

    lines = [{"role": h["role"], "speaker": h["speaker"], "text": h["text"], "procedural": True}
             for h in opening_history(script["topic"])]
    return {"id": f"{label}:{script['slug']}:k{k}", "topic_id": script["slug"], "topic": script["topic"],
            "source": "system", "system": label, "k": k, "audience": list(script["audience"]), "lines": lines,
            "floor_state": {}}


def add_audience(ep: dict, text: str) -> None:
    from agent.show.show_history import human_name

    ep["lines"].append({"role": "human", "speaker": human_name(), "text": text})


def prefix_of(ep: dict) -> dict:
    return {"lines": [{"role_id": l["role"], "speaker": l["speaker"], "text": l["text"],
                       "procedural": bool(l.get("procedural"))} for l in ep["lines"]]}


def last_speaker(ep: dict) -> str:
    return next(l["role"] for l in reversed(ep["lines"]) if not l.get("procedural"))


def finish(ep: dict) -> dict:
    ep.pop("floor_state", None)
    ep["scored"] = [i for i, l in enumerate(ep["lines"]) if l["role"] in PANEL]
    ep["audience_idx"] = [i for i, l in enumerate(ep["lines"]) if l["role"] == "human"]
    return ep


# ------------------------------------------------------------------------------------------------ think_action

def _parse(raw: str, interval_text: str) -> Any:
    from agent.interaction.think_action import parse_think_action

    return parse_think_action(raw, floor_state="free", interval_text=interval_text, allow_backchannels=True).action


def run_think_action(system: Any, backend: Any, eps: list[dict], slots: int = SLOTS) -> None:
    """Every episode in lockstep: per slot one batch of think calls, one of controller calls, one of fallbacks.

    A fact-pack arm's pack is set for the whole run (the show system clears TALKSHOW_FACT_PACK when it is built, and
    single-item tasks set it per request; T6 episodes use the show's own pack, keyed by the episode's topic)."""
    own_pack = getattr(system, "own_pack", "") or ""
    saved = os.environ.get("TALKSHOW_FACT_PACK")
    if own_pack:
        os.environ["TALKSHOW_FACT_PACK"] = own_pack
    try:
        _run_think_action(system, backend, eps, slots)
    finally:
        if saved is None:
            os.environ.pop("TALKSHOW_FACT_PACK", None)
        else:
            os.environ["TALKSHOW_FACT_PACK"] = saved


def _run_think_action(system: Any, backend: Any, eps: list[dict], slots: int = SLOTS) -> None:
    from agent.floor.floor_parser import strip_speech_control_tags
    from eval.bench.floor import arbitrate, host_fallback, record_grant
    from eval.bench.systems import Request, strip_think
    from eval.model_only import TEMPERATURE, build_request, planner_messages, postprocess
    from eval.think_action_prompt import think_request
    from agent.panel.select import SAMPLE_TEMPERATURE, select_n

    names = _names()
    n_sel = select_n()  # TALKSHOW_PANEL_SELECT: extra think samples per panelist (agent/panel/select.py)
    n_beats = min(len(e["audience"]) for e in eps)
    for beat in range(n_beats):
        for e in eps:
            add_audience(e, e["audience"][beat])
        plans: dict[str, str] = {}
        if system.planner:  # one plan per audience turn (production planner prompt, temp 0)
            texts = sorted({e["audience"][beat] for e in eps})
            outs = backend.generate([Request("planner", planner_messages(t), 0.0, 160, None, "planner") for t in texts])
            plans = {t: strip_think(o) for t, o in zip(texts, outs)}
        for slot in range(slots):
            # 1. every floor role but the last speaker thinks on the finished turn
            thinks, reqs = [], []
            for e in eps:
                spoke = last_speaker(e)
                for role in FLOOR_ROLES:
                    if role == spoke:
                        continue
                    tr = think_request(prefix_of(e), role, plan=plans.get(e["audience"][beat], ""),
                                       topic_id=e["topic_id"])
                    thinks.append((e, role, tr, 0))
                    reqs.append(Request(f"think:{role}", tr["messages"], tr["temperature"], tr["max_tokens"],
                                        seed_of(e["topic_id"], e["k"], beat, slot, role), e["id"]))
                    for j in range(1, n_sel if role in PANEL else 1):  # same prompt, the selection temperature
                        thinks.append((e, role, tr, j))
                        reqs.append(Request(f"think_extra:{role}", tr["messages"], SAMPLE_TEMPERATURE,
                                            tr["max_tokens"], seed_of(e["topic_id"], e["k"], beat, slot, role, j),
                                            e["id"]))
            outs = backend.generate(reqs)
            actions: dict[str, dict] = defaultdict(dict)
            extras: dict[str, dict] = defaultdict(lambda: defaultdict(list))
            prompts: dict[str, dict] = defaultdict(dict)
            for (e, role, tr, j), raw in zip(thinks, outs):
                if j:
                    extras[e["id"]][role].append(_parse(raw, tr["interval_text"]))
                    continue
                actions[e["id"]][role] = _parse(raw, tr["interval_text"])
                prompts[e["id"]][role] = tr["messages"][-1]["content"]
                e.setdefault("think_prompts", 0)
                e["think_prompts"] += 1
                e.setdefault("opener_block_prompts", 0)
                e["opener_block_prompts"] += OPENER_MARK in tr["messages"][-1]["content"]
            # 2. the production free-floor selection (controller call when no rule picks)
            got: dict[str, dict] = {}
            ctl = []
            for e in eps:
                g = arbitrate(prefix_of(e), None, actions[e["id"]], roles=FLOOR_ROLES, state=e["floor_state"])
                got[e["id"]] = g
                if g["controller"] is not None:
                    c = g["controller"]
                    ctl.append((e, Request("controller", c["messages"], c["temperature"], c["max_tokens"],
                                           seed_of("controller", e["topic_id"], e["k"], beat, slot), e["id"])))
            for (e, _r), raw in zip(ctl, backend.generate([r for _e, r in ctl])):
                got[e["id"]] = arbitrate(prefix_of(e), None, actions[e["id"]], answer=raw, roles=FLOOR_ROLES,
                                         state=e["floor_state"])
                got[e["id"]]["controller_raw"] = raw
            # 3. the selected role's line: its think content, the worker draft, or the host fallback line
            work, host, spoken = [], [], []
            for e in eps:
                g = got[e["id"]]
                out = g["outcome"]
                e["floor_state"] = g["state"]
                info = {"decision": out.decision, "reason_code": out.reason_code, "candidates": g["candidates"],
                        "controller_called": g["controller"] is not None, "beat": beat, "slot": slot}
                role = out.selected_role
                act = actions[e["id"]].get(role)
                if out.decision == "host_fallback":
                    info.update(strategy=out.host_strategy, target=out.host_target)
                    hf = host_fallback(prefix_of(e), e["floor_state"])
                    host.append((e, info, hf))
                elif act is not None and act.decision == "speak" and act.content:
                    spoken.append((e, info, role, act))
                elif role in PANEL:  # selected without a line of its own: the worker draft, as live
                    r = build_request(prefix_of(e), "", role, False, include_human=False, live_fixes=system.live,
                                      plan=plans.get(e["audience"][beat], "") if system.live else "",
                                      topic_id=e["topic_id"])
                    work.append((e, info, role, r))
                else:  # the host selected with no content (not expected): nothing said this slot
                    e["lines"].append({"role": HOST, "speaker": names[HOST], "text": "…", "source": "empty",
                                       "floor": info})
            # the selected think lines: the decision model's pick among the candidates, the punch-up pass (both off by
            # default; agent/panel/select.py), then the line is recorded
            for (e, info, role, _act), (text, extra) in zip(spoken, _finish_think_lines(backend, spoken, extras,
                                                                                         names, beat, slot)):
                line = {"role": role, "speaker": names[role], "text": text or "…", "source": "think", "floor": info}
                if role in prompts[e["id"]]:
                    line["opener_block"] = OPENER_MARK in prompts[e["id"]][role]
                line.update(extra)
                e["lines"].append(line)
                e["floor_state"] = record_grant(e["floor_state"], role)
            w_out = backend.generate([Request("fallback", r["messages"], TEMPERATURE, 512,
                                              seed_of("fallback", e["topic_id"], e["k"], beat, slot), e["id"])
                                      for e, _i, _role, r in work])
            for (e, info, role, r), raw in zip(work, w_out):
                text = postprocess(raw, role, r["panel_roles"], r["data"].show_history)
                e["lines"].append({"role": role, "speaker": names[role], "text": text or "…",
                                   "source": "worker_fallback", "floor": info})
                e["floor_state"] = record_grant(e["floor_state"], role)
            h_out = backend.generate([Request("host", hf["request"]["messages"], hf["request"]["temperature"],
                                              hf["request"]["max_tokens"],
                                              seed_of("host", e["topic_id"], e["k"], beat, slot), e["id"])
                                      for e, _i, hf in host])
            for (e, info, _hf), raw in zip(host, h_out):
                done = host_fallback(prefix_of(e), e["floor_state"], answer=raw)
                e["floor_state"] = record_grant(done["state"], HOST)
                e["lines"].append({"role": HOST, "speaker": names[HOST], "text": done["text"] or "…",
                                   "source": "host_fallback", "floor": info})
            print(f"[think_action] beat {beat} slot {slot}: {len(eps)} episodes, {len(ctl)} controller calls, "
                  f"{len(work)} worker drafts, {len(host)} host fallbacks", flush=True)


def _finish_think_lines(backend: Any, spoken: list[tuple], extras: dict, names: dict[str, str], beat: int,
                        slot: int) -> list[tuple[str, dict]]:
    """(text, extra line fields) per selected think line, as the live ThinkActionInteraction.finish_proposed_reply:
    with TALKSHOW_PANEL_SELECT a panelist with several candidates gets the decision model's pick (one Jev call each,
    run concurrently). With the flag off: the think content, no extra fields."""
    import asyncio

    from agent.adapters.gemma_mm_client import GemmaMMClient
    from agent.floor.floor_parser import strip_speech_control_tags
    from agent.panel.select import candidate_contents, pick_best, recent_transcript, select_n
    from eval.bench.systems import Request
    from eval.think_action_prompt import _data_for

    out: list[list] = [[act.content, {}] for _e, _i, _r, act in spoken]
    data = {}

    def data_of(e: dict) -> Any:
        if e["id"] not in data:
            data[e["id"]] = _data_for(prefix_of(e), None, "", e["topic_id"])
        return data[e["id"]]

    if select_n() > 1:
        todo = []
        for n, (e, _i, role, act) in enumerate(spoken):
            cands = candidate_contents(act, extras[e["id"]].get(role, [])) if role in PANEL else []
            if role in PANEL:
                out[n][1]["candidates"] = [strip_speech_control_tags(c).strip() for c in cands]
            if len(cands) > 1:
                todo.append((n, cands, pick_best(recent_transcript(data_of(e).show_history), role, cands)))

        async def run() -> list:
            return list(await asyncio.gather(*(c for _n, _c, c in todo)))

        for (n, cands, _c), (idx, info) in zip(todo, asyncio.run(run()) if todo else []):
            out[n][0] = cands[idx]
            out[n][1]["select"] = {k: v for k, v in info.items() if k != "select_probs"}
    for n in range(len(out)):
        out[n][0] = strip_speech_control_tags(out[n][0] or "").strip()
    return [(t, x) for t, x in out]


# ------------------------------------------------------------------------------------------------ plain

def run_plain(system: Any, backend: Any, eps: list[dict], slots: int = SLOTS) -> None:
    """No floor of its own: the T2f floor question picks the next panelist, then the PlainLLM prompt answers."""
    from eval.bench.build import persona_cards
    from eval.bench.floor import parse_name, plain_floor_messages
    from eval.bench.systems import Request, plain_messages, plain_postprocess

    names = _names()
    cards = persona_cards(False)

    def obs(e: dict, role: str | None = None) -> dict:
        o = {"topic": e["topic"], "names": names, "personas": cards, "audience_turn": None,
             "history": [{"role": l["role"], "speaker": l["speaker"], "text": l["text"]} for l in e["lines"]]}
        if role:
            o.update(role=role, name=names[role], item_id=e["id"])
        return o

    n_beats = min(len(e["audience"]) for e in eps)
    for beat in range(n_beats):
        for e in eps:
            add_audience(e, e["audience"][beat])
        for slot in range(slots):
            outs = backend.generate([Request("floor", plain_floor_messages(obs(e)), 0.0, 16,
                                             seed_of("floor", e["topic_id"], e["k"], beat, slot), e["id"])
                                     for e in eps])
            picks = []
            for e, raw in zip(eps, outs):
                role = parse_name(raw, names)
                parsed = role is not None
                if role is None:  # unreadable answer: the panelist who spoke less recently
                    spoke = [l["role"] for l in e["lines"] if l["role"] in PANEL]
                    role = "guest" if spoke and spoke[-1] == "commentator" else "commentator"
                picks.append((role, {"decision": "speak", "reason_code": "floor_question" if parsed else
                                     "unparsed_floor_answer", "beat": beat, "slot": slot}))
            outs = backend.generate([Request("reply", plain_messages(obs(e, role)), system.temperature,
                                             system.max_tokens, seed_of(e["topic_id"], e["k"], beat, slot, role),
                                             e["id"]) for e, (role, _i) in zip(eps, picks)])
            for e, (role, info), raw in zip(eps, picks, outs):
                text = plain_postprocess(raw, names[role])
                e["lines"].append({"role": role, "speaker": names[role], "text": text or "…", "source": "plain",
                                   "floor": info})
            print(f"[plain] beat {beat} slot {slot}: {len(eps)} episodes", flush=True)


def _names() -> dict[str, str]:
    from eval.bench.build import names

    return names()


# ------------------------------------------------------------------------------------------------ generate

def generate(args: argparse.Namespace) -> Path:
    from eval.bench.run import git_commit
    from eval.bench.systems import DryRunBackend, FakeBackend, make_system

    scripts = load_scripts(Path(args.scripts), args.topics)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run or args.fake:
        os.environ.setdefault("TALKSHOW_TEXT_TOKENIZER", "word")
        backend = DryRunBackend() if args.dry_run else FakeBackend()
        model = args.model or "none"
    else:
        if not args.model:
            sys.exit("--model is required (or --dry-run / --fake)")
        from eval.bench.systems import VLLMBackend
        from eval.hf_local import local_snapshot

        if args.model.startswith("openai:"):
            from eval.bench.systems import OpenAIBackend

            Path(args.out).mkdir(parents=True, exist_ok=True)
            backend = OpenAIBackend(args.model.split(":", 1)[1], usage_path=str(Path(args.out) / "usage.jsonl"))
        else:
            model_path = local_snapshot(args.model)
            if not (os.environ.get("VLLM_MODEL_PATH") or os.environ.get("VLLM_MODEL")):
                os.environ["VLLM_MODEL_PATH"] = model_path  # the think path splits intervals with the model's tokenizer
            backend = VLLMBackend(model_path, tp=args.tp, max_model_len=args.max_model_len, seed=args.seed)
        model = args.model
    system = make_system(args.system, backend, args.arm)
    label = args.system if args.system == "plain" else f"{args.system}-{args.arm}"
    if args.model and "gemma-4-E4B" not in args.model:  # the same arm on another panel model stays a separate system
        label += "@" + args.model.rstrip("/").split("/")[-1]
    reruns = 1 if args.dry_run else args.reruns
    eps = [new_episode(s, label, k) for k in range(reruns) for s in scripts]
    t0 = time.time()
    (run_plain if args.system == "plain" else run_think_action)(system, backend, eps, args.slots)
    wall = time.time() - t0
    for e in eps:
        finish(e)
        e.update(model=model, arm=None if args.system == "plain" else args.arm)
    from eval.quality_ruler.arms import ARMS

    config = {"system": system.describe(), "label": label, "model": model, "scripts": str(args.scripts),
              "topics": [s["slug"] for s in scripts], "reruns": reruns, "slots": args.slots, "seed": args.seed,
              "arm_flags": ARMS.get(args.arm, {}).get("live_flags", {}) if args.system != "plain" else {},
              "env_flags": {k: v for k, v in sorted(os.environ.items()) if k.startswith("TALKSHOW_")
                            and k not in ("TALKSHOW_ROOT",)},
              "commit": git_commit(), "wall_s": round(wall, 1),
              "mode": "dry_run" if args.dry_run else "fake" if args.fake else "model"}
    (out / "run.json").write_text(json.dumps(config, indent=1))
    if args.dry_run:
        with (out / "prompts.jsonl").open("w") as f:
            for r in backend.records:
                f.write(json.dumps({"tag": r.tag, "kind": r.kind, "temperature": r.temperature,
                                    "max_tokens": r.max_tokens, "seed": r.seed, "messages": r.messages},
                                   ensure_ascii=False) + "\n")
    from eval.quality_ruler.episodes import save

    save(out / "episodes.jsonl", eps)
    print(f"{len(eps)} episodes ({len(scripts)} topics x {reruns} reruns) in {wall:.0f}s -> {out}/episodes.jsonl")
    return out


# ------------------------------------------------------------------------------------------------ counted metrics

def first_word(t: str) -> str:
    w = re.findall(r"[A-Za-z']+", (t or "").lower())
    return w[0] if w else ""


def _bigrams(t: str) -> set:
    from eval.quality_ruler.auto import content

    c = content(t)
    return set(zip(c, c[1:]))


def counted(ep: dict) -> dict[str, float]:
    """Counted measures for one episode (no judge). Higher = more repetition for the opener and self-repeat keys."""
    from eval.quality_ruler.auto import metrics

    L = ep["lines"]
    panel = [i for i in ep["scored"] if L[i]["role"] in PANEL]
    m: dict[str, float] = {}
    same, tops = [], []
    for role in PANEL:
        mine = [first_word(L[i]["text"]) for i in panel if L[i]["role"] == role]
        same += [a == b for a, b in zip(mine, mine[1:])]
        if len(mine) >= 2:
            tops.append(Counter(mine).most_common(1)[0][1] / len(mine))
    m["opener.same_first_word_as_own_previous"] = sum(same) / len(same) if same else float("nan")
    m["opener.top_first_word_share"] = sum(tops) / len(tops) if tops else float("nan")
    adj = [first_word(L[a]["text"]) == first_word(L[b]["text"]) for a, b in zip(panel, panel[1:])]
    m["opener.same_first_word_adjacent_lines"] = sum(adj) / len(adj) if adj else float("nan")
    rep = []
    for n, i in enumerate(panel):
        own = [L[j]["text"] for j in panel[:n] if L[j]["role"] == L[i]["role"]]
        if own:
            b = _bigrams(L[i]["text"])
            rep.append(max(len(b & _bigrams(o)) / max(len(b | _bigrams(o)), 1) for o in own))
    m["self_repeat.max_bigram_jaccard"] = sum(rep) / len(rep) if rep else float("nan")
    m["self_repeat.share_lines_over_0.2"] = sum(r > 0.2 for r in rep) / len(rep) if rep else float("nan")
    auto = metrics([ep], Path("/nonexistent"))[ep["id"]]  # no embeddings: lexical measures only
    for k in ("repetition.distinct2", "repetition.1-repeated_opener", "interaction.lexical_link",
              "flow.human_answered", "length.words_per_line"):
        m[k] = auto[k]
    show = [l for l in L if not l.get("procedural") and l["role"] != "human"]
    m["floor.panel_lines"] = len(panel)
    m["floor.host_lines"] = sum(l["role"] == HOST for l in show)
    m["floor.worker_fallback_lines"] = sum(l.get("source") == "worker_fallback" for l in show)
    m["floor.controller_share"] = (sum(bool(l.get("floor", {}).get("controller_called")) for l in show)
                                   / len(show)) if show else float("nan")
    roles = [L[i]["role"] for i in panel]
    m["floor.ryan_share"] = roles.count("commentator") / len(roles) if roles else float("nan")
    return m


def counted_table(runs: dict[str, list[dict]], pack: dict | None) -> dict[str, dict[str, float]]:
    from eval.quality_ruler.specifics import score_episodes

    out = {}
    for label, eps in runs.items():
        per = [counted(e) for e in eps]
        keys = per[0].keys() if per else []
        row = {}
        for k in keys:
            v = [p[k] for p in per if p[k] == p[k]]
            row[k] = sum(v) / len(v) if v else float("nan")
        if pack is not None:
            sp = score_episodes(eps, pack)
            row["specifics.unsupported_per100w"] = sp["unsupported_per100w"]
            row["specifics.share_lines_unsupported"] = sp["share_lines_unsupported"]
            row["specifics.share_supported"] = sp["share_supported"]
        row["episodes"] = len(eps)
        out[label] = row
    return out


# ------------------------------------------------------------------------------------------------ judge requests

def episode_text(ep: dict) -> str:
    out = []
    for l in ep["lines"]:
        if l.get("procedural"):
            continue
        who = AUDIENCE if l["role"] == "human" else l["speaker"]
        out.append(f"{who}: {l['text']}")
    return "\n".join(out)


def pairwise_messages(topic: str, a: dict, b: dict) -> list[dict]:
    user = (f"Topic: {topic}\n\nEpisode A:\n{episode_text(a)}\n\nEpisode B:\n{episode_text(b)}\n\n{PAIRWISE_QUESTION}")
    return [{"role": "system", "content": PAIRWISE_SYSTEM}, {"role": "user", "content": user}]


def uptake_messages(ep: dict, h: int, i: int) -> list[dict]:
    from eval.quality_ruler.items import JUDGE_SYSTEM

    ctx = [l for l in ep["lines"][max(0, h - 6): h + 1] if not l.get("procedural")]
    ctx_text = "\n".join(f"{AUDIENCE if l['role'] == 'human' else l['speaker']}: {l['text']}" for l in ctx)
    between = [l for l in ep["lines"][h + 1: i] if not l.get("procedural")]
    if between:  # e.g. the host spoke first
        ctx_text += "\n" + "\n".join(f"{l['speaker']}: {l['text']}" for l in between)
    L = ep["lines"][i]
    user = (f"Topic: {ep['topic']}\n\nTranscript so far:\n{ctx_text}\n\nLine to judge - {L['speaker']}: {L['text']}"
            f"\n\nQuestion: {UPTAKE_QUESTION}")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def uptake_requests(runs: dict[str, list[dict]]) -> list[dict]:
    out = []
    for label, eps in runs.items():
        for e in eps:
            for n, h in enumerate(e["audience_idx"]):
                nxt = next((i for i in range(h + 1, len(e["lines"])) if e["lines"][i]["role"] == "human"),
                           len(e["lines"]))
                i = next((j for j in range(h + 1, nxt) if e["lines"][j]["role"] in PANEL), None)
                if i is None:
                    continue
                out.append({"id": f"up|{e['id']}|a{n}", "task": "T6", "system": label, "episode": e["id"],
                            "topic_id": e["topic_id"], "rerun": e["k"], "audience_turn": n, "line": i,
                            "messages": uptake_messages(e, h, i)})
    return out


def pairwise_requests(runs: dict[str, list[dict]], base: str | None, perturbations: tuple[str, ...] = PERTURBATIONS,
                      seed: int = 0) -> list[dict]:
    """System pairs (same topic and rerun, both orders) and, for the base system, each perturbed episode against its
    original (both orders)."""
    from eval.quality_ruler.perturb import perturb

    out = []
    labels = sorted(runs)
    by = {lab: {(e["topic_id"], e["k"]): e for e in runs[lab]} for lab in labels}
    for x in range(len(labels)):
        for y in range(x + 1, len(labels)):
            sa, sb = labels[x], labels[y]
            for key in sorted(set(by[sa]) & set(by[sb])):
                ea, eb = by[sa][key], by[sb][key]
                for order, (a, b) in (("ab", (ea, eb)), ("ba", (eb, ea))):
                    out.append({"id": f"pw|{key[0]}|k{key[1]}|{sa}|{sb}|{order}", "task": "T6", "kind": "systems",
                                "topic_id": key[0], "rerun": key[1], "sys_a": a["system"], "sys_b": b["system"],
                                "order": order, "words_a": _words(a), "words_b": _words(b),
                                "messages": pairwise_messages(ea["topic"], a, b)})
    if base:
        rng = random.Random(seed)
        for e in runs[base]:
            for kind in perturbations:
                p = perturb(e, kind, rng)
                if p is None:
                    continue
                for order, (a, b) in (("ab", (e, p)), ("ba", (p, e))):
                    out.append({"id": f"pt|{e['id']}|{kind}|{order}", "task": "T6", "kind": "perturbation",
                                "perturbation": kind, "topic_id": e["topic_id"], "rerun": e["k"], "episode": e["id"],
                                "orig_is": "A" if order == "ab" else "B", "order": order,
                                "messages": pairwise_messages(e["topic"], a, b)})
    return out


def _words(ep: dict) -> int:
    return sum(len(re.findall(r"[A-Za-z0-9$%']+", ep["lines"][i]["text"])) for i in ep["scored"])


# ------------------------------------------------------------------------------------------------ judge results

def load_answers(cmp_dir: Path) -> dict[str, dict[str, dict]]:
    """tag -> request id -> answer row."""
    out = {}
    for f in sorted((cmp_dir / "judge").glob("answers_*.jsonl")):
        if f.name.endswith(".partial.jsonl"):
            continue
        tag = f.name[len("answers_"):-len(".jsonl")]
        out[tag] = {r["id"]: r for r in (json.loads(x) for x in f.read_text().splitlines() if x.strip())}
    return out


def judge_results(answers: dict[str, dict[str, dict]], boot: int = 1000) -> dict:
    from eval.quality_ruler.stats import bt_bootstrap, krippendorff_alpha

    res: dict[str, Any] = {"judges": sorted(answers)}
    tags = sorted(answers)
    # pairwise systems: s = P(first-named system wins), averaged over both orders
    pair_s: dict[str, dict[tuple, float]] = {}
    order_ok: dict[str, list[bool]] = {}
    for tag in tags:
        acc: dict[tuple, list[float]] = defaultdict(list)
        pref: dict[tuple, list[bool]] = defaultdict(list)
        for r in answers[tag].values():
            if r.get("request") != "pairwise" or r.get("kind") != "systems":
                continue
            _p, topic, k, sa, sb, order = r["id"].split("|")
            win = r["p_a"] if order == "ab" else 1 - r["p_a"]
            acc[(topic, k, sa, sb)].append(win)
            pref[(topic, k, sa, sb)].append(win > 0.5)
        pair_s[tag] = {key: sum(v) / len(v) for key, v in acc.items()}
        order_ok[tag] = [len(set(v)) == 1 for v in pref.values() if len(v) == 2]
    if any(pair_s.values()):
        res["bt"] = {}
        for name, use in [*((t, [t]) for t in tags), ("pooled", tags)]:
            keys = set.intersection(*(set(pair_s[t]) for t in use)) if use else set()
            rows = [{"a": key[2], "b": key[3], "topic": key[0],
                     "s": sum(pair_s[t][key] for t in use) / len(use)} for key in sorted(keys)]
            if not rows:
                continue
            point, ci = bt_bootstrap(rows, "a", "b", "topic", boot=boot)
            wins: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                wins[r["a"]].append(r["s"])
                wins[r["b"]].append(1 - r["s"])
            res["bt"][name] = {s: {"strength": round(point[s], 3), "ci95": [round(ci[s][0], 3), round(ci[s][1], 3)],
                                   "win_rate": round(sum(wins[s]) / len(wins[s]), 3)} for s in sorted(point)}
        res["order_consistency"] = {t: round(sum(v) / len(v), 3) for t, v in order_ok.items() if v}
        if len(tags) >= 2:
            common = set.intersection(*(set(pair_s[t]) for t in tags))
            units = {key: [pair_s[t][key] > 0.5 for t in tags] for key in common}
            res["agreement_pairwise_alpha"] = round(krippendorff_alpha(units), 3) if units else None
            res["agreement_pairwise_n"] = len(units)
    # perturbation gate: P(original wins) averaged over both orders, per kind
    gate: dict[str, dict] = {}
    for tag in tags:
        acc = defaultdict(list)
        for r in answers[tag].values():
            if r.get("request") != "pairwise" or r.get("kind") != "perturbation":
                continue
            win = r["p_a"] if r["orig_is"] == "A" else 1 - r["p_a"]
            acc[(r["episode"], r["perturbation"])].append(win)
        by_kind: dict[str, list[float]] = defaultdict(list)
        for (ep, kind), v in acc.items():
            by_kind[kind].append(sum(v) / len(v))
        if by_kind:
            gate[tag] = {kind: {"orig_wins": round(sum(x > 0.5 for x in v) / len(v), 3), "n": len(v),
                                "mean_p_orig": round(sum(v) / len(v), 3)} for kind, v in sorted(by_kind.items())}
    if gate:
        res["perturbation_gate"] = gate
    # audience uptake: share of first panel lines after an audience turn judged as responding to it
    up: dict[str, dict] = {}
    up_units: dict[str, list] = defaultdict(list)
    for tag in tags:
        acc = defaultdict(list)
        for r in answers[tag].values():
            if r.get("request") != "uptake":
                continue
            acc[r["system"]].append(r["p_yes"] > 0.5)
            up_units[r["id"]].append(r["p_yes"] > 0.5)
        if acc:
            up[tag] = {s: {"yes": round(sum(v) / len(v), 3), "n": len(v)} for s, v in sorted(acc.items())}
    if up:
        res["uptake"] = up
        if len(tags) >= 2:
            units = {k: v for k, v in up_units.items() if len(v) == len(tags)}
            res["agreement_uptake_alpha"] = round(krippendorff_alpha(units), 3) if units else None
    return res


# ------------------------------------------------------------------------------------------------ compare

def compare(args: argparse.Namespace) -> dict:
    from eval.quality_ruler.episodes import load

    runs: dict[str, list[dict]] = {}
    configs = {}
    for d in (Path(x) for x in args.runs.split(",") if x):
        eps = load(d / "episodes.jsonl")
        if not eps:
            continue
        runs[eps[0]["system"]] = eps
        configs[eps[0]["system"]] = json.loads((d / "run.json").read_text()) if (d / "run.json").is_file() else {}
    base = None
    if args.base:
        base = json.loads((Path(args.base) / "run.json").read_text())["label"]
    out = Path(args.out)
    (out / "judge").mkdir(parents=True, exist_ok=True)
    pack = None
    pack_path = args.pack or os.environ.get("TALKSHOW_FACT_PACK", "")
    if pack_path:
        from eval.quality_ruler.specifics import load_pack

        pack = load_pack(pack_path)
    res: dict[str, Any] = {"systems": sorted(runs), "counted": counted_table(runs, pack),
                           "flags": {s: c.get("arm_flags", {}) for s, c in configs.items()},
                           "differ": differ(runs)}
    for name, reqs in (("pairwise", pairwise_requests(runs, base, seed=args.seed)), ("uptake", uptake_requests(runs))):
        path = out / "judge" / f"{name}.jsonl"
        if not path.is_file() or args.rewrite_requests:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs))
        res[f"n_{name}_requests"] = len(reqs)
    answers = load_answers(out)
    if answers:
        res["judge"] = judge_results(answers, boot=args.boot)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "results.md").write_text(report(res))
    print(report(res))
    return res


def differ(runs: dict[str, list[dict]]) -> dict:
    """Check that systems really differ: per system the share of think prompts with the opener-variety block, and per
    pair of systems the share of show lines identical at the same (topic, rerun, position)."""
    out: dict[str, Any] = {"opener_block_prompts": {}, "identical_lines": {}}
    for lab, eps in runs.items():
        n = sum(e.get("think_prompts", 0) for e in eps)
        out["opener_block_prompts"][lab] = (round(sum(e.get("opener_block_prompts", 0) for e in eps) / n, 3)
                                            if n else None)
    labels = sorted(runs)
    for x in range(len(labels)):
        for y in range(x + 1, len(labels)):
            a = {(e["topic_id"], e["k"]): e for e in runs[labels[x]]}
            b = {(e["topic_id"], e["k"]): e for e in runs[labels[y]]}
            same = tot = 0
            for key in set(a) & set(b):
                la = [l["text"] for l in a[key]["lines"] if l["role"] in FLOOR_ROLES and not l.get("procedural")]
                lb = [l["text"] for l in b[key]["lines"] if l["role"] in FLOOR_ROLES and not l.get("procedural")]
                same += sum(p == q for p, q in zip(la, lb))
                tot += min(len(la), len(lb))
            out["identical_lines"][f"{labels[x]} vs {labels[y]}"] = round(same / tot, 3) if tot else None
    return out


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def report(res: dict) -> str:
    systems = res["systems"]
    keys = list(next(iter(res["counted"].values())).keys()) if res["counted"] else []
    lines = ["# T6 full show: results", "", "## Counted (no judge), mean per episode", "",
             "| measure | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(_fmt(res["counted"][s].get(k)) for s in systems) + " |")
    lines += ["", "## Do the systems differ?", "", f"opener-variety block in think prompts: "
              f"{json.dumps(res['differ']['opener_block_prompts'])}",
              f"identical show lines at the same position: {json.dumps(res['differ']['identical_lines'])}", "",
              f"requests: {res.get('n_pairwise_requests')} pairwise, {res.get('n_uptake_requests')} uptake"]
    j = res.get("judge")
    if j:
        lines += ["", f"## Judges: {', '.join(j['judges'])}"]
        for name, bt in j.get("bt", {}).items():
            lines += ["", f"Bradley-Terry ({name}): strength [95% CI over topics], win rate", ""]
            for s, v in sorted(bt.items(), key=lambda kv: -kv[1]["strength"]):
                lines.append(f"- {s}: {v['strength']} [{v['ci95'][0]}, {v['ci95'][1]}], win rate {v['win_rate']}")
        if "order_consistency" in j:
            lines.append(f"\nsame winner in both A/B orders: {json.dumps(j['order_consistency'])}")
        if "agreement_pairwise_alpha" in j:
            lines.append(f"judge agreement, pairwise winner: alpha {j['agreement_pairwise_alpha']} "
                         f"(n={j['agreement_pairwise_n']})")
        if "perturbation_gate" in j:
            lines += ["", "Perturbation gate (share of episodes where the original beats its perturbed copy):"]
            for tag, g in j["perturbation_gate"].items():
                lines.append(f"- {tag}: " + ", ".join(f"{k} {v['orig_wins']} (n={v['n']})" for k, v in g.items()))
        if "uptake" in j:
            lines += ["", "Audience uptake (share yes):"]
            for tag, u in j["uptake"].items():
                lines.append(f"- {tag}: " + ", ".join(f"{s} {v['yes']} (n={v['n']})" for s, v in u.items()))
            if j.get("agreement_uptake_alpha") is not None:
                lines.append(f"judge agreement, uptake: alpha {j['agreement_uptake_alpha']}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="run one system on the scripts")
    g.add_argument("--system", choices=("plain", "think_action"), required=True)
    g.add_argument("--arm", default="production")
    g.add_argument("--model", default="")
    g.add_argument("--scripts", default=str(DEV_SCRIPTS))
    g.add_argument("--topics", default="", help="comma list of script slugs (default all)")
    g.add_argument("--reruns", type=int, default=3)
    g.add_argument("--slots", type=int, default=SLOTS, help="show lines after each audience turn")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--tp", type=int, default=1)
    g.add_argument("--max-model-len", type=int, default=8192)
    g.add_argument("--out", required=True)
    mode = g.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="no model: placeholder lines, every prompt recorded")
    mode.add_argument("--fake", action="store_true", help="no model: placeholder lines")
    c = sub.add_parser("compare", help="counted metrics, judge requests, judge results")
    c.add_argument("--runs", required=True, help="comma list of generate output dirs")
    c.add_argument("--base", default="", help="the run whose episodes are perturbed for the gate")
    c.add_argument("--out", required=True)
    c.add_argument("--pack", default="", help="fact pack for unsupported specifics (default TALKSHOW_FACT_PACK)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--boot", type=int, default=1000)
    c.add_argument("--rewrite-requests", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        generate(args)
    else:
        compare(args)


if __name__ == "__main__":
    from eval.bench.proc import main_exit

    main_exit(main)
