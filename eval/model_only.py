#!/usr/bin/env python3
"""Listening test without the live stack: the same replayed prefix and edited human turn, model only.

No show graph, floor control or host reply: the addressed panelist answers the human turn directly, with exactly the
prompt the panel worker would build (agent/multi_agent/worker.py: panelist_system_prompt + panel_speech_prompt +
speaker-perspective history), at production sampling (temperature 0.45, max_tokens 512), and the same post-processing
(parse_host_speech -> strip_speech_control_tags -> normalize).

    python -m eval.model_only trace [--out DIR]         # CPU: dump one full prompt per condition
    python -m eval.model_only generate -k 5 [--out DIR] # GPU: Gemma via vLLM offline

`generate` writes <out>/lines.jsonl in eval/responsiveness.py format, so the same embed / logprob / report stages and
the judge apply (deploy/slurm-eval-listen.sh).

Speaker: BOTH panelists answer every condition (each as if it had the first line), so every comparison holds the speaker
fixed, e.g. Amy addressed (`addressee`) vs Amy not addressed (`orig`, which addresses Ryan). Compare with the
guest_first / commentator_first rows of the responsiveness report (pos1 mixes speakers here and is not meaningful).
Differences from the live system, by design: no host reply between the human turn and the panelist, mood state =
prefix state (no appraisal of the edited turn).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS = ROOT / "eval" / "data" / "minimal_pairs" / "pairs.yaml"
PREFIX_DIR = ROOT / "eval" / "data" / "minimal_pairs" / "prefix"
CONDITIONS = ("orig", "paraphrase", "negate", "number", "entity", "addressee", "persona_swap")
TEMPERATURE = 0.45  # GemmaMMClient.complete_text default (panel worker passes none)


def _data_for(prefix: dict, human_text: str, include_human: bool = True):
    """A TalkShowData carrying only what the prompt builders read."""
    from agent.config import load_scenario
    from agent.data import TalkShowData
    from agent.session.ghost_replay import apply_prefix

    data = TalkShowData(scenario=load_scenario(), runtime=None)  # type: ignore[arg-type]
    data.room_name = "model-only"
    data.turn_log = None
    apply_prefix(data, {k: v for k, v in prefix.items() if k != "persona_swap"})
    if include_human:
        data.show_history.append("human", human_text)
    return data


def build_minimal_request(prefix: dict, human_text: str, role: str, persona_swap: bool) -> dict:
    """Scaffold ablation: same persona text and transcript, none of the panel mechanics
    (no FROZEN-floor / "respond to the latest speaker" instructions, no dialogue hints,
    no mood block). Isolates the model from the system's prompt scaffold."""
    from agent.config import load_persona_name, set_persona_swap
    from agent.panel.panel_prompts import panelist_personality_block

    set_persona_swap(persona_swap)
    data = _data_for(prefix, human_text)
    name = load_persona_name(role)
    system = panelist_personality_block(role)
    hist = data.show_history.prior_messages(perspective=role)
    prompt = (
        f'The human guest just said: "{human_text}"\n'
        f"It is your turn, {name}. Reply with your spoken line only, in this format:\n"
        "[reply]: <your spoken lines>"
    )
    messages = [{"role": "system", "content": system}, *[dict(m) for m in hist]]
    if messages[-1]["role"] == "user":
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{prompt}"
    else:
        messages.append({"role": "user", "content": prompt})
    set_persona_swap(False)
    controller_roles = ["commentator", "guest"]
    return {"messages": messages, "panel_roles": controller_roles, "data": data}


def build_request(prefix: dict, human_text: str, role: str, persona_swap: bool, include_human: bool = True,
                  dialogue_pick: str | None = None, dialogue_library: str | None = None) -> dict:
    """Production panel prompt. dialogue_pick / dialogue_library override the scenario's move catalogue (generation
    knobs; defaults = production). include_human=False: no new human line (the prefix already ends where the panelist
    speaks)."""
    from agent.config import load_persona_name, set_persona_swap
    from agent.floor import TurnController
    from agent.panel.dialogue_library import format_dialogue_hint
    from agent.panel.panel_prompts import panel_speech_prompt, panelist_system_prompt

    set_persona_swap(persona_swap)
    data = _data_for(prefix, human_text, include_human)
    controller = TurnController(data.scenario, data)
    panel_roles = controller.panel_speaker_roles()
    dialogue_hint = ""
    dlg = data.scenario.dialogue
    if dlg and dlg.library:
        dialogue_hint, _ = format_dialogue_hint(
            dialogue_library or dlg.library, role=role, pick=dialogue_pick or dlg.pick,
            index=data.dialogue_example_index
        )
    prompt = panel_speech_prompt(
        role=role,
        name=load_persona_name(role),
        host_name=load_persona_name("host"),
        scenario=data.scenario,
        panel_roles=panel_roles,
        history=data.show_history,
        latest_human=data.show_history.latest_human_text(),
        data=data,
    )
    system = panelist_system_prompt(
        role,
        scenario=data.scenario,
        panel_roles=panel_roles,
        history=data.show_history,
        dialogue_hint=dialogue_hint,
        data=data,
    )
    hist = data.show_history.prior_messages(perspective=role)
    # GemmaMMClient._build_messages: fold the instruction into a trailing user turn.
    messages = [{"role": "system", "content": system}, *[dict(m) for m in hist]]
    if messages[-1]["role"] == "user":
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{prompt}"
    else:
        messages.append({"role": "user", "content": prompt})
    set_persona_swap(False)
    return {"messages": messages, "panel_roles": panel_roles, "data": data}


# ---- planner prompt (also used by the `lite_plan` generation knob, eval/quality_ruler/arms.py)

PLANNER_SYSTEM = "You extract the concrete commitments in one line a guest said on a talk show. Be literal and short."


def planner_messages(human_text: str) -> list[dict]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": (
            f'Guest line: "{human_text}"\n\n'
            "List up to 4 concrete items the guest committed to: every figure (with its unit), named people/places/"
            "organisations, the stance they took, and who they addressed. One item per line, starting with \"- \". "
            "No commentary."
        )},
    ]


def add_plan(messages: list[dict], plan: str, lite: bool = False) -> list[dict]:
    """Show the panelist the extracted commitments and make it say which one it engages and how.

    lite=True: same plan, but no hidden [engages] output line (the extra ~120 decoded characters were most of the
    live latency cost)."""
    if lite:
        block = (
            "\n\nThe human guest's concrete points (extracted from their latest line):\n" + plan.strip() + "\n"
            "Make your [reply] engage the item that matters most for your argument explicitly, and let it change what you argue."
        )
    else:
        block = (
            "\n\nThe human guest's concrete points (extracted from their latest line):\n" + plan.strip() + "\n"
            "Before [reply], output one line:\n[engages]: <the item you engage> — <how it changes your argument>\n"
            "Then make your [reply] engage that item explicitly."
        )
    out = [dict(m) for m in messages]
    out[-1]["content"] = out[-1]["content"] + block
    return out


def postprocess(raw: str, role: str, panel_roles: list[str], history) -> str:
    from agent.adapters.response_parser import parse_host_speech
    from agent.floor.floor_parser import strip_speech_control_tags
    from agent.panel.panel_speech_normalize import normalize_panelist_speech

    import re as _re

    raw = _re.sub(r"(?im)^\s*\[engages\]\s*:.*$", "", raw)  # the planning line is never spoken
    speech = strip_speech_control_tags(parse_host_speech(raw).reply.strip())
    return normalize_panelist_speech(role, speech, panel_roles=panel_roles, history=history)


def _items(topics_filter: str = "", roles: tuple[str, ...] = ("commentator", "guest"),
           conditions: tuple[str, ...] = CONDITIONS, pairs: Path = PAIRS):
    import yaml

    topics = yaml.safe_load(Path(pairs).read_text())["topics"]
    keep = set(topics_filter.split(",")) if topics_filter else None
    for t in topics:
        if keep and t["slug"] not in keep:
            continue
        pf = PREFIX_DIR / f"{t['slug']}.json"
        if not pf.is_file():
            print(f"skip {t['slug']}: no prefix", file=sys.stderr)
            continue
        prefix = json.loads(pf.read_text())
        for cond in conditions:
            text = t["turn2"]["orig" if cond == "persona_swap" else cond]
            for role in roles:
                yield t, prefix, cond, text, role


def cmd_trace(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = args.topic
    builder = build_minimal_request if args.scaffold == "minimal" else build_request
    conds = tuple(args.conditions.split(",")) if args.conditions else CONDITIONS
    for t, prefix, cond, text, role in _items(slug, roles=("guest",) if args.guest else ("commentator",),
                                              conditions=conds, pairs=args.pairs):
        req = builder(prefix, text, role, cond == "persona_swap")
        path = out / f"prompt_{slug}__{cond}__{role}.txt"
        with path.open("w") as f:
            f.write(f"# topic={slug} condition={cond} speaker={role}\n")
            for m in req["messages"]:
                f.write(f"\n===== {m['role']} =====\n{m['content']}\n")
        n_human = sum(m["content"].count(text) for m in req["messages"])
        print(f"{cond:13s} speaker={role:11s} messages={len(req['messages'])} "
              f"chars={sum(len(m['content']) for m in req['messages'])} edited-turn verbatim ×{n_human} → {path.name}")


def cmd_generate(args: argparse.Namespace) -> None:
    import re

    from transformers import AutoTokenizer  # noqa: F401  (fail early if env is broken)
    from vllm import LLM, SamplingParams

    scaffolds = ("system", "minimal") if args.scaffold == "both" else (args.scaffold,)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.88,
        seed=args.seed,
        # tp>1 hung at worker init on some nodes (custom all-reduce / P2P); plain NCCL is enough here.
        disable_custom_all_reduce=args.tp > 1,
    )
    # --arm applies a set of generation knobs (eval/quality_ruler/arms.py) to this prompt; default = production.
    arm = {}
    if args.arm:
        import hashlib

        from eval.quality_ruler.arms import ARMS, BANNED_OPENERS, ban_text, opener, opener_plan_text, rerank_score

        arm = ARMS[args.arm]
    n_cand = int(arm.get("best_of", 1))
    params = SamplingParams(temperature=arm.get("temperature", TEMPERATURE), min_p=arm.get("min_p", 0.0),
                            max_tokens=512, seed=None, n=n_cand,
                            **({"bad_words": [v for w in BANNED_OPENERS for v in (w, " " + w)]}
                               if arm.get("ban_openers") else {}))
    # Qwen3-style templates: no hidden reasoning — the production client gets plain text.
    chat_kwargs = {"enable_thinking": False} if "qwen3" in args.model.lower() else None

    def chat(convs, sp):
        return llm.chat(convs, sp, chat_template_kwargs=chat_kwargs) if chat_kwargs else llm.chat(convs, sp)

    for scaffold in scaffolds:
        builder = build_minimal_request if scaffold == "minimal" else build_request
        conds = tuple(args.conditions.split(",")) if args.conditions else CONDITIONS
        items, convs = [], []
        for t, prefix, cond, text, role in _items(args.topics, conditions=conds, pairs=args.pairs):
            if arm:
                req = build_request(prefix, text, role, cond == "persona_swap",
                                    dialogue_pick=arm.get("dialogue_pick"), dialogue_library=arm.get("dialogue_library"))
            else:
                req = builder(prefix, text, role, cond == "persona_swap")
            msgs = req["messages"]
            if arm:
                earlier = [l["text"] for l in prefix["lines"] if l.get("role_id") in ("commentator", "guest")]
                req["earlier"] = earlier
                extra = ([arm["instruction"]] if arm.get("instruction") else []) + (
                    [ban_text([opener(x) for x in earlier])] if arm.get("opener_ban") else [])
                if extra:
                    msgs = [dict(m) for m in msgs]
                    msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + "\n".join(extra)
            for k in range(args.k):
                m_k = msgs
                if arm.get("opener_plan"):
                    # Same plan for every condition of a (topic, k, speaker): the minimal pair differs only in the edit.
                    seed = int(hashlib.sha256(f"plan:{t['slug']}:{k}:{role}".encode()).hexdigest()[:8], 16)
                    m_k = [dict(m) for m in msgs]
                    m_k[-1]["content"] += "\n\n" + opener_plan_text(seed, role if arm["opener_plan"] == "persona" else None)
                items.append((t, cond, k, role, text, req))
                convs.append(m_k)

        outs = chat(convs, params)
        out = Path(args.out) / scaffold if args.scaffold == "both" else Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        n_empty = 0
        tag = args.model.split("/")[-1]
        name = "lines.jsonl"
        with (out / name).open("w") as f, (out / "raw.jsonl").open("w") as fr:
            for (t, cond, k, role, text, req), o in zip(items, outs):
                cands = []
                for c_i, comp in enumerate(o.outputs):
                    raw = re.sub(r"<think>.*?</think>", "", comp.text, flags=re.S).strip()
                    line = postprocess(raw, role, req["panel_roles"], req["data"].show_history)
                    fr.write(json.dumps({"slug": t["slug"], "cond": cond, "k": k, "role": role, "cand": c_i, "raw": raw}) + "\n")
                    if line:
                        cands.append(line)
                if not cands:
                    n_empty += 1
                    continue
                if arm and len(cands) > 1:  # best-of-N: the same judge-free rerank as the episode generator
                    used = {opener(x) for x in req["earlier"]}
                    cands.sort(key=lambda c: -rerank_score(c, req["earlier"], used, arm.get("opener_penalty", 1.0)))
                row = {
                    "slug": t["slug"], "topic": t["topic"], "cond": cond, "k": k, "pos": 1,
                    "role": role, "text": cands[0], "heard": text, "between": [],
                    "log": f"{tag}/{scaffold}/{t['slug']}/{cond}/{role}/{k}",
                }
                if n_cand > 1:
                    row["cands"] = cands
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{args.model} | {scaffold} | {args.arm or 'production'}] generated {len(items)} items "
              f"({n_empty} empty after post-processing) → {out}/{name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "responsiveness-model"))
    ap.add_argument("--scaffold", choices=("system", "minimal", "both"), default="system",
                    help="system = the panel worker's prompt; minimal = persona + transcript only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("trace", help="write one full prompt per condition (CPU)")
    p.add_argument("--topic", default="nuclear-power")
    p.add_argument("--guest", action="store_true", help="trace Amy's prompt instead of Ryan's")
    p.add_argument("--conditions", default="")
    p.add_argument("--pairs", default=str(PAIRS))
    p.set_defaults(fn=cmd_trace)
    p = sub.add_parser("generate", help="generate replies -> lines.jsonl (GPU)")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--topics", default="")
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--conditions", default="", help="comma list of pairs.yaml turn2 keys (default: the 7 core conditions)")
    p.add_argument("--pairs", default=str(PAIRS), help="pairs file (default: the 30 hand-written topics)")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--arm", default="", help="generation knobs from eval/quality_ruler/arms.py ARMS; default = production")
    p.set_defaults(fn=cmd_generate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
