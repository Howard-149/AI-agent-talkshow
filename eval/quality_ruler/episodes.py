"""Episodes for the quality ruler: one JSON object per episode, `lines` = [{role, speaker, text}] in show order.

role ∈ host | commentator (Ryan) | guest (Amy) | human. A *beat* = the panel lines between two human/host-topic turns.

Sources
- system   : model-only multi-beat episodes on the public test motions, generated with the production panel prompts
             (eval/model_only.build_request) by `generate` (GPU). Host lines are the production tee-up templates
             (agent/floor/host_lines._INTRO_SPEAKER), not LLM host replies (a simplification, stated in the README).
             `--path think_action`: each panel line is the panelist's Think & Action think `content` instead
             (eval/think_action_prompt.py), with the worker prompt as fallback when the think does not choose speak.
- anchor   : real IQ2 panel exchanges on the same motions (eval/data/public_set/build/anchors*.jsonl).
- live     : live ghost replay sessions (live_episodes), with real LLM host replies.
- perturbed: known-bad versions of system/anchor/live beats (perturb.py).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT / "eval" / "data" / "public_set"
PANEL = ("commentator", "guest")
NAMES = {"commentator": "Ryan", "guest": "Amy", "host": "Lessac", "human": "Human guest"}


def model_name(m: str) -> str:
    """Readable model name, also for a resolved local snapshot path (…/models--org--name/snapshots/<hash>)."""
    import re as _re

    hit = _re.search(r"models--[^/]+--([^/]+)/snapshots", m)
    return hit.group(1) if hit else m.rstrip("/").split("/")[-1]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def save(path: Path, eps: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps))


def panel_idx(ep: dict) -> list[int]:
    return [i for i, l in enumerate(ep["lines"]) if l["role"] in PANEL]


def beats(ep: dict) -> list[list[int]]:
    """Indices of panel lines grouped into beats (a beat ends at a human line)."""
    out, cur = [], []
    for i, l in enumerate(ep["lines"]):
        if l["role"] == "human" and cur:
            out.append(cur)
            cur = []
        elif l["role"] in PANEL:
            cur.append(i)
    if cur:
        out.append(cur)
    return out


def beat_episode(ep: dict, b: int, tag: str) -> dict:
    """A one-beat episode: everything up to and including beat b (earlier lines stay as context)."""
    bs = beats(ep)
    end = bs[b][-1] + 1
    start_ctx = 0
    return {**ep, "id": f"{ep['id']}#b{b}{tag}", "base_id": ep["id"], "lines": ep["lines"][start_ctx:end],
            "scored": bs[b]}


def anchors(version: str = "") -> list[dict]:
    """Human anchors → one-beat episodes: [human turn1 (template)] + host + 4 panel lines.
    version="_v2": the most interactive discussion windows (`build.py anchors --v2`)."""
    p = PUBLIC / "build" / f"anchors{version}.jsonl"
    if not p.is_file():
        sys.exit(f"missing {p}: run `python -m eval.data.public_set.build anchors` first (eval/data/public_set/README.md)")
    out = []
    for a in load(p):
        lines = [{"role": "human", "speaker": NAMES["human"],
                  "text": f'I want to put a motion to the panel: "{a["topic"].rstrip(".")}." Where do the two of you land?'}]
        for l in a["lines"]:
            lines.append({"role": l["role_id"], "speaker": l["speaker"], "text": l["text"],
                          **({"side": l["side"]} if l.get("side") else {})})
        ep = {"id": f"anchor:{a['topic_id']}", "topic_id": a["topic_id"], "topic": a["topic"], "source": "anchor",
              "lines": lines}
        ep["scored"] = panel_idx(ep)
        out.append(ep)
    return out


# --------------------------------------------------------------------------------------------- live episodes

def _host_template_res() -> list:
    """Regexes for every production host template (agent/floor/host_lines.py tuples + the welcome line): a host line
    matching one is procedural floor management; anything else is a real (LLM) host reply."""
    import re as _re

    from agent.floor import host_lines as H

    pats = [_re.escape("Welcome — I'm Lessac, your host.") + ".*"]
    for name in dir(H):
        v = getattr(H, name)
        if name.startswith("_") and isinstance(v, tuple) and v and all(isinstance(x, str) for x in v):
            for tmpl in v:
                esc = _re.escape(tmpl)
                esc = esc.replace(_re.escape("{name}"), r"\w+").replace(_re.escape("{topic}"), r".+")
                pats.append(esc)
    return [_re.compile("^" + p + "$") for p in pats]


def live_episodes(log_dir: Path, arm: str = "A", limit: int = 0, seed: int = 0) -> list[dict]:
    """One episode per live ghost session of the given arm (room suffix __v<arm>; "" = sessions without a suffix): the
    replayed prefix (history_snapshot), the human's
    turn 2, then every spoken line in order (line_start). Host lines that match a production template are marked
    procedural; the others (the host LLM's replies) are `host_scored` for the host-craft items."""
    import ast
    import re as _re

    res = _host_template_res()
    out = []
    for f in sorted(Path(log_dir).glob("session-*.jsonl")):
        ev = []
        for line in f.read_text().splitlines():
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        room = next((e.get("room", "") for e in ev if e.get("room")), "")
        m = _re.match(r"ghost-(?P<slug>[\w-]+?)__(?P<cond>\w+?)__k(?P<k>\d+)(?:__v(?P<v>[A-Za-z0-9]+))?", room)
        if not m or (m["v"] or "") != arm:
            continue
        snap = next((e for e in ev if e.get("event") == "history_snapshot"), None)
        human = next((e for e in ev if e.get("event") == "ghost_human_start"), None)
        if snap is None or human is None:
            continue
        prefix = snap["lines"] if isinstance(snap["lines"], list) else ast.literal_eval(snap["lines"])
        lines = [{"role": l["role_id"], "speaker": NAMES.get(l["role_id"], l.get("speaker", "")), "text": l["text"],
                  **({"procedural": True} if l.get("procedural") else {})} for l in prefix]
        lines.append({"role": "human", "speaker": NAMES["human"], "text": human.get("heard", "")})
        t_h = human.get("ts", 0)
        for e in ev:
            if e.get("event") == "line_start" and e.get("ts", 0) > t_h and e.get("text"):
                role = e.get("role")
                if role not in ("host", "commentator", "guest"):
                    continue
                l = {"role": role, "speaker": NAMES[role], "text": e["text"]}
                if role == "host" and any(r.match(e["text"].strip()) for r in res):
                    l["procedural"] = True
                lines.append(l)
        h = max(i for i, l in enumerate(lines) if l["role"] == "human")
        ep = {"id": f"live:{m['slug']}:{m['cond']}:k{m['k']}:v{arm}", "topic_id": m["slug"], "topic": m["slug"],
              "source": "live", "lines": lines,
              "scored": [i for i in range(h + 1, len(lines)) if lines[i]["role"] in PANEL],
              "host_scored": [i for i in range(h + 1, len(lines)) if lines[i]["role"] == "host"
                              and not lines[i].get("procedural")]}
        if ep["scored"]:
            out.append(ep)
    if limit and len(out) > limit:
        out = sorted(random.Random(seed).sample(out, limit), key=lambda e: e["id"])
    return out


# --------------------------------------------------------------------------------------------- generation (GPU)

def generate(args) -> None:
    """Model-only episodes: turn 1 → beat 1 (4 panel lines) → turn 2 (orig) → beat 2 (4 lines), production panel prompts.
    Panel speakers alternate starting with Ryan; before each panel line the host says a production tee-up line.
    `--arm` / `--arm-spec` (eval/quality_ruler/arms.py) set the generation knobs; `--pairs` picks the topic set (the 30
    hand-written topics in minimal_pairs/pairs.yaml, or the public test set build/pairs_public.yaml; a comma list
    concatenates files). Seeds are per (topic, k, step), shared across arms.

    Arms with `live_flags` build every panel prompt with the live worker's fixes (model_only.build_request(
    live_fixes=True)): the uptake plan only on the first panel line after a human turn, the fact pack, the concrete
    instruction. `--dry-run DIR`: no model; every prompt is written to DIR (prompts.jsonl, one readable .txt per
    episode) and each panel line is a placeholder, so later prompts show the placeholder in the transcript.

    `--path think_action`: each panel line is what the Think & Action show would speak, the `content` of the panelist's
    think call on the finished previous turn (made before the host's tee-up, as live). The speaker order stays fixed
    (no floor controller). A think that does not choose speak falls back to the worker prompt (with the arm's live
    fixes), as the live show does when a selected panelist has no proposed reply. Only arms made of `live_flags` and
    `fact_pack` apply on this path (they set the TALKSHOW_* flags both prompts read); the other knobs shape worker
    prompts and are refused."""
    import hashlib
    import json as _json
    import os
    import re

    import yaml

    from agent.floor.host_lines import _INTRO_SPEAKER
    from agent.panel.uptake import uptake_planner_enabled
    from eval.model_only import TEMPERATURE, add_plan, build_request, planner_messages, postprocess
    from eval.quality_ruler.arms import (ARMS, BANNED_OPENERS, apply_env, ban_text, opener, opener_plan_text,
                                         rerank_score)

    arm = dict(ARMS.get(args.arm, {})) if args.arm in ARMS else {}
    if args.arm_spec:
        arm.update(_json.loads(args.arm_spec))
    think_path = getattr(args, "path", "worker") == "think_action"
    if think_path and (set(arm) - {"live_flags", "fact_pack", "best_of"} or int(arm.get("best_of", 1)) != 1):
        raise SystemExit("--path think_action takes only arms made of live_flags / fact_pack (TALKSHOW_* flags); "
                         "the other knobs shape worker prompts")
    apply_env(arm)
    live = bool(arm.get("live_flags"))
    plan_beats = bool(arm.get("lite_plan")) or ((live or think_path) and uptake_planner_enabled())
    if think_path:
        from eval.think_action_prompt import spoken_content, think_request

        # The final interval of a panel line is split with the show's tokenizer: the generating model's.
        if not (os.environ.get("VLLM_MODEL_PATH") or os.environ.get("VLLM_MODEL")):
            os.environ["VLLM_MODEL_PATH"] = args.model
    pairs = [t for f in str(args.pairs).split(",") if f for t in yaml.safe_load(Path(f).read_text())["topics"]]
    if args.topics:
        keep = set(args.topics.split(","))
        pairs = [t for t in pairs if t["slug"] in keep]
    dry = Path(args.dry_run) if getattr(args, "dry_run", "") else None
    if dry:
        llm, SamplingParams = DryRunLLM(), _DrySampling
    else:
        from vllm import LLM, SamplingParams

        llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=8192, gpu_memory_utilization=0.88,
                  seed=args.seed, disable_custom_all_reduce=args.tp > 1)
    chat_kwargs = {"enable_thinking": False} if "qwen3" in args.model.lower() else None

    def chat(convs, sps):
        return llm.chat(convs, sps, chat_template_kwargs=chat_kwargs) if chat_kwargs else llm.chat(convs, sps)

    def seed_of(*parts) -> int:
        return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)

    rng = random.Random(args.seed)
    temp = arm.get("temperature", TEMPERATURE)
    bad_words = [v for w in BANNED_OPENERS for v in (w, " " + w)]
    n = int(arm.get("best_of", 1))
    eps = []
    for t in pairs:
        for k in range(args.k):
            label = "think_action" if think_path else args.arm
            eps.append({"id": f"{label}:{t['slug']}:k{k}", "topic_id": t["slug"], "topic": t["topic"],
                        "source": "system", "arm": label, "arm_spec": arm, "model": model_name(args.model), "k": k,
                        "plan": [t["turn1"], t["turn2"]["orig"]], "fact_plan": {},
                        "lines": [{"role": "host", "speaker": "Lessac", "procedural": True,
                                   "text": f"You wanted to talk about {t['topic'].rstrip('.?')} — the floor is yours, go ahead."},
                                  {"role": "human", "speaker": NAMES["human"], "text": t["turn1"]}]})
    for beat in range(2):
        if beat == 1:
            for e in eps:
                e["lines"].append({"role": "human", "speaker": NAMES["human"], "text": e["plan"][1]})
        if plan_beats:  # one plan per beat from the latest human turn (production planner prompt, temp 0)
            texts = sorted({e["plan"][beat] for e in eps})
            outs = chat([planner_messages(x) for x in texts], SamplingParams(temperature=0.0, max_tokens=160))
            plans = {x: re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.S).strip() for x, o in zip(texts, outs)}
            for e in eps:
                e["fact_plan"][beat] = plans[e["plan"][beat]]
        for step in range(args.lines_per_beat):
            role = PANEL[step % 2]
            convs, reqs, sps = [], [], []
            for e in eps:
                if think_path:
                    # Live, the speaker's content comes from its think on the previous turn, before the tee-up.
                    before = {"lines": [{"role_id": l["role"], "speaker": l["speaker"], "text": l["text"],
                                         "procedural": bool(l.get("procedural"))} for l in e["lines"]]}
                    think = think_request(before, role, plan=e["fact_plan"].get(beat, ""), topic_id=e["topic_id"])
                e["lines"].append({"role": "host", "speaker": "Lessac", "procedural": True,
                                   "text": rng.choice(_INTRO_SPEAKER).format(name=NAMES[role])})
                prefix = {"lines": [{"role_id": l["role"], "speaker": l["speaker"], "text": l["text"],
                                     "procedural": bool(l.get("procedural"))} for l in e["lines"]]}
                if think_path:
                    if think.get("fact_pack"):
                        think["think_fact_pack"] = {"line": len(e["lines"]), **think["fact_pack"]}
                    think["fallback"] = build_request(prefix, "", role, False, include_human=False, live_fixes=live,
                                                      plan=e["fact_plan"].get(beat, "") if live else "",
                                                      topic_id=e["topic_id"])
                    reqs.append(think)
                    convs.append(think["messages"])
                    if dry:
                        llm.record(e["id"], len(e["lines"]), role, think["messages"])
                    sps.append(SamplingParams(temperature=think["temperature"], max_tokens=think["max_tokens"],
                                              seed=seed_of(e["topic_id"], e["k"], beat, step)))
                    continue
                r = build_request(prefix, "", role, False, include_human=False,
                                  dialogue_pick=arm.get("dialogue_pick"), dialogue_library=arm.get("dialogue_library"),
                                  live_fixes=live, plan=e["fact_plan"].get(beat, "") if live else "",
                                  topic_id=e["topic_id"])
                if r.get("fact_pack"):
                    e.setdefault("fact_pack", []).append({"line": len(e["lines"]), **r["fact_pack"]})
                msgs = [dict(m) for m in r["messages"]]
                extra = []
                if arm.get("instruction"):
                    extra.append(arm["instruction"])
                if arm.get("opener_ban"):
                    extra.append(ban_text([opener(l["text"]) for l in e["lines"] if l["role"] in PANEL]))
                if arm.get("opener_plan"):
                    extra.append(opener_plan_text(seed_of("plan", e["topic_id"], e["k"], beat, step),
                                                  role if arm["opener_plan"] == "persona" else None))
                if extra:
                    msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + "\n".join(extra)
                if arm.get("lite_plan") and not live:
                    msgs = add_plan(msgs, e["fact_plan"][beat], lite=True)
                reqs.append(r)
                convs.append(msgs)
                if dry:
                    llm.record(e["id"], len(e["lines"]), role, msgs)
                sps.append(SamplingParams(temperature=temp, min_p=arm.get("min_p", 0.0), max_tokens=512, n=n,
                                          seed=seed_of(e["topic_id"], e["k"], beat, step),
                                          **({"bad_words": bad_words} if arm.get("ban_openers") else {})))
            outs = chat(convs, sps)
            if think_path:
                spoken = [spoken_content(re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.S).strip(),
                                         r["interval_text"])[1] for r, o in zip(reqs, outs)]
                redo = [i for i, x in enumerate(spoken) if not x]
                if redo:  # worker fallback, production sampling
                    f_outs = chat([reqs[i]["fallback"]["messages"] for i in redo],
                                  [SamplingParams(temperature=TEMPERATURE, max_tokens=512,
                                                  seed=seed_of("fallback", eps[i]["topic_id"], eps[i]["k"], beat, step))
                                   for i in redo])
                    for i, o in zip(redo, f_outs):
                        r = reqs[i]["fallback"]
                        raw = re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.S).strip()
                        spoken[i] = postprocess(raw, role, r["panel_roles"], r["data"].show_history)
                        if r.get("fact_pack"):
                            eps[i].setdefault("fact_pack", []).append({"line": len(eps[i]["lines"]), **r["fact_pack"]})
                for i, (e, line) in enumerate(zip(eps, spoken)):
                    if i not in redo and reqs[i].get("think_fact_pack"):
                        e.setdefault("fact_pack", []).append(reqs[i]["think_fact_pack"])
                    e["lines"].append({"role": role, "speaker": NAMES[role], "text": line or "…",
                                       "source": "worker_fallback" if i in redo else "think"})
                print(f"[think_action] beat {beat} step {step} done ({len(eps)} episodes, "
                      f"{len(redo)} worker fallbacks)", flush=True)
                continue
            for e, r, o in zip(eps, reqs, outs):
                cands = []
                for comp in o.outputs:
                    raw = re.sub(r"<think>.*?</think>", "", comp.text, flags=re.S).strip()
                    line = postprocess(raw, role, r["panel_roles"], r["data"].show_history)
                    if line:
                        cands.append(line)
                if not cands:
                    cands = ["…"]
                if len(cands) > 1:
                    earlier = [l["text"] for l in e["lines"] if l["role"] in PANEL]
                    used = {opener(x) for x in earlier}
                    cands.sort(key=lambda c: -rerank_score(c, earlier, used, arm.get("opener_penalty", 1.0)))
                e["lines"].append({"role": role, "speaker": NAMES[role], "text": cands[0],
                                   **({"n_candidates": len(cands)} if n > 1 else {})})
            print(f"[{args.arm}] beat {beat} step {step} done ({len(eps)} episodes)", flush=True)
    for e in eps:
        e.pop("plan")
        e["scored"] = panel_idx(e)
    if dry:
        llm.dump(dry, eps)
        print(f"dry run: {len(llm.rows)} prompts for {len(eps)} episodes → {dry}")
        return
    save(Path(args.out), eps)
    print(f"wrote {len(eps)} episodes → {args.out}")


class _DrySampling:
    def __init__(self, **kw) -> None:
        self.kw = kw


class DryRunLLM:
    """Stands in for vllm.LLM in `generate --dry-run`: records prompts, returns placeholder text (no model)."""

    PLACEHOLDER = "[reply]: (dry run: no model was called; the panelist's line goes here)"
    # A Think & Action think prompt (--path think_action) gets a "speak" action with the same placeholder line.
    THINK = json.dumps({"think": "", "action": {
        "decision": "speak", "intent": {"type": "contribute", "goal": "dry run", "target": None},
        "content": "(dry run: no model was called; the panelist's line goes here)", "backchannels": []}})
    PLAN = "- (dry run: the planner's extracted points go here)"

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, eid: str, line: int, role: str, msgs: list[dict]) -> None:
        self.rows.append({"episode": eid, "line": line, "role": role, "messages": msgs})

    def chat(self, convs, sps, **_kw):
        from types import SimpleNamespace

        def text(conv) -> str:
            if conv[0]["content"].startswith("You extract"):
                return self.PLAN
            return self.THINK if "JSON schema:" in conv[-1]["content"] else self.PLACEHOLDER

        return [SimpleNamespace(outputs=[SimpleNamespace(text=text(c))]) for c in convs]

    def dump(self, out: Path, eps: list[dict]) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "prompts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows))
        by_ep: dict[str, list[dict]] = {}
        for r in self.rows:
            by_ep.setdefault(r["episode"], []).append(r)
        for e in eps:
            name = e["id"].replace(":", "__").replace("/", "_") + ".txt"
            with (out / name).open("w") as f:
                f.write(f"# episode={e['id']} topic={e['topic']}\n")
                for r in by_ep.get(e["id"], []):
                    f.write(f"\n\n######## panel line at index {r['line']} — speaker={r['role']}\n")
                    offered = next((x for x in e.get("fact_pack", []) if x["line"] == r["line"]), None)
                    if offered:
                        f.write(f"# fact pack: {json.dumps({k: v for k, v in offered.items() if k != 'line'})}\n")
                    for m in r["messages"]:
                        f.write(f"\n===== {m['role']} =====\n{m['content']}\n")
