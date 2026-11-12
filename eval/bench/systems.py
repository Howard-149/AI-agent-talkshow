"""The system interface and three adapters.

A system answers one observation (items.observation: what the role under test sees) with

    respond(obs) -> {"text": str, "speaker": role,
                     "floor_action": "speak" | "yield" | "wait" | "interrupt" | "backchannel" | None}

plus bookkeeping fields (prompt_hash, raw, calls). ``floor_action`` is None when the system does not decide whether to
take the floor (it was simply asked for a line). On T2 (obs["floor_choice"]) every system may leave the floor:
PlainLLM answers PASS (floor "yield"), Think & Action stays silent (floor "wait" or "backchannel"); the worker makes
no floor decision. ``respond_batch`` answers many observations with batched model calls; ``respond`` is the one-item
case. A new system subclasses ``System`` and implements ``requests`` (build the prompts) and
``finish`` (turn model outputs into the reply), or overrides ``respond_batch`` entirely (e.g. a remote service).

Adapters:
  PlainLLM             one public prompt template (PLAIN_SYSTEM / PLAIN_USER below), no show machinery
  TalkshowWorker       the show's panel worker prompt (eval/model_only.build_request), arms = flag presets
  TalkshowThinkAction  the show's Think & Action think call (eval/think_action_prompt.think_request); its decision is
                       the floor action; a panelist who does not choose to speak gets the worker line as fallback
                       (what the live show says if the floor still goes to that panelist)

Backends (the model): VLLMBackend (offline vLLM, batched), FakeBackend (no model; fixed placeholder or a callable),
DryRunBackend (FakeBackend that also records every request; run.py --dry-run).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

PLAIN_TEMPERATURE = 0.7
PLAIN_MAX_TOKENS = 256
PLAIN_SYSTEM = "{persona}"
PLAIN_USER = (
    "You are on a live talk show panel. Topic: {topic}\n"
    "\n"
    "Transcript so far:\n"
    "{transcript}\n"
    "{facts}"
    "\n"
    "You are {name}. Reply with your next spoken line only: 1-3 sentences, plain text, no speaker label."
)
PLAIN_FACTS = "\nChecked facts on this topic:\n{facts}\n"
# Floor-choice tasks only (T2): the one line added after the instruction, so the panelist can leave the floor.
PLAIN_PASS = "If the question was addressed to someone else, reply with exactly PASS instead of a line."
PASS_RE = re.compile(r"\s*pass\s*[.!]*\s*", re.I)
AUDIENCE_LABEL = "Audience member"
FLOOR_ACTIONS = ("speak", "yield", "wait", "interrupt", "backchannel")
YIELDS = ("yield", "wait", "backchannel")  # floor actions that leave the floor to someone else


def prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


# ------------------------------------------------------------------------------------------------ backends

@dataclass
class Request:
    """One model call: chat messages plus sampling. ``kind``: planner | reply | think | fallback."""
    kind: str
    messages: list[dict]
    temperature: float
    max_tokens: int
    seed: int | None = None
    tag: str = ""  # item id / rerun, for dry-run records


class FakeBackend:
    """No model. Answers with the episode generator's dry-run placeholders (a think call gets a "speak" action), or
    with ``responder(request) -> str`` when given."""

    def __init__(self, responder: Callable[[Request], str] | None = None) -> None:
        self.responder = responder

    def default(self, req: Request) -> str:
        from eval.quality_ruler.episodes import DryRunLLM

        if req.kind == "planner":
            return DryRunLLM.PLAN
        if req.kind.startswith("think"):
            return DryRunLLM.THINK
        if req.kind == "controller":  # the free-floor controller: take the first panelist
            return json.dumps({"decision": "speak", "selected_role": "commentator",
                               "reason_code": "high_contribution", "reason": "placeholder (no model was called)"})
        if req.kind == "floor":
            return "Ryan"
        return DryRunLLM.PLACEHOLDER

    def generate(self, reqs: list[Request]) -> list[str]:
        return [self.responder(r) if self.responder else self.default(r) for r in reqs]


class DryRunBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[Request] = []

    def generate(self, reqs: list[Request]) -> list[str]:
        self.records += reqs
        return super().generate(reqs)


class VLLMBackend:
    """Offline vLLM: one engine, every request of a stage in one batched chat call."""

    def __init__(self, model: str, tp: int = 1, max_model_len: int = 8192, seed: int = 0) -> None:
        from vllm import LLM

        self.model = model
        self.llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len, gpu_memory_utilization=0.88,
                       seed=seed, disable_custom_all_reduce=tp > 1)
        # Qwen3-style templates: no hidden reasoning (the show's client gets plain text).
        self.chat_kwargs = {"enable_thinking": False} if "qwen3" in model.lower() else None

    def generate(self, reqs: list[Request]) -> list[str]:
        from vllm import SamplingParams

        if not reqs:
            return []
        sps = [SamplingParams(temperature=r.temperature, max_tokens=r.max_tokens, seed=r.seed) for r in reqs]
        convs = [r.messages for r in reqs]
        kw = {"chat_template_kwargs": self.chat_kwargs} if self.chat_kwargs else {}
        outs = self.llm.chat(convs, sps, use_tqdm=False, **kw)
        return [strip_think(o.outputs[0].text) for o in outs]


# ------------------------------------------------------------------------------------------------ interface

@dataclass
class Pending:
    """One observation in flight through the stages of respond_batch."""
    obs: dict
    seed: int | None
    state: dict = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)
    result: dict | None = None


class System:
    name = "system"

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def describe(self) -> dict:
        return {"system": self.name}

    def respond(self, obs: dict, seed: int | None = None) -> dict:
        return self.respond_batch([obs], [seed])[0]

    def respond_batch(self, obs_list: list[dict], seeds: list[int | None] | None = None) -> list[dict]:
        pend = [Pending(o, s) for o, s in zip(obs_list, seeds or [None] * len(obs_list))]
        stage = 0
        while True:
            todo = [(p, r) for p in pend if p.result is None for r in self.requests(p, stage)]
            if not todo and all(p.result is not None for p in pend):
                break
            t0 = time.perf_counter()
            texts = self.backend.generate([r for _p, r in todo])
            per = (time.perf_counter() - t0) / max(1, len(todo))
            for (p, r), text in zip(todo, texts):
                p.calls.append({"kind": r.kind, "prompt_hash": prompt_hash(r.messages), "latency_s": round(per, 4),
                                "raw": text})
                p.state.setdefault("outputs", {})[r.kind] = text
            for p in pend:
                if p.result is None:
                    p.result = self.finish(p, stage)
            stage += 1
            if stage > 5:
                raise RuntimeError("system did not finish in 5 stages")
        out = []
        for p in pend:
            main = next((c for c in p.calls if c["kind"] in ("reply", "think")), p.calls[0] if p.calls else {})
            out.append({**p.result, "prompt_hash": main.get("prompt_hash", ""), "calls": p.calls,
                        "latency_s": round(sum(c["latency_s"] for c in p.calls), 4)})
        return out

    # subclasses
    def requests(self, p: Pending, stage: int) -> list[Request]:
        raise NotImplementedError

    def finish(self, p: Pending, stage: int) -> dict | None:
        """The reply once p's outputs are in, or None to run another stage."""
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ plain LLM

def plain_messages(obs: dict) -> list[dict]:
    """The public PlainLLM prompt: persona card, the transcript, optional checked facts, the instruction."""
    names = obs.get("names", {})
    lines = []
    for h in obs["history"]:
        who = AUDIENCE_LABEL if h["role"] == "human" else names.get(h["role"], h["speaker"])
        lines.append(f"{who}: {h['text']}")
    if obs.get("audience_turn"):
        lines.append(f"{AUDIENCE_LABEL}: {obs['audience_turn']}")
    facts = ""
    if obs.get("fact_pack"):
        facts = PLAIN_FACTS.format(facts="\n".join(f"- {f}" for f in obs["fact_pack"]))
        if obs.get("fact_instruction"):  # T3b: one line right after the facts
            facts += obs["fact_instruction"] + "\n"
    persona = obs["personas"].get(obs["role"], f"You are {obs['name']} on a live talk show panel.")
    user = PLAIN_USER.format(topic=obs["topic"], transcript="\n".join(lines), facts=facts, name=obs["name"])
    if obs.get("floor_choice"):
        user += "\n" + PLAIN_PASS
    return [{"role": "system", "content": PLAIN_SYSTEM.format(persona=persona)}, {"role": "user", "content": user}]


def is_pass(text: str) -> bool:
    """An exact PASS: any case, surrounding spaces, optional trailing . or !"""
    return bool(PASS_RE.fullmatch(text or ""))


def plain_postprocess(text: str, name: str) -> str:
    text = strip_think(text)
    text = re.sub(rf"^\s*(?:\[reply\]\s*:|{re.escape(name)}\s*(?:\([^)]*\))?\s*:)\s*", "", text, flags=re.I)
    text = text.strip().strip('"“”').strip()
    return re.sub(r"\s+", " ", text)


class PlainLLM(System):
    name = "plain"

    def __init__(self, backend: Any, temperature: float = PLAIN_TEMPERATURE, max_tokens: int = PLAIN_MAX_TOKENS) -> None:
        super().__init__(backend)
        self.temperature, self.max_tokens = temperature, max_tokens

    def describe(self) -> dict:
        return {"system": self.name, "temperature": self.temperature, "max_tokens": self.max_tokens}

    def requests(self, p: Pending, stage: int) -> list[Request]:
        if stage:
            return []
        if p.obs.get("floor_item"):  # T2f: one floor question (eval/bench/floor.py)
            from eval.bench.floor import plain_floor_messages

            return [Request("floor", plain_floor_messages(p.obs), 0.0, 16, p.seed, p.obs["item_id"])]
        return [Request("reply", plain_messages(p.obs), self.temperature, self.max_tokens, p.seed, p.obs["item_id"])]

    def finish(self, p: Pending, stage: int) -> dict:
        if p.obs.get("floor_item"):
            from eval.bench.floor import parse_name

            raw = p.state["outputs"]["floor"]
            role = parse_name(strip_think(raw), p.obs.get("names", {}))
            return {"text": "", "speaker": role, "floor_action": "speak" if role else None, "selected_role": role,
                    "floor_mode": "question", "raw": raw}
        text = plain_postprocess(p.state["outputs"]["reply"], p.obs["name"])
        floor = None
        if p.obs.get("floor_choice"):  # exact PASS = yield the floor, no line
            floor = "yield" if is_pass(text) else "speak"
            text = "" if floor == "yield" else text
        return {"text": text, "speaker": p.obs["role"], "floor_action": floor, "raw": p.state["outputs"]["reply"]}


# ------------------------------------------------------------------------------------------------ the show

def _show_arms() -> tuple[str, ...]:
    """Arm presets made of the show's own flags (live_flags, fact_pack; best-of 1): the ones the show adapters take."""
    from eval.quality_ruler.arms import ARMS

    return tuple(k for k, v in ARMS.items()
                 if set(v) <= {"live_flags", "fact_pack", "best_of"} and int(v.get("best_of", 1)) == 1)


SHOW_ARMS = _show_arms()


def show_fact_instruction(messages: list[dict], obs: dict) -> tuple[list[dict], str | None]:
    """T3b on the show's prompts. (messages, how the instruction was given):
    "show_rule"     the prompt already tells the panelist to ground its line in the checked facts (the concrete
                    instruction that comes with a fact block on the fast and fact-pack arms): unchanged, T3b = T3a;
    "adapter_line"  otherwise the instruction line is added right after the fact block's rule;
    "not_applicable" no fact block found (the prompt is left unchanged)."""
    from agent.panel.fact_pack import CONCRETE_WITH_PACK, RULE

    line = obs.get("fact_instruction")
    if not line:
        return messages, None
    text = messages[-1]["content"]
    if CONCRETE_WITH_PACK in text:
        return messages, "show_rule"
    if RULE not in text:
        return messages, "not_applicable"
    out = [dict(m) for m in messages]
    out[-1]["content"] = text.replace(RULE, RULE + "\n" + line, 1)
    return out, "adapter_line"


def show_persona_swap(obs: dict) -> bool:
    """The show takes no free persona cards, only its own pair or the pair swapped; anything else is refused."""
    from eval.bench.build import persona_cards

    panel = {r: obs["personas"].get(r) for r in ("commentator", "guest")}
    if panel == persona_cards(False):
        return False
    if panel == persona_cards(True):
        return True
    raise ValueError(f"{obs['item_id']}: the show cannot take these persona cards (only its own, or swapped)")


@contextmanager
def _env(**kv: str | None) -> Iterator[None]:
    from eval.bench.build import _env as env

    with env(**kv):
        yield


class _Show(System):
    """Shared by the two show adapters: arm flags, the fact pack, the uptake planner."""

    def __init__(self, backend: Any, arm: str = "production", own_pack: str = "") -> None:
        from eval.quality_ruler.arms import ARMS, apply_env

        super().__init__(backend)
        if arm not in SHOW_ARMS:
            raise ValueError(f"arm must be one of {SHOW_ARMS}")
        self.arm = arm
        spec = ARMS[arm]
        # The show's own fact pack (arm concrete_fact_pack) for items that bring no facts.
        self.own_pack = (own_pack or os.environ.get("TALKSHOW_FACT_PACK", "")) if spec.get("fact_pack") else ""
        if spec.get("fact_pack") and not self.own_pack:
            raise SystemExit(f"arm {arm} needs the show's fact pack: TALKSHOW_FACT_PACK=<json> or own_pack")
        os.environ["TALKSHOW_FACT_PACK"] = self.own_pack or "unused"  # apply_env checks it is set for pack arms
        apply_env(spec)
        os.environ.pop("TALKSHOW_FACT_PACK", None)
        self.live = bool(spec.get("live_flags"))
        self._packs: dict[tuple, str] = {}
        self._pack_dir = tempfile.mkdtemp(prefix="bench-pack-")
        atexit.register(shutil.rmtree, self._pack_dir, True)
        from agent.panel.uptake import uptake_planner_enabled

        self.planner = uptake_planner_enabled()

    def describe(self) -> dict:
        return {"system": self.name, "arm": self.arm, "planner": self.planner, "own_pack": bool(self.own_pack)}

    def pack_env(self, obs: dict) -> tuple[dict[str, str | None], str | None]:
        """(environment, topic id) for the fact block: the item's facts if it has any, else the show's own pack (pack
        arms), else none. The item's facts go into a one-topic pack offering all of them."""
        facts = obs.get("fact_pack")
        if facts:
            key = tuple(facts)
            if key not in self._packs:
                tid = f"bench-{len(self._packs)}"
                path = Path(self._pack_dir) / f"{tid}.json"
                path.write_text(json.dumps({tid: {"topic": obs["topic"], "facts": [{"fact": f} for f in facts]}}))
                self._packs[key] = f"{path}::{tid}"
            path, tid = self._packs[key].split("::")
            return {"TALKSHOW_FACT_PACK": path, "TALKSHOW_FACT_PACK_K": str(len(facts)),
                    "TALKSHOW_FACT_PACK_TOPIC": None}, tid
        if self.own_pack:
            return {"TALKSHOW_FACT_PACK": self.own_pack, "TALKSHOW_FACT_PACK_K": None,
                    "TALKSHOW_FACT_PACK_TOPIC": None}, obs["topic_id"]
        return {"TALKSHOW_FACT_PACK": None, "TALKSHOW_FACT_PACK_K": None, "TALKSHOW_FACT_PACK_TOPIC": None}, None

    def prefix(self, obs: dict) -> dict:
        from eval.bench.build import prefix_from_history

        return prefix_from_history(obs["history"], obs.get("show_state"))

    def planner_request(self, p: Pending) -> list[Request]:
        """Stage 0 when the arm runs the uptake planner on the audience line (production planner prompt, temp 0)."""
        from eval.model_only import planner_messages

        if self.planner and p.obs.get("audience_turn"):
            return [Request("planner", planner_messages(p.obs["audience_turn"]), 0.0, 160, None, p.obs["item_id"])]
        return []

    def plan(self, p: Pending) -> str:
        return strip_think(p.state.get("outputs", {}).get("planner", ""))


class TalkshowWorker(_Show):
    name = "worker"

    def worker_request(self, p: Pending, kind: str = "reply", role: str | None = None) -> tuple[Request, dict]:
        """The worker prompt for ``role`` (default the role under test)."""
        from eval.model_only import TEMPERATURE, build_request

        obs = p.obs
        role = role or obs["role"]
        env, tid = self.pack_env(obs)
        swap = show_persona_swap(obs)
        human = obs.get("audience_turn")
        with _env(**env):
            req = build_request(self.prefix(obs), human or "", role, swap, include_human=human is not None,
                                live_fixes=self.live or bool(env["TALKSHOW_FACT_PACK"]),
                                plan=self.plan(p), topic_id=tid)
        req["messages"], p.state["fact_instruction"] = show_fact_instruction(req["messages"], obs)
        return Request(kind, req["messages"], TEMPERATURE, 512, p.seed, obs["item_id"]), req

    def requests(self, p: Pending, stage: int) -> list[Request]:
        if p.obs.get("floor_item"):
            return []  # the worker path has no floor selection: T2f does not apply
        if stage == 0 and self.planner_request(p):
            p.state["planned"] = True
            return self.planner_request(p)
        if stage == (1 if p.state.get("planned") else 0):
            r, req = self.worker_request(p)
            p.state["req"] = req
            return [r]
        return []

    def finish(self, p: Pending, stage: int) -> dict | None:
        from eval.model_only import postprocess

        if p.obs.get("floor_item"):
            return {"text": "", "speaker": None, "floor_action": None, "selected_role": None,
                    "floor_mode": "not_applicable", "raw": ""}
        if "reply" not in p.state.get("outputs", {}):
            return None
        req = p.state["req"]
        raw = p.state["outputs"]["reply"]
        text = postprocess(raw, p.obs["role"], req["panel_roles"], req["data"].show_history)
        return {"text": text, "speaker": p.obs["role"], "floor_action": None, "raw": raw,
                **({"fact_instruction": p.state["fact_instruction"]} if p.state.get("fact_instruction") else {}),
                "fact_pack_offered": req.get("fact_pack") or None}


def floor_action(decision: str, backchannels: list) -> str:
    """The think decision as a floor action: speak, interrupt; silent -> wait (backchannel if it sent one)."""
    if decision in ("speak", "interrupt"):
        return decision
    return "backchannel" if backchannels else "wait"


class TalkshowThinkAction(TalkshowWorker):
    """Think call first (its decision is the floor action); a panelist who does not speak gets the worker line."""
    name = "think_action"

    def __init__(self, backend: Any, arm: str = "production", own_pack: str = "", fallback: bool = True) -> None:
        super().__init__(backend, arm, own_pack)
        self.fallback = fallback

    def describe(self) -> dict:
        return {**super().describe(), "fallback": self.fallback}

    # T2f (eval/bench/floor.py): both panelists think on the same state, then the production free-floor selection
    # picks; the controller call, when the selection needs it, is one more model request.

    def floor_requests(self, p: Pending, stage: int) -> list[Request]:
        from eval.bench.floor import PANEL, arbitrate
        from eval.think_action_prompt import think_request

        obs = p.obs
        if stage == 0 and self.planner_request(p):
            p.state["planned"] = True
            return self.planner_request(p)
        first = 1 if p.state.get("planned") else 0
        if stage == first:
            reqs = []
            for k, role in enumerate(PANEL):
                req = think_request(self.prefix(obs), role, human_text=obs.get("audience_turn"), plan=self.plan(p))
                p.state[f"think:{role}"] = req
                reqs.append(Request(f"think:{role}", req["messages"], req["temperature"], req["max_tokens"],
                                    None if p.seed is None else p.seed + k, obs["item_id"]))
            return reqs
        if stage == first + 1:
            p.state["actions"] = self.floor_actions(p)
            got = arbitrate(self.prefix(obs), obs.get("audience_turn") or "", p.state["actions"])
            p.state["floor"] = got
            if got["controller"] is not None:
                c = got["controller"]
                return [Request("controller", c["messages"], c["temperature"], c["max_tokens"], p.seed,
                                obs["item_id"])]
            return self.floor_settle(p, got)  # a rule picked (addressee priority): no controller call
        if stage == first + 2 and "controller" in p.state.get("outputs", {}) and "outcome" not in p.state:
            got = arbitrate(self.prefix(obs), obs.get("audience_turn") or "", p.state["actions"],
                            p.state["outputs"]["controller"])
            return self.floor_settle(p, got)
        return []

    def floor_settle(self, p: Pending, got: dict) -> list[Request]:
        """The final selection; a selected panelist with no proposed line (e.g. forced by ADDRESSEE_FLOOR=force)
        gets the worker draft, as in the live show (actuators.generate_panelist_line)."""
        p.state["outcome"] = got
        out = got["outcome"]
        action = p.state["actions"].get(out.selected_role)
        if out.decision == "speak" and action is not None and not (action.decision == "speak" and action.content):
            r, req = self.worker_request(p, "fallback", role=out.selected_role)
            p.state["req"] = req
            return [r]
        return []

    def floor_actions(self, p: Pending) -> dict:
        from agent.interaction.think_action import parse_think_action
        from eval.bench.floor import PANEL

        outs = p.state.get("outputs", {})
        return {role: parse_think_action(outs[f"think:{role}"], floor_state="free",
                                         interval_text=p.state[f"think:{role}"]["interval_text"],
                                         allow_backchannels=True).action for role in PANEL}

    def floor_finish(self, p: Pending) -> dict | None:
        from eval.bench.floor import PANEL
        from eval.model_only import postprocess

        outs = p.state.get("outputs", {})
        if "outcome" not in p.state:
            return None
        got = p.state["outcome"]
        out = got["outcome"]
        actions = p.state["actions"]
        selected = out.selected_role
        text, source = "", "none"
        if "req" in p.state:  # the selected panelist had no line of their own: worker draft
            if "fallback" not in outs:
                return None
            req = p.state["req"]
            text = postprocess(outs["fallback"], selected, req["panel_roles"], req["data"].show_history)
            source = "worker_fallback"
        elif selected in actions and out.decision == "speak":
            text, source = actions[selected].content or "", "think"
        return {"text": text, "speaker": selected, "floor_action": out.decision, "selected_role": selected,
                "floor_mode": "show", "floor_reason_code": out.reason_code, "text_source": source,
                "decisions": {r: actions[r].decision for r in PANEL}, "candidates": got["candidates"],
                "controller_called": got["controller"] is not None, "controller_raw": outs.get("controller"),
                "addressee_rule": got["addressee"], "raw": json.dumps({r: outs.get(f"think:{r}") for r in PANEL})}

    def requests(self, p: Pending, stage: int) -> list[Request]:
        from eval.think_action_prompt import think_request

        if p.obs.get("floor_item"):
            return self.floor_requests(p, stage)
        if stage == 0 and self.planner_request(p):
            p.state["planned"] = True
            return self.planner_request(p)
        first = 1 if p.state.get("planned") else 0
        obs = p.obs
        if stage == first:
            env, tid = self.pack_env(obs)
            prefix = self.prefix(obs)
            if show_persona_swap(obs):
                prefix["persona_swap"] = True
            with _env(**env):
                req = think_request(prefix, obs["role"], human_text=obs.get("audience_turn"), plan=self.plan(p),
                                    topic_id=tid)
            req["messages"], p.state["fact_instruction"] = show_fact_instruction(req["messages"], obs)
            p.state["think"] = req
            return [Request("think", req["messages"], req["temperature"], req["max_tokens"], p.seed, obs["item_id"])]
        if stage == first + 1 and p.state.get("need_fallback"):
            r, req = self.worker_request(p, "fallback")
            p.state["req"] = req
            return [r]
        return []

    def finish(self, p: Pending, stage: int) -> dict | None:
        from agent.interaction.think_action import parse_think_action
        from eval.model_only import postprocess
        from eval.think_action_prompt import spoken_content

        if p.obs.get("floor_item"):
            return self.floor_finish(p)
        outs = p.state.get("outputs", {})
        if "think" not in outs:
            return None
        think = p.state["think"]
        if "decision" not in p.state:
            decision, line = spoken_content(outs["think"], think["interval_text"])
            parsed = parse_think_action(outs["think"], floor_state="free", interval_text=think["interval_text"],
                                        allow_backchannels=True).action
            p.state["decision"] = decision
            p.state["action"] = floor_action(decision, list(getattr(parsed, "backchannels", ()) or ()))
            p.state["line"] = line
            if not line and self.fallback:
                p.state["need_fallback"] = True
                return None
        if p.state.get("need_fallback") and "fallback" not in outs:
            return None
        res = {"speaker": p.obs["role"], "floor_action": p.state["action"], "decision": p.state["decision"],
               **({"fact_instruction": p.state["fact_instruction"]} if p.state.get("fact_instruction") else {}),
               "raw": outs["think"], "fact_pack_offered": think.get("fact_pack") or None}
        if p.state.get("line"):
            return {**res, "text": p.state["line"], "text_source": "think"}
        if "fallback" in outs:
            req = p.state["req"]
            text = postprocess(outs["fallback"], p.obs["role"], req["panel_roles"], req["data"].show_history)
            return {**res, "text": text, "text_source": "worker_fallback"}
        return {**res, "text": "", "text_source": "none"}


SYSTEMS = {"plain": PlainLLM, "worker": TalkshowWorker, "think_action": TalkshowThinkAction}


def make_system(name: str, backend: Any, arm: str = "production", **kw: Any) -> System:
    if name == "plain":
        if arm != "production":
            raise SystemExit("--arm applies to the show's systems only")
        return PlainLLM(backend, **kw)
    return SYSTEMS[name](backend, arm=arm, **kw)
