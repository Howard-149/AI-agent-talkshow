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
        if req.kind == "think":
            return DryRunLLM.THINK
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
        return [Request("reply", plain_messages(p.obs), self.temperature, self.max_tokens, p.seed, p.obs["item_id"])]

    def finish(self, p: Pending, stage: int) -> dict:
        text = plain_postprocess(p.state["outputs"]["reply"], p.obs["name"])
        floor = None
        if p.obs.get("floor_choice"):  # exact PASS = yield the floor, no line
            floor = "yield" if is_pass(text) else "speak"
            text = "" if floor == "yield" else text
        return {"text": text, "speaker": p.obs["role"], "floor_action": floor, "raw": p.state["outputs"]["reply"]}


# ------------------------------------------------------------------------------------------------ the show

SHOW_ARMS = ("production", "fast", "concrete_fact_pack")


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

    def worker_request(self, p: Pending, kind: str = "reply") -> tuple[Request, dict]:
        from eval.model_only import TEMPERATURE, build_request

        obs = p.obs
        env, tid = self.pack_env(obs)
        swap = show_persona_swap(obs)
        human = obs.get("audience_turn")
        with _env(**env):
            req = build_request(self.prefix(obs), human or "", obs["role"], swap, include_human=human is not None,
                                live_fixes=self.live or bool(env["TALKSHOW_FACT_PACK"]),
                                plan=self.plan(p), topic_id=tid)
        return Request(kind, req["messages"], TEMPERATURE, 512, p.seed, obs["item_id"]), req

    def requests(self, p: Pending, stage: int) -> list[Request]:
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

        if "reply" not in p.state.get("outputs", {}):
            return None
        req = p.state["req"]
        raw = p.state["outputs"]["reply"]
        text = postprocess(raw, p.obs["role"], req["panel_roles"], req["data"].show_history)
        return {"text": text, "speaker": p.obs["role"], "floor_action": None, "raw": raw,
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

    def requests(self, p: Pending, stage: int) -> list[Request]:
        from eval.think_action_prompt import think_request

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
