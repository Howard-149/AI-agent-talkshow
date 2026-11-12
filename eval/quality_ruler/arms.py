"""Generation knobs for model-only episodes (episodes.generate, model_only.py generate --arm).

An arm is a dict of knobs; the empty dict is production. Knobs:
  dialogue_pick     override the scenario's move catalogue pick (production: all) — none | random | rotate
  dialogue_library  override the move library (production: panel) — e.g. panel_concrete
  instruction       extra text appended to the panelist's turn prompt
  opener_ban        True → the prompt lists openers already used in this episode + GLOBAL_TOP_OPENERS as "do not start with"
  lite_plan         True → planner-lite: the latest human turn's figures/entities/stance in the prompt (no [engages] line)
  temperature, min_p  sampling (production: 0.45, no min-p)
  best_of           N candidates, reranked by the judge-free score below
  opener_penalty    weight of the reused/top-opener penalty in that rerank (default 1.0)
  ban_openers       True → vLLM bad_words: the template opener phrases (BANNED_OPENERS) cannot be decoded anywhere
  opener_plan       True → each line is told how to OPEN, sampled per line from OPENER_PLANS (seeded);
                    "persona" → sampled from the speaker's own list (OPENER_PLANS_ROLE)
  live_flags        {env var: value}: the live show's panel flags (agent/panel/*). The prompt is built by the live
                    worker's own code path (model_only.build_request(live_fixes=True)), so offline episodes see exactly
                    the live prompt. Flags not listed are switched off for the run (apply_env).
  fact_pack         True → keep TALKSHOW_FACT_PACK (the pack path, from the environment; required); otherwise it is unset
Any other combination: `ruler.py generate --arm-spec '{"best_of": 4}'`.

Presets: production (no knobs); concrete (move library with facts + concrete instruction + best-of-4, as knobs, without
the uptake and scaffold fixes); concrete_varied_openers; fast (the live FAST flag set of deploy/arms/dialogue-quality.arms,
best-of 1); concrete_fact_pack (fast + the fact pack). TALKSHOW_ADDRESSEE_FLOOR has no effect offline: model-only
episodes fix the speaker order.
"""

from __future__ import annotations

import os
import re

# Production's 10 most frequent line openers (first 3 words) in the model-only production baseline episodes.
GLOBAL_TOP_OPENERS = ["i hear ryan", "i hear ryan's", "amy is right", "i hear what", "amy's point about", "ryan is right",
                      "ryan i understand", "i hear you", "i think the", "i agree with"]

CONCRETE = ("Ground your line in one concrete fact, number, name or example — from the discussion or from what you "
            "know. No generic statements.")

# Concrete moves + concrete instruction + best-of-4 (the live TALKSHOW_PANEL_CONCRETE / TALKSHOW_PANEL_BEST_OF fixes).
CONCRETE_KNOBS = {"dialogue_library": "panel_concrete", "instruction": CONCRETE, "best_of": 4}

# Template opener phrases (from the production top openers, cut to their templated 2–3-word core; Ryan/Amy are the panel's
# fixed names). Banned as token sequences anywhere in the output (vLLM bad_words), with and without a leading space.
BANNED_OPENERS = ["I hear", "Amy is right", "Ryan is right", "Amy's point", "Ryan's point", "I agree with",
                  "I understand", "You're right", "That's a great point", "That's a fair point"]

# Rhetorical entries for the opener plan (one sampled per line). Each names HOW to begin, never a phrase to copy.
OPENER_PLANS = [
    "Open with the concrete fact, number or example itself — no preamble.",
    "Open with a direct, pointed question to the other panelist.",
    "Open with your conclusion, stated plainly in a few words.",
    "Open with a short concession, then turn to your counterpoint.",
    "Open with what is at stake for a real person or place.",
    "Open with a one-sentence case or story from the real world.",
    "Open by bluntly naming what you disagree with.",
    "Open with a comparison to another country, era or industry.",
]


# Persona-aware entries: the generic plan cost persona fidelity (the guest persona is written to acknowledge before
# disagreeing, so a blanket "do not credit the previous speaker" fights it). Each role gets entries that fit its card; the guest may
# acknowledge briefly in her own words, just not with the stock formula.
OPENER_PLANS_ROLE = {
    "commentator": [
        "Open with a direct, pointed question to the other panelist.",
        "Open by bluntly naming what you disagree with.",
        "Open with your conclusion, stated plainly in a few words.",
        "Open with the concrete fact, number or example itself — no preamble.",
        "Open with a comparison to another country, era or industry.",
    ],
    "guest": [
        "Open with a one-sentence case or story from the real world.",
        "Open with what is at stake for a real person or place.",
        "Open with a short concession in your own words, then turn to your counterpoint.",
        "Open with the concrete fact, number or example itself — no preamble.",
        "Open with a gentle question that reframes the issue.",
    ],
}
ROLE_TAIL = {
    "commentator": " Do not begin by restating or crediting what the previous speaker said.",
    "guest": " If you acknowledge the other speaker, do it briefly in your own words — never with \"I hear…\" or \"X is right\".",
}


def opener_plan_text(seed: int, role: str | None = None) -> str:
    import random as _r

    if role in OPENER_PLANS_ROLE:
        return _r.Random(seed).choice(OPENER_PLANS_ROLE[role]) + ROLE_TAIL[role]
    return (_r.Random(seed).choice(OPENER_PLANS) + " Do not begin by restating or crediting what the previous speaker "
            "said.")


# The live FAST set (deploy/arms/dialogue-quality.arms), best-of 1.
FAST_FLAGS = {"TALKSHOW_ADDRESSEE_FLOOR": "1", "TALKSHOW_UPTAKE_PLANNER": "lite", "TALKSHOW_PANEL_SCAFFOLD_FIX": "1",
              "TALKSHOW_DIALOGUE_LIBRARY": "panel_concrete", "TALKSHOW_PANEL_CONCRETE": "1",
              "TALKSHOW_PANEL_BEST_OF": "1"}
# Every flag apply_env sets or clears (the replayable panel flags, agent/session/ghost_replay.REPLAY_FLAGS).
LIVE_FLAG_NAMES = ("TALKSHOW_UPTAKE_PLANNER", "TALKSHOW_PANEL_SCAFFOLD_FIX", "TALKSHOW_ADDRESSEE_FLOOR",
                   "TALKSHOW_DIALOGUE_LIBRARY", "TALKSHOW_PANEL_CONCRETE", "TALKSHOW_PANEL_BEST_OF")

ARMS: dict[str, dict] = {
    "production": {},
    "concrete": CONCRETE_KNOBS,
    # concrete + template-opener ban + best-of-8 with a strong reused-opener penalty + persona-aware opener plan
    "concrete_varied_openers": {**CONCRETE_KNOBS, "ban_openers": True, "best_of": 8, "opener_penalty": 3.0,
                                "opener_plan": "persona"},
    "fast": {"live_flags": FAST_FLAGS, "best_of": 1},
    "concrete_fact_pack": {"live_flags": FAST_FLAGS, "best_of": 1, "fact_pack": True},
    # Same as concrete_fact_pack, but the style examples carry no figures or named cases of their own.
    "concrete_fact_pack_nofig": {"live_flags": {**FAST_FLAGS, "TALKSHOW_DIALOGUE_LIBRARY": "panel_concrete_nofig"},
                                 "best_of": 1, "fact_pack": True},
}


def apply_env(arm: dict) -> None:
    """Set this process's panel flags for one arm: the arm's live_flags, every other panel flag off, and the fact pack
    only for arms that use it (so a value in the environment or .env cannot leak into another arm)."""
    flags = arm.get("live_flags") or {}
    for name in LIVE_FLAG_NAMES:
        if name in flags:
            os.environ[name] = str(flags[name])
        else:
            os.environ.pop(name, None)
    os.environ.pop("TALKSHOW_FACT_PACK_TOPIC", None)  # offline episodes pass their topic id
    if arm.get("fact_pack"):
        if not os.environ.get("TALKSHOW_FACT_PACK", "").strip():
            raise SystemExit("this arm needs TALKSHOW_FACT_PACK=<fact pack json> in the environment")
    else:
        os.environ.pop("TALKSHOW_FACT_PACK", None)


def opener(t: str, n: int = 3) -> str:
    return " ".join(re.findall(r"[A-Za-z']+", t.lower())[:n])


def ban_text(used: list[str]) -> str:
    seen, out = set(), []
    for o in used + GLOBAL_TOP_OPENERS:
        if o and o not in seen:
            seen.add(o)
            out.append(o)
    return ("Do not begin your line with any of these openings (they have been overused): "
            + "; ".join(f'"{o}…"' for o in out) + ". Open differently — with your point itself.")


# ---- judge-free reranking for best-of-N (lexical; no embedding model needed inside the generation job)

def _words(t: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$%']+", t)


def rerank_score(cand: str, earlier: list[str], used_openers: set[str], opener_penalty: float = 1.0) -> float:
    """Higher = more specific, more novel, less templated. Figures+names per 100 words, minus overlap with the generic
    filler pool and with earlier lines (content-bigram Jaccard), minus a penalty for a reused/top opener."""
    from eval.quality_ruler.auto import NUM, content, names_figs
    from eval.quality_ruler.items import GENERIC_POOL

    nw = max(len(_words(cand)), 1)
    spec = 100 * (len(NUM.findall(cand)) + len(names_figs(cand))) / nw
    mine = content(cand)
    bi = set(zip(mine, mine[1:]))

    def jac(other: str) -> float:
        o = content(other)
        ob = set(zip(o, o[1:]))
        return len(bi & ob) / max(len(bi | ob), 1)

    generic = max((jac(g) for g in GENERIC_POOL), default=0.0)
    novelty_pen = max((jac(e) for e in earlier), default=0.0)
    op = opener(cand)
    op_pen = opener_penalty if (op in used_openers or op in GLOBAL_TOP_OPENERS) else 0.0
    return spec / 10 - 2 * generic - 2 * novelty_pen - op_pen
