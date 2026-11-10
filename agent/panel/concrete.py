"""Concrete panel lines (all flags default OFF).

Offline ablations found the panel's weakest quality dimension is specificity, and that three cheap changes
together fix most of it without hurting persona fidelity or listening:

TALKSHOW_DIALOGUE_LIBRARY=<id>   use another move library (config/dialogue_libraries/<id>.yaml), e.g. panel_concrete:
                                 the same moves whose examples each name a figure, place, person or case.
TALKSHOW_PANEL_CONCRETE=1        append the "ground your line in one concrete fact" instruction to every panel prompt.
TALKSHOW_PANEL_BEST_OF=N         draft N candidates in one request (vLLM n) and keep the best by a judge-free lexical
                                 score: specificity − generic filler − overlap with earlier lines − reused/top opener.

The rerank is judge-free so it adds no model call.
"""

from __future__ import annotations

import os
import re

CONCRETE = ("Ground your line in one concrete fact, number, name or example — from the discussion or from what you "
            "know. No generic statements.")

# Production's 10 most frequent line openers (first 3 words) in baseline episodes.
GLOBAL_TOP_OPENERS = ["i hear ryan", "i hear ryan's", "amy is right", "i hear what", "amy's point about", "ryan is right",
                      "ryan i understand", "i hear you", "i think the", "i agree with"]

STOP = set("""a an the and or but so to of in on at for with by from as is are was were be been being it its this that these those
i you he she we they me him her us them my your his our their not no do does did have has had will would can could should
just very really more most much many some any all about than then there here what which who whom when where why how if
because also only even still too into out up down over again like well yes yeah okay ok think know mean say said going
get got make one way thing things people right""".split())
NUM = re.compile(r"\d|\b(?:percent|million|billion|trillion|hundred|thousand|half|third|quarter|dozen|two|three|four|five|"
                 r"six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b", re.IGNORECASE)

GENERIC_POOL = [
    'I think this is a really complex issue, and there are good points on both sides.',
    "I hear what you're saying, but I think we need to look at the bigger picture here.",
    'At the end of the day, it really comes down to the people who are affected by this.',
    'I think we have to be careful not to oversimplify something this important.',
    "That's a fair point, but I'd also consider the long-term consequences.",
    'We need to find a balance that works for everyone involved.',
    "I think the real issue is that we're not having an honest conversation about this.",
    "There's no easy answer here, and I think we have to acknowledge that.",
    'I agree with part of that, but I think the situation is more nuanced.',
    'What matters most is that we keep the human element at the center of this debate.',
    'I think both sides are missing something important in this discussion.',
    "It's not about one side winning; it's about finding a solution that actually works.",
    "We can't ignore the reality on the ground, but we also can't stop moving forward.",
    'I think the system itself is the problem, not the individuals caught up in it.',
    "Let's not lose sight of what's really at stake here.",
    'I understand the concern, and I think it deserves a serious answer.',
    "That's exactly the kind of thinking that got us into this situation in the first place.",
    "I'd push back on that a little — it's not as simple as it sounds.",
    'We have to start by asking the right questions before we jump to conclusions.',
    "Change is hard, but I think it's necessary if we want things to get better.",
    'Honestly, I think everyone here wants the same thing; we just disagree on how to get there.',
    'The evidence is mixed, and I think we should be humble about what we know.',
    "I think we're framing this as an either-or choice when it doesn't have to be.",
    'That sounds good in theory, but the real world is a lot messier.',
]


def dialogue_library_override() -> str | None:
    v = os.environ.get("TALKSHOW_DIALOGUE_LIBRARY", "").strip()
    return v or None


def concrete_instruction_enabled() -> bool:
    return os.environ.get("TALKSHOW_PANEL_CONCRETE", "0").strip().lower() in ("1", "true", "yes", "on")


def best_of() -> int:
    try:
        return max(1, int(os.environ.get("TALKSHOW_PANEL_BEST_OF", "1") or 1))
    except ValueError:
        return 1


def words(t: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$%']+", t)


def content(t: str) -> list[str]:
    return [w.lower() for w in words(t) if w.lower() not in STOP and len(w) > 2]


def names_figs(t: str) -> set[str]:
    out = set(m.group(0).lower() for m in re.finditer(r"\d[\d,.]*%?", t))
    toks = t.split()
    for k, w in enumerate(toks):
        if k > 0 and not toks[k - 1].endswith((".", "?", "!")) and re.match(r"^[A-Z][a-z]{2,}", w):
            out.add(re.sub(r"\W", "", w).lower())
    return out


def opener(t: str, n: int = 3) -> str:
    return " ".join(re.findall(r"[A-Za-z']+", t.lower())[:n])


def rerank_score(cand: str, earlier: list[str], used_openers: set[str], opener_penalty: float = 1.0) -> float:
    """Higher = more specific, more novel, less templated."""
    nw = max(len(words(cand)), 1)
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


def pick_best(cands: list[str], earlier: list[str]) -> tuple[int, list[float]]:
    """Index of the best spoken candidate + all scores (ties → first, i.e. the first sample)."""
    used = {opener(x) for x in earlier}
    scores = [rerank_score(c, earlier, used) for c in cands]
    return max(range(len(cands)), key=lambda i: (scores[i], -i)), scores
