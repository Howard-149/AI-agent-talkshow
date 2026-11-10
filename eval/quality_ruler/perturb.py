"""Known-bad perturbations (FBI-style) of one-beat episodes. Each perturbation targets specific dimensions (EXPECT);
validation gate (a) requires the own dimension to drop in ≥ 90% of cases and the other dimensions to stay put.

All perturbations act only on the scored panel lines of the beat; context lines are untouched.
persona_inversion, strip_reasons and verbose_pad need an LLM rewrite (GPU stages `invert` / `rewrite`); the others
are deterministic (seeded).
"""

from __future__ import annotations

import copy
import random

from eval.quality_ruler.items import GENERIC_POOL

# own measures per perturbation: jury dimensions and/or automatic metrics ("auto:<metric>")
EXPECT = {
    "generic_swap": ["specific", "auto:specific.1-generic_sim"],
    "label_swap": ["fidelity", "distinct"],
    "duplicate_argument": ["novelty", "repetition", "auto:novelty.share_novel"],
    "same_opener": ["auto:repetition.1-repeated_opener"],  # the argument-repetition item cannot see openers
    "length_pad": [],  # content-free padding: should move NOTHING except length (length-control check)
    "turn_shuffle": ["interaction", "flow"],
    "persona_inversion": ["fidelity", "distinct"],
}
# Final measure set (ruler.py --v2): content-neutral padding (LLM verbose rewrite) replaces length_pad; new own dims.
EXPECT_V2 = {
    "generic_swap": ["specific", "auto:specific.1-generic_sim"],
    "label_swap": ["fidelity", "distinct"],
    "duplicate_argument": ["auto:novelty.share_novel", "repetition2"],
    "same_opener": ["auto:repetition.1-repeated_opener"],
    "turn_shuffle": ["interaction2", "flow"],
    "persona_inversion": ["fidelity", "distinct"],
    "offthread_generic": ["interaction2"],
    "strip_reasons": ["argument"],
    "verbose_pad": [],   # same content in ~1.8× words: must move nothing but length
    "host_template": ["host"],
}
OFFTHREAD_OPENER = {"commentator": "I hear Amy's point, and ", "guest": "I hear Ryan's point, and "}
HOST_TEMPLATE = "All right, open floor — who'd like to speak?"
PAD = " And honestly, that is something all of us should really be thinking about very carefully going forward."
OPENER = "I hear you, and I'd add that "
SWAP = {"commentator": ("guest", "Amy"), "guest": ("commentator", "Ryan")}


OTHER_POOL: dict[str, dict[str, list[str]]] = {}  # topic_id → role → panel lines from OTHER motions (set by build)


def _lower_first(t: str) -> str:
    return t[0].lower() + t[1:] if t and not t.startswith("I ") and not t.startswith("I'") else t


def perturb(ep: dict, kind: str, rng: random.Random) -> dict | None:
    e = copy.deepcopy(ep)
    e.update({"id": f"{ep['id']}~{kind}", "base_id": ep["id"], "source": "perturbed", "perturbation": kind,
              "base_source": ep["source"], "unit": False})  # a perturbed copy is never itself a unit
    sc = e["scored"]
    L = e["lines"]
    if kind == "generic_swap":  # half of the scored lines (≥ 2) → generic filler, same speaker
        for i in rng.sample(sc, max(2, len(sc) // 2)):
            L[i]["text"] = rng.choice(GENERIC_POOL)
    elif kind == "label_swap":  # the words stay, the speaker labels swap (Ryan's lines now attributed to Amy …)
        for i in sc:
            L[i]["role"], L[i]["speaker"] = SWAP[L[i]["role"]]
    elif kind == "duplicate_argument":  # the second half of the beat repeats the first half (same speakers)
        half = len(sc) // 2
        for a, b in zip(sc[:half], sc[half:]):
            if L[a]["role"] == L[b]["role"]:
                L[b]["text"] = L[a]["text"]
            else:  # find the same speaker's earlier line
                src = next((x for x in sc[:half] if L[x]["role"] == L[b]["role"]), a)
                L[b]["text"] = L[src]["text"]
    elif kind == "same_opener":
        for i in sc:
            L[i]["text"] = OPENER + _lower_first(L[i]["text"])
    elif kind == "length_pad":
        for i in sc:
            L[i]["text"] = L[i]["text"].rstrip() + PAD
    elif kind == "turn_shuffle":  # reorder the scored panel lines (each keeps its speaker) so that the order of
        # speakers no longer alternates — otherwise ~30% of shuffles of a 4-line, 2-speaker beat keep alternation
        order = sc[:]
        for _ in range(200):
            rng.shuffle(order)
            roles = [L[i]["role"] for i in order]
            if order != sc and any(a == b for a, b in zip(roles, roles[1:])):
                break
        moved = [copy.deepcopy(L[i]) for i in order]
        for i, m in zip(sc, moved):
            L[i] = m
    elif kind == "persona_inversion":
        return None  # built by the GPU `invert` stage (needs an LLM rewrite)
    elif kind == "offthread_generic":  # formulaic opener + a same-role line from ANOTHER motion (needs other_pool)
        pool = OTHER_POOL.get(ep["topic_id"], {})
        for i in sc[1:]:
            cand = pool.get(L[i]["role"], [])
            if cand:
                L[i]["text"] = OFFTHREAD_OPENER[L[i]["role"]] + _lower_first(rng.choice(cand))
    elif kind == "host_template":
        hs = e.get("host_scored", [])
        if not hs:
            return None
        for i in hs:
            L[i]["text"] = HOST_TEMPLATE
    else:
        raise ValueError(kind)
    return e


INVERT_PROMPT = {
    "commentator": ("Rewrite this talk-show line so that it keeps exactly the same content and position, but is spoken in "
                    "the OPPOSITE style: warm, gentle, measured, acknowledging the other view first, 2–3 sentences, "
                    "like a thoughtful friend. Output only the rewritten line.\n\nLine: {text}"),
    "guest": ("Rewrite this talk-show line so that it keeps exactly the same content and position, but is spoken in the "
              "OPPOSITE style: sharp, blunt, ironic and challenging, 1–2 sentences, like live TV commentary. "
              "Output only the rewritten line.\n\nLine: {text}"),
}


REWRITE_PROMPT = {
    "strip_reasons": ("Rewrite this talk-show line keeping ONLY its claim or position. Remove every reason, explanation, "
                      "example, number and piece of evidence. Keep the speaker's voice and keep it one or two short "
                      "sentences. Output only the rewritten line.\n\nLine: {text}"),
    "verbose_pad": ("Rewrite this talk-show line to say exactly the same thing in about twice as many words. Restate and "
                    "elaborate the SAME points; add NO new claims, reasons, examples, numbers or names. Keep the speaker's "
                    "voice. Output only the rewritten line.\n\nLine: {text}"),
}
