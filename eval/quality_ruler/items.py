"""Quality ruler: item wording, persona traits, generic-filler pool, persuasiveness anchors.

Every checklist item is a yes/no question about ONE panel line (or one host line) read in its context (the previous lines
of the episode). Score per item = P(yes) from the judge's next-token logprobs; `reverse` items count as passed when the
answer is "no". A dimension's checklist score = mean share of its items passed. Literature: CheckEval (binary
checklists), Wachsmuth et al. 2017 (argument cogency), MPCEval (multi-party novelty/interaction), PersonaEval
(attribution), TreeDebater (1–5 persuasiveness anchors).
"""

from __future__ import annotations

# dimension → (description, [(item_id, question, reverse)])
LINE_ITEMS: dict[str, tuple[str, list[tuple[str, str, bool]]]] = {
    "specific": ("Specific, not generic", [
        ("S1", "Would this line fit almost any debate topic if only the topic words were changed?", True),
        ("S2", "Does this line mention a concrete detail — a number, a name, a place, an event, or a specific example?", False),
        ("S3", "Is this line about the specific point under discussion, rather than the topic in general?", False),
    ]),
    "novelty": ("Adds new information", [
        ("N1", "Does this line add a fact, an example, or an angle that has not been said earlier in the conversation?", False),
        ("N2", "Was the main point of this line already made earlier in the conversation, by anyone?", True),
    ]),
    "interaction": ("Engages earlier turns", [
        ("I1", "Does this line directly respond to a specific point made by an earlier speaker (rebutting it, building on "
               "it, or questioning it)?", False),
        ("I2", "Does this line make clear whose point it is responding to (by name or unmistakable reference)?", False),
        ("I3", "Could this line have been said without having heard any of the previous turns?", True),
    ]),
    "argument": ("Argument quality (cogency)", [
        ("A1", "Does this line state a clear claim or position?", False),
        ("A2", "Does this line give a reason for its claim?", False),
        ("A3", "Does this line support its claim with evidence or a concrete example?", False),
        ("A4", "Does this line address the opposing view or a likely objection?", False),
    ]),
    "repetition": ("Non-repetition", [
        ("P1", "Does this line repeat an argument that was already made earlier in this conversation?", True),
    ]),
}

# Persona fidelity: items for the line's OWN persona (wording from the production persona cards).
PERSONA_ITEMS: dict[str, list[tuple[str, str, bool]]] = {
    "commentator": [  # Ryan: sharp, provocative, punchy, 1–2 sentences, challenges, pushes for a position
        ("R1", "Is this line sharp and punchy — blunt, ironic or provocative, like live TV commentary?", False),
        ("R2", "Is this line short — one or two sentences?", False),
        ("R3", "Does this line challenge someone or push them to take a real position?", False),
    ],
    "guest": [  # Amy: warm, grounded, personal stories/lived experience, acknowledges nuance before disagreeing, 2–3 sentences
        ("M1", "Is the tone of this line warm and gentle?", False),
        ("M2", "Does this line bring in personal experience, a personal story, or how real people are affected?", False),
        ("M3", "Does this line acknowledge the other person's view or the nuance before disagreeing or adding to it?", False),
    ],
}

HOST_ITEMS: list[tuple[str, str, bool]] = [
    ("H1", "Does the host's line follow up on what was just said?", False),
    ("H2", "Does the host's line move the discussion forward (a sharper question, a new angle, or a reason to bring in "
           "the next speaker)?", False),
]

PERSONA_CARDS = {
    "commentator": "Speaker A: the panel's sharp, provocative edge — punchy, ironic, blunt; challenges people who dodge; "
                   "keeps lines to 1–2 sentences; sounds like live TV commentary.",
    "guest": "Speaker B: the panel's warm, grounded, empathetic voice — personal stories, measured opinions, acknowledges "
             "nuance before gently disagreeing; 2–3 sentences; sounds like a thoughtful friend.",
}

# Pool of generic panel filler (used by the 'specific' metric and the generic-swap perturbation). Topic-free on purpose.
GENERIC_POOL = [
    "I think this is a really complex issue, and there are good points on both sides.",
    "I hear what you're saying, but I think we need to look at the bigger picture here.",
    "At the end of the day, it really comes down to the people who are affected by this.",
    "I think we have to be careful not to oversimplify something this important.",
    "That's a fair point, but I'd also consider the long-term consequences.",
    "We need to find a balance that works for everyone involved.",
    "I think the real issue is that we're not having an honest conversation about this.",
    "There's no easy answer here, and I think we have to acknowledge that.",
    "I agree with part of that, but I think the situation is more nuanced.",
    "What matters most is that we keep the human element at the center of this debate.",
    "I think both sides are missing something important in this discussion.",
    "It's not about one side winning; it's about finding a solution that actually works.",
    "We can't ignore the reality on the ground, but we also can't stop moving forward.",
    "I think the system itself is the problem, not the individuals caught up in it.",
    "Let's not lose sight of what's really at stake here.",
    "I understand the concern, and I think it deserves a serious answer.",
    "That's exactly the kind of thinking that got us into this situation in the first place.",
    "I'd push back on that a little — it's not as simple as it sounds.",
    "We have to start by asking the right questions before we jump to conclusions.",
    "Change is hard, but I think it's necessary if we want things to get better.",
    "Honestly, I think everyone here wants the same thing; we just disagree on how to get there.",
    "The evidence is mixed, and I think we should be humble about what we know.",
    "I think we're framing this as an either-or choice when it doesn't have to be.",
    "That sounds good in theory, but the real world is a lot messier.",
]

# TreeDebater-style 1–5 persuasiveness, scored per beat (the panel exchange as a whole).
PERSUASION_ANCHORS = (
    "1 = no real argument: assertions, slogans or filler; nothing a listener could weigh.\n"
    "2 = weak: a position with vague support; reasons are generic or unrelated to the point at hand.\n"
    "3 = adequate: clear positions with some relevant reasons, but little evidence and little engagement with the other side.\n"
    "4 = strong: clear claims, relevant reasons and concrete evidence or examples, and the speakers answer each other.\n"
    "5 = compelling: as 4, plus the strongest objection is met head-on and the exchange would change a thoughtful "
    "listener's mind or sharpen the disagreement."
)

PAIRWISE_QUESTION = ("Which panel exchange is better as a live talk-show discussion — more specific, more informative, "
                     "with the panelists actually engaging each other and each sounding like themselves? "
                     "Answer with a single letter: A or B.")

JUDGE_SYSTEM = ("You are a careful annotator of talk-show transcripts. Answer exactly what is asked, using only the "
                "transcript you are shown.")
