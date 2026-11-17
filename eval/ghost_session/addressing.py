"""Who a show line is handing the floor to: the guest, another participant, or nobody.

Panelists use names as vocatives all the time ("Howard, that's fair."), so a name alone
does not hand anyone the floor. A participant is *handed* the floor when a vocative of
their name (or "our guest", for the guest) is tied to a question or an explicit
invitation ("the floor is yours", "what do you think") in the same or the next sentence,
and that question / invitation is in the line's last two sentences (a real hand-over
ends the line). When several are handed the floor, the last one wins
("Howard, good point. Amy, what do you think?" → Amy).

``addressing`` returns:

* ``"direct"``   — the floor goes to the guest: he must answer.
* ``"other:<Name>"`` — the floor goes to someone else: the guest stays out of it.
* ``"asked"``    — the guest's name is tied to a question, but it is not a hand-over
  (mid-line, rhetorical): the LLM decides, outside the cooldown.
* ``"mentioned"`` — his name only (a bare vocative, "like Howard said"): the LLM
  decides, within the cooldown.
* ``""``         — neither.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SENTENCE = re.compile(r"[^.!?]+[.!?]*")
_GUEST_WORDS = re.compile(r"\b(our (human |studio |special )?guest|our caller)\b", re.I)
_INVITE = re.compile(
    r"\b(the floor is yours|what do you think|what'?s your take|your take|over to you|"
    r"tell us|jump in|weigh in|how about you|what about you|go ahead|your turn|"
    r"walk us through|help us out|i want to hear from you|let'?s hear from)\b",
    re.I,
)
# Capitalized words that open a sentence with a comma but are not names.
_NOT_NAMES = {
    w.lower()
    for w in (
        "Yes Yeah Yep No Nope Oh Ooh Ah Aha Hmm Ha Haha So Well Okay Ok OK Look Listen Now Honestly "
        "Exactly Totally Bingo Right Sure Wow Hey Absolutely Seriously Agreed True Fair Alright "
        "Again Also Plus Still Actually Basically See Come Please Thanks Thank First Second Third "
        "Finally Anyway Folks Everyone Everybody Friends Guys Indeed Definitely Fine Great Good "
        "Perfect Amen Correct Precisely Boom Man Dude Wait Hold But And Or Because Then Sadly "
        "Luckily Frankly Personally Obviously Clearly Look Remember Imagine Think Here There "
        "Today Tonight Yesterday Tomorrow Meanwhile However Plus Also Like Sorry Wonderful Lovely "
        "Brilliant Nice Cool Awesome Ouch Yikes Whoa Gosh Geez God Lord Ladies Gentlemen Hi Hello "
        "Welcome Bye Cheers Congrats Congratulations Listen Speaking Personally"
    ).split()
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text.replace("…", ".")) if s.strip()]


def _is_question_or_invite(sentence: str) -> bool:
    return sentence.rstrip().endswith("?") or bool(_INVITE.search(sentence))


def _vocative(sentence: str, name: str) -> bool:
    """``name`` is spoken to in ``sentence`` (not about: "Amy's right", "like Ryan")."""
    who = re.escape(name)
    if not re.search(rf"\b{who}\b", sentence, re.I):
        return False
    third = re.search(
        rf"\b(like|as|what|that|said|says|by|from|to|with|and|about|than|for)\s+{who}\b(?!\s*,)"
        rf"|\b{who}(['’]s|\s+(said|says|just|was|is|has|had|thinks|mentioned|nailed|brought|got|made))\b",
        sentence,
        re.I,
    )
    if third:
        return False
    return bool(
        re.search(rf"(^|[,;:—–-]\s*|\b(oh|so|well|now|okay|yes|yeah|and|but)\s+){who}\b", sentence, re.I)
        or re.search(rf"\b{who}\s*([,?!.;:—–-]|$)", sentence, re.I)
    )


def _heuristic_names(sentences: list[str], guest: str) -> set[str]:
    """Vocative-looking capitalized words ("Amy, what…", "…, Ryan?") when no roster
    is known yet (the opening hands over to a panelist who has not spoken)."""
    found: set[str] = set()
    for s in sentences:
        for m in re.finditer(r"(?:^|[,;:—–-]\s*)([A-Z][a-z]{1,14})\s*,", s):
            found.add(m.group(1))
        for m in re.finditer(r",\s*([A-Z][a-z]{1,14})\s*[?!.]$", s):
            found.add(m.group(1))
    return {w for w in found if w.lower() not in _NOT_NAMES and w.lower() != guest.lower()}


def floor_target(text: str, guest: str, others: Iterable[str] = ()) -> str | None:
    """The name the line hands the floor to ("" = the guest via "our guest"), or None."""
    sentences = _sentences(text)
    if not sentences:
        return None
    last2 = len(sentences) - 2
    names = {n.strip() for n in others if n and n.strip() and n.strip().lower() != guest.lower()}
    names |= _heuristic_names(sentences[max(0, last2):], guest)
    candidates = [guest, *sorted(names)]
    best: tuple[int, int, str] | None = None
    for name in candidates:
        for i, sent in enumerate(sentences):
            voc = _vocative(sent, name) or (name == guest and bool(_GUEST_WORDS.search(sent)))
            if not voc:
                continue
            for j in (i, i + 1):
                if j < len(sentences) and j >= last2 and _is_question_or_invite(sentences[j]):
                    key = (j, i, name)
                    if best is None or key[:2] > best[:2]:
                        best = key
                    break
    return None if best is None else best[2]


def addressing(text: str, guest: str, others: Iterable[str] = ()) -> str:
    target = floor_target(text, guest, others)
    if target is not None:
        return "direct" if target.lower() == guest.lower() else f"other:{target}"
    sentences = _sentences(text)
    named = [i for i, s in enumerate(sentences) if re.search(rf"\b{re.escape(guest)}\b", s, re.I)]
    if not named:
        return ""
    for i in named:
        if any(_is_question_or_invite(s) for s in sentences[i : i + 2]) and _vocative(sentences[i], guest):
            return "asked"
    return "mentioned"
