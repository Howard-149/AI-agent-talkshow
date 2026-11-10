"""The grounding task's fact edit: change one number in a checked fact by a fixed rule.

Rule (applied to the first editable number of a fact, reading left to right):
- a year (4 digits, 1000-2100, no currency, percent or scale word): minus 10 ("1971" -> "1961");
- a percentage: times 2 if the result stays below 100, else divided by 2 ("38 percent" -> "76 percent", "75%" -> "37.5%");
- any other number (counts, money, decimals): times 2 ("$30" -> "$60", "1.5 billion" -> "3 billion").
Format is kept: the currency sign, the unit and scale words around the number are untouched, thousands separators are
kept when the original had them, the number of decimals is the original's (one more for a halved value), and a zero
fraction is dropped unless the original was money with cents ("$7.25" -> "$14.50").

Not editable, so skipped: zero, ordinals and decades ("21st", "1970s"), numbers glued to a word, hyphen or plus
("COVID-19", "3-year", "P5+1"), fractions and ratios ("24/7"), ends of a range ("10-20", "between 10 and 20",
"500,000 to nearly 2 million"), both parts of "X of Y" ("62 of 84 deputies", "30th out of 197"), days of the month,
numbers next to a capitalised word that is not a month or a sentence opener (names such as "Section 230", "Boeing 737",
"4 Day Week Global"), and ages whose double passes 100. Doubling or halving can still make a fact implausible (a
percentage that "increased to 45.5% from 55%"); that is accepted, the rule stays fixed.

An edit is kept only if the unsupported-specifics extractor (eval/quality_ruler/specifics.py) sees exactly one value
change (old value -> new value), and neither value appears anywhere else the panelist can see (topic, transcript,
audience turn, the other offered facts). Then a reply that states the new value followed the edited fact, one that
states the old value did not.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

NUMBER = re.compile(r"(?<![\w.,/+\-–])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w/]|-\w|\.\d|,\d)")
PERCENT_AFTER = re.compile(r"^\s*(?:%|percent\b|per cent\b|percentage points?\b)", re.I)
SCALE_AFTER = re.compile(r"^\s*(?:hundred|thousand|million|billion|trillion|bn|tn|m|k)\b", re.I)
MONEY_AFTER = re.compile(r"^\s*(?:hundred |thousand |million |billion |trillion )?(?:dollars?|cents?|euros?|pounds?)\b",
                         re.I)
RANGE_AFTER = re.compile(r"^\s*(?:%|percent)?\s*(?:-|–|—|to|and|or)\s*(?:[a-z]+\s+)?[$£€]?\d", re.I)
PART_OF_AFTER = re.compile(r"^\s*(?:%|percent)?\s+(?:out\s+)?of\s+(?:[a-z]+\s+)?[$£€]?\d", re.I)
PART_OF_BEFORE = re.compile(r"\d(?:st|nd|rd|th)?\s*(?:%|percent)?\s+(?:out\s+)?of\s+(?:[a-z]+\s+)?[$£€]?$", re.I)
NAME_AFTER = re.compile(r"^\s+[A-Z][a-z]")
AGE_BEFORE = re.compile(r"\b(?:age|aged|ages)(?:\s+(?:of|is|was|at|to|will be))?\s*$", re.I)
AGE_AFTER = re.compile(r"^[\s-]*(?:years?|yrs?)[\s-]*old\b", re.I)
RANGE_BEFORE = re.compile(r"\d\s*(?:%|percent)?\s*(?:-|–|—|to|and|or)\s*(?:[a-z]+\s+)?[$£€]?$", re.I)
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
OPENERS = {"in", "by", "since", "from", "until", "after", "before", "about", "around", "over", "nearly", "some", "only",
           "of", "at", "on", "as", "almost", "roughly", "between", "more", "less", "fewer", "up", "within", "under",
           "for", "to", "the", "a", "an", "and", "but", "with", "every", "each", "per", "than"}


@dataclass(frozen=True)
class NumberEdit:
    start: int
    end: int
    old: str  # the number as written ("1,500", "7.25")
    new: str
    kind: str  # year | percent | number
    old_key: str = ""  # value key of the specifics extractor ("n:1500"), filled by edit_fact
    new_key: str = ""

    def apply(self, text: str) -> str:
        return text[: self.start] + self.new + text[self.end:]


def _fmt(value: float, decimals: int, commas: bool, keep_zero_fraction: bool) -> str:
    s = f"{value:,.{decimals}f}" if commas else f"{value:.{decimals}f}"
    if decimals and not keep_zero_fraction and float(s.replace(",", "")) == int(float(s.replace(",", ""))):
        s = s.split(".")[0]
    elif decimals and not keep_zero_fraction:
        s = s.rstrip("0").rstrip(".")
    return s


def candidates(text: str) -> list[NumberEdit]:
    """Every editable number in text, with its rule-edited replacement, in reading order."""
    out = []
    for m in NUMBER.finditer(text):
        whole, frac = m.group(1), m.group(2) or ""
        start, end = m.start(), m.end()
        before, after = text[:start], text[end:]
        value = float(whole.replace(",", "") + frac)
        if value == 0:
            continue
        if (RANGE_AFTER.match(after) or RANGE_BEFORE.search(before) or PART_OF_AFTER.match(after)
                or PART_OF_BEFORE.search(before)):
            continue
        if NAME_AFTER.match(after):  # part of a name or a date ("4 Day Week Global", "24 April")
            continue
        if (AGE_BEFORE.search(before) or AGE_AFTER.match(after)) and value * 2 > 100:  # no 134-year-olds
            continue
        money_sign = before.endswith(("$", "£", "€"))
        prev = re.findall(r"[A-Za-z][A-Za-z'’.]*", before[-40:])
        prev_word = prev[-1] if prev and re.search(r"[A-Za-z][A-Za-z'’.]*\s*[$£€]?$", before) else ""
        pw = prev_word.lower().rstrip(".")
        if prev_word[:1].isupper() and pw not in MONTHS | OPENERS:
            continue
        if pw in MONTHS and not (len(whole) == 4 and not frac):  # a day of the month
            continue
        percent = bool(PERCENT_AFTER.match(after))
        money = money_sign or bool(MONEY_AFTER.match(after))
        decimals = len(frac) - 1 if frac else 0
        commas = "," in whole
        if (not frac and not commas and len(whole) == 4 and 1000 <= value <= 2100 and not money and not percent
                and not SCALE_AFTER.match(after)):
            kind, new = "year", str(int(value) - 10)
        elif percent:
            kind = "percent"
            new = (_fmt(value * 2, decimals, commas, False) if value * 2 < 100
                   else _fmt(value / 2, decimals + 1, commas, False))
        else:
            kind = "number"
            new = _fmt(value * 2, decimals, commas, money and decimals == 2)
        out.append(NumberEdit(start, end, m.group(0), new, kind))
    return out


def edit_fact(fact: str, context: Iterable[str] = ()) -> tuple[str, NumberEdit] | None:
    """(edited fact, the edit) for the first number whose edit passes the checks, else None. ``context``: every other
    text the panelist sees at this turn (neither the old nor the new value may appear there)."""
    from eval.quality_ruler.specifics import Support, extract

    seen = Support(list(context))
    old_keys = Counter(s["key"] for s in extract(fact) if s["kind"] != "name")
    for c in candidates(fact):
        new_fact = c.apply(fact)
        new_keys = Counter(s["key"] for s in extract(new_fact) if s["kind"] != "name")
        gone, came = old_keys - new_keys, new_keys - old_keys
        if sum(gone.values()) != 1 or sum(came.values()) != 1:
            continue
        (ok,), (nk,) = gone.keys(), came.keys()
        if ok in seen.values or nk in seen.values or nk in old_keys:
            continue
        return new_fact, NumberEdit(c.start, c.end, c.old, c.new, c.kind, ok, nk)
    return None


def follows(reply: str, old_key: str, new_key: str) -> str:
    """Which value of an edited fact a reply states: "new", "old", "both" or "neither"."""
    from eval.quality_ruler.specifics import extract

    keys = {s["key"] for s in extract(reply) if s["kind"] != "name"}
    has_new, has_old = new_key in keys, old_key in keys
    return "both" if has_new and has_old else "new" if has_new else "old" if has_old else "neither"
