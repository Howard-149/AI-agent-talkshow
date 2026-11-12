"""Fact pack: specific facts for panel lines come only from a checked list (flag default OFF).

Asking panelists to be concrete makes a small model invent figures, names and studies. The fact pack gives each topic a
short list of checked facts (sentences taken from Wikipedia), offers the most relevant ones in every panel prompt, and
tells the panelist to take specific numbers, names, studies and cases only from that list or from what was said in the
show.

TALKSHOW_FACT_PACK=<path>     JSON file {topic_id: {"topic": str, "pages": [...], "facts": [{"fact", "source_sentence",
                              "page", "url", ...}]}}. Unset = off: the panel prompt is unchanged.
TALKSHOW_FACT_PACK_K=<n>      facts offered per panel line (default 6).
TALKSHOW_FACT_PACK_TOPIC=<id> pin the topic for a live show whose topic is known in advance (optional).

Both panelists get the same facts (any "side" label in the pack is ignored).

Topic matching, first hit wins:
  1. the topic id the caller passes (offline episodes know their topic id);
  2. TALKSHOW_FACT_PACK_TOPIC;
  3. the room name of a ghost replay session (``ghost-<topic id>__...`` or ``ghost-pfx-<topic id>-...``);
  4. the topic text: the human's hand-raise topic, then the host and human lines from newest to oldest; a line matches
     a pack topic when the normalised topic text appears inside the normalised line (the host's floor line quotes the
     topic, e.g. "About <topic>: the mic is yours...").
No match = no block (logged once per room and topic).

Selection per panel line: the K facts with the most content words in common with the latest human turn plus the last
2 lines; facts a panelist already used in this show (shares a figure with an earlier panel line, or most of its content
words) go last. Selection reads only the show history, so a live show and an offline replay of the same history get the
same facts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_K = 6
PANEL = ("commentator", "guest")

RULE = ("Use specific numbers, names, studies or cases ONLY from these facts or from what was said in the show. "
        "Otherwise speak in general terms, or say 'for example' and keep it clearly hypothetical.")
HEADER = "Checked facts on this topic (from Wikipedia). Both panelists see the same list:"
# Replaces agent.panel.concrete.CONCRETE while a fact block is in the prompt: "from what you know" would contradict RULE.
CONCRETE_WITH_PACK = ("Ground your line in one concrete fact, number, name or example taken from the checked facts above "
                      "or from what was said in the show. Never add a figure, name or study of your own.")

_STOP = set("""a an the and or but so to of in on at for with by from as is are was were be been being it its this that these
those i you he she we they me him her us them my your his our their not no do does did have has had will would can could
should just very really more most much many some any all about than then there here what which who whom when where why
how if because also only even still too into out up down over again like well yes yeah okay ok think know mean say said
going get got make one way thing things people right other such per while after before since during under""".split())
_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")
_logged: set[tuple[str, str]] = set()


# ------------------------------------------------------------------------------------------------- flags and pack

def pack_path() -> str:
    return os.environ.get("TALKSHOW_FACT_PACK", "").strip()


def fact_pack_enabled() -> bool:
    return bool(pack_path())


def k_facts() -> int:
    try:
        return max(1, int(os.environ.get("TALKSHOW_FACT_PACK_K", str(DEFAULT_K)) or DEFAULT_K))
    except ValueError:
        return DEFAULT_K


@lru_cache(maxsize=4)
def load_pack(path: str) -> dict[str, dict]:
    """The pack at path ({} if it cannot be read; logged)."""
    try:
        with open(path, encoding="utf-8") as f:
            pack = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("fact pack: cannot read %s (%s); panel prompts get no facts", path, exc)
        return {}
    return {str(k): v for k, v in pack.items() if isinstance(v, dict) and v.get("facts")}


# ------------------------------------------------------------------------------------------------- topic matching

def norm(text: str) -> str:
    """Lowercase, letters and digits only, single spaces."""
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


@lru_cache(maxsize=4)
def _topic_index(path: str) -> tuple[tuple[str, str], ...]:
    """(normalised topic text, topic id), longest text first, so a longer motion wins over one it contains."""
    pack = load_pack(path)
    rows = [(norm(v.get("topic", "")), k) for k, v in pack.items()]
    return tuple(sorted((r for r in rows if r[0]), key=lambda r: -len(r[0])))


def match_text(text: str, path: str) -> str | None:
    """Topic id whose normalised topic text appears in text."""
    t = f" {norm(text)} "
    for topic, tid in _topic_index(path):
        if f" {topic} " in t:
            return tid
    return None


def match_room(room: str, pack: dict) -> str | None:
    """Topic id from a ghost replay room name (longest id wins: ids can share a prefix)."""
    hits = [k for k in pack if room.startswith((f"ghost-{k}__", f"ghost-pfx-{k}-"))]
    return max(hits, key=len) if hits else None


def resolve_topic(data: Any, pack: dict, path: str, topic_id: str | None = None) -> tuple[str | None, str]:
    """(topic id, how it was found) or (None, "")."""
    if topic_id and topic_id in pack:
        return topic_id, "topic_id"
    pinned = os.environ.get("TALKSHOW_FACT_PACK_TOPIC", "").strip()
    if pinned in pack:
        return pinned, "env"
    hit = match_room(str(getattr(data, "room_name", "") or ""), pack)
    if hit:
        return hit, "room"
    texts = []
    try:
        texts.append(data.human_hand_topic or "")
    except AttributeError:
        pass
    texts += [ln.text for ln in reversed(data.show_history.lines) if ln.role_id in ("host", "human")]
    for text in texts:
        hit = match_text(text, path) if text else None
        if hit:
            return hit, "text"
    return None, ""


# ------------------------------------------------------------------------------------------------- fact selection

def _content(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.add(w[:-1] if len(w) > 4 and w.endswith("s") else w)
    return out


def _figures(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _DIGITS.finditer(text or "")}


def fact_used(fact: str, panel_lines: list[str]) -> bool:
    """A panelist already used this fact: an earlier panel line shares one of its figures or most of its content words."""
    figs, words = _figures(fact), _content(fact)
    for line in panel_lines:
        if figs & _figures(line):
            return True
        if words and len(words & _content(line)) / len(words) >= 0.5:
            return True
    return False


def select_facts(facts: list[dict], query: str, panel_lines: list[str], k: int) -> list[int]:
    """Indices of the k facts to offer: unused before used, then most content words shared with query, then pack order."""
    q = _content(query)

    def key(i: int) -> tuple:
        text = facts[i].get("fact", "")
        return (fact_used(text, panel_lines), -len(_content(text) & q), i)

    return sorted(range(len(facts)), key=key)[:k]


def query_text(data: Any, history: Any = None) -> str:
    """Latest human turn plus the last 2 lines of the show (``history``: default the show's)."""
    hist = history if history is not None else data.show_history
    last = [ln.text for ln in hist.lines[-2:]]
    return " ".join([hist.latest_human_text() or "", *last])


def fact_block(facts: list[dict], idx: list[int]) -> str:
    return "\n\n" + HEADER + "\n" + "\n".join(f"- {facts[i]['fact'].strip()}" for i in idx) + "\n" + RULE


# ------------------------------------------------------------------------------------------------- prompt

def pack_block(data: Any, topic_id: str | None = None, history: Any = None) -> tuple[str, dict[str, Any]]:
    """(text to append to the panel turn prompt, turn-log fields). ("", {}) when the flag is off.

    Shared by the worker prompt and the Think & Action think prompt (agent/panel/think_prompt.py), which passes the
    ``history`` its think call sees (default: the show's)."""
    path = pack_path()
    if not path:
        return "", {}
    pack = load_pack(path)
    tid, how = resolve_topic(data, pack, path, topic_id) if pack else (None, "")
    if tid is None:
        key = (str(getattr(data, "room_name", "") or ""), str(topic_id or ""))
        if key not in _logged:
            _logged.add(key)
            logger.info("fact pack: no topic match (room=%r topic_id=%r); no facts offered", key[0], key[1])
        return "", {"fact_pack_topic": None}
    facts = pack[tid]["facts"]
    hist = history if history is not None else data.show_history
    panel_lines = [ln.text for ln in hist.lines if ln.role_id in PANEL]
    idx = select_facts(facts, query_text(data, hist), panel_lines, k_facts())
    return fact_block(facts, idx), {"fact_pack_topic": tid, "fact_pack_match": how, "fact_pack_offered": idx}


def finish_turn_prompt(data: Any, prompt: str, topic_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """Last step of the panel turn prompt, shared by the live worker and the offline builders: the fact block (flag
    TALKSHOW_FACT_PACK), then the concrete instruction (flag TALKSHOW_PANEL_CONCRETE), which points at the facts when a
    block is present. Both flags off = prompt unchanged."""
    from agent.panel.concrete import CONCRETE, concrete_instruction_enabled

    block, info = pack_block(data, topic_id)
    prompt += block
    if concrete_instruction_enabled():
        prompt += "\n\n" + (CONCRETE_WITH_PACK if block else CONCRETE)
    return prompt, info
