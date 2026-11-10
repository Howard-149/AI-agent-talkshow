"""Benchmark items: one fixed input for the role under test, plus what a good reply should show.

An item is everything a system sees at one turn of a talk show (topic, persona cards, transcript, the audience member's
new line, optional checked facts) and the expected behaviour that scoring checks. Items that differ only in the edited
element share a ``pair_group``; the group's reference version is ``meta["reference_version"]``.

Tasks (README.md): T1 listening, T2 addressing, T3 grounding, T4 persona, T5 hosting, T6 full show.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

TASKS = {
    "T1": "listening",
    "T2": "addressing",
    "T3": "grounding",
    "T4": "persona",
    "T5": "hosting",
    "T6": "full_show",
}
SPLITS = ("dev", "test")
ROLES = ("host", "commentator", "guest")
PANEL = ("commentator", "guest")
HISTORY_ROLES = ("host", "commentator", "guest", "human")


@dataclass
class Item:
    item_id: str
    task: str  # T1..T6
    split: str  # dev | test
    topic_id: str
    topic: str
    pair_group: str  # items that differ only in the edited element share it
    version: str  # "orig", "number", "paraphrase", "addr_guest", "fact_edit_1", ...
    is_control: bool  # an edit that should NOT change the reply (e.g. a paraphrase)
    role_under_test: str  # "commentator" | "guest" | "host"
    personas: dict[str, str]  # role -> persona card text, as given to this version
    history: list[dict[str, Any]]  # [{"speaker", "role", "text"[, "procedural": true]}], oldest first
    audience_turn: str | None  # the audience member's new line (None: the role reacts to the history as it is)
    fact_pack: list[str] | None  # checked facts offered at this turn (None: no facts)
    expected: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown item fields: {sorted(unknown)}")
        item = cls(**{k: d[k] for k in names if k in d})
        validate(item)
        return item


def validate(item: Item) -> None:
    """Raise ValueError if the item breaks the format."""
    if item.task not in TASKS:
        raise ValueError(f"{item.item_id}: unknown task {item.task!r}")
    if item.split not in SPLITS:
        raise ValueError(f"{item.item_id}: unknown split {item.split!r}")
    if item.role_under_test not in ROLES:
        raise ValueError(f"{item.item_id}: unknown role {item.role_under_test!r}")
    for i, line in enumerate(item.history):
        if not {"speaker", "role", "text"} <= set(line) or line["role"] not in HISTORY_ROLES:
            raise ValueError(f"{item.item_id}: history line {i} needs speaker/role/text with a known role")
    if item.fact_pack is not None and not all(isinstance(f, str) for f in item.fact_pack):
        raise ValueError(f"{item.item_id}: fact_pack must be a list of strings or null")
    if item.role_under_test in PANEL and item.role_under_test not in item.personas:
        raise ValueError(f"{item.item_id}: no persona card for {item.role_under_test}")


def item_hash(item: Item | dict[str, Any]) -> str:
    """Content hash of an item (canonical JSON), to check a locally rebuilt item against a committed manifest."""
    d = item.to_dict() if isinstance(item, Item) else item
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def write_items(path: Path | str, items: Iterable[Item]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            validate(it)
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_items(path: Path | str) -> list[Item]:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                out.append(Item.from_dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{i + 1}: {exc}") from exc
    ids = [it.item_id for it in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return out


def observation(item: Item) -> dict[str, Any]:
    """What the role under test sees: the input every system gets (systems.py)."""
    role = item.role_under_test
    return {
        "item_id": item.item_id,
        "topic_id": item.topic_id,
        "topic": item.topic,
        "role": role,
        "name": item.meta.get("names", {}).get(role, role),
        "names": dict(item.meta.get("names", {})),
        "personas": dict(item.personas),
        "history": [dict(h) for h in item.history],
        "audience_turn": item.audience_turn,
        "fact_pack": list(item.fact_pack) if item.fact_pack is not None else None,
        # A recorded show state to resume from (mood, move-example index). Systems that keep no such state ignore it.
        "show_state": dict(item.meta.get("show_state", {})),
    }
