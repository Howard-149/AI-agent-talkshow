"""Load dialogue-move libraries and format Gemma hints for panel speech."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from agent.config import REPO_ROOT

logger = logging.getLogger(__name__)

LIB_DIR = REPO_ROOT / "config" / "dialogue_libraries"


@dataclass(frozen=True)
class DialogueMove:
    id: str
    label: str
    example: str
    tone: str
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class DialogueLibrary:
    id: str
    description: str
    preamble: str
    moves: tuple[DialogueMove, ...]


@lru_cache(maxsize=16)
def load_dialogue_library(library_id: str) -> DialogueLibrary:
    path = LIB_DIR / f"{library_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Dialogue library not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    moves: list[DialogueMove] = []
    for item in raw.get("moves") or []:
        roles = tuple(item.get("roles") or ())
        moves.append(
            DialogueMove(
                id=str(item.get("id", "")),
                label=str(item.get("label", "")),
                example=str(item.get("example", "")).strip(),
                tone=str(item.get("tone", "")),
                roles=roles,
            )
        )
    return DialogueLibrary(
        id=str(raw.get("id", library_id)),
        description=str(raw.get("description", "")),
        preamble=str(raw.get("preamble", "")).strip(),
        moves=tuple(moves),
    )


def _moves_for_role(library: DialogueLibrary, role: str) -> list[DialogueMove]:
    eligible = [m for m in library.moves if not m.roles or role in m.roles]
    return eligible or list(library.moves)


def format_dialogue_all(library_id: str, *, role: str) -> str:
    """
    Full style catalog for this role — collaborative + pushback together.
    Used when scenario dialogue.pick is ``all``.
    """
    library = load_dialogue_library(library_id)
    moves = _moves_for_role(library, role)
    if not moves:
        return ""

    collab = [m for m in moves if m.tone == "collaborative"]
    push = [m for m in moves if m.tone == "pushback"]
    other = [m for m in moves if m.tone not in ("collaborative", "pushback")]

    sections: list[str] = []
    if library.preamble:
        sections.append(library.preamble)

    def _block(title: str, items: list[DialogueMove]) -> str:
        if not items:
            return ""
        lines = [f"{title}:"]
        for m in items:
            lines.append(f"  [{m.label}]\n{m.example}")
        return "\n".join(lines)

    for block in (
        _block(
            "Pushback (when you disagree — examples show how to challenge clearly)",
            push,
        ),
        _block(
            "Collaborative (when you agree or build on the thread)",
            collab,
        ),
        _block("Other", other),
    ):
        if block:
            sections.append(block)

    return "\n\n".join(sections)


def format_dialogue_hint(
    library_id: str,
    *,
    role: str,
    pick: str,
    index: int,
) -> tuple[str, int]:
    """
    pick=all — inject full move catalog every turn.
    pick=rotate|random — one move per turn.
    pick=none — no dialogue hint (personality + panel mechanics only).
    """
    if pick == "none":
        logger.info(
            "dialogue hint role=%s library=%s pick=%s mode=disabled",
            role,
            library_id,
            pick,
        )
        return "", index

    if pick == "all":
        hint = format_dialogue_all(library_id, role=role)
        logger.info(
            "dialogue hint role=%s library=%s pick=all mode=all chars=%d",
            role,
            library_id,
            len(hint),
        )
        return hint, index

    library = load_dialogue_library(library_id)
    moves = _moves_for_role(library, role)
    if not moves:
        logger.info(
            "dialogue hint role=%s library=%s pick=%s mode=empty",
            role,
            library_id,
            pick,
        )
        return "", index

    if pick == "random":
        move = random.choice(moves)
        next_index = index
    else:
        move = moves[index % len(moves)]
        next_index = index + 1

    parts = []
    if library.preamble:
        parts.append(library.preamble)
    tone_note = ""
    if move.tone == "collaborative":
        tone_note = " (collaborative — use when you agree or extend)"
    elif move.tone == "pushback":
        tone_note = " (pushback — use boldly when you disagree; follow the example's tone)"
    parts.append(f"Optional style hint{tone_note} — [{move.label}]:\n{move.example}")
    hint = "\n\n".join(parts)
    logger.info(
        "dialogue hint role=%s library=%s pick=%s mode=single move=%s tone=%s chars=%d",
        role,
        library_id,
        pick,
        move.id,
        move.tone,
        len(hint),
    )
    return hint, next_index
