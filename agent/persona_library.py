"""Persona library: structured personas that any panel seat can take (flag TALKSHOW_PERSONA_LIBRARY, default off).

A seat (config/personas/<seat>.yaml: host, commentator, guest) owns the voice, avatar, UI colour and the spoken name.
A persona (config/persona_library/<id>.yaml) owns what is said and how, in fixed short fields (SCHEMA below).

TALKSHOW_PERSONA_LIBRARY=1            panel seats use library personas instead of their free-text ``personality``
TALKSHOW_PERSONA_<SEAT>=<id>          the persona for a seat (default DEFAULT_PERSONA: ryan_skeptic, amy_warm)
TALKSHOW_PERSONA_LIBRARY_NAMES=1      the seat's spoken name becomes the persona's ``name`` (default: the seat keeps
                                      its own name, so "Dana" in the commentator seat is still called Ryan)

The spoken name is what addressee detection, transcript labels and the UI use (agent.config.load_persona_name). With
library names on, an audience member who says "Ryan, ..." no longer names anyone if the commentator seat holds Dana;
keeping the seat's name (the default) keeps addressing and labels stable whatever persona sits there.

Rendering (agent.config.load_persona_personality and agent/panel/think_prompt.near_block):
- system message: a short card from name, role, background, leaning and relationships (render_card);
- right before the line is written (the Think & Action panelist think prompt): one compact block with the style,
  the moves and the avoid list (render_near). It supersedes TALKSHOW_THINK_PERSONA_NEAR when both are on.
The worker prompt (the fallback when a selected panelist has no proposed line) gets the card only.

The host seat is not rendered from the library yet: host prompts are instruction based (config/personas/host.yaml
``instructions``), so lessac_host is a library entry for later use. With the flag off nothing here is read and every
prompt is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

LIBRARY_DIR = Path(__file__).resolve().parents[1] / "config" / "persona_library"
DEFAULT_PERSONA = {"commentator": "ryan_skeptic", "guest": "amy_warm", "host": "lessac_host"}
LIBRARY_SEATS = ("commentator", "guest")  # seats rendered from the library (see the module docstring for the host)
REQUIRED = ("id", "name", "role", "style", "moves")
STYLE_KEYS = ("length", "tone", "register")


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    role: str
    style: dict[str, str]
    moves: tuple[str, ...]
    background: str = ""
    leaning: str = ""
    avoid: tuple[str, ...] = ()
    relationships: dict[str, str] = field(default_factory=dict)
    example_lines: tuple[str, ...] = ()


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def library_enabled() -> bool:
    return _flag("TALKSHOW_PERSONA_LIBRARY")


def library_names_enabled() -> bool:
    return _flag("TALKSHOW_PERSONA_LIBRARY_NAMES")


def parse(data: dict[str, Any], source: str = "") -> Persona:
    """A Persona from one library file's mapping; ValueError naming the problem if a field is missing or malformed."""
    where = f" ({source})" if source else ""
    missing = [k for k in REQUIRED if not data.get(k)]
    if missing:
        raise ValueError(f"persona{where}: missing {missing}")
    style = data["style"]
    if not isinstance(style, dict) or any(not style.get(k) for k in STYLE_KEYS):
        raise ValueError(f"persona{where}: style needs {list(STYLE_KEYS)}")
    moves = data["moves"]
    if not isinstance(moves, list) or not 1 <= len(moves) <= 5:
        raise ValueError(f"persona{where}: moves must be a list of 1 to 5")
    unknown = set(data) - {f for f in Persona.__dataclass_fields__}
    if unknown:
        raise ValueError(f"persona{where}: unknown fields {sorted(unknown)}")
    return Persona(
        id=str(data["id"]), name=str(data["name"]), role=str(data["role"]),
        style={k: str(style[k]) for k in STYLE_KEYS}, moves=tuple(str(m) for m in moves),
        background=str(data.get("background") or ""), leaning=str(data.get("leaning") or ""),
        avoid=tuple(str(a) for a in data.get("avoid") or ()),
        relationships={str(k): str(v) for k, v in (data.get("relationships") or {}).items()},
        example_lines=tuple(str(x) for x in data.get("example_lines") or ()),
    )


@lru_cache(maxsize=64)
def load(persona_id: str, library_dir: str = "") -> Persona:
    import yaml

    path = Path(library_dir or LIBRARY_DIR) / f"{persona_id}.yaml"
    if not path.is_file():
        raise ValueError(f"no persona {persona_id!r} in {path.parent}")
    persona = parse(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, path.name)
    if persona.id != persona_id:
        raise ValueError(f"{path.name}: id {persona.id!r} does not match the file name")
    return persona


def available(library_dir: str = "") -> list[str]:
    return sorted(p.stem for p in Path(library_dir or LIBRARY_DIR).glob("*.yaml"))


def persona_id_for_seat(seat: str) -> str:
    return os.environ.get(f"TALKSHOW_PERSONA_{seat.upper()}", "").strip() or DEFAULT_PERSONA[seat]


def persona_for_seat(seat: str) -> Persona:
    return load(persona_id_for_seat(seat))


def render_card(persona: Persona, name: str, names: dict[str, str] | None = None) -> str:
    """The system-message card: who they are (name, role, background, leaning) and how they relate to other seats.
    ``name`` is the spoken name; ``names`` maps seats to their spoken names (for the relationship lines)."""
    lines = [f"You are {name}. On this show: {persona.role}."]
    if persona.background:
        lines.append(f"Background: {persona.background}.")
    if persona.leaning:
        lines.append(f"Leaning: {persona.leaning}.")
    for seat, rel in persona.relationships.items():
        lines.append(f"With {(names or {}).get(seat, seat)}: {rel}.")
    return "\n".join(lines)


def render_near(persona: Persona, name: str) -> str:
    """The block right before the line is written: style, moves, things to avoid."""
    s = persona.style
    lines = [f"Write the content as {name} ({s['length']}, {s['tone']}, {s['register']}).",
             "Use one of your moves if it fits: " + "; ".join(persona.moves) + "."]
    if persona.avoid:
        lines.append("Never: " + "; ".join(persona.avoid) + ".")
    return "\n".join(lines)
