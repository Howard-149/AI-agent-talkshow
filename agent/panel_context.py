from __future__ import annotations

from agent.config import ScenarioConfig, load_persona_name, load_scenario
from agent.show_history import HUMAN_LABEL

_ROLE_LABELS = {
    "host": "Host",
    "guest": "Guest",
    "commentator": "Commentator",
}


def panel_speaker_roles(scenario: ScenarioConfig) -> list[str]:
    """AI panelists who may take the floor (excludes human and listen/host role)."""
    listen = scenario.turn_control.listen_role
    return [r for r in scenario.turn_control.order if r not in ("human", listen)]


def panel_speaker_names(scenario: ScenarioConfig) -> list[str]:
    return [load_persona_name(r) for r in panel_speaker_roles(scenario)]


def format_name_list(names: list[str]) -> str:
    if not names:
        return "the panel"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def role_label(role: str) -> str:
    return _ROLE_LABELS.get(role, role.replace("_", " ").title())


def other_panelist_roles(speak_role: str, panel_roles: list[str]) -> list[str]:
    return [r for r in panel_roles if r != speak_role]


def other_panelist_names(speak_role: str, panel_roles: list[str]) -> list[str]:
    return [load_persona_name(r) for r in other_panelist_roles(speak_role, panel_roles)]


def any_other_panelist_spoken(
    history: object, speak_role: str, panel_roles: list[str]
) -> bool:
    from agent.show_history import ShowHistory

    if not isinstance(history, ShowHistory):
        return False
    return any(history.role_has_spoken(r) for r in other_panelist_roles(speak_role, panel_roles))


def unspoken_panelist_names(
    history: object, panel_roles: list[str]
) -> list[str]:
    from agent.show_history import ShowHistory

    if not isinstance(history, ShowHistory):
        return panel_speaker_names(load_scenario())
    return [
        load_persona_name(r)
        for r in panel_roles
        if not history.role_has_spoken(r)
    ]


def human_disambiguation_note(scenario: ScenarioConfig | None = None) -> str:
    sc = scenario or load_scenario()
    ai = format_name_list(panel_speaker_names(sc))
    return f'"{HUMAN_LABEL}" is the real person — not the AI panelists ({ai}).'


def ai_guest_role_name() -> str:
    """Display name for the AI guest role (often confused with Human guest label)."""
    return load_persona_name("guest")


def session_welcome_line(scenario: ScenarioConfig | None = None) -> str:
    host = load_persona_name("host")
    return (
        f"Welcome — I'm {host}, your host. "
        "Raise your hand if you'd like to speak or bring a topic — I'll call on you."
    )


def panel_opening_line(scenario: ScenarioConfig | None = None) -> str:
    sc = scenario or load_scenario()
    host = load_persona_name("host")
    panel = format_name_list(panel_speaker_names(sc))
    return (
        f"Welcome — I'm {host}, your host. When you have a topic, take the floor; "
        f"I'll tee it up for {panel}, then back to you."
    )


def next_tag_options(panel_roles: list[str], *, include_host: bool = True) -> str:
    roles = list(panel_roles)
    if include_host:
        roles.append("host")
    return " | ".join(roles)


def name_to_panel_role(name: str, panel_roles: list[str]) -> str | None:
    key = name.strip().lower()
    for role in panel_roles:
        if load_persona_name(role).lower() == key:
            return role
    return None


def panel_role_name_map(panel_roles: list[str]) -> str:
    """One line for prompts: display name = role id, …"""
    return ", ".join(f"{load_persona_name(r)} = {r}" for r in panel_roles)


def role_name_aliases(*, include_host: bool = True) -> dict[str, str]:
    """Lowercase display name → role id (for parsing model output)."""
    from agent.config import load_scenario

    sc = load_scenario()
    aliases: dict[str, str] = {}
    if include_host:
        aliases[load_persona_name("host").lower()] = "host"
    for role in panel_speaker_roles(sc):
        aliases[load_persona_name(role).lower()] = role
    return aliases


def floor_valid_next_roles(scenario: ScenarioConfig | None = None) -> frozenset[str]:
    sc = scenario or load_scenario()
    return frozenset(panel_speaker_roles(sc)) | frozenset({"close", "host", "human"})


def is_panel_speaker_role(role: str, scenario: ScenarioConfig | None = None) -> bool:
    key = role.strip().lower()
    sc = scenario or load_scenario()
    return key in panel_speaker_roles(sc)
