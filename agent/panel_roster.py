from __future__ import annotations

from agent.config import ScenarioConfig, load_persona_name, load_persona_yaml

_ROLE_LABELS = {
    "host": "Host",
    "guest": "Guest",
    "commentator": "Commentator",
}

_DEFAULT_COLORS = {
    "host": "#4f46e5",
    "commentator": "#059669",
    "guest": "#db2777",
}


def panel_roster_entries(scenario: ScenarioConfig) -> list[dict[str, str]]:
    """AI panelists for this scenario (excludes human slots)."""
    entries: list[dict[str, str]] = []
    for role in scenario.turn_control.order:
        if role == "human":
            continue
        persona = load_persona_yaml(role)
        ui = persona.get("ui") or {}
        entries.append(
            {
                "role": role,
                "name": load_persona_name(role),
                "label": str(ui.get("label") or _ROLE_LABELS.get(role, role.title())),
                "color": str(ui.get("color") or _DEFAULT_COLORS.get(role, "#6366f1")),
            }
        )
    return entries
