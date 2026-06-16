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


def panel_roster_entries(scenario: ScenarioConfig) -> list[dict[str, object]]:
    """AI panelists for this scenario (excludes human slots)."""
    entries: list[dict[str, object]] = []
    for role in scenario.turn_control.order:
        if role == "human":
            continue
        persona = load_persona_yaml(role)
        ui = persona.get("ui") or {}
        entry: dict[str, object] = {
            "role": role,
            "name": load_persona_name(role),
            "label": str(ui.get("label") or _ROLE_LABELS.get(role, role.title())),
            "color": str(ui.get("color") or _DEFAULT_COLORS.get(role, "#6366f1")),
        }
        avatar_cfg = ui.get("avatar")
        if isinstance(avatar_cfg, dict) and avatar_cfg.get("vrm"):
            avatar: dict[str, object] = {"vrm": str(avatar_cfg["vrm"])}
            if avatar_cfg.get("scale") is not None:
                avatar["scale"] = float(avatar_cfg["scale"])
            entry["avatar"] = avatar
        entries.append(entry)
    return entries
