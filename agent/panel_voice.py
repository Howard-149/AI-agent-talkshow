from __future__ import annotations

import re

from agent.config import load_persona_name


def normalize_panelist_speech(
    role: str,
    text: str,
    *,
    panel_roles: list[str],
    history: object | None = None,
) -> str:
    """
    Fix third-person self-reference and wrong openers to panelists who have not spoken yet.
    """
    from agent.show_history import ShowHistory

    name = load_persona_name(role)
    t = text.strip()
    if not t:
        return text

    other_roles = [r for r in panel_roles if r != role]
    other_names = {r: load_persona_name(r) for r in other_roles}

    unspoken_names = [
        other_names[r]
        for r in other_roles
        if isinstance(history, ShowHistory) and not history.role_has_spoken(r)
    ]

    for other in unspoken_names:
        t = re.sub(
            rf"^{re.escape(other)},?\s+",
            "",
            t,
            count=1,
            flags=re.I,
        )
        t = re.sub(
            rf"\b{re.escape(other)},?\s+you\b",
            "our colleague",
            t,
            flags=re.I,
        )
        t = re.sub(
            rf"\bthank you,?\s+{re.escape(other)}\b",
            "thank you",
            t,
            flags=re.I,
        )

    for other in other_names.values():
        t = re.sub(
            rf"^{re.escape(other)},?\s*I think\s+{re.escape(name)}\s+",
            "I think ",
            t,
            count=1,
            flags=re.I,
        )
        if other != name:
            t = re.sub(
                rf"^I think\s+{re.escape(other)}\s+is\b",
                f"{other} is",
                t,
                count=1,
                flags=re.I,
            )
            t = re.sub(
                rf"^I think\s+{re.escape(other)}\s+has\b",
                f"{other} has",
                t,
                count=1,
                flags=re.I,
            )
            t = re.sub(
                rf"^I think\s+{re.escape(other)}\s+",
                f"{other} made a good point — ",
                t,
                count=1,
                flags=re.I,
            )

    t = re.sub(rf"^I think\s+{re.escape(name)}\s+has\b", "I have", t, flags=re.I)
    t = re.sub(rf"\bI think\s+{re.escape(name)}\s+", "I ", t, flags=re.I)
    t = re.sub(rf"\b{re.escape(name)}\s+has\b", "I have", t, flags=re.I)
    t = re.sub(rf"\b{re.escape(name)}\s+had\b", "I had", t, flags=re.I)

    host_name = load_persona_name("host")
    if (
        isinstance(history, ShowHistory)
        and history.latest_human_text()
        and name.lower() != host_name.lower()
    ):
        # Retarget replies that name the host for a claim the guest actually made.
        m_you = re.match(
            rf"^{re.escape(host_name)},?\s+you(?:'re|'re|\s+are)\s+(.+)$",
            t,
            flags=re.I,
        )
        if m_you:
            t = f"Human guest, you're {m_you.group(1).strip()}"
        else:
            m_host = re.match(
                rf"^{re.escape(host_name)},?\s+(.+)$",
                t,
                flags=re.I,
            )
            if m_host:
                t = f"Human guest — {m_host.group(1).strip()}"

    return t.strip() or text.strip()
