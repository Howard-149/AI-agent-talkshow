from __future__ import annotations

import re

from agent.config import load_persona_name


def normalize_panelist_speech(role: str, text: str) -> str:
    """
    Fix third-person self-reference (e.g. Ryan saying 'Ryan has a good point').
    """
    name = load_persona_name(role)
    t = text.strip()
    if not t:
        return text

    other = load_persona_name(
        "guest" if role == "commentator" else "commentator"
    )

    # "Amy, I think Ryan has..." → drop wrong opener when speaker is Ryan
    if role == "commentator":
        t = re.sub(
            rf"^{re.escape(other)},?\s*I think\s+{re.escape(name)}\s+",
            "I think ",
            t,
            count=1,
            flags=re.I,
        )
        t = re.sub(rf"\bI think\s+{re.escape(name)}\s+", "I ", t, flags=re.I)
        t = re.sub(rf"\b{re.escape(name)}\s+has\b", "I have", t, flags=re.I)
        t = re.sub(rf"\b{re.escape(name)}\s+had\b", "I had", t, flags=re.I)

    if role == "guest":
        ryan = load_persona_name("commentator")
        t = re.sub(
            rf"^I think\s+{re.escape(ryan)}\s+",
            f"{ryan} made a good point — I ",
            t,
            count=1,
            flags=re.I,
        )
        t = re.sub(rf"\bI think\s+{re.escape(name)}\s+", "I ", t, flags=re.I)

    return t.strip() or text.strip()
