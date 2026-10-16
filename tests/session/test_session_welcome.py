"""The opening's welcome line must build without a LiveKit job (text-only mode, tests)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("livekit.agents")


def test_session_welcome_line_without_job_context() -> None:
    import agent.agents  # noqa: F401 — import order that avoids the session_handoff cycle
    from agent.config import load_scenario
    from agent.floor.host_floor import session_welcome_line

    data = SimpleNamespace(scenario=load_scenario(), needed_locales={"en"})
    text, texts = session_welcome_line(data)
    assert text and texts.get("en") == text
