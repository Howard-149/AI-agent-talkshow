"""Coordinator for concurrent drafting of multiple persona roles."""

from __future__ import annotations

import asyncio

from agent.multi_agent.types import AgentDraft
from agent.multi_agent.worker import PersonaWorker
from agent.show.talkshow_data import TalkShowData


class MultiAgentCoordinator:
    """Owns one PersonaWorker per role and drafts them concurrently."""

    def __init__(self, workers: dict[str, PersonaWorker]) -> None:
        self.workers = workers

    async def draft_roles(
        self, roles: list[str], data: TalkShowData, *, step: str = "draft"
    ) -> dict[str, AgentDraft]:
        tasks = {
            role: asyncio.create_task(self.workers[role].draft(data, step=step))
            for role in roles
        }
        results = await asyncio.gather(*tasks.values())
        return {draft.role: draft for draft in results}
