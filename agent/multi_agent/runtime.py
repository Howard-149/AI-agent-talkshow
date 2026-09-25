"""Lifecycle owner for persistent persona workers."""

from __future__ import annotations

from agent.multi_agent.coordinator import MultiAgentCoordinator
from agent.multi_agent.worker import PersonaWorker
from agent.multi_agent.types import AgentDraft
from agent.show.talkshow_data import TalkShowData


class MultiAgentRuntime:
    """Owns persistent per-persona workers for one talk-show session."""

    def __init__(self) -> None:
        self.workers: dict[str, PersonaWorker] = {}
        self.coordinator = MultiAgentCoordinator(self.workers)

    def worker(self, role: str) -> PersonaWorker:
        worker = self.workers.get(role)

        if worker is None:
            worker = PersonaWorker(role)
            self.workers[role] = worker

        return worker

    async def draft_role(
        self,
        role: str,
        data: TalkShowData,
        *,
        step: str,
    ) -> AgentDraft:
        return await self.worker(role).draft(
            data,
            step=step,
        )

    async def draft_roles(
        self,
        roles: list[str],
        data: TalkShowData,
        *,
        step: str,
    ) -> dict[str, AgentDraft]:
        for role in roles:
            self.worker(role)

        return await self.coordinator.draft_roles(
            roles,
            data,
            step=step,
        )