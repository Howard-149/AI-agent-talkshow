from __future__ import annotations

import logging

from livekit.agents import Agent, RunContext

logger = logging.getLogger(__name__)


class HostAgent(Agent):
    def __init__(self, *, instructions: str) -> None:
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        logger.info("HostAgent entered session")
        await self.session.generate_reply(
            instructions="Greet the user briefly in English as the talk-show host and invite them to speak."
        )
