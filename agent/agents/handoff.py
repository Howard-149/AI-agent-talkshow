from __future__ import annotations

import logging

from livekit.agents import Agent, RunContext
from livekit.agents.llm import function_tool

from agent.data import TalkShowData
from agent.supervisor import TurnController

logger = logging.getLogger(__name__)



class HandoffToolsMixin:
    """function_tool handoffs between Host / Guest / Commentator."""

    _role: str
    _data: TalkShowData

    def _controller(self) -> TurnController:
        return TurnController(self._data.scenario, self._data)

    def _handoff(self, ctx: RunContext[TalkShowData], to_role: str, reason: str) -> Agent:
        from agent.agents.factory import build_agent

        ctrl = self._controller()
        ctrl.record_handoff(to_role=to_role, reason=reason)
        ctrl.apply_persona_for_role(to_role)
        logger.info("%s handing off to %s", self._role, to_role)
        return build_agent(to_role, ctx.userdata)

    @function_tool
    async def handoff_to_host(self, context: RunContext[TalkShowData]) -> Agent:
        """Return control to the host. Use when the segment should be led by the host again."""
        return self._handoff(context, "host", f"tool:{self._role}_to_host")

    @function_tool
    async def handoff_to_guest(self, context: RunContext[TalkShowData]) -> Agent:
        """Hand off to the guest panelist for a substantive answer or story."""
        return self._handoff(context, "guest", f"tool:{self._role}_to_guest")

    @function_tool
    async def handoff_to_commentator(self, context: RunContext[TalkShowData]) -> Agent:
        """Hand off to the commentator for a brief aside or reaction."""
        return self._handoff(context, "commentator", f"tool:{self._role}_to_commentator")
