"""Human hand-raise UI: dequeue a raised hand once the human gets the floor."""

from __future__ import annotations

import logging

from agent.data import TalkShowData
from agent.ui.ui_events import emit_hand_raise

logger = logging.getLogger(__name__)


async def dequeue_hand_raise(data: TalkShowData, role: str) -> None:
    """Remove role from queue after grant or hand down."""
    popped = data.hand_raise_queue.pop(role)
    if popped is None:
        return
    await emit_hand_raise(role, False)
    from agent.ui.ui_events import emit_queue_state

    await emit_queue_state(data, phase=f"dequeue:{role}")
    if data.turn_log:
        data.turn_log.log(
            "hand_raise_dequeue",
            role=role,
            room=data.room_name,
            queue_remaining=data.hand_raise_queue.roles(),
        )
