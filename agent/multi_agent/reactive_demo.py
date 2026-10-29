"""Explicit preset topic/stance; every AI line and intent is generated live."""
from __future__ import annotations

from agent.floor.floor_control import apply_floor_next, peek_floor_next

TOPIC = "Should students use AI for homework?"


async def run_reactive_demo(session, data, controller, *, mode: str) -> bool:
    broker = data.reactive_session
    if broker is None or data.panel_chain_running or (not broker.ai_only and peek_floor_next(data) != "human"):
        if broker is not None:
            await broker.emit("reactive_error", message="Finish the current round before starting a demo.")
        return False
    show = data.show_session
    if show is None or not show.running:
        await broker.emit("reactive_error", message="The session graph is not ready.")
        return False
    # Claim synchronously: double-clicks cannot enqueue two independent demos.
    data.panel_chain_running = True
    from agent.floor.hand_raise_queue import HandRaiseQueue
    from agent.show.show_history import ShowHistory
    from agent.ui.ui_events import emit_transcript, emit_queue_state, emit_floor_grant

    data.show_history = ShowHistory()
    data.hand_raise_queue = HandRaiseQueue()
    broker.demo_active = broker.demo_opening = True
    broker.pending_human = False
    broker.demo_stop = False
    broker.mode = mode
    broker.topic = TOPIC
    broker.policy.last_ai_cut = float("-inf")
    try:
        await broker.emit("reactive_reset", mode=mode, topic=TOPIC, ai_only=broker.ai_only)
        if broker.ai_only:
            # Topic is configuration, not a fabricated human/host utterance.
            data.last_human_heard = ""
        else:
            data.show_history.append("human", TOPIC)
            data.last_human_heard = TOPIC
            await emit_transcript("human", TOPIC, step="demo_topic")
        await emit_queue_state(data, phase="demo")
        apply_floor_next(data, "guest")
        await emit_floor_grant("guest", reason="demo_opening")
        broker.log("reactive_demo_start", mode=mode, topic=TOPIC,
                   preset="Amy pro-AI opening stance; live generated dialogue and decisions",
                   followups="Amy reply, Lessac follow-up, Ryan proposal, Amy reply, Lessac wrap",
                   max_speeches=7)
        result = await show.post("panel", trigger="reactive_demo")
        if result is None:
            apply_floor_next(data, "host" if broker.ai_only else "human")
            if not broker.ai_only:
                await emit_floor_grant("human", reason="demo_error_recovery")
            await broker.emit("reactive_error", message="Generation failed. You can retry the demo.")
        await broker.emit("reactive_complete", mode=mode, outcome=(result or {}).get("last_outcome", ""))
        broker.log("reactive_demo_end", mode=mode,
                   ai_only=broker.ai_only,
                   history=[{"role": x.role_id, "text": x.text} for x in data.show_history.lines])
        return result is not None
    finally:
        # Demo-only queued intents must not start an extra round after wrapping.
        data.hand_raise_queue.clear_roles(controller.panel_speaker_roles())
        broker.demo_active = broker.demo_opening = False
        broker.mode = "live"
        data.panel_chain_running = False
        await emit_queue_state(data, phase="demo_end")
