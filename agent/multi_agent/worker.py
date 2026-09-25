"""Per-persona worker: builds an AgentDraft without touching the shared floor."""

from __future__ import annotations

import logging
import os
import time

from agent.adapters.response_parser import parse_host_speech
from agent.config import load_persona_name
from agent.emotion import normalize_emotion
from agent.floor.floor_parser import strip_speech_control_tags
from agent.floor.turn_controller import TurnController
from agent.multi_agent.types import AgentDraft
from agent.panel.dialogue_library import format_dialogue_hint
from agent.panel.panel_prompts import panel_speech_prompt, panelist_system_prompt
from agent.panel.panel_speech_normalize import normalize_panelist_speech
from agent.show.talkshow_data import TalkShowData

logger = logging.getLogger(__name__)


class PersonaWorker:
    """Generates drafts for one persona/role. Shares the runtime's Gemma client;
    persona identity and prompt context are the only per-worker state."""

    def __init__(self, role: str) -> None:
        self.role = role

    async def draft(self, data: TalkShowData, *, step: str) -> AgentDraft:
        role = self.role
        controller = TurnController(data.scenario, data)
        panel_roles = controller.panel_speaker_roles()
        name = load_persona_name(role)
        host_name = load_persona_name("host")
        hist = data.show_history.prior_messages()
        latest_human = data.show_history.latest_human_text()
        history_version = len(data.show_history.lines)

        dialogue_hint = ""
        next_dialogue_index: int | None = None
        dlg = data.scenario.dialogue
        if dlg and dlg.library:
            # Read-only here: advancing data.dialogue_example_index is deferred to
            # speak_draft() so an un-spoken draft doesn't consume the next example.
            dialogue_hint, next_dialogue_index = format_dialogue_hint(
                dlg.library,
                role=role,
                pick=dlg.pick,
                index=data.dialogue_example_index,
            )
        else:
            logger.info("dialogue hint role=%s skipped (no library in scenario)", role)

        prompt = panel_speech_prompt(
            role=role,
            name=name,
            host_name=host_name,
            scenario=data.scenario,
            panel_roles=panel_roles,
            history=data.show_history,
            latest_human=latest_human,
            data=data,
        )
        system_prompt = panelist_system_prompt(
            role,
            scenario=data.scenario,
            panel_roles=panel_roles,
            history=data.show_history,
            dialogue_hint=dialogue_hint,
            data=data,
        )
        logger.info(
            "panel draft prompt role=%s step=%s system_chars=%d user_chars=%d hint_chars=%d pick=%s",
            role,
            step,
            len(system_prompt),
            len(prompt),
            len(dialogue_hint),
            dlg.pick if dlg and dlg.library else "none",
        )
        if os.environ.get("TALKSHOW_LOG_PROMPTS", "").lower() in ("1", "true", "yes"):
            logger.info(
                "TALKSHOW_LOG_PROMPTS role=%s\n--- system ---\n%s\n--- user ---\n%s",
                role,
                system_prompt,
                prompt,
            )

        t0 = time.monotonic()
        raw = await data.runtime.gemma_client.complete_text(
            prompt,
            system_prompt=system_prompt,
            history_messages=hist,
        )
        model_latency_s = time.monotonic() - t0

        parsed = parse_host_speech(raw)
        emotion = normalize_emotion(parsed.emotion)
        speech = strip_speech_control_tags(parsed.reply.strip())
        if not speech:
            logger.warning("empty panel draft for role=%s", role)
            speech = (
                "I want to make sure we speak to what the room is discussing — "
                "let me add my view on that."
            )
        normalized = normalize_panelist_speech(
            role,
            speech,
            panel_roles=panel_roles,
            history=data.show_history,
        )
        if normalized != speech:
            logger.info("panel voice normalized for role=%s", role)
            speech = normalized

        valid_next = set(panel_roles) | {"host", "human", "close"}
        next_role = parsed.next_speaker if parsed.next_speaker in valid_next else "host"

        if getattr(data, "turn_log", None) is not None:
            data.turn_log.log(
                "panel_model_done",
                role=role,
                step=step,
                reply=speech,
                reply_len=len(speech),
                next_tag=next_role,
                emotion=emotion,
                model_latency_s=round(model_latency_s, 3),
                dialogue_library=dlg.library if dlg and dlg.library else None,
                dialogue_pick=dlg.pick if dlg and dlg.library else None,
                hint_chars=len(dialogue_hint),
                system_chars=len(system_prompt),
                user_chars=len(prompt),
                room=getattr(data, "room_name", ""),
            )

        return AgentDraft(
            role=role,
            speech=speech,
            next_role=next_role,
            emotion=emotion,
            dialogue_example_index=next_dialogue_index,
            history_version=history_version,
            model_latency_s=model_latency_s,
        )
