"""Per-persona worker: builds an AgentDraft without touching the shared floor."""

from __future__ import annotations

import logging
import os
import time

from agent.adapters.response_parser import parse_host_speech
from agent.config import load_persona_name
from agent.emotion import emotion_source, get_role_emotion, normalize_emotion
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
        # Legacy (non-graph) turn modes appraise in the background; make sure this
        # persona's own pending appraisal has landed before its prompt is built.
        # The show graph joins appraisals before the next line, so this is a no-op there.
        from agent.emotion.appraisal import await_pending_appraisal

        await await_pending_appraisal(data, role)
        controller = TurnController(data.scenario, data)
        panel_roles = controller.panel_speaker_roles()
        name = load_persona_name(role)
        host_name = load_persona_name("host")
        hist = data.show_history.prior_messages(perspective=role)
        latest_human = data.show_history.latest_human_text()
        history_version = len(data.show_history.lines)

        # Concrete panel lines (flags default off): move library / instruction / best-of-N.
        from agent.panel.concrete import (
            CONCRETE,
            best_of,
            concrete_instruction_enabled,
            dialogue_library_override,
            pick_best,
        )

        dialogue_hint = ""
        next_dialogue_index: int | None = None
        dlg = data.scenario.dialogue
        if dlg and dlg.library:
            # Read-only here: advancing data.dialogue_example_index is deferred to
            # speak_draft() so an un-spoken draft doesn't consume the next example.
            dialogue_hint, next_dialogue_index = format_dialogue_hint(
                dialogue_library_override() or dlg.library,
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
        # Uptake fix (flags default off): scaffold fix + uptake plan, first answer to a human only.
        from agent.panel.uptake import (
            adapt_panel_prompt,
            await_uptake_plan,
            engages_chars,
            first_answer_to_human,
            plan_block,
            strip_engages,
            uptake_planner_enabled,
        )

        system_prompt, prompt, uptake_info = adapt_panel_prompt(data, system_prompt, prompt)
        planner_on = uptake_planner_enabled()
        human_now = first_answer_to_human(data) if planner_on else None
        if human_now is not None:
            plan, waited = await await_uptake_plan(data, human_now)
            if plan:
                prompt += plan_block(plan)
            uptake_info.update(plan_used=bool(plan), plan_wait_s=round(waited, 3))
        if concrete_instruction_enabled():
            prompt += "\n\n" + CONCRETE
        n_best = best_of()
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
        concrete_info: dict = {}
        if n_best > 1:
            raws = await data.runtime.gemma_client.complete_text_n(
                prompt,
                n_best,
                system_prompt=system_prompt,
                history_messages=hist,
                output_role=role,
                reasoning_effort="none",
            )
            raws = [strip_engages(r) if planner_on else r for r in raws] or [""]
            spoken = [
                normalize_panelist_speech(
                    role,
                    strip_speech_control_tags(parse_host_speech(r).reply.strip()),
                    panel_roles=panel_roles,
                    history=data.show_history,
                )
                for r in raws
            ]
            earlier = [ln.text for ln in data.show_history.lines if ln.role_id in panel_roles]
            ok = [i for i, x in enumerate(spoken) if x.strip()] or [0]
            best_i, scores = pick_best([spoken[i] for i in ok], earlier)
            raw = raws[ok[best_i]]
            concrete_info = {"best_of": n_best, "best_index": ok[best_i],
                             "rerank_scores": [round(x, 3) for x in scores]}
        else:
            raw = await data.runtime.gemma_client.complete_text(
                prompt,
                system_prompt=system_prompt,
                history_messages=hist,
                output_role=role,
                raw=True,  # parse [next] from the full output, not the stripped reply
                reasoning_effort="none",
            )
        model_latency_s = time.monotonic() - t0
        if planner_on:
            uptake_info["engages_chars"] = engages_chars(raw)
            raw = strip_engages(raw)

        parsed = parse_host_speech(raw)
        # PAD mode: mood comes from listeners' appraisals, not the speaker's own tag;
        # read the current label without mutating shared state (drafts may go unspoken).
        if emotion_source() == "llm":
            emotion = normalize_emotion(parsed.emotion)
        else:
            emotion = get_role_emotion(data, role)
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
                raw=(raw or "")[:1500],
                model_latency_s=round(model_latency_s, 3),
                dialogue_library=(dialogue_library_override() or dlg.library) if dlg and dlg.library else None,
                concrete_instruction=concrete_instruction_enabled(),
                dialogue_pick=dlg.pick if dlg and dlg.library else None,
                hint_chars=len(dialogue_hint),
                system_chars=len(system_prompt),
                user_chars=len(prompt),
                room=getattr(data, "room_name", ""),
                **uptake_info,
                **concrete_info,
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
