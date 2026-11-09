"""Build TalkShowRuntime (TurnStore + Gemma client + app config)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agent.adapters.gemma_mm_client import GemmaMMClient
from agent.adapters.turn_store import TurnStore
from agent.config import AppConfig, load_config, load_persona_instructions, text_only_enabled
from agent.multi_agent.runtime import MultiAgentRuntime
from agent.telemetry.model_output_logger import ModelOutputLogger


@dataclass
class TalkShowRuntime:
    turn_store: TurnStore
    gemma_client: GemmaMMClient
    config: AppConfig
    multi_agent: MultiAgentRuntime


def build_runtime(config: AppConfig | None = None) -> TalkShowRuntime:
    app_cfg = config or load_config()
    locale = app_cfg.locale()
    turn_store = TurnStore()
    persona = load_persona_instructions("host")
    gemma_client = GemmaMMClient(locale.llm, system_prompt=persona)
    gemma_client.set_persona("host", persona)
    if text_only_enabled():
        gemma_client.set_output_logger(
            ModelOutputLogger(Path(os.environ.get("LOG_DIR", "logs")) / "model-outputs")
        )
    multi_agent = MultiAgentRuntime()
    return TalkShowRuntime(turn_store=turn_store, gemma_client=gemma_client, config=app_cfg, multi_agent=multi_agent)
