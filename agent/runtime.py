from __future__ import annotations

from dataclasses import dataclass

from agent.adapters.gemma_mm_client import GemmaMMClient
from agent.adapters.turn_store import TurnStore
from agent.config import AppConfig, load_config, load_persona_instructions


@dataclass
class TalkShowRuntime:
    turn_store: TurnStore
    gemma_client: GemmaMMClient
    config: AppConfig


def build_runtime(config: AppConfig | None = None) -> TalkShowRuntime:
    app_cfg = config or load_config()
    locale = app_cfg.locale()
    turn_store = TurnStore()
    persona = load_persona_instructions("host")
    gemma_client = GemmaMMClient(locale.llm, system_prompt=persona)
    gemma_client.set_persona("host", persona)
    return TalkShowRuntime(turn_store=turn_store, gemma_client=gemma_client, config=app_cfg)
