from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _expand_path(path: str) -> str:
    return path.replace("${USER}", os.environ.get("USER", ""))


@dataclass(frozen=True)
class LocaleLLMConfig:
    model: str
    base_url: str
    api_key: str
    max_tokens: int
    reply_language: str
    include_heard_summary: bool


@dataclass(frozen=True)
class LocaleTTSConfig:
    engine: str
    model_path: str
    sample_rate: int


@dataclass(frozen=True)
class LocaleConfig:
    llm: LocaleLLMConfig
    tts: LocaleTTSConfig


@dataclass(frozen=True)
class AppConfig:
    default_locale: str
    locales: dict[str, LocaleConfig]

    def locale(self, code: str | None = None) -> LocaleConfig:
        key = code or self.default_locale
        if key not in self.locales:
            raise KeyError(f"Unknown locale: {key}")
        return self.locales[key]


def load_persona_yaml(persona_id: str) -> dict:
    path = REPO_ROOT / "config" / "personas" / f"{persona_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_persona_name(persona_id: str) -> str:
    """Display / in-character name (aligned with Piper voice per role)."""
    return str(load_persona_yaml(persona_id).get("name", persona_id.title()))


def load_persona_instructions(
    persona_id: str = "host", *, panel_mode: bool = False
) -> str:
    base = str(load_persona_yaml(persona_id)["instructions"]).strip()
    if not panel_mode:
        return base
    from agent.show_context import panel_host_gemma_addon

    if persona_id == "host":
        return f"{base}\n\n{panel_host_gemma_addon()}".strip()
    return base


def load_persona_personality(persona_id: str) -> str:
    """In-character personality for panel turns (speaking + hand-raise polls)."""
    data = load_persona_yaml(persona_id)
    raw = data.get("personality") or data.get("panel_voice")  # panel_voice: legacy alias
    if raw:
        return str(raw).strip()
    return ""


@dataclass(frozen=True)
class DialogueConfig:
    library: str | None = None
    pick: str = "all"  # all | rotate | random | none


@dataclass(frozen=True)
class TurnControlConfig:
    mode: str  # manual_only | rotate_after_user | panel_round_robin | host_moderated
    order: tuple[str, ...]
    listen_role: str  # who hears the human (Gemma audio-in) in panel mode


@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    default_room: str
    turn_control: TurnControlConfig
    dialogue: DialogueConfig | None = None


def load_scenario(path: Path | None = None) -> ScenarioConfig:
    scenario_path = path or Path(
        os.environ.get(
            "SCENARIO_PATH",
            REPO_ROOT / "config" / "scenarios" / "default.yaml",
        )
    )
    raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    tc = raw.get("turn_control", {})
    meet = raw.get("meet", {})
    order = tuple(tc.get("order", ["host", "guest", "commentator"]))
    dialogue_raw = raw.get("dialogue")
    dialogue: DialogueConfig | None = None
    if isinstance(dialogue_raw, dict):
        library = dialogue_raw.get("library")
        pick_raw = str(dialogue_raw.get("pick", "all")).strip().lower()
        dialogue = DialogueConfig(
            library=str(library).strip() if library else None,
            pick=pick_raw,
        )
    return ScenarioConfig(
        id=str(raw.get("id", "default")),
        default_room=str(meet.get("default_room", "talkshow-dev")),
        turn_control=TurnControlConfig(
            mode=str(tc.get("mode", "manual_only")),
            order=order,
            listen_role=str(tc.get("listen_role", "host")),
        ),
        dialogue=dialogue,
    )


def load_persona_tts(persona_id: str, app_cfg: AppConfig) -> LocaleTTSConfig:
    """Per-role Piper: .env PIPER_MODEL_PATH_<ROLE> → persona yaml tts → PIPER_MODEL_PATH (host)."""
    import logging

    logger = logging.getLogger(__name__)
    data = load_persona_yaml(persona_id)
    base = app_cfg.locale()
    role_key = persona_id.upper()

    if persona_id == "host":
        env_path = os.environ.get("PIPER_MODEL_PATH", "").strip()
    else:
        env_path = os.environ.get(f"PIPER_MODEL_PATH_{role_key}", "").strip()

    if env_path:
        cfg = LocaleTTSConfig(
            engine=base.tts.engine,
            model_path=_expand_path(env_path),
            sample_rate=base.tts.sample_rate,
        )
        logger.info("Piper %s ← %s", persona_id, cfg.model_path)
        return cfg

    tts_block = data.get("tts")
    if tts_block:
        cfg = LocaleTTSConfig(
            engine=tts_block.get("engine", base.tts.engine),
            model_path=_expand_path(tts_block.get("model_path", base.tts.model_path)),
            sample_rate=int(tts_block.get("sample_rate", base.tts.sample_rate)),
        )
        logger.info("Piper %s ← %s (persona yaml)", persona_id, cfg.model_path)
        return cfg

    logger.warning(
        "Piper %s: no PIPER_MODEL_PATH_%s / persona tts — using default %s",
        persona_id,
        role_key,
        base.tts.model_path,
    )
    return base.tts


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or Path(
        os.environ.get("CONFIG_PATH", REPO_ROOT / "config" / "multimodal.yaml")
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    locales: dict[str, LocaleConfig] = {}
    for code, block in raw["locales"].items():
        llm = block["llm"]
        tts = block["tts"]
        locales[code] = LocaleConfig(
            llm=LocaleLLMConfig(
                model=os.environ.get("VLLM_MODEL", llm["model"]),
                base_url=os.environ.get("VLLM_BASE_URL", llm["base_url"]),
                api_key=os.environ.get("VLLM_API_KEY", llm.get("api_key", "EMPTY")),
                max_tokens=int(llm.get("max_tokens", 512)),
                reply_language=llm.get("reply_language", "English"),
                include_heard_summary=bool(llm.get("include_heard_summary", True)),
            ),
            tts=LocaleTTSConfig(
                engine=tts.get("engine", "piper"),
                model_path=_expand_path(
                    os.environ.get("PIPER_MODEL_PATH", tts["model_path"])
                ),
                sample_rate=int(tts.get("sample_rate", 22050)),
            ),
        )

    return AppConfig(
        default_locale=raw.get("default_locale", "en"),
        locales=locales,
    )
