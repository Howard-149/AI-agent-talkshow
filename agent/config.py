"""Load scenario, persona, locale LLM/TTS YAML into typed talk-show config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COSYVOICE_PROMPTS_DIR = REPO_ROOT / "cosyvoice-prompts"


def _expand_path(path: str) -> str:
    """Expand ``${USER}`` in absolute/cluster paths (Piper ONNX, etc.)."""
    return path.replace("${USER}", os.environ.get("USER", ""))


def _cosyvoice_prompt_path(persona_id: str, locale: str, persona_yaml: dict) -> Path:
    """Resolve clone prompt under ``<repo>/cosyvoice-prompts/``.

    Persona yaml ``tts.cosyvoice.prompt_wav`` is a **basename** (language-agnostic
    for instruct2/zero_shot), or a legacy locale→basename map. Missing key tries
    ``{persona_id}.wav``, then ``en_{persona_id}.wav``, then ``{locale}_{persona_id}.wav``.
    """
    cosy = ((persona_yaml.get("tts") or {}).get("cosyvoice") or {})
    wav_map = cosy.get("prompt_wav") or {}
    candidates: list[str] = []
    if isinstance(wav_map, str) and wav_map.strip():
        candidates.append(wav_map.strip())
    elif isinstance(wav_map, dict):
        for key in (locale, "default", "en", "zh"):
            name = str(wav_map.get(key) or "").strip()
            if name:
                candidates.append(name)
        for name in wav_map.values():
            n = str(name or "").strip()
            if n:
                candidates.append(n)
    for fallback in (
        f"{persona_id}.wav",
        f"en_{persona_id}.wav",
        f"{locale}_{persona_id}.wav",
    ):
        candidates.append(fallback)

    seen: set[str] = set()
    for name in candidates:
        base = Path(name).name
        if not base or base in seen:
            continue
        seen.add(base)
        path = COSYVOICE_PROMPTS_DIR / base
        if path.is_file():
            return path
    # Last resort: first candidate path (even if missing — caller/logs surface it).
    first = next(iter(seen), f"{persona_id}.wav")
    return COSYVOICE_PROMPTS_DIR / first

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
    from agent.show.show_context import panel_host_gemma_addon

    if persona_id == "host":
        return f"{base}\n\n{panel_host_gemma_addon()}".strip()
    return base


@dataclass(frozen=True)
class PersonaAvatarConfig:
    portrait: str | None = None
    idle_video: str | None = None
    dystream_enabled: bool = True


def load_persona_avatar(persona_id: str) -> PersonaAvatarConfig:
    """DyStream portrait + idle loop paths from persona yaml ui.avatar."""
    data = load_persona_yaml(persona_id)
    ui = data.get("ui") or {}
    avatar_cfg = ui.get("avatar")
    if not isinstance(avatar_cfg, dict):
        return PersonaAvatarConfig()

    dystream_raw = avatar_cfg.get("dystream")
    dystream_enabled = True
    if isinstance(dystream_raw, dict) and dystream_raw.get("enabled") is False:
        dystream_enabled = False

    portrait = avatar_cfg.get("portrait")
    idle_video = avatar_cfg.get("idle_video")
    return PersonaAvatarConfig(
        portrait=str(portrait).strip() if portrait else None,
        idle_video=str(idle_video).strip() if idle_video else None,
        dystream_enabled=dystream_enabled,
    )


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


def load_persona_tts(
    persona_id: str,
    app_cfg: AppConfig,
    *,
    locale: str | None = None,
) -> LocaleTTSConfig:
    """Per-role TTS: CosyVoice prompt wav or Piper ONNX, by locale/role env → yaml → default."""
    import logging

    from agent.adapters.tts_synthesize import resolve_tts_engine
    from agent.locale.viewer_locales import DEFAULT_LOCALE, normalize_locale

    logger = logging.getLogger(__name__)
    data = load_persona_yaml(persona_id)
    loc = normalize_locale(locale) if locale else DEFAULT_LOCALE
    try:
        base = app_cfg.locale(loc)
    except KeyError:
        base = app_cfg.locale()
        loc = app_cfg.default_locale
    role_key = persona_id.upper()
    loc_key = loc.upper()
    engine = resolve_tts_engine(base.tts)

    if engine == "cosyvoice":
        from agent.adapters.cosyvoice_voices import cosyvoice_mode, mode_needs_prompt_wav, resolve_spk_id

        sample_rate = int(
            os.environ.get("COSYVOICE_SAMPLE_RATE", "").strip()
            or base.tts.sample_rate
            or 22050
        )
        mode = cosyvoice_mode()
        if mode_needs_prompt_wav(mode):
            # Optional absolute .env override; else file under cosyvoice-prompts/.
            # Prefer role-only (language-agnostic clone ref); locale-specific is legacy.
            env_path = (
                os.environ.get(f"COSYVOICE_PROMPT_WAV_{role_key}", "").strip()
                or os.environ.get(f"COSYVOICE_PROMPT_WAV_{loc_key}_{role_key}", "").strip()
            )
            if not env_path and persona_id == "host":
                env_path = (
                    os.environ.get("COSYVOICE_PROMPT_WAV", "").strip()
                    or os.environ.get(f"COSYVOICE_PROMPT_WAV_{loc_key}", "").strip()
                )
            prompt = (
                Path(_expand_path(env_path))
                if env_path
                else _cosyvoice_prompt_path(persona_id, loc, data)
            )
            logger.info(
                "CosyVoice %s locale=%s mode=%s ← %s",
                persona_id,
                loc,
                mode,
                prompt,
            )
            return LocaleTTSConfig(
                engine="cosyvoice",
                model_path=str(prompt),
                sample_rate=sample_rate,
            )

        # Built-in spk_id (CosyVoice-300M-Instruct / SFT) — no wav.
        spk = resolve_spk_id(role=persona_id, locale=loc, model_path="")
        cosy_block = ((data.get("tts") or {}).get("cosyvoice") or {})
        if cosy_block.get("spk_id"):
            spk = str(cosy_block["spk_id"]).strip() or spk
        logger.info(
            "CosyVoice %s locale=%s mode=%s ← spk %s",
            persona_id,
            loc,
            mode,
            spk,
        )
        return LocaleTTSConfig(
            engine="cosyvoice",
            model_path=spk,
            sample_rate=sample_rate,
        )

    # e.g. PIPER_MODEL_PATH_ZH_HOST, PIPER_MODEL_PATH_ZH_GUEST
    env_path = os.environ.get(f"PIPER_MODEL_PATH_{loc_key}_{role_key}", "").strip()
    if not env_path and persona_id == "host":
        env_path = os.environ.get(f"PIPER_MODEL_PATH_{loc_key}", "").strip()
    if not env_path and loc == DEFAULT_LOCALE:
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
        logger.info("Piper %s locale=%s ← %s", persona_id, loc, cfg.model_path)
        return cfg

    tts_block = data.get("tts")
    if tts_block and loc == DEFAULT_LOCALE:
        cfg = LocaleTTSConfig(
            engine=tts_block.get("engine", base.tts.engine),
            model_path=_expand_path(tts_block.get("model_path", base.tts.model_path)),
            sample_rate=int(tts_block.get("sample_rate", base.tts.sample_rate)),
        )
        logger.info("Piper %s ← %s (persona yaml)", persona_id, cfg.model_path)
        return cfg

    logger.warning(
        "Piper %s locale=%s: no PIPER_MODEL_PATH_%s_%s — using locale default %s",
        persona_id,
        loc,
        loc_key,
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
                    (
                        os.environ.get("PIPER_MODEL_PATH", tts["model_path"])
                        if code == "en"
                        else os.environ.get(
                            f"PIPER_MODEL_PATH_{code.upper()}", tts["model_path"]
                        )
                    )
                ),
                sample_rate=int(tts.get("sample_rate", 22050)),
            ),
        )

    return AppConfig(
        default_locale=raw.get("default_locale", "en"),
        locales=locales,
    )
