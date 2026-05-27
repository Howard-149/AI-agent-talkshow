from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def load_persona_instructions(persona_id: str = "host") -> str:
    path = REPO_ROOT / "config" / "personas" / f"{persona_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(data["instructions"]).strip()


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
                model_path=os.environ.get("PIPER_MODEL_PATH", tts["model_path"]),
                sample_rate=int(tts.get("sample_rate", 22050)),
            ),
        )

    return AppConfig(
        default_locale=raw.get("default_locale", "en"),
        locales=locales,
    )
