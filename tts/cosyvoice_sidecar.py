#!/usr/bin/env python3
"""CosyVoice TTS sidecar — warm model once, serve /synthesize.

Endpoints:
  GET  /health
  GET  /spks
  POST /synthesize  JSON:
       mode=instruct|sft|instruct2|zero_shot
       { text, spk_id?, prompt_wav?, instruct?, stream? }
     → audio/wav (16-bit mono)  OR  JSON {pcm_s16le_b64, sample_rate, num_channels}

Default path for talkshow: CosyVoice-300M-Instruct + mode=instruct (spk_id, no wav).
Requires a CosyVoice conda env (not the talkshow agent env). See deploy/RUN-BABEL.md.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cosyvoice_sidecar")

_synth_lock = threading.Lock()
_model = None
_models_ready = False
_warmup_sec: float | None = None
_model_dir: str = ""
_default_mode: str = "instruct"
# prompt_wav absolute path → registered zero_shot_spk_id (speech tokens cached in frontend.spk2info)
_prompt_spk_ids: dict[str, str] = {}


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _synth_speed(payload: dict[str, Any] | None = None) -> float:
    """CosyVoice speaking-rate multiplier (upstream ``tts(..., speed=)``).

    ``1.0`` = normal; ``>1`` faster / shorter audio; ``<1`` slower.
    Typical range 0.5–2.0. Prefer non-stream synth (our default).
    """
    raw = ""
    if payload is not None:
        raw = str(payload.get("speed") or "").strip()
    if not raw:
        raw = os.environ.get("COSYVOICE_SPEED", "1.1").strip()
    try:
        speed = float(raw or "1.1")
    except ValueError:
        speed = 1.1
    return max(0.5, min(2.0, speed))


def _strip_inline_comment(val: str) -> str:
    """Strip quotes then unquoted ``# ...`` tails (common .env foot-guns)."""
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        val = val[1:-1].strip()
    if "#" in val:
        val = val.split("#", 1)[0].rstrip()
    return val.strip().strip("'").strip('"')


def _clean_mode(raw: str) -> str:
    """First token only; strip inline ``#`` comments from .env pollution."""
    val = _strip_inline_comment(str(raw or "")).lower()
    return (val.split() or [""])[0]


def _load_repo_env() -> None:
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = _strip_inline_comment(val)
        os.environ.setdefault(key, val)


def _register_prompt_spk(prompt_wav: str) -> str:
    """Cache prompt-wav speech tokens once (whisper/ONNX) — biggest instruct2 win."""
    assert _model is not None
    key = str(Path(prompt_wav).resolve())
    existing = _prompt_spk_ids.get(key)
    if existing:
        return existing
    spk_id = f"talkshow_{abs(hash(key)) % (10**12)}"
    # Dummy prompt_text; instruct2 overrides prompt_text tokens per request.
    placeholder = "You are a helpful assistant.<|endofprompt|>"
    t0 = time.monotonic()
    ok = _model.add_zero_shot_spk(placeholder, key, spk_id)
    if not ok:
        raise RuntimeError(f"add_zero_shot_spk failed for {key}")
    _prompt_spk_ids[key] = spk_id
    logger.info(
        "cached prompt_wav spk_id=%s path=%s extract_s=%.3f",
        spk_id,
        key,
        time.monotonic() - t0,
    )
    return spk_id


def _inference_instruct2_cached(
    text: str,
    instruct: str,
    prompt_wav: str,
    *,
    stream: bool,
    speed: float = 1.0,
) -> list:
    """instruct2 with cached prompt features; refresh instruct tokens each call."""
    assert _model is not None
    spk_id = _register_prompt_spk(prompt_wav)
    frontend = _model.frontend
    chunks = []
    for i in frontend.text_normalize(text, split=True, text_frontend=True):
        model_input = frontend.frontend_instruct2(
            i, instruct, prompt_wav, _model.sample_rate, spk_id
        )
        # Cached spk2info freezes registration-time prompt_text — replace with this turn's instruct.
        prompt_text_token, prompt_text_token_len = frontend._extract_text_token(instruct)
        model_input["prompt_text"] = prompt_text_token
        model_input["prompt_text_len"] = prompt_text_token_len
        for item in _model.model.tts(**model_input, stream=stream, speed=speed):
            chunks.append(item["tts_speech"])
    return chunks


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _discover_prompt_wavs() -> list[Path]:
    """Collect talkshow clone prompts to pre-cache at warm (no cold first request)."""
    found: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if not resolved.is_file():
            return
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        found.append(resolved)

    prompts_dir = Path(
        os.environ.get("COSYVOICE_PROMPTS_DIR", "").strip()
        or str(_repo_root() / "cosyvoice-prompts")
    )
    if prompts_dir.is_dir():
        for wav in sorted(prompts_dir.glob("*.wav")):
            _add(wav)

    # Absolute env overrides (same keys agent/config.py understands).
    for key, val in os.environ.items():
        if not key.startswith("COSYVOICE_PROMPT_WAV"):
            continue
        raw = (val or "").strip()
        if raw:
            _add(Path(raw))

    # Persona yaml basenames → prompts dir (no agent import; sidecar stays standalone).
    personas = _repo_root() / "config" / "personas"
    if personas.is_dir():
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None  # noqa: N816
        if yaml is not None:
            for yml in sorted(personas.glob("*.yaml")):
                try:
                    data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
                except Exception:  # noqa: BLE001
                    continue
                cosy = ((data.get("tts") or {}).get("cosyvoice") or {})
                wav_map = cosy.get("prompt_wav") or {}
                names: list[str] = []
                if isinstance(wav_map, str) and wav_map.strip():
                    names.append(wav_map.strip())
                elif isinstance(wav_map, dict):
                    for name in wav_map.values():
                        n = str(name or "").strip()
                        if n:
                            names.append(n)
                for name in names:
                    _add(prompts_dir / Path(name).name)
                # Role basename fallbacks (language-agnostic + legacy en_*).
                role = yml.stem
                for name in (f"{role}.wav", f"en_{role}.wav"):
                    _add(prompts_dir / name)

    return found


def _warm_prompt_caches() -> None:
    """Register all prompt wavs into CosyVoice spk2info during sidecar warm."""
    if not _env_flag("COSYVOICE_CACHE_PROMPT", True):
        logger.info("COSYVOICE_CACHE_PROMPT=0 — skip prompt cache warm")
        return
    if not _env_flag("COSYVOICE_SIDECAR_WARM_PROMPTS", True):
        logger.info("COSYVOICE_SIDECAR_WARM_PROMPTS=0 — skip prompt cache warm")
        return
    paths = _discover_prompt_wavs()
    if not paths:
        logger.warning(
            "no prompt wavs found under cosyvoice-prompts/ — "
            "first instruct2 request will pay extract cost"
        )
        return
    t0 = time.monotonic()
    for path in paths:
        try:
            _register_prompt_spk(str(path))
        except Exception:  # noqa: BLE001
            logger.exception("prompt cache warm failed path=%s", path)
    logger.info(
        "prompt cache warm done n=%d elapsed=%.2fs ids=%s",
        len(_prompt_spk_ids),
        time.monotonic() - t0,
        list(_prompt_spk_ids.values()),
    )


def _warm_synth_probe() -> None:
    """One short instruct2 (or instruct) synth to warm CUDA / wetext / flow kernels."""
    if not _env_flag("COSYVOICE_SIDECAR_WARM_PROBE", True):
        logger.info("COSYVOICE_SIDECAR_WARM_PROBE=0 — skip synth probe")
        return
    assert _model is not None
    mode = _clean_mode(os.environ.get("COSYVOICE_MODE", _default_mode)) or "instruct"
    instruct = (
        "You are a helpful assistant. Speak in a calm, natural tone."
        "<|endofprompt|>"
    )
    text = "Hi."
    t0 = time.monotonic()
    try:
        if mode in ("instruct2", "zero_shot") and _prompt_spk_ids:
            prompt_wav = next(iter(_prompt_spk_ids.keys()))
            if mode == "instruct2":
                chunks = _inference_instruct2_cached(
                    text,
                    instruct,
                    prompt_wav,
                    stream=False,
                    speed=_synth_speed(),
                )
            else:
                chunks = []
                for item in _model.inference_zero_shot(
                    text,
                    instruct,
                    prompt_wav,
                    stream=False,
                    speed=_synth_speed(),
                ):
                    chunks.append(item["tts_speech"])
            _ = chunks
        elif mode in ("instruct", "sft"):
            spks = _list_spks()
            if not spks:
                logger.warning("synth probe skipped — no built-in spks for mode=%s", mode)
                return
            spk = spks[0]
            if mode == "instruct":
                for _ in _model.inference_instruct(text, spk, instruct, stream=False):
                    pass
            else:
                for _ in _model.inference_sft(text, spk, stream=False):
                    pass
        else:
            logger.warning(
                "synth probe skipped — mode=%s needs prompt wavs (none cached)",
                mode,
            )
            return
        logger.info("synth probe ok mode=%s elapsed=%.2fs", mode, time.monotonic() - t0)
    except Exception:  # noqa: BLE001
        logger.exception(
            "synth probe failed mode=%s (sidecar still serves; first request may be slow)",
            mode,
        )


def warm_model(model_dir: str) -> None:
    global _model, _models_ready, _warmup_sec, _model_dir
    if _models_ready and _model is not None:
        return
    _model_dir = model_dir
    cosy_root = os.environ.get("COSYVOICE_ROOT", "").strip()
    if not cosy_root:
        # .../CosyVoice/pretrained_models/<ckpt>
        md = Path(model_dir).resolve()
        if md.parent.name == "pretrained_models" and (md.parent.parent / "cosyvoice").is_dir():
            cosy_root = str(md.parent.parent)
    if cosy_root:
        matcha = str(Path(cosy_root) / "third_party" / "Matcha-TTS")
        if cosy_root not in sys.path:
            sys.path.insert(0, cosy_root)
        if matcha not in sys.path:
            sys.path.insert(0, matcha)
    else:
        raise RuntimeError(
            "COSYVOICE_ROOT unset and could not infer from model_dir. "
            "Set COSYVOICE_ROOT to the CosyVoice git clone (directory that contains cosyvoice/)."
        )

    fp16 = _env_flag("COSYVOICE_FP16", True)
    load_vllm = _env_flag("COSYVOICE_LOAD_VLLM", False)
    load_trt = _env_flag("COSYVOICE_LOAD_TRT", False)
    # Official CosyVoice3 vllm_example uses fp16=False with load_vllm+load_trt.
    if load_vllm and fp16 and not os.environ.get("COSYVOICE_FP16"):
        fp16 = False
        logger.info("COSYVOICE_LOAD_VLLM=1 — defaulting fp16=False (override with COSYVOICE_FP16=1)")
    if load_trt and fp16:
        logger.warning(
            "load_trt + fp16 can be unstable on CosyVoice3; prefer COSYVOICE_FP16=0"
        )

    logger.info(
        "loading CosyVoice from %s (root=%s fp16=%s vllm=%s trt=%s) …",
        model_dir,
        cosy_root,
        fp16,
        load_vllm,
        load_trt,
    )
    t0 = time.monotonic()
    try:
        from cosyvoice.cli.cosyvoice import AutoModel  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot import cosyvoice (PYTHONPATH/COSYVOICE_ROOT={cosy_root!r}). "
            "Install CosyVoice deps in that conda env and set COSYVOICE_ROOT."
        ) from exc

    if load_vllm:
        # Required by CosyVoice upstream vllm_example.py before AutoModel.
        try:
            from vllm import ModelRegistry  # noqa: WPS433
            from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM  # noqa: WPS433

            ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        except Exception as exc:
            raise RuntimeError(
                "COSYVOICE_LOAD_VLLM=1 but vLLM/CosyVoice vllm backend import failed. "
                "Use a dedicated env (see deploy/RUN-BABEL.md CosyVoice vLLM section): "
                f"{exc}"
            ) from exc

    auto_kwargs: dict[str, Any] = {
        "model_dir": model_dir,
        "fp16": fp16,
        "load_vllm": load_vllm,
        "load_trt": load_trt,
    }
    if load_trt:
        auto_kwargs["trt_concurrent"] = int(
            os.environ.get("COSYVOICE_TRT_CONCURRENT", "1") or "1"
        )

    _model = AutoModel(**auto_kwargs)
    load_s = time.monotonic() - t0

    # Mark not ready until caches + probe finish — avoid first-request cold start.
    _warm_prompt_caches()
    _warm_synth_probe()

    _warmup_sec = time.monotonic() - t0
    _models_ready = True
    spks: list[str] = []
    try:
        spks = list(_model.list_available_spks())
    except Exception:  # noqa: BLE001
        logger.warning("list_available_spks failed (ok for CosyVoice2/3 zero-shot models)")
    logger.info(
        "models ready (load=%.1fs warmup_total=%.1fs) fp16=%s vllm=%s trt=%s "
        "prompt_cache=%d spks=%s — sidecar can accept /synthesize",
        load_s,
        _warmup_sec,
        fp16,
        load_vllm,
        load_trt,
        len(_prompt_spk_ids),
        spks,
    )


def _list_spks() -> list[str]:
    if _model is None:
        return []
    try:
        return list(_model.list_available_spks())
    except Exception:  # noqa: BLE001
        return []


def _tensor_to_wav_bytes(speech, sample_rate: int) -> bytes:
    import torch

    if hasattr(speech, "detach"):
        speech = speech.detach().cpu()
    if speech.dim() > 1:
        speech = speech.squeeze()
    clipped = speech.clamp(-1.0, 1.0)
    pcm = (clipped * 32767.0).to(torch.int16).numpy().tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def _run_synth(payload: dict[str, Any]) -> tuple[bytes, int]:
    assert _model is not None
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")
    mode = _clean_mode(str(payload.get("mode") or _default_mode or "instruct")) or "instruct"
    stream = bool(payload.get("stream", False))
    instruct = str(payload.get("instruct") or "").strip()
    spk_id = str(payload.get("spk_id") or payload.get("spk") or "").strip()
    prompt_wav = str(payload.get("prompt_wav") or "").strip()
    speed = _synth_speed(payload)

    chunks = []
    if mode == "sft":
        if not spk_id:
            raise ValueError("mode=sft requires spk_id")
        for item in _model.inference_sft(text, spk_id, stream=stream, speed=speed):
            chunks.append(item["tts_speech"])
    elif mode == "instruct":
        # CosyVoice-300M-Instruct: text + built-in spk + instruct (emotion), no wav.
        if not spk_id:
            raise ValueError("mode=instruct requires spk_id")
        if not instruct:
            instruct = (
                "You are a helpful assistant. Speak in a calm, natural tone."
                "<|endofprompt|>"
            )
        for item in _model.inference_instruct(
            text, spk_id, instruct, stream=stream, speed=speed
        ):
            chunks.append(item["tts_speech"])
    elif mode == "zero_shot":
        if not prompt_wav or not Path(prompt_wav).is_file():
            raise ValueError(f"mode=zero_shot needs prompt_wav file: {prompt_wav!r}")
        prompt_text = str(
            payload.get("prompt_text")
            or instruct
            or "You are a helpful assistant.<|endofprompt|>"
        )
        for item in _model.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=stream, speed=speed
        ):
            chunks.append(item["tts_speech"])
    elif mode == "instruct2":
        if not prompt_wav or not Path(prompt_wav).is_file():
            raise ValueError(f"mode=instruct2 needs prompt_wav file: {prompt_wav!r}")
        if not instruct:
            instruct = (
                "You are a helpful assistant. Speak in a calm, natural tone."
                "<|endofprompt|>"
            )
        if _env_flag("COSYVOICE_CACHE_PROMPT", True):
            chunks = _inference_instruct2_cached(
                text, instruct, prompt_wav, stream=stream, speed=speed
            )
        else:
            for item in _model.inference_instruct2(
                text, instruct, prompt_wav, stream=stream, speed=speed
            ):
                chunks.append(item["tts_speech"])
    else:
        raise ValueError(f"unknown mode={mode!r} (instruct|sft|instruct2|zero_shot)")

    if not chunks:
        raise RuntimeError("CosyVoice returned no audio chunks")
    import torch

    speech = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
    rate = int(getattr(_model, "sample_rate", 22050))
    return _tensor_to_wav_bytes(speech, rate), rate


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            payload = {
                "ok": _models_ready,
                "model_dir": _model_dir,
                "warmup_sec": _warmup_sec,
                "default_mode": _default_mode,
                "fp16": _env_flag("COSYVOICE_FP16", True),
                "prompt_cache": len(_prompt_spk_ids),
                "spks": _list_spks() if _models_ready else [],
            }
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(200 if _models_ready else 503, raw, "application/json")
            return
        if path == "/spks":
            if not _models_ready:
                self._send(503, b"models not ready", "text/plain")
                return
            raw = json.dumps(
                {"speakers": _list_spks()}, ensure_ascii=False
            ).encode("utf-8")
            self._send(200, raw, "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/synthesize":
            self._send(404, b"not found", "text/plain")
            return
        if not _models_ready:
            self._send(503, b"models not ready", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, b"invalid json", "text/plain")
            return

        t0 = time.monotonic()
        try:
            with _synth_lock:
                wav_bytes, rate = _run_synth(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("synthesize failed")
            err = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(500, err, "application/json")
            return

        want_json = "application/json" in (self.headers.get("Accept") or "")
        if want_json or str(payload.get("format") or "").lower() == "json":
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
                rate = wf.getframerate()
            out = {
                "sample_rate": rate,
                "num_channels": 1,
                "pcm_s16le_b64": base64.b64encode(pcm).decode("ascii"),
                "latency_s": round(time.monotonic() - t0, 3),
            }
            self._send(200, json.dumps(out).encode("utf-8"), "application/json")
            return

        logger.info(
            "synthesize ok mode=%s spk=%s chars=%d rate=%d speed=%.2f latency_s=%.3f",
            payload.get("mode") or _default_mode,
            payload.get("spk_id") or payload.get("spk") or "",
            len(str(payload.get("text") or "")),
            rate,
            _synth_speed(payload),
            time.monotonic() - t0,
        )
        self._send(200, wav_bytes, "audio/wav")


def main() -> None:
    global _default_mode
    _load_repo_env()
    _default_mode = _clean_mode(os.environ.get("COSYVOICE_MODE", "instruct")) or "instruct"
    parser = argparse.ArgumentParser(description="CosyVoice TTS sidecar")
    parser.add_argument(
        "--host", default=os.environ.get("COSYVOICE_SIDECAR_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("COSYVOICE_SIDECAR_PORT", "8767")),
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("COSYVOICE_MODEL_DIR", ""),
    )
    args = parser.parse_args()
    model_dir = args.model_dir.strip()
    if not model_dir:
        logger.error("Set COSYVOICE_MODEL_DIR or --model-dir")
        sys.exit(1)

    warm = os.environ.get("COSYVOICE_SIDECAR_WARM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if warm:
        warm_model(model_dir)
    else:
        logger.warning(
            "COSYVOICE_SIDECAR_WARM=0 — /synthesize will fail until warm_model"
        )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info(
        "listening on http://%s:%s default_mode=%s",
        args.host,
        args.port,
        _default_mode,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
