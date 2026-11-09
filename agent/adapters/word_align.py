"""Word timings for synthesized speech by CTC forced alignment.

TTS gives us both the text and its audio, so instead of recognizing what is being
said we align the known words to the PCM (torchaudio wav2vec2, English letters) and
get where each word ends in the clip. Playback then knows the exact moment a word is
heard: the sentence's playback start plus that word's end.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")
_lock = threading.Lock()
_model: Any = None  # (model, label index, sample rate, device) once loaded
_failed = False


def word_align_enabled() -> bool:
    return os.environ.get("TALKSHOW_WORD_ALIGN", "1").lower() not in ("0", "false", "no")


def _load() -> Any:
    global _model, _failed
    with _lock:
        if _model is not None or _failed:
            return _model
        try:
            import torch
            import torchaudio

            t0 = time.monotonic()
            device = os.environ.get("TALKSHOW_WORD_ALIGN_DEVICE", "cpu")
            threads = int(os.environ.get("TALKSHOW_WORD_ALIGN_THREADS", "4"))
            if device == "cpu" and threads > 0:
                torch.set_num_threads(threads)
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            model = bundle.get_model().to(device).eval()
            labels = {label: i for i, label in enumerate(bundle.get_labels())}
            _model = (model, labels, bundle.sample_rate, device)
            logger.info(
                "word aligner ready device=%s in %.1fs", device, time.monotonic() - t0
            )
        except Exception:
            _failed = True
            logger.exception("word aligner unavailable — backchannels fall back to sentence timing")
        return _model


def warm_word_aligner() -> None:
    """Load the model off the event loop so the first line doesn't pay for it."""
    if word_align_enabled():
        threading.Thread(target=_load, name="word-align-warm", daemon=True).start()


def _letters(word: str) -> str:
    return re.sub(r"[^A-Z']", "", word.upper().replace("’", "'"))


def align_word_ends(pcm: bytes, sample_rate: int, text: str) -> list[float] | None:
    """Seconds into ``pcm`` (int16 mono) at which each whitespace word of ``text`` ends.

    Words with no letters to align (numbers, dashes) take the end of the word before
    them. None when alignment isn't possible.
    """
    words = _WORD_RE.findall(text)
    loaded = _load()
    if not words or not pcm or loaded is None:
        return None
    model, labels, model_rate, device = loaded

    import torch
    import torchaudio.functional as F

    letters = [_letters(w) for w in words]
    aligned = [i for i, w in enumerate(letters) if w]
    if not aligned:
        return None
    tokens: list[int] = []
    for n, i in enumerate(aligned):
        if n:
            tokens.append(labels["|"])
        tokens.extend(labels[c] for c in letters[i])

    with torch.inference_mode():
        wave = torch.frombuffer(bytearray(pcm), dtype=torch.int16).float() / 32768.0
        wave = wave.unsqueeze(0)
        if sample_rate != model_rate:
            wave = F.resample(wave, sample_rate, model_rate)
        emission, _ = model(wave.to(device))
        emission = torch.log_softmax(emission, dim=-1).cpu()
    frames = emission.shape[1]
    if frames < len(tokens):
        return None
    path, scores = F.forced_align(
        emission, torch.tensor([tokens], dtype=torch.int32), blank=0
    )
    spans = F.merge_tokens(path[0], scores[0].exp())
    seconds_per_frame = wave.shape[1] / frames / model_rate

    # merge_tokens yields one span per target token, in order.
    ends: dict[int, float] = {}
    pos = 0
    for n, i in enumerate(aligned):
        if n:
            pos += 1  # the "|" between words
        pos += len(letters[i])
        ends[i] = spans[pos - 1].end * seconds_per_frame

    result: list[float] = []
    last = 0.0
    for i in range(len(words)):
        last = ends.get(i, last)
        result.append(last)
    return result


async def align_word_ends_async(pcm: bytes, sample_rate: int, text: str) -> list[float] | None:
    try:
        t0 = time.monotonic()
        ends = await asyncio.to_thread(align_word_ends, pcm, sample_rate, text)
        logger.debug(
            "word align words=%d ms=%d text=%.40r",
            len(ends or []),
            round((time.monotonic() - t0) * 1000),
            text,
        )
        return ends
    except Exception:
        logger.exception("word align failed text=%.60r", text)
        return None
