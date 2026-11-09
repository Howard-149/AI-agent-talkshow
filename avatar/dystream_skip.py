"""Skip the wav2vec pass DyStream runs on every window and then throws away.

``one_clip_only_inference`` (DyStream Audio2FaceGPT, checked at f587541) calls
``get_audio2face_fea`` and ``get_audio2face_fea_other`` on the current audio window,
then replaces both results with the precomputed features the caller passes in,
because ``use_pre_compute_audio_feature`` is hard-coded True. The online synth always
passes precomputed features, so those two calls cost time (7.6 of 43.5 ms per frame
on Nov 3, plus a GPU→CPU copy each) without changing a single frame.

Kept free of torch so it can be tested without the DyStream env.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_METHODS = ("get_audio2face_fea", "get_audio2face_fea_other")
_MARKER = "use_pre_compute_audio_feature = True"
_checked: dict[type, bool] = {}


def inner_wav2vec_unused(model: Any) -> bool:
    """True when this model's ``one_clip_only_inference`` discards the inner wav2vec
    features, judged from its source; False if the source can't be read or changed."""
    cls = type(model)
    if cls not in _checked:
        try:
            src = inspect.getsource(cls.one_clip_only_inference)
        except (AttributeError, OSError, TypeError):
            src = ""
        _checked[cls] = _MARKER in src
        if not _checked[cls]:
            logger.warning(
                "dystream: one_clip_only_inference may use its own wav2vec features; "
                "not skipping them"
            )
    return _checked[cls]


def _unused(*_args: Any, **_kwargs: Any) -> None:
    return None


@contextmanager
def unused_wav2vec_skipped(model: Any, skip: bool) -> Iterator[bool]:
    """While open, the model's two inner wav2vec calls return None at once.

    Yields whether the skip is active. The previous attributes (including any
    profiler wrappers) come back on exit. Renders run one at a time under the
    sidecar's bake lock, so the swap can't leak into another render.
    """
    if not skip or not inner_wav2vec_unused(model):
        yield False
        return
    saved = {name: model.__dict__[name] for name in _METHODS if name in model.__dict__}
    for name in _METHODS:
        setattr(model, name, _unused)
    try:
        yield True
    finally:
        for name in _METHODS:
            if name in saved:
                setattr(model, name, saved[name])
            else:
                delattr(model, name)
