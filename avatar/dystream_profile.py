"""Stage timers for DyStream online synth (attention vs flow-matching vs render).

``one_clip_only_inference`` (DyStream Audio2FaceGPT) is:

  1. get_audio2face_fea / _other  — wav2vec on the window (often discarded;
     we already pass precomputed features)
  2. GPT ``self.blocks``          — causal + cross attention (AR transformer)
  3. ``diffusion_head`` × steps   — flow-matching velocity net (CFG batch=5)
  4. ``noise_scheduler.step``     — Euler / ODE state update (usually cheap)

Stages (see ``summary()`` keys):

  audio_load     wav decode (librosa)
  audio2face     audio encoder → face feature (once per clip)
  ar_fm          whole ``one_clip_only_inference``
  ar_a2f_inner   wav2vec *inside* one_clip (duplicate of audio2face)
  ar_attn        GPT transformer blocks (attention)
  fm_net         DiffusionHead (flow-matching network)
  fm_ode         ``noise_scheduler.step`` (ODE integrator math)
  ar_other       remainder of one_clip (proj / CFG / masks)
  vis_flow       visualization ``flow_estimator``
  face_gen       visualization ``face_generator``
  xfer           GPU→CPU + RGBA pack

CUDA event totals are flushed once per clip (accurate GPU ms without
per-frame host stall). CPU ms are always accumulated as a fallback.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

_STAGE_ORDER = (
    "audio_load",
    "audio2face",
    "ar_fm",
    "ar_a2f_inner",
    "ar_attn",
    "fm_net",
    "fm_ode",
    "ar_other",
    "vis_flow",
    "face_gen",
    "xfer",
)


def profile_enabled() -> bool:
    raw = os.environ.get("DYSTREAM_PROFILE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _cuda_events_enabled() -> bool:
    raw = os.environ.get("DYSTREAM_PROFILE_CUDA", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


class SynthProfiler:
    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = profile_enabled() if enabled is None else enabled
        self.cpu_ms: dict[str, float] = defaultdict(float)
        self.gpu_ms: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._gpu_pairs: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        self._use_cuda = False
        if self.enabled and _cuda_events_enabled():
            try:
                import torch

                self._use_cuda = bool(torch.cuda.is_available())
            except Exception:
                self._use_cuda = False

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.monotonic()
        start_ev = end_ev = None
        if self._use_cuda:
            import torch

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        try:
            yield
        finally:
            if self._use_cuda and start_ev is not None and end_ev is not None:
                end_ev.record()
                self._gpu_pairs[name].append((start_ev, end_ev))
            self.cpu_ms[name] += (time.monotonic() - t0) * 1000.0
            self.counts[name] += 1

    def maybe(self, name: str):
        return self.span(name) if self.enabled else nullcontext()

    def flush_gpu(self) -> None:
        if not self._use_cuda or not self._gpu_pairs:
            return
        import torch

        try:
            torch.cuda.synchronize()
        except Exception:
            return
        for name, pairs in self._gpu_pairs.items():
            acc = 0.0
            for start_ev, end_ev in pairs:
                try:
                    acc += float(start_ev.elapsed_time(end_ev))
                except Exception:
                    pass
            self.gpu_ms[name] = acc
        self._gpu_pairs.clear()

    def _pick_ms(self, name: str) -> float:
        gpu = self.gpu_ms.get(name, 0.0)
        if gpu > 0:
            return gpu
        return float(self.cpu_ms.get(name, 0.0))

    def summary(self, *, frames: int, steps: int, window: int) -> dict[str, Any]:
        self.flush_gpu()
        ar_fm = self._pick_ms("ar_fm")
        ar_attn = self._pick_ms("ar_attn")
        fm_net = self._pick_ms("fm_net")
        fm_ode = self._pick_ms("fm_ode")
        ar_a2f = self._pick_ms("ar_a2f_inner")
        accounted = ar_attn + fm_net + fm_ode + ar_a2f
        ar_other = max(0.0, ar_fm - accounted) if ar_fm > 0 else 0.0
        # Legacy alias: "ar_net" ≈ GPT attention (not "AR minus scheduler").
        ar_net = ar_attn if ar_attn > 0 else max(0.0, ar_fm - fm_ode - fm_net - ar_a2f)
        vis_flow = self._pick_ms("vis_flow")
        face_gen = self._pick_ms("face_gen")
        xfer = self._pick_ms("xfer")
        render = vis_flow + face_gen + xfer
        audio_load = self._pick_ms("audio_load")
        audio2face = self._pick_ms("audio2face")
        n = max(int(frames), 1)
        clock = "cuda" if self.gpu_ms else "cpu"

        def _r(ms: float) -> int:
            return int(round(ms))

        return {
            "clock": clock,
            "frames": int(frames),
            "steps": int(steps),
            "window": int(window),
            "audio_load_ms": _r(audio_load),
            "audio2face_ms": _r(audio2face),
            "ar_fm_ms": _r(ar_fm),
            "ar_a2f_inner_ms": _r(ar_a2f),
            "ar_attn_ms": _r(ar_attn),
            "ar_net_ms": _r(ar_net),
            "fm_net_ms": _r(fm_net),
            "fm_ode_ms": _r(fm_ode),
            "ar_other_ms": _r(ar_other),
            "vis_flow_ms": _r(vis_flow),
            "face_gen_ms": _r(face_gen),
            "xfer_ms": _r(xfer),
            "render_ms": _r(render),
            "ms_per_frame_ar_fm": round(ar_fm / n, 3),
            "ms_per_frame_ar_a2f_inner": round(ar_a2f / n, 3),
            "ms_per_frame_ar_attn": round(ar_attn / n, 3),
            "ms_per_frame_ar_net": round(ar_net / n, 3),
            "ms_per_frame_fm_net": round(fm_net / n, 3),
            "ms_per_frame_fm_ode": round(fm_ode / n, 3),
            "ms_per_frame_ar_other": round(ar_other / n, 3),
            "ms_per_frame_vis_flow": round(vis_flow / n, 3),
            "ms_per_frame_face_gen": round(face_gen / n, 3),
            "ms_per_frame_xfer": round(xfer / n, 3),
            "ms_per_frame_render": round(render / n, 3),
        }


def _noop_restore() -> None:
    return None


def wrap_forward(module: Any, profiler: SynthProfiler, name: str) -> Callable[[], None]:
    """Time ``module.forward`` under ``name`` (restored after synth)."""
    if module is None or not profiler.enabled:
        return _noop_restore
    if getattr(module, "_talkshow_profile_wrapped", False):
        return _noop_restore
    orig = module.forward

    def timed(*args: Any, **kwargs: Any) -> Any:
        with profiler.span(name):
            return orig(*args, **kwargs)

    module.forward = timed
    module._talkshow_profile_wrapped = True

    def restore() -> None:
        module.forward = orig
        module._talkshow_profile_wrapped = False

    return restore


def wrap_module_list(modules: Any, profiler: SynthProfiler, name: str) -> Callable[[], None]:
    """Time every submodule ``forward`` under the same span name (sums)."""
    if modules is None or not profiler.enabled:
        return _noop_restore
    restores = [wrap_forward(m, profiler, name) for m in modules]

    def restore() -> None:
        for r in restores:
            r()

    return restore


def wrap_method(obj: Any, method: str, profiler: SynthProfiler, name: str) -> Callable[[], None]:
    if obj is None or not profiler.enabled:
        return _noop_restore
    orig = getattr(obj, method, None)
    if orig is None or getattr(orig, "_talkshow_profile_wrapped", False):
        return _noop_restore

    def timed(*args: Any, **kwargs: Any) -> Any:
        with profiler.span(name):
            return orig(*args, **kwargs)

    timed._talkshow_profile_wrapped = True  # type: ignore[attr-defined]
    setattr(obj, method, timed)

    def restore() -> None:
        setattr(obj, method, orig)

    return restore


def wrap_scheduler_step(scheduler: Any, profiler: SynthProfiler) -> Callable[[], None]:
    """Time ``noise_scheduler.step`` (flow-matching Euler update)."""
    orig = getattr(scheduler, "step", None)
    if not profiler.enabled or orig is None:
        return _noop_restore
    if getattr(scheduler, "_talkshow_profile_wrapped", False):
        return _noop_restore

    def timed_step(*args: Any, **kwargs: Any) -> Any:
        with profiler.span("fm_ode"):
            return orig(*args, **kwargs)

    scheduler.step = timed_step
    scheduler._talkshow_profile_wrapped = True

    def restore() -> None:
        scheduler.step = orig
        scheduler._talkshow_profile_wrapped = False

    return restore


def install_model_hooks(model: Any, scheduler: Any, profiler: SynthProfiler) -> Callable[[], None]:
    """Hook GPT blocks / DiffusionHead / inner wav2vec / scheduler for one clip."""
    restores = [
        wrap_scheduler_step(scheduler, profiler),
        wrap_module_list(getattr(model, "blocks", None), profiler, "ar_attn"),
        wrap_forward(getattr(model, "diffusion_head", None), profiler, "fm_net"),
        wrap_method(model, "get_audio2face_fea", profiler, "ar_a2f_inner"),
        wrap_method(model, "get_audio2face_fea_other", profiler, "ar_a2f_inner"),
    ]

    def restore() -> None:
        for r in reversed(restores):
            r()

    return restore


def log_summary(profile: dict[str, Any]) -> None:
    frames = max(int(profile.get("frames") or 0), 1)
    logger.info(
        "dystream profile clock=%s frames=%d steps=%d window=%d | "
        "audio_load=%dms audio2face=%dms | "
        "ar_fm=%dms (a2f_inner=%dms ar_attn=%dms fm_net=%dms fm_ode=%dms other=%dms) | "
        "render=%dms (vis_flow=%dms face_gen=%dms xfer=%dms)",
        profile.get("clock"),
        frames,
        int(profile.get("steps") or 0),
        int(profile.get("window") or 0),
        int(profile.get("audio_load_ms") or 0),
        int(profile.get("audio2face_ms") or 0),
        int(profile.get("ar_fm_ms") or 0),
        int(profile.get("ar_a2f_inner_ms") or 0),
        int(profile.get("ar_attn_ms") or 0),
        int(profile.get("fm_net_ms") or 0),
        int(profile.get("fm_ode_ms") or 0),
        int(profile.get("ar_other_ms") or 0),
        int(profile.get("render_ms") or 0),
        int(profile.get("vis_flow_ms") or 0),
        int(profile.get("face_gen_ms") or 0),
        int(profile.get("xfer_ms") or 0),
    )
    per = {
        "a2f_inner": float(profile.get("ms_per_frame_ar_a2f_inner") or 0),
        "ar_attn": float(profile.get("ms_per_frame_ar_attn") or 0),
        "fm_net": float(profile.get("ms_per_frame_fm_net") or 0),
        "fm_ode": float(profile.get("ms_per_frame_fm_ode") or 0),
        "ar_other": float(profile.get("ms_per_frame_ar_other") or 0),
        "vis_flow": float(profile.get("ms_per_frame_vis_flow") or 0),
        "face_gen": float(profile.get("ms_per_frame_face_gen") or 0),
        "xfer": float(profile.get("ms_per_frame_xfer") or 0),
    }
    total = sum(per.values()) or 1.0
    parts = " ".join(
        f"{k}={v:.1f}ms/{100.0 * v / total:.0f}%" for k, v in per.items() if v > 0
    )
    logger.info("dystream profile ms/frame (AR+render): %s", parts or "(empty)")
    if float(profile.get("ar_a2f_inner_ms") or 0) > 0:
        logger.info(
            "dystream profile note: a2f_inner is wav2vec inside one_clip "
            "(features already precomputed — candidate to skip)"
        )
