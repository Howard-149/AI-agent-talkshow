"""DyStream synthesize: motion → RGBA frames (no MP4).

Online path: sliding-window AR + single-frame render (O(window) VRAM).
"""
from __future__ import annotations

import gc
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineMeta:
    """Sent before the first RGBA frame on the online stream path."""

    width: int
    height: int
    fps: float
    expected_frames: int
    audio_feat_ms: int


@dataclass
class OnlineStreamStats:
    frame_count: int
    motion_ms: int
    render_ms: int
    total_ms: int
    ttff_ms: int
    cuda_mb_peak: float | None


_role_cache: dict[str, dict] = {}


def _device() -> torch.device:
    from app import DEVICE  # noqa: WPS433

    return DEVICE


def cuda_mem_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
    except Exception:
        return None


def release_cuda() -> None:
    """Free fragmented CUDA cache after a bake/synthesize (sidecar stays warm)."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def list_persona_portraits() -> list[tuple[str, Path]]:
    """(role_id, portrait_path) from config/personas/*.yaml ui.avatar.portrait."""
    import yaml

    from avatar.paths import REPO_ROOT, resolve_avatar_asset_path

    personas_dir = REPO_ROOT / "config" / "personas"
    out: list[tuple[str, Path]] = []
    if not personas_dir.is_dir():
        return out
    for path in sorted(personas_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("skip persona yaml %s: %s", path.name, exc)
            continue
        role = str(data.get("id") or path.stem).strip()
        ui = data.get("ui") or {}
        avatar = ui.get("avatar") if isinstance(ui, dict) else None
        if not isinstance(avatar, dict):
            continue
        dystream = avatar.get("dystream")
        if isinstance(dystream, dict) and dystream.get("enabled") is False:
            continue
        portrait = avatar.get("portrait")
        if not portrait:
            continue
        try:
            p = resolve_avatar_asset_path(str(portrait).strip())
        except Exception:
            continue
        if p.is_file():
            out.append((role, p))
    return out


def warm_role_caches(*, probe_render: bool = True) -> None:
    """Pre-build per-role portrait latents (+ optional 1-frame vis) at sidecar start.

    Avoids first-turn spike from ``process_image`` / face_encoder on Amy/Ryan/etc.
    """
    from app import load_visualization_model  # noqa: WPS433

    load_visualization_model()
    roles = list_persona_portraits()
    if not roles:
        logger.warning("warm_role_caches: no persona portraits found")
        return
    t0 = time.monotonic()
    for role, portrait in roles:
        cache = _get_or_build_role_cache(cache_key=role, portrait=portrait, npz=None)
        if not probe_render:
            continue
        # Warm FaceRenderer path for this portrait (face_encoder + one generator step).
        renderer = _FrameRenderer(cache["resized_pil"])
        try:
            lat = cache["motion_latent_cpu"]
            if isinstance(lat, torch.Tensor):
                if lat.dim() == 1:
                    lat = lat.unsqueeze(0)
                renderer.render(lat[0] if lat.dim() >= 2 else lat)
        finally:
            renderer.close()
        release_cuda()
        logger.info("warmed role cache+render key=%s portrait=%s", role, portrait.name)
    logger.info(
        "warm_role_caches done roles=%d elapsed=%.1fs",
        len(roles),
        time.monotonic() - t0,
    )


def warm_ar_probe(*, duration_sec: float = 0.4) -> None:
    """One short silence stream on the first persona — warms audio encoder + AR kernels."""
    import tempfile
    import wave

    roles = list_persona_portraits()
    if not roles:
        return
    role, portrait = roles[0]
    sr = 16000
    n = max(int(sr * duration_sec), sr // 5)
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dystream-warm-") as tmp:
        wav_path = Path(tmp) / "silence.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * n)
        frames = 0
        for item in iter_synthesize_online(
            portrait=portrait,
            audio_wav=wav_path,
            cache_key=role,
            denoising_steps=5,
        ):
            if isinstance(item, OnlineMeta):
                continue
            frames += 1
            if frames >= 3:
                break
    release_cuda()
    logger.info(
        "warm_ar_probe done role=%s frames=%d elapsed=%.1fs",
        role,
        frames,
        time.monotonic() - t0,
    )


def _get_or_build_role_cache(
    *,
    cache_key: str,
    portrait: Path,
    npz: Path | None,
) -> dict:
    """Cache portrait motion latent + resized PIL per role (CPU-side latent)."""
    from app import load_visualization_model, process_image  # noqa: WPS433

    mtime = portrait.stat().st_mtime if portrait.is_file() else 0.0
    npz_key = str(npz.resolve()) if npz and npz.is_file() else ""
    hit = _role_cache.get(cache_key)
    if (
        hit is not None
        and hit.get("portrait") == str(portrait.resolve())
        and hit.get("mtime") == mtime
        and hit.get("npz") == npz_key
    ):
        return hit

    load_visualization_model()

    if npz is not None and npz.is_file():
        data = np.load(npz, allow_pickle=True)
        try:
            motion_latent_np = data["motion_latent"]
        except KeyError:
            motion_latent_np = data["random_data"]
        motion_latent_cpu = torch.from_numpy(motion_latent_np)
        ref_img_path = str(data.get("ref_img_path", ""))
        if ref_img_path and Path(ref_img_path).is_file():
            resized_pil = Image.open(ref_img_path).convert("RGB")
        else:
            resized_pil = Image.open(portrait).convert("RGB")
            resized_pil, _, motion_latent_cpu = process_image(resized_pil)
    else:
        resized_pil, _, motion_latent_cpu = process_image(
            Image.open(portrait).convert("RGB")
        )

    # Keep latent on CPU in cache — move to GPU only for each request.
    if isinstance(motion_latent_cpu, torch.Tensor):
        motion_latent_cpu = motion_latent_cpu.detach().cpu()

    entry = {
        "portrait": str(portrait.resolve()),
        "mtime": mtime,
        "npz": npz_key,
        "resized_pil": resized_pil,
        "motion_latent_cpu": motion_latent_cpu,
    }
    _role_cache[cache_key] = entry
    logger.info("synth role cache ready key=%s portrait=%s", cache_key, portrait.name)
    return entry


def _rgb_to_rgba_bytes(rgb: np.ndarray) -> bytes:
    """rgb: [H,W,3] uint8 → RGBA bytes."""
    h, w, _ = rgb.shape
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1).tobytes()


def _rgb_batch_to_rgba_list(video_np: np.ndarray) -> list[bytes]:
    """video_np: [T,H,W,3] uint8 → list of RGBA bytes."""
    if video_np.ndim != 4 or video_np.shape[-1] != 3:
        raise ValueError(f"unexpected video shape {video_np.shape}")
    return [_rgb_to_rgba_bytes(video_np[i]) for i in range(video_np.shape[0])]


class _FrameRenderer:
    """Single-frame vis render — never accumulates recon tensors on GPU."""

    def __init__(self, ref_image_pil: Image.Image) -> None:
        from app import DEVICE, load_visualization_model  # noqa: WPS433

        load_visualization_model()
        from app import _vis_ctx  # noqa: WPS433

        self._device = DEVICE
        self._transform = _vis_ctx["transform"]
        self._face_encoder = _vis_ctx["face_encoder"]
        self._flow_estimator = _vis_ctx["flow_estimator"]
        self._face_generator = _vis_ctx["face_generator"]
        ref = self._transform(ref_image_pil.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            self._face_feat = self._face_encoder(ref)
        del ref
        self._ref_latent: torch.Tensor | None = None  # [1, 512] on device
        self.width: int | None = None
        self.height: int | None = None

    def render(self, latent_1x512: torch.Tensor) -> bytes:
        """latent: [1, 512] or [512] → RGBA bytes; frees step GPU tensors."""
        lat = latent_1x512.detach()
        if lat.dim() == 1:
            lat = lat.unsqueeze(0)
        if lat.dim() == 3:
            lat = lat.squeeze(0)
        if lat.shape[0] != 1:
            lat = lat[:1]
        lat = lat.to(self._device).float()

        if self._ref_latent is None:
            self._ref_latent = lat.detach().clone()

        with torch.inference_mode():
            tgt = self._flow_estimator(self._ref_latent, lat)
            recon = self._face_generator(tgt, self._face_feat)
            rgb = recon.permute(0, 2, 3, 1).detach().float().cpu().numpy()[0]
        del tgt, recon, lat
        rgb_u8 = np.clip((rgb + 1) / 2 * 255, 0, 255).astype("uint8")
        del rgb
        if self.height is None:
            self.height, self.width = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
        return _rgb_to_rgba_bytes(rgb_u8)

    def close(self) -> None:
        self._ref_latent = None
        self._face_feat = None


def _precompute_audio2face(
    model,
    audio: torch.Tensor,
    pose_fps: int,
    *,
    other: bool = False,
) -> torch.Tensor:
    """Match upstream inference audio2face feature path (self or other encoder)."""
    audio_list = [i.cpu().numpy() for i in audio]
    inputs = model.audio_processor(
        audio_list, sampling_rate=16000, return_tensors="pt", padding=True
    ).to(audio.device)
    encoder = model.audio_encoder_face_other if other else model.audio_encoder_face
    fea = encoder(
        torch.concat(
            [inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)],
            dim=-1,
        )
    )["high_level"]
    fea = F.interpolate(
        fea.transpose(1, 2),
        scale_factor=(pose_fps / 50),
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    return fea


def iter_synthesize_online(
    *,
    portrait: Path,
    audio_wav: Path,
    denoising_steps: int = 5,
    npz: Path | None = None,
    cache_key: str = "default",
    cfg_audio: float = 0.5,
    cfg_audio_other: float = 0.5,
    cfg_anchor: float = 0.0,
    cfg_all: float = 1.0,
    log_every: int = 25,
) -> Iterator[OnlineMeta | bytes]:
    """Yield OnlineMeta once, then RGBA frames as each AR stride completes.

    GPU working set stays O(window): no full-T motion tensor, no recon_list.
    """
    from app import (  # noqa: WPS433
        _dystream_cfg,
        _dystream_ema,
        _dystream_model,
        _noise_scheduler,
        load_dystream_model,
        load_visualization_model,
    )

    t_all = time.monotonic()
    mem0 = cuda_mem_mb()
    peak = mem0
    load_dystream_model()
    load_visualization_model()

    cache = _get_or_build_role_cache(
        cache_key=cache_key, portrait=portrait, npz=npz
    )
    motion_latent_cpu = cache["motion_latent_cpu"]
    resized_pil = cache["resized_pil"]

    audio_sr = int(OmegaConf.select(_dystream_cfg.config, "model.audio_sr", default=16000))
    pose_fps = int(OmegaConf.select(_dystream_cfg.config, "model.pose_fps", default=25))
    samples_per_pose = int(audio_sr / pose_fps)

    t_feat = time.monotonic()
    audio_self, _ = librosa.load(str(audio_wav), sr=audio_sr)
    additional_motion_seq = int(_dystream_model.inpainting_length)
    audio_self = np.concatenate(
        [
            np.zeros(additional_motion_seq * samples_per_pose, dtype=np.float32),
            audio_self.astype(np.float32),
        ],
        axis=0,
    )
    device = _device()
    # Keep full waveform on CPU; move window slices to GPU per step.
    audio_cpu = torch.from_numpy(audio_self).float().unsqueeze(0)
    audio_other_cpu = torch.zeros_like(audio_cpu)
    audio_feat_ms = round((time.monotonic() - t_feat) * 1000)

    motion_latent = motion_latent_cpu
    if motion_latent.dim() == 1:
        motion_latent = motion_latent.unsqueeze(0)
    if motion_latent.dim() == 2:
        motion_latent = motion_latent.unsqueeze(0)
    # [1, 1, C] on GPU — not full T.
    anchor_motion = motion_latent[:, 0:1, :].to(device)
    pre_frames = additional_motion_seq
    past_motion = anchor_motion.repeat(1, pre_frames, 1)

    total_len = int(audio_cpu.shape[1] // samples_per_pose)
    window = int(_dystream_model.cfg.cbh_window_length)
    stride = 1
    expected_frames = max(0, total_len - window + 1)

    _dystream_model.cfg_audio = cfg_audio
    _dystream_model.cfg_audio_other = cfg_audio_other
    _dystream_model.cfg_anchor = cfg_anchor
    _dystream_model.cfg_all = cfg_all
    denoising_steps = int(denoising_steps)

    if _dystream_ema is not None:
        _dystream_ema.to(device)
        ctx = _dystream_ema.average_parameters(_dystream_model.parameters())
    else:
        ctx = nullcontext()

    renderer = _FrameRenderer(resized_pil)
    motion_ms = 0
    render_ms = 0
    frame_count = 0
    ttff_ms = 0
    meta_sent = False
    audio_ratio = int(_dystream_model.cfg.audio_fps // _dystream_model.cfg.pose_fps)
    w0, h0 = resized_pil.size

    try:
        audio_gpu = audio_cpu.to(device)
        audio_other_gpu = audio_other_cpu.to(device)
        # Critical: whole-clip Gradio path uses @torch.no_grad on run_inference.
        # Without this, past_motion=concat(past, out) retains the full AR graph and
        # VRAM grows ~linearly with frame index (streaming OOM; offline does not).
        with ctx, torch.inference_mode():
            t_af = time.monotonic()
            audio2face_fea = _precompute_audio2face(
                _dystream_model, audio_gpu, pose_fps, other=False
            ).detach()
            audio_other2face_fea = _precompute_audio2face(
                _dystream_model, audio_other_gpu, pose_fps, other=True
            ).detach()
            motion_ms += round((time.monotonic() - t_af) * 1000)

            # Keep waveform for one_clip_only_inference signature (feats are precomputed).
            past_audio = None
            past_audio_other = None
            past_motion = past_motion.detach()
            anchor_motion = anchor_motion.detach()

            for i in range(0, total_len, stride):
                start_idx = i
                end_idx = min(start_idx + window, total_len)
                window_size = end_idx - start_idx
                if window_size < window:
                    break

                audio_slice_len = window_size * audio_ratio
                audio_slice_start = start_idx * audio_ratio
                audio_slice = audio_gpu[
                    :, audio_slice_start : audio_slice_start + audio_slice_len
                ]
                audio_slice_other = audio_other_gpu[
                    :, audio_slice_start : audio_slice_start + audio_slice_len
                ]

                t_m = time.monotonic()
                out = _dystream_model.one_clip_only_inference(
                    per_compute_audio_feature=audio2face_fea[:, start_idx:end_idx],
                    per_compute_audio_other_feature=audio_other2face_fea[
                        :, start_idx:end_idx
                    ],
                    past_audio_self=past_audio,
                    audio_self=audio_slice,
                    past_audio_other=past_audio_other,
                    audio_other=audio_slice_other,
                    past_motion=past_motion,
                    gen_frames=stride,
                    anchor_latent=anchor_motion,
                    noise_scheduler=_noise_scheduler,
                    num_inference_steps=denoising_steps,
                )
                out = out.detach()
                motion_ms += round((time.monotonic() - t_m) * 1000)

                # Detach so the AR chain cannot retain previous steps.
                past_motion = torch.concat([past_motion, out], dim=1)[
                    :, -additional_motion_seq:
                ].detach()

                for fi in range(out.shape[1]):
                    lat = out[:, fi, :].detach()
                    t_r = time.monotonic()
                    rgba = renderer.render(lat)
                    render_ms += round((time.monotonic() - t_r) * 1000)
                    del lat
                    frame_count += 1
                    if not meta_sent:
                        ttff_ms = round((time.monotonic() - t_all) * 1000)
                        width = int(renderer.width or w0)
                        height = int(renderer.height or h0)
                        yield OnlineMeta(
                            width=width,
                            height=height,
                            fps=float(pose_fps),
                            expected_frames=expected_frames,
                            audio_feat_ms=audio_feat_ms,
                        )
                        meta_sent = True
                    # Frame bytes are on CPU; drop GPU temps for this step.
                    yield rgba

                del out, audio_slice, audio_slice_other
                mem = cuda_mem_mb()
                if mem is not None and (peak is None or mem > peak):
                    peak = mem
                if frame_count > 0 and frame_count % log_every == 0:
                    logger.info(
                        "online synth frames=%d cuda_mb=%s peak=%s",
                        frame_count,
                        mem,
                        peak,
                    )
                    # Occasional cache trim — cheap insurance vs fragmentation.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        del audio_gpu, audio_other_gpu, audio2face_fea, audio_other2face_fea
        del past_motion, anchor_motion
    finally:
        renderer.close()
        release_cuda()

    if not meta_sent:
        yield OnlineMeta(
            width=int(w0),
            height=int(h0),
            fps=float(pose_fps),
            expected_frames=0,
            audio_feat_ms=audio_feat_ms,
        )

    total_ms = round((time.monotonic() - t_all) * 1000)
    logger.info(
        "iter_synthesize_online frames=%d expected=%d ttff_ms=%d motion_ms=%d "
        "render_ms=%d total_ms=%d cuda_mb_before=%s peak=%s",
        frame_count,
        expected_frames,
        ttff_ms,
        motion_ms,
        render_ms,
        total_ms,
        mem0,
        peak,
    )
    iter_synthesize_online.last_stats = OnlineStreamStats(  # type: ignore[attr-defined]
        frame_count=frame_count,
        motion_ms=motion_ms,
        render_ms=render_ms,
        total_ms=total_ms,
        ttff_ms=ttff_ms,
        cuda_mb_peak=peak,
    )

