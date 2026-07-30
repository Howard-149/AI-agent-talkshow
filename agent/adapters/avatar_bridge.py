from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import load_persona_avatar
from agent.adapters.avatar_video import FRAME_EOS
from avatar.dystream_job import (
    run_dystream_bake,
    run_dystream_synthesize_stream,
    wait_for_dystream_sidecar_ready,
)
from avatar.paths import (
    avatar_assets_dir,
    avatar_clip_public_url,
    avatar_clips_dir,
    ensure_avatar_dirs,
    resolve_avatar_asset_path,
    speech_clip_path,
    speech_wav_path,
)
from avatar.pcm_utils import pcm16_to_wav, resample_pcm16_mono

logger = logging.getLogger(__name__)

DYSTREAM_AUDIO_RATE = 16000


def avatar_enabled() -> bool:
    raw = os.environ.get("TALKSHOW_AVATAR_ENABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def avatar_synth_mode() -> str:
    """stream = online preroll (default); mp4 = legacy bake."""
    raw = os.environ.get("TALKSHOW_AVATAR_SYNTH", "stream").strip().lower()
    if raw in ("mp4", "bake", "legacy"):
        return "mp4"
    if raw in ("frames", "pack", "a_lite", "alite"):
        logger.warning(
            "TALKSHOW_AVATAR_SYNTH=%s is removed; using stream",
            raw,
        )
    return "stream"


def avatar_preroll_frames() -> int:
    """Minimum preroll floor (dynamic target is at least this)."""
    return max(1, int(os.environ.get("TALKSHOW_AVATAR_PREROLL_FRAMES", "8")))


def avatar_preroll_probe_frames() -> int:
    """Frames used to measure gen_fps (need ≥2 for inter-frame rate)."""
    return max(2, int(os.environ.get("TALKSHOW_AVATAR_PREROLL_PROBE_FRAMES", "2")))


def avatar_preroll_margin_frames() -> int:
    """Extra frames beyond the deficit estimate."""
    return max(0, int(os.environ.get("TALKSHOW_AVATAR_PREROLL_MARGIN_FRAMES", "4")))


def avatar_preroll_margin_factor() -> float:
    """Scale deficit estimate (>1 = more buffer)."""
    return max(1.0, float(os.environ.get("TALKSHOW_AVATAR_PREROLL_MARGIN_FACTOR", "1.0")))


def compute_dynamic_preroll(
    *,
    expected_frames: int,
    gen_fps: float,
    play_fps: float,
    margin_factor: float,
    margin_frames: int,
    min_frames: int,
) -> int:
    """Preroll so buffer lasts the whole utterance at constant gen/play rates.

    preroll >= N * (1 - gen/play) * margin_factor + margin_frames, clamped to [min, N].
    """
    if expected_frames <= 0:
        return max(1, min_frames)
    play_fps = max(play_fps, 1.0)
    if gen_fps <= 0:
        return expected_frames
    if gen_fps >= play_fps:
        return min(expected_frames, max(min_frames, margin_frames))
    deficit_ratio = 1.0 - (gen_fps / play_fps)
    need = int(math.ceil(expected_frames * deficit_ratio * margin_factor)) + margin_frames
    return max(min_frames, min(expected_frames, need))


@dataclass(frozen=True)
class SyncClipResult:
    """Legacy MP4 bake result (HTTP clip URL path)."""

    url: str
    bake_ms: int
    clip_path: Path


@dataclass
class SyncStreamResult:
    """Online stream: frames arrive on queue after preroll; None/FRAME_EOS ends."""

    frame_queue: asyncio.Queue
    fps: float
    width: int
    height: int
    preroll_frames: int
    ttff_ms: int
    preroll_ms: int
    bake_ms: int  # time until preroll ready (play can start)
    audio_feat_ms: int = 0
    expected_frames: int = 0
    gen_fps: float = 0.0
    cuda_mb_peak: float | None = None
    error: BaseException | None = None
    _reader_task: asyncio.Task | None = field(default=None, repr=False)


def _resize_rgba_frame(
    fr: bytes,
    *,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> bytes:
    if src_w == dst_w and src_h == dst_h:
        return fr
    from PIL import Image

    img = Image.frombytes("RGBA", (src_w, src_h), fr)
    img = img.resize((dst_w, dst_h), Image.Resampling.BILINEAR)
    return img.tobytes()


class AvatarBridge:
    """DyStream before/during audio playout; stream (default) or legacy MP4."""

    def __init__(self, *, turn_log: object | None = None) -> None:
        self._turn_log = turn_log
        ensure_avatar_dirs()
        self._out_dir = avatar_clips_dir()
        self.synth_mode = avatar_synth_mode()

    async def bake_sync(
        self,
        role: str,
        step: str,
        pcm: bytes,
        sample_rate: int,
        *,
        slot: str = "current",
    ) -> SyncClipResult | SyncStreamResult | None:
        avatar_cfg = load_persona_avatar(role)
        if not avatar_cfg.dystream_enabled or not avatar_cfg.portrait:
            return None

        portrait_path = resolve_avatar_asset_path(avatar_cfg.portrait)
        if not portrait_path.is_file():
            logger.warning("avatar portrait missing role=%s path=%s", role, portrait_path)
            return None

        wav_path = speech_wav_path(slot)
        wav_path.unlink(missing_ok=True)

        t0 = time.monotonic()
        try:
            pcm16 = resample_pcm16_mono(pcm, sample_rate, DYSTREAM_AUDIO_RATE)
            pcm16_to_wav(pcm16, wav_path, sample_rate=DYSTREAM_AUDIO_RATE)

            if self.synth_mode == "stream":
                pcm_dur = len(pcm16) / float(DYSTREAM_AUDIO_RATE)
                return await self._synthesize_stream(
                    role=role,
                    step=step,
                    slot=slot,
                    portrait_path=portrait_path,
                    wav_path=wav_path,
                    t0=t0,
                    pcm_duration_sec=pcm_dur,
                )
            return await self._bake_mp4(
                role=role,
                step=step,
                slot=slot,
                portrait_path=portrait_path,
                wav_path=wav_path,
                t0=t0,
            )
        except Exception as exc:
            logger.warning("avatar bake failed role=%s slot=%s: %s", role, slot, exc)
            if self._turn_log is not None:
                self._turn_log.log(
                    "avatar_bake_failed",
                    role=role,
                    step=step,
                    slot=slot,
                    avatar_bake_ms=round((time.monotonic() - t0) * 1000),
                    error=str(exc)[:500],
                )
            wav_path.unlink(missing_ok=True)
            return None
        finally:
            # Stream mode keeps wav until reader finishes (deleted in reader).
            if self.synth_mode != "stream":
                wav_path.unlink(missing_ok=True)

    async def _synthesize_stream(
        self,
        *,
        role: str,
        step: str,
        slot: str,
        portrait_path: Path,
        wav_path: Path,
        t0: float,
        pcm_duration_sec: float,
    ) -> SyncStreamResult:
        dst_w = int(os.environ.get("TALKSHOW_AVATAR_VIDEO_WIDTH", "512"))
        dst_h = int(os.environ.get("TALKSHOW_AVATAR_VIDEO_HEIGHT", "512"))
        min_preroll = avatar_preroll_frames()
        probe_n = avatar_preroll_probe_frames()
        margin_frames = avatar_preroll_margin_frames()
        margin_factor = avatar_preroll_margin_factor()
        # TTS length → provisional N @ 25fps until sidecar meta arrives.
        expected_from_tts = max(1, int(math.ceil(pcm_duration_sec * 25.0)))
        loop = asyncio.get_running_loop()
        # Room for a near-full buffer if gen << play.
        frame_queue: asyncio.Queue = asyncio.Queue(
            maxsize=max(expected_from_tts + 64, 256)
        )
        preroll_ready = asyncio.Event()
        meta_box: dict = {}
        err_box: list[BaseException] = []

        def _reader() -> None:
            buffered = 0
            src_w = dst_w
            src_h = dst_h
            play_fps = 25.0
            expected = expected_from_tts
            target_preroll = max(min_preroll, probe_n)
            t_frame1: float | None = None
            gen_fps = 0.0
            try:
                for kind, payload in run_dystream_synthesize_stream(
                    portrait=portrait_path,
                    audio_wav=wav_path,
                    cache_key=role,
                ):
                    if kind == "meta":
                        meta_box.update(payload)
                        src_w = int(payload.get("width") or dst_w)
                        src_h = int(payload.get("height") or dst_h)
                        play_fps = float(payload.get("fps") or 25.0)
                        meta_expected = int(payload.get("expected_frames") or 0)
                        if meta_expected > 0:
                            expected = meta_expected
                        elif pcm_duration_sec > 0 and play_fps > 0:
                            expected = max(
                                1, int(math.ceil(pcm_duration_sec * play_fps))
                            )
                        meta_box["expected_frames"] = expected
                        meta_box["ttff_ms"] = round((time.monotonic() - t0) * 1000)
                        continue

                    fr = _resize_rgba_frame(
                        payload,
                        src_w=src_w,
                        src_h=src_h,
                        dst_w=dst_w,
                        dst_h=dst_h,
                    )
                    fut = asyncio.run_coroutine_threadsafe(
                        frame_queue.put(fr), loop
                    )
                    fut.result(timeout=600)
                    buffered += 1
                    now = time.monotonic()

                    if buffered == 1:
                        t_frame1 = now
                        if "ttff_ms" not in meta_box:
                            meta_box["ttff_ms"] = round((now - t0) * 1000)

                    # After probe frames: measure steady gen rate (frame1 → frame probe).
                    if (
                        buffered == probe_n
                        and t_frame1 is not None
                        and not preroll_ready.is_set()
                    ):
                        dt = now - t_frame1
                        if dt > 1e-6 and probe_n > 1:
                            gen_fps = (probe_n - 1) / dt
                        elif meta_box.get("ttff_ms"):
                            # Fallback: first-frame latency as ms/frame (conservative).
                            gen_fps = 1000.0 / max(float(meta_box["ttff_ms"]), 1.0)
                        target_preroll = compute_dynamic_preroll(
                            expected_frames=expected,
                            gen_fps=gen_fps,
                            play_fps=play_fps,
                            margin_factor=margin_factor,
                            margin_frames=margin_frames,
                            min_frames=max(min_preroll, probe_n),
                        )
                        meta_box["gen_fps"] = round(gen_fps, 3)
                        meta_box["preroll_frames_target"] = target_preroll
                        logger.info(
                            "avatar dynamic preroll role=%s expected=%d gen_fps=%.2f "
                            "play_fps=%.1f target=%d (min=%d margin_f=%.2f margin_n=%d)",
                            role,
                            expected,
                            gen_fps,
                            play_fps,
                            target_preroll,
                            min_preroll,
                            margin_factor,
                            margin_frames,
                        )

                    if buffered >= target_preroll and not preroll_ready.is_set():
                        # If probe not reached yet, keep waiting (target still min).
                        if buffered < probe_n:
                            continue
                        meta_box["preroll_ms"] = round((now - t0) * 1000)
                        meta_box["preroll_frames"] = buffered
                        if "gen_fps" not in meta_box and t_frame1 is not None:
                            dt = now - t_frame1
                            if dt > 1e-6 and buffered > 1:
                                meta_box["gen_fps"] = round((buffered - 1) / dt, 3)
                        loop.call_soon_threadsafe(preroll_ready.set)
            except BaseException as exc:
                err_box.append(exc)
                logger.exception("avatar stream reader failed")
            finally:
                wav_path.unlink(missing_ok=True)
                if not preroll_ready.is_set():
                    meta_box.setdefault("preroll_frames", buffered)
                    meta_box.setdefault(
                        "preroll_ms", round((time.monotonic() - t0) * 1000)
                    )
                    loop.call_soon_threadsafe(preroll_ready.set)
                fut = asyncio.run_coroutine_threadsafe(
                    frame_queue.put(FRAME_EOS), loop
                )
                try:
                    fut.result(timeout=30)
                except Exception:
                    pass

        reader_task = asyncio.create_task(asyncio.to_thread(_reader))

        await preroll_ready.wait()
        if err_box and frame_queue.empty():
            await reader_task
            raise err_box[0]

        ttff_ms = int(meta_box.get("ttff_ms") or round((time.monotonic() - t0) * 1000))
        preroll_ms = int(
            meta_box.get("preroll_ms") or round((time.monotonic() - t0) * 1000)
        )
        preroll_n = int(meta_box.get("preroll_frames") or min_preroll)
        fps = float(meta_box.get("fps") or 25.0)
        expected = int(meta_box.get("expected_frames") or expected_from_tts)
        audio_feat_ms = int(meta_box.get("audio_feat_ms") or 0)
        gen_fps = float(meta_box.get("gen_fps") or 0.0)
        target = int(meta_box.get("preroll_frames_target") or preroll_n)

        if self._turn_log is not None:
            self._turn_log.log(
                "avatar_bake",
                role=role,
                step=step,
                slot=slot,
                avatar_bake_ms=preroll_ms,
                avatar_video_compose_ms=preroll_ms,
                audio_feat_ms=audio_feat_ms,
                ttff_ms=ttff_ms,
                preroll_ms=preroll_ms,
                preroll_frames=preroll_n,
                preroll_frames_target=target,
                expected_frames=expected,
                gen_fps=gen_fps,
                pcm_duration_sec=round(pcm_duration_sec, 3),
                synth="stream",
            )
        logger.info(
            "avatar stream preroll ready role=%s slot=%s ttff_ms=%d preroll_ms=%d "
            "preroll_frames=%d target=%d expected=%d gen_fps=%.2f",
            role,
            slot,
            ttff_ms,
            preroll_ms,
            preroll_n,
            target,
            expected,
            gen_fps,
        )
        return SyncStreamResult(
            frame_queue=frame_queue,
            fps=fps,
            width=dst_w,
            height=dst_h,
            preroll_frames=preroll_n,
            ttff_ms=ttff_ms,
            preroll_ms=preroll_ms,
            bake_ms=preroll_ms,
            audio_feat_ms=audio_feat_ms,
            expected_frames=expected,
            gen_fps=gen_fps,
            error=err_box[0] if err_box else None,
            _reader_task=reader_task,
        )

    async def _bake_mp4(
        self,
        *,
        role: str,
        step: str,
        slot: str,
        portrait_path: Path,
        wav_path: Path,
        t0: float,
    ) -> SyncClipResult:
        mp4_path = speech_clip_path(slot)
        mp4_path.unlink(missing_ok=True)
        _, bake_sec = await asyncio.to_thread(
            run_dystream_bake,
            portrait=portrait_path,
            audio_wav=wav_path,
            output_mp4=mp4_path,
        )
        bake_ms = round(bake_sec * 1000)
        url = avatar_clip_public_url(mp4_path.name)
        if self._turn_log is not None:
            self._turn_log.log(
                "avatar_bake",
                role=role,
                step=step,
                slot=slot,
                avatar_bake_ms=bake_ms,
                avatar_video_compose_ms=bake_ms,
                url=url,
                synth="mp4",
            )
        logger.info(
            "avatar bake done role=%s slot=%s bake_ms=%d url=%s",
            role,
            slot,
            bake_ms,
            url,
        )
        return SyncClipResult(url=url, bake_ms=bake_ms, clip_path=mp4_path)


_bridge: AvatarBridge | None = None


def get_avatar_bridge() -> AvatarBridge | None:
    return _bridge


def init_avatar_bridge(*, turn_log: object | None = None) -> AvatarBridge | None:
    global _bridge
    if not avatar_enabled():
        _bridge = None
        return None
    wait_for_dystream_sidecar_ready()
    _bridge = AvatarBridge(turn_log=turn_log)
    sidecar = os.environ.get("DYSTREAM_SIDECAR_URL", "").strip() or "(subprocess)"
    logger.info(
        "AvatarBridge enabled synth=%s sidecar=%s assets_dir=%s clips_dir=%s dystream=%s",
        avatar_synth_mode(),
        sidecar,
        avatar_assets_dir(),
        avatar_clips_dir(),
        os.environ.get("DYSTREAM_ROOT", "(unset)"),
    )
    return _bridge
