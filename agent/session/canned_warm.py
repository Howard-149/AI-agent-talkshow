"""Render the host's canned lines into the disk cache.

Run on the stack's node once the sidecars are up and before the agent worker
starts (or any time after; the worker reads the cache per line):

    python -m agent.session.canned_warm              # English lines
    python -m agent.session.canned_warm --locales en,zh
    python -m agent.session.canned_warm --force      # re-render everything

Lines already cached under the current voice and portrait are skipped, so a
re-run only renders what is missing. Chinese is warmed for the session welcome
only (it ships pre-translated); other Chinese canned lines still render live.
The worker uses the cache only with TALKSHOW_PRERENDER_CANNED=1.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from dotenv import load_dotenv

from agent.session import canned_cache as cc

logger = logging.getLogger("canned_warm")


async def _render_frames(bridge: object, role: str, step: str, pcm: bytes, sample_rate: int) -> tuple[list[bytes], float]:
    """All RGBA frames of one DyStream render, plus its fps."""
    from agent.adapters.avatar_video import FRAME_EOS, _decode_mp4_frames
    from agent.adapters.dystream_bridge import SyncClipResult, SyncStreamResult

    res = await bridge.bake_sync(role, step, pcm, sample_rate, slot="canned")  # type: ignore[attr-defined]
    if isinstance(res, SyncStreamResult):
        frames: list[bytes] = []
        while True:
            fr = await res.frame_queue.get()
            if fr is FRAME_EOS or fr is None:
                break
            frames.append(fr)
        if res._reader_task is not None:
            await res._reader_task
        if res.error is not None:
            raise res.error
        return frames, res.fps
    if isinstance(res, SyncClipResult):
        w, h = cc.video_size()
        return await asyncio.to_thread(_decode_mp4_frames, res.clip_path, w, h)
    raise RuntimeError(f"avatar render returned nothing for {step}")


async def warm(locales: list[str], *, force: bool = False) -> int:
    from agent.adapters.dystream_bridge import avatar_enabled, init_avatar_bridge
    from agent.adapters.tts_synthesize import synthesize_pcm_for_config
    from agent.config import (
        load_config,
        load_persona_avatar,
        load_persona_name,
        load_persona_tts,
        load_scenario,
    )
    # agent.agents first: session_handoff and agents.host import each other, and the
    # cycle only resolves in this order (as in agent.main).
    import agent.agents  # noqa: F401
    from agent.panel.panel_context import panel_speaker_roles, session_welcome_texts
    from agent.panel.panel_speech import PANEL_HOST_CLOSE
    from avatar.paths import resolve_avatar_asset_path

    config = load_config()
    scenario = load_scenario()
    host = scenario.turn_control.listen_role
    names = [load_persona_name(r) for r in panel_speaker_roles(scenario)]
    welcome = session_welcome_texts(scenario)
    bridge = init_avatar_bridge() if avatar_enabled() else None
    width, height = cc.video_size()

    portrait = None
    if bridge is not None:
        av = load_persona_avatar(host)
        if av.dystream_enabled and av.portrait:
            portrait = resolve_avatar_asset_path(av.portrait)

    jobs: list[tuple[str, str, str]] = []  # (locale, kind, text)
    for loc in locales:
        if loc == "en":
            jobs += [("en", k, t) for k, t in cc.canned_lines(names, close_line=PANEL_HOST_CLOSE, welcome=welcome["en"])]
        elif loc in welcome:
            jobs.append((loc, "welcome", welcome[loc]))

    rendered = skipped = failed = 0
    t_all = time.monotonic()
    for loc, kind, text in jobs:
        tts_cfg = load_persona_tts(host, config, locale=loc)
        key = cc.clip_key(
            role=host,
            locale=loc,
            text=text,
            voice=cc.voice_identity(tts_cfg, role=host, locale=loc),
            portrait=portrait,
            width=width,
            height=height,
        )
        if not force and cc.load(key) is not None:
            skipped += 1
            continue
        t0 = time.monotonic()
        try:
            pcm, sample_rate, _ = await synthesize_pcm_for_config(
                tts_cfg, text, emotion=None, locale=loc, role=host, step=f"canned:{kind}"
            )
            if not pcm:
                raise RuntimeError("TTS returned no audio")
            tts_s = time.monotonic() - t0
            frames: list[bytes] = []
            fps = 25.0
            if portrait is not None:
                frames, fps = await _render_frames(bridge, host, f"canned:{kind}", pcm, sample_rate)
            jpegs = await asyncio.to_thread(cc.encode_frames, frames, width, height) if frames else []
            clip = cc.CannedClip(
                key=key,
                role=host,
                locale=loc,
                text=text,
                pcm=pcm,
                sample_rate=sample_rate,
                frames_jpeg=tuple(jpegs),
                fps=fps,
                width=width if frames else 0,
                height=height if frames else 0,
            )
            cc.store(clip, extra={"kind": kind, "tts_s": round(tts_s, 3), "render_s": round(time.monotonic() - t0, 3)})
            rendered += 1
            logger.info("cached %s/%s %.1fs audio, %d frames, %.1fs: %r", loc, kind, clip.duration_sec, len(jpegs), time.monotonic() - t0, text)
        except Exception:
            failed += 1
            logger.exception("could not cache %s/%s: %r", loc, kind, text)

    logger.info(
        "canned cache %s: %d rendered, %d already cached, %d failed in %.0fs (avatar %s)",
        cc.cache_dir(),
        rendered,
        skipped,
        failed,
        time.monotonic() - t_all,
        "on" if portrait is not None else "off",
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locales", default="en", help="comma-separated, e.g. en,zh")
    ap.add_argument("--force", action="store_true", help="re-render lines that are already cached")
    args = ap.parse_args(argv)
    locales = [x.strip() for x in args.locales.split(",") if x.strip()]
    return asyncio.run(warm(locales, force=args.force))


if __name__ == "__main__":
    raise SystemExit(main())
