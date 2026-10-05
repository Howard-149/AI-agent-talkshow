"""Pre-rendered canned host lines: key, disk round trip, frame codec, line list."""

from __future__ import annotations

import pytest

from agent.session import canned_cache as cc

VOICE = {"engine": "cosyvoice", "model": "prompts/en_host.wav:100:1", "emotion": "neutral"}


def _key(**over):
    kw = dict(role="host", locale="en", text="Floor's open.", voice=VOICE, portrait=None, width=512, height=512)
    kw.update(over)
    return cc.clip_key(**kw)


def test_key_is_stable_and_tracks_what_changes_the_clip(tmp_path):
    portrait = tmp_path / "lessac.png"
    portrait.write_bytes(b"png")
    base = _key(portrait=portrait)
    assert base == _key(portrait=portrait)
    assert base != _key(portrait=portrait, text="Who wants to take this next?")
    assert base != _key(portrait=portrait, locale="zh")
    assert base != _key(portrait=portrait, voice={**VOICE, "speed": "1.1"})
    assert base != _key(portrait=portrait, width=256)
    assert base != _key()  # avatar off is a different clip
    # Surrounding whitespace in the text does not change the key.
    assert base == _key(portrait=portrait, text="  Floor's open. ")


def test_key_misses_after_the_portrait_file_changes(tmp_path):
    portrait = tmp_path / "lessac.png"
    portrait.write_bytes(b"png")
    before = _key(portrait=portrait)
    portrait.write_bytes(b"a different portrait")
    assert _key(portrait=portrait) != before


def test_frames_survive_the_jpeg_round_trip():
    w = h = 16
    frame = bytes([200, 40, 40, 255]) * (w * h)
    jpegs = cc.encode_frames([frame, frame], w, h)
    assert len(jpegs) == 2 and all(j[:2] == b"\xff\xd8" for j in jpegs)
    back = cc.decode_frames(jpegs, w, h)
    assert len(back) == 2 and len(back[0]) == w * h * 4
    r, g, b, a = back[0][:4]
    assert abs(r - 200) < 8 and abs(g - 40) < 8 and abs(b - 40) < 8 and a == 255


@pytest.mark.parametrize("with_frames", [True, False])
def test_store_then_load(tmp_path, monkeypatch, with_frames):
    monkeypatch.setenv("TALKSHOW_CANNED_CACHE_DIR", str(tmp_path / "canned"))
    frames = (b"\xff\xd8one", b"\xff\xd8two-longer") if with_frames else ()
    clip = cc.CannedClip(
        key=_key(), role="host", locale="en", text="Floor's open.",
        pcm=b"\x01\x00" * 22050, sample_rate=22050, frames_jpeg=frames,
        fps=25.0, width=512 if with_frames else 0, height=512 if with_frames else 0,
    )
    cc.store(clip, extra={"kind": "open_floor"})
    got = cc.load(clip.key)
    assert got == clip
    assert got.duration_sec == pytest.approx(1.0)
    # Storing again replaces the clip.
    cc.store(clip)
    assert cc.load(clip.key) == clip


def test_load_misses_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("TALKSHOW_CANNED_CACHE_DIR", str(tmp_path))
    assert cc.load("nope") is None
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "meta.json").write_text("{not json")
    assert cc.load("broken") is None


def test_canned_lines_cover_every_fixed_host_line():
    from agent.floor.host_lines import _HUMAN_FLOOR, _INTRO_SPEAKER, _OPEN_FLOOR

    lines = cc.canned_lines(["Amy", "Ryan"], close_line="That wraps it.", welcome="Welcome!")
    kinds = [k for k, _ in lines]
    assert kinds.count("open_floor") == len(_OPEN_FLOOR)
    assert kinds.count("intro") == 2 * len(_INTRO_SPEAKER)
    assert kinds.count("grant_human") == len(_HUMAN_FLOOR)
    assert ("close", "That wraps it.") in lines and ("welcome", "Welcome!") in lines
    assert ("intro", "Ryan, you're up.") in lines
    assert set(kinds) <= cc.CANNED_KINDS
    assert all("{" not in t for _, t in lines)


@pytest.mark.parametrize("raw,on", [("1", True), ("true", True), ("on", True), ("0", False), ("", False)])
def test_flag(monkeypatch, raw, on):
    monkeypatch.setenv("TALKSHOW_PRERENDER_CANNED", raw)
    assert cc.prerender_enabled() is on
