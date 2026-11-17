"""Ghost session live turns: script format, transcript tracking, prompt, parsing, voice path.

No network: the LLM is an httpx MockTransport and the TTS sidecar is monkeypatched.
"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import httpx
import pytest

from eval.ghost_session import live
from eval.ghost_session.client import (
    GhostRoomClient,
    GhostScript,
    ScriptedTurn,
    load_script,
    prepare_live_turn,
)
from eval.ghost_session.live import (
    LiveWriter,
    LLMConfig,
    TranscriptLine,
    build_messages,
    clean_utterance,
    parse_reply,
    spell_numbers,
)

GOOD = (
    '{"emotion": "Anger", "utterance": "Honestly Maya, I am not sure I buy that. I used to spend whole '
    'nights stuck on one bug and I learned a ton, and now I just paste it into a chatbot and move on."}'
)


def _run(coro):
    return asyncio.run(coro)


# --- script format --------------------------------------------------------------


def test_load_script_live_and_scripted_turns(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(
        """
name: Howard
topic: Is AI making us smarter?
persona: A grad student.
turns:
  - text: A scripted line.
    after_lines: 1
    emotion: Anger
  - generate: true
    goal: push back
    after_lines: 4
    when: mid_line
    after_sec: 3
  - after_lines: 2          # no text, no generate: skipped as before
""",
        encoding="utf-8",
    )
    script = load_script(path)
    assert script.topic == "Is AI making us smarter?"
    assert script.persona == "A grad student."
    assert len(script.turns) == 2
    first, second = script.turns
    assert (first.text, first.generate, first.emotion) == ("A scripted line.", False, "Anger")
    assert second.generate and second.text == "" and second.goal == "push back"
    assert (second.after_lines, second.when, second.after_sec) == (4, "mid_line", 3.0)


def test_text_wins_over_generate(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("turns:\n  - text: fixed words\n    generate: true\n", encoding="utf-8")
    (turn,) = load_script(path).turns
    assert turn.text == "fixed words" and not turn.generate


def test_live_demo_script_parses() -> None:
    path = Path(__file__).resolve().parents[2] / "logs" / "ghost-demo" / "ai_dependence_live.yaml"
    if not path.is_file():
        pytest.skip("demo script not present")
    script = load_script(path)
    assert script.name == "Howard" and script.topic and script.persona
    assert all(t.generate and t.goal for t in script.turns)
    assert script.reactive or script.turns
    if script.reactive:
        assert script.max_turns >= len(script.turns) and script.max_minutes > 0


# --- transcript tracking --------------------------------------------------------------


def _tr(role: str, step: str, text: str, speaker: str = "") -> dict:
    return {"type": "transcript", "role": role, "step": step, "text": text, "speaker": speaker}


def test_transcript_tracking() -> None:
    c = GhostRoomClient()
    c.on_ui_event(_tr("host", "s1", "Welcome to the show.", "Max"))
    c.on_ui_event(_tr("host", "s1", "Welcome to the show, everyone."))  # same line, fuller text
    c.on_ui_event({"type": "line_end", "role": "host", "step": "s1"})
    c.on_ui_event(_tr("human", "h1", "stt of my own words"))  # ignored: we record our own
    c.record_own_line("Howard", "My own words.")
    c.on_ui_event(_tr("guest", "s2", "This line never played.", "Maya"))
    c.on_ui_event({"type": "line_end", "role": "guest", "step": "s2", "played": False})
    c.on_ui_event(_tr("commentator", "s3", "One two three four five six.", "Leo"))
    c.on_ui_event({"type": "interrupt", "role": "commentator", "by_role": "human", "heard_text": "One two three"})
    c.on_ui_event({"type": "line_end", "role": "commentator", "step": "s3"})
    # The same step id in a later beat is a new line.
    c.on_ui_event(_tr("host", "s1", "Next beat.", "Max"))

    lines = c.transcript
    assert [(l.role, l.speaker, l.text) for l in lines] == [
        ("host", "Max", "Welcome to the show, everyone."),
        ("human", "Howard", "My own words."),
        ("commentator", "Leo", "One two three —"),
        ("host", "Max", "Next beat."),
    ]
    assert c.agent_lines == 4


def test_on_target_fires_when_trigger_line_is_announced() -> None:
    async def scenario() -> list[str]:
        c = GhostRoomClient()
        events: list[str] = []
        turn = ScriptedTurn(text="", after_lines=1, when="line_end", max_wait_sec=5, generate=True)
        task = asyncio.create_task(
            c.wait_for_turn(turn, lines_before=0, on_target=lambda: events.append("target"))
        )
        await asyncio.sleep(0.05)
        c.on_ui_event(_tr("host", "s1", "Opening line."))
        c.on_ui_event({"type": "role_active", "role": "host"})
        await asyncio.sleep(0.3)
        events.append("line still playing")
        c.on_ui_event({"type": "role_idle"})
        c.on_ui_event({"type": "line_end", "role": "host", "step": "s1"})
        events.append(await task)
        return events

    import eval.ghost_session.client as client_mod

    old = client_mod.LINE_END_QUIET_S
    client_mod.LINE_END_QUIET_S = 0.05
    try:
        assert _run(scenario()) == ["target", "line still playing", "line_end"]
    finally:
        client_mod.LINE_END_QUIET_S = old


# --- prompt --------------------------------------------------------------------


def test_build_messages_recent_lines_and_cut_line() -> None:
    lines = [TranscriptLine("guest", "Maya", f"line {i}") for i in range(20)]
    cut = TranscriptLine("guest", "Maya", " ".join(f"w{i}" for i in range(40)))
    msgs = build_messages(
        name="Howard", persona="A grad student.", topic="AI", transcript=lines,
        goal="push back", cut_line=cut, cut_after_sec=3,
    )
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "Howard" in system and "A grad student." in system and "Show topic: AI" in system
    assert "line 7" not in user and "line 8" in user and "line 19" in user  # last 12
    assert "push back" in user
    assert "w0 w1" in user and "w39" not in user  # only the part heard before cutting in


def test_build_messages_line_end_has_no_cut() -> None:
    msgs = build_messages(name="Howard", persona="", topic="", transcript=[], goal="")
    assert "just finished" in msgs[1]["content"] and "just starting" in msgs[1]["content"]


# --- parsing --------------------------------------------------------------------


def test_parse_json_and_fenced() -> None:
    emo, text = parse_reply(GOOD)
    assert emo == "Anger" and text.startswith("Honestly Maya")
    emo, _ = parse_reply("```json\n" + GOOD.replace("Anger", "surprised") + "\n```")
    assert emo == "Surprise"


def test_parse_label_format_and_unknown_emotion() -> None:
    raw = "emotion: Bored\nutterance: I mean, I get it, but I really think we are losing something here when nobody struggles."
    emo, text = parse_reply(raw)
    assert emo == "Neutral" and text.startswith("I mean")


def test_parse_rejects_garbage() -> None:
    assert parse_reply("") is None
    assert parse_reply('{"emotion": "Happiness", "utterance": "Yes."}') is None
    assert parse_reply("I cannot help with that.") is None


def test_clean_and_trim() -> None:
    raw = "*laughs* Okay so (sighs) I spent 3 hours on it 😅 and 45% of that was [pause] waiting, back in 2019."
    out = clean_utterance(raw)
    assert out == "Okay so I spent three hours on it and forty-five percent of that was waiting, back in twenty nineteen."
    long = " ".join(["word"] * 50) + ". " + " ".join(["more"] * 50) + "."
    _, text = parse_reply(json.dumps({"emotion": "Neutral", "utterance": long}))
    assert len(text.split()) <= live.MAX_WORDS and text.endswith(".")


def test_spell_numbers() -> None:
    assert spell_numbers("1,200 people and 2.5 hours in 2005") == (
        "one thousand two hundred people and two point five hours in two thousand five"
    )


# --- LLM endpoint ---------------------------------------------------------------------


def _writer(replies: list[str], seen: list[dict]) -> LiveWriter:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gemma-test"}, {"id": "other"}]})
        seen.append(json.loads(request.content))
        content = replies.pop(0)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    cfg = LLMConfig(base_url="http://llm.test/v1", model="", api_key="EMPTY")
    return LiveWriter(cfg, transport=httpx.MockTransport(handler))


def test_writer_uses_first_model_and_retries_once() -> None:
    seen: list[dict] = []
    writer = _writer(["sure! here you go", GOOD], seen)
    gen = _run(writer.write_turn([{"role": "user", "content": "hi"}]))
    assert gen.attempts == 2 and not gen.fallback and gen.emotion == "Anger"
    assert [s["model"] for s in seen] == ["gemma-test", "gemma-test"]


def test_writer_falls_back_to_neutral() -> None:
    writer = _writer(["no", '{"emotion": "Anger"}'], [])
    gen = _run(writer.write_turn([{"role": "user", "content": "hi"}]))
    assert gen.fallback and gen.emotion == "Neutral" and gen.text == live.FALLBACK_TEXT


def test_llm_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("GHOST_LLM_BASE_URL", "GHOST_LLM_MODEL", "GHOST_LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    cfg = LLMConfig.from_env()
    assert (cfg.base_url, cfg.model, cfg.api_key) == ("http://127.0.0.1:18000/v1", "", "EMPTY")


# --- voice ------------------------------------------------------------------------


def _fake_ref(folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with wave.open(str(folder / name), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\0\0" * 160)


def test_remote_reference_path_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.ghost_session import voice

    local = tmp_path / "voice"
    _fake_ref(local, "Neutral.wav")
    _fake_ref(local, "Anger.wav")
    monkeypatch.setenv("GHOST_VOICE_DIR", str(local))
    monkeypatch.setenv("GHOST_VOICE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("GHOST_VOICE_REMOTE_DIR", "/remote/voice/")
    calls: list[str] = []

    async def fake_synth(**kwargs):
        calls.append(kwargs["prompt_wav"])
        return b"\1\0" * 2205, 22050, 1

    monkeypatch.setattr(voice, "_sidecar_synth", lambda: fake_synth)
    speech, path = _run(voice.speech_for_turn("Hello there.", emotion="Anger"))
    assert calls == ["/remote/voice/Anger.wav"]
    assert path.parent == tmp_path / "cache" and path.is_file()
    _run(voice.speech_for_turn("Hello there.", emotion="Anger"))  # cached
    _run(voice.speech_for_turn("Hello again.", emotion="Sadness"))  # no Sadness.wav locally
    assert calls == ["/remote/voice/Anger.wav", "/remote/voice/Neutral.wav"]

    monkeypatch.delenv("GHOST_VOICE_REMOTE_DIR")
    assert voice.sidecar_reference_path(local / "Neutral.wav") == str((local / "Neutral.wav").resolve())


# --- prepare_live_turn ------------------------------------------------------------------


def test_prepare_live_turn_mid_line_uses_cut_line(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict] = []
    writer = _writer([GOOD], seen)
    c = GhostRoomClient()
    c.on_ui_event(_tr("host", "s1", "Tonight we talk about AI.", "Max"))
    c.on_ui_event(_tr("guest", "s2", "I think AI is the best thing that ever happened to students everywhere.", "Maya"))
    script = GhostScript(name="Howard", persona="A grad student.", topic="AI")
    turn = ScriptedTurn(text="", after_lines=2, when="mid_line", after_sec=1, generate=True, goal="push back")
    result = _run(prepare_live_turn(c, writer, script, turn, idx=2, voice=False))
    assert result.emotion == "Anger" and result.speech is None
    user = seen[0]["messages"][1]["content"]
    assert "Max: Tonight we talk about AI." in user
    assert 'Maya has just started saying: "I think AI is the best' in user


def test_sidecar_client_loads_without_agents_sdk() -> None:
    from eval.ghost_session import voice

    synth = voice._sidecar_synth()
    assert synth.__name__ == "synthesize_pcm_via_sidecar"
