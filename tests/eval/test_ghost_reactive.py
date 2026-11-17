"""Reactive ghost guest: addressing rule, decision timing, budget. No network, no room."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from eval.ghost_session.client import GhostRoomClient, load_script
from eval.ghost_session.live import build_reaction_messages, parse_reaction, TranscriptLine
from eval.ghost_session.addressing import addressing
from eval.ghost_session.reactive import PreparedReply, ReactiveGuest, ReactiveLimits

SAY = "Okay but honestly, I think you're skipping the part where people stop learning how to debug anything on their own, and that scares me a bit."


def test_addressing_rule() -> None:
    others = ["Claire", "Ryan", "Amy"]
    cases = {
        # Hand-overs to Howard: a question / invitation tied to his name, at the line's end.
        "Howard, what do you think?": "direct",
        "So Howard, is that true for you?": "direct",
        "Howard, I'm curious. Do you still write code by hand?": "direct",
        "Welcome back. Howard, the floor is yours.": "direct",
        "Let's hear from our guest. What's your take?": "direct",
        "Howard, that's a real win. But here's my question: could you explain the fix?": "direct",
        # Hand-overs to someone else: Howard stays out (the last one named wins).
        "Hi everyone, I'm Claire. Today we ask if AI makes us smarter. Ryan, is AI a crutch?": "other:Ryan",
        "Howard, that is the line of the night. Amy, how do you teach kids that?": "other:Amy",
        "Great point. Over to you, Ryan.": "other:Ryan",
        "Hi everyone. Today: AI and learning. Priya, you teach — what do you see?": "other:Priya",  # not on the roster yet
        # A question to him mid-line, then more talk: not a hand-over.
        "Howard, did you learn anything? Because I doubt it. Anyway the data says otherwise.": "asked",
        # Bare vocatives and third person.
        "That's fair, Howard.": "mentioned",
        "Yeah, Howard, that's not harsh, if you can't explain it, it's a problem.": "mentioned",
        "Totally, Howard — two minutes beats two hours.": "mentioned",
        "I agree with what Howard said earlier. But Amy is wrong.": "mentioned",
        "Like Howard, I use AI daily.": "mentioned",
        "Oh, I love that. Honestly, that is the whole debate.": "",
        "Our guests are split. Exactly, the data is mixed.": "",
    }
    for text, want in cases.items():
        assert addressing(text, "Howard", others) == want, text


def test_parse_reaction() -> None:
    assert parse_reaction('{"action": "silent", "emotion": "Neutral", "utterance": ""}') == ("silent", "", "")
    action, emo, text = parse_reaction(f'{{"action": "cut_in", "emotion": "Anger", "utterance": "{SAY}"}}')
    assert (action, emo) == ("cut_in", "Anger") and text.startswith("Okay but")
    assert parse_reaction(f'{{"emotion": "Happiness", "utterance": "{SAY}"}}')[0] == "respond"
    assert parse_reaction("hmm") is None


def test_reaction_prompt() -> None:
    line = TranscriptLine("host", "Max", "Howard, what do you think?")
    direct = build_reaction_messages(name="Howard", persona="p", topic="t", transcript=[], line=line,
                                     addressed="direct", allow_cut_in=False)[1]["content"]
    assert "must answer" in direct and "cut_in" not in direct
    other = build_reaction_messages(name="Howard", persona="p", topic="t", transcript=[], line=line,
                                    turns_spoken=2, lines_since_turn=4)[1]["content"]
    assert '"silent"' in other and '"cut_in"' in other and "spoken 2 times" in other


def test_load_reactive_script(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text("reactive: true\nmax_turns: 5\nmin_lines_between_turns: 2\nmax_minutes: 4\n"
                 "turns:\n  - generate: true\n    after_lines: 1\n", encoding="utf-8")
    s = load_script(p)
    assert (s.reactive, s.max_turns, s.min_lines_between_turns, s.max_minutes) == (True, 5, 2, 4.0)
    assert len(s.turns) == 1
    assert load_script(Path(__file__).resolve().parents[2] / "eval/ghost_session/scripts/smoke.yaml").reactive is False


# --- the controller, driven by fake UI events ------------------------------------------


class Harness:
    def __init__(self, actions: dict[int, str], *, gen_delay: float = 0.0, limits: ReactiveLimits | None = None):
        self.client = GhostRoomClient()
        self.actions = actions  # line ordinal → action the "LLM" picks (default silent)
        self.gen_delay = gen_delay
        self.prepared: list[tuple[int, bool]] = []  # (line, must_answer)
        self.spoken: list[tuple[str, float]] = []
        self.guest = ReactiveGuest(
            self.client,
            name="Howard",
            limits=limits or ReactiveLimits(max_turns=8, min_lines_between_turns=2, max_minutes=1),
            prepare=self.prepare,
            speak=self.speak,
            build_messages=lambda **kw: [{"line": kw["line"], "addressed": kw["addressed"]}],  # type: ignore[list-item]
            pause_range=(0.01, 0.02),
            cut_in_after_s=0.05,
            max_line_wait_s=5,
        )
        self.step = 0

    async def prepare(self, messages, must_answer):
        ordinal = self.client.agent_lines
        self.prepared.append((ordinal, must_answer))
        await asyncio.sleep(self.gen_delay)
        action = "respond" if must_answer else self.actions.get(ordinal, "silent")
        return PreparedReply(action, "Neutral", "" if action == "silent" else f"reply to {ordinal}", None, 0.1, 0.1)

    async def speak(self, text, speech):
        t = time.monotonic()
        self.client.record_own_line("Howard", text)
        self.spoken.append((text, t))
        await asyncio.sleep(0.02)
        return t

    def announce(self, role: str, text: str) -> int:
        self.step += 1
        self.client.on_ui_event({"type": "transcript", "role": role, "step": f"s{self.step}", "text": text, "speaker": role})
        return self.step

    def play(self, role: str) -> None:
        self.client.on_ui_event({"type": "role_active", "role": role})

    def end(self, role: str, step: int, played: bool = True) -> None:
        self.client.on_ui_event({"type": "role_idle"})
        ev = {"type": "line_end", "role": role, "step": f"s{step}"}
        if not played:
            ev["played"] = False
        self.client.on_ui_event(ev)


async def _tick(sec: float = 0.08) -> None:
    await asyncio.sleep(sec)


def test_direct_question_is_answered_at_line_end_even_in_cooldown() -> None:
    async def scenario():
        h = Harness({})
        task = asyncio.create_task(h.guest.run(last_turn_ordinal=0))
        s = h.announce("host", "Howard, what do you think about that?")  # line 1: in cooldown, but direct
        h.play("host")
        await _tick()
        assert h.spoken == []  # not while the line plays
        h.end("host", s)
        end_t = time.monotonic()
        await _tick(0.2)
        assert [t for t, _ in h.spoken] == ["reply to 1"]
        assert h.spoken[0][1] - end_t >= 0.005
        assert h.prepared == [(1, True)]
        h.guest.stop()
        await asyncio.wait_for(task, 1)
        return h
    h = asyncio.run(scenario())
    assert h.guest.decisions[0]["decision"] == "respond" and "latency_s" in h.guest.decisions[0]


def test_cooldown_skips_llm_and_silent_is_default() -> None:
    async def scenario():
        h = Harness({})
        task = asyncio.create_task(h.guest.run(last_turn_ordinal=0))
        for i in range(3):
            s = h.announce("guest", f"Point number {i}. AI is great.")
            h.play("guest")
            await _tick(0.03)
            h.end("guest", s)
            await _tick(0.05)
        h.guest.stop()
        await asyncio.wait_for(task, 1)
        return h
    h = asyncio.run(scenario())
    assert [p[0] for p in h.prepared] == [2, 3]  # line 1 is within min_lines_between_turns=2
    assert h.spoken == []
    assert [d["decision"] for d in h.guest.decisions] == ["silent (cooldown)", "silent", "silent"]


def test_reply_dropped_when_next_line_is_already_on_air() -> None:
    async def scenario():
        h = Harness({2: "respond"}, gen_delay=0.15)
        task = asyncio.create_task(h.guest.run(last_turn_ordinal=0))
        s1 = h.announce("guest", "First.")
        h.end("guest", s1)
        await _tick(0.02)
        s2 = h.announce("guest", "AI tutors beat human tutors, period.")
        h.play("guest")
        await _tick(0.02)
        h.end("guest", s2)
        s3 = h.announce("commentator", "I disagree with that completely.")
        h.play("commentator")  # on air before Howard's reply to line 2 is ready
        await _tick(0.3)
        assert h.spoken == []
        h.guest.stop()
        h.end("commentator", s3)
        await asyncio.wait_for(task, 1)
        return h
    h = asyncio.run(scenario())
    assert "dropped" in h.guest.decisions[1]["decision"] or h.guest.decisions[1]["decision"] == "next line on air"


def test_reply_spoken_in_the_gap_even_with_a_queued_line() -> None:
    async def scenario():
        h = Harness({2: "respond"})
        task = asyncio.create_task(h.guest.run(last_turn_ordinal=0))
        h.end("guest", h.announce("guest", "First."))
        await _tick(0.02)
        s2 = h.announce("guest", "AI tutors beat human tutors, period.")
        h.play("guest")
        await _tick()
        h.announce("host", "Next up, the kids question.")  # queued, not playing
        h.end("guest", s2)
        await _tick(0.2)
        h.guest.stop()
        await asyncio.wait_for(task, 1)
        return h
    h = asyncio.run(scenario())
    assert [t for t, _ in h.spoken] == ["reply to 2"]


def test_cut_in_speaks_mid_line_and_no_double_turn() -> None:
    async def scenario():
        h = Harness({2: "cut_in", 3: "respond"}, limits=ReactiveLimits(max_turns=8, min_lines_between_turns=0, max_minutes=1))
        task = asyncio.create_task(h.guest.run(last_turn_ordinal=0))
        h.end("guest", h.announce("guest", "First."))
        await _tick(0.02)
        s2 = h.announce("guest", "Struggle is overrated, honestly.")
        h.play("guest")
        await _tick(0.15)
        assert [t for t, _ in h.spoken] == ["reply to 2"]  # before the line ended
        h.end("guest", s2)
        await _tick(0.15)
        assert len(h.spoken) == 1  # no new agent line since: no second turn
        h.guest.stop()
        await asyncio.wait_for(task, 1)
        return h
    asyncio.run(scenario())


def test_max_turns_stops_reacting() -> None:
    async def scenario():
        h = Harness({}, limits=ReactiveLimits(max_turns=2, min_lines_between_turns=0, max_minutes=1))
        h.guest.turns_done = 1  # one seed turn already spoken
        task = asyncio.create_task(h.guest.run(last_turn_ordinal=0))
        for _ in range(3):
            s = h.announce("host", "Howard, what do you think?")
            h.end("host", s)
            await _tick()
        await asyncio.wait_for(task, 1)
        return h
    h = asyncio.run(scenario())
    assert len(h.spoken) == 1 and h.guest.turns_done == 2


# --- run_ghost_session wiring: seed turn → reactive → leave after N panel lines -----------


def test_run_ghost_session_reactive_wiring(monkeypatch) -> None:
    """Reactive run end to end: answer when asked → budget spent → goodbye at a gap →
    wrap_up → leave only after the host's close_round line ended and ~quiet."""
    import eval.ghost_session.client as client_mod
    import eval.ghost_session.ending as ending_mod
    from eval.ghost_session import live
    from eval.ghost_session.client import GhostScript, run_ghost_session

    said: list[str] = []
    control: list[dict] = []
    order: list[tuple] = []
    holder: dict = {}

    class FakeWriter:
        config = type("C", (), {"base_url": "http://llm.test/v1"})()

        async def write_turn(self, messages):
            assert "leave the show" in messages[1]["content"]
            return live.GeneratedTurn("Happiness", "Thanks all, I have to head out.", 0.0, 1)

        async def react(self, messages, *, must_answer=False):
            if must_answer:
                return live.Reaction("respond", "Happiness", "answer words", 0.0, 1)
            return live.Reaction("silent", "", "", 0.0, 1)

        async def aclose(self):
            pass

    async def fake_connect(self, url, token):
        holder["client"] = self

    async def fake_noop(self, *a, **k):
        return None

    async def fake_inject(client, text, **kwargs):
        said.append(text)
        order.append(("said", text))

    async def fake_control(self, payload):
        control.append(payload)
        order.append(("control", payload["type"]))

    monkeypatch.setattr(client_mod, "mint_ghost_token", lambda **kw: ("url", "tok"))
    monkeypatch.setattr(client_mod.GhostRoomClient, "connect", fake_connect)
    monkeypatch.setattr(client_mod.GhostRoomClient, "disconnect", fake_noop)
    monkeypatch.setattr(client_mod.GhostRoomClient, "publish_control", fake_control)
    monkeypatch.setattr(client_mod, "_inject_until_heard", fake_inject)
    monkeypatch.setattr(live, "LiveWriter", FakeWriter)
    monkeypatch.setattr(ending_mod, "GAP_PAUSE_S", 0.05)
    monkeypatch.setattr(ending_mod, "CLOSE_QUIET_S", 0.2)

    script = GhostScript(name="Howard", topic="AI", persona="p", reactive=True, max_turns=1,
                         min_lines_between_turns=3, max_minutes=1)
    events: list[str] = []

    async def drive():
        while "client" not in holder:
            await asyncio.sleep(0.01)
        c = holder["client"]

        async def line(role, step, text, play=0.05):
            c.on_ui_event({"type": "transcript", "role": role, "step": step, "text": text, "speaker": role})
            c.on_ui_event({"type": "role_active", "role": role})
            await asyncio.sleep(play)
            c.on_ui_event({"type": "role_idle"})
            c.on_ui_event({"type": "line_end", "role": role, "step": step})

        await line("host", "opening", "Welcome to the show. Howard, what do you think?")
        while len(said) < 2:  # answer, then (budget spent) the goodbye
            await asyncio.sleep(0.01)
        while not control:
            await asyncio.sleep(0.01)
        events.append("wrap_up")
        await asyncio.sleep(0.1)
        await line("host", "close_round", "That's our show, thanks everyone!", play=0.3)
        events.append("close ended")

    async def main():
        driver = asyncio.create_task(drive())
        t = await asyncio.wait_for(
            run_ghost_session(room="r", script=script, text_turns=True, hold_sec=0,
                              quiet_sec=30, stop_after_panel=6, welcome_timeout_sec=5),
            timeout=20,
        )
        events.append("left")
        await driver
        return t

    asyncio.run(main())
    assert said == ["answer words", "Thanks all, I have to head out."]
    assert control == [{"type": "wrap_up"}]
    # wrap_up goes out as the goodbye starts, not after it.
    assert order[-2:] == [("control", "wrap_up"), ("said", "Thanks all, I have to head out.")]
    assert events == ["wrap_up", "close ended", "left"]


def test_a_question_that_never_played_is_not_answered() -> None:
    async def scenario():
        h = Harness({})
        task = asyncio.create_task(h.guest.run(last_turn_ordinal=0))
        s = h.announce("host", "Howard, what do you think?")
        await _tick(0.05)
        h.end("host", s, played=False)  # rolled back before it played
        await _tick(0.2)
        h.guest.stop()
        await asyncio.wait_for(task, 1)
        return h
    h = asyncio.run(scenario())
    assert h.spoken == [] and "line dropped" in h.guest.decisions[0]["decision"]


# --- floor to someone else, cooldown for mentions, stale replies, seed hand-over ---------


def _drive(h: Harness, steps) -> None:
    async def scenario():
        task = asyncio.create_task(h.guest.run(last_turn_ordinal=0))
        await steps(h)
        h.guest.stop()
        await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_floor_handed_to_someone_else_is_left_alone() -> None:
    h = Harness({}, limits=ReactiveLimits(max_turns=8, min_lines_between_turns=0, max_minutes=1))

    async def steps(h):
        h.end("host", h.announce("host", "Great point, Howard. Amy, how do you teach kids that?"))
        await _tick(0.2)

    _drive(h, steps)
    assert h.prepared == [] and h.spoken == []
    assert h.guest.decisions[0]["decision"] == "silent (floor to Amy)"


def test_mention_respects_cooldown_but_a_question_does_not() -> None:
    h = Harness({}, limits=ReactiveLimits(max_turns=8, min_lines_between_turns=3, max_minutes=1))

    async def steps(h):
        h.end("guest", h.announce("guest", "That's fair, Howard."))  # mentioned, line 1: cooldown
        await _tick(0.1)
        h.end("guest", h.announce("guest", "Howard, did you learn anything? I doubt it. Anyway."))  # asked
        await _tick(0.1)

    _drive(h, steps)
    assert [d["addressed"] for d in h.guest.decisions] == ["mentioned", "asked"]
    assert h.guest.decisions[0]["decision"] == "silent (cooldown)"
    assert h.prepared == [(2, False)]


def test_reply_dropped_when_line_end_never_comes(monkeypatch) -> None:
    import eval.ghost_session.reactive as reactive_mod

    monkeypatch.setattr(reactive_mod, "STALE_AFTER_NEXT_PLAYS_S", 0.1)
    h = Harness({})

    async def steps(h):
        h.announce("host", "Howard, what do you think?")  # its line_end never arrives
        h.play("host")
        await _tick(0.05)
        h.announce("guest", "Let me jump in first.")
        h.play("guest")
        await _tick(0.4)

    _drive(h, steps)
    assert h.spoken == []
    assert "end never seen" in h.guest.decisions[0]["decision"]


def test_reply_dropped_after_two_newer_lines() -> None:
    h = Harness({}, gen_delay=0.1)

    async def steps(h):
        h.announce("host", "Howard, what do you think?")
        await _tick(0.03)  # he starts on line 1
        h.announce("guest", "Queued one.")
        h.announce("commentator", "Queued two.")
        await _tick(0.3)

    _drive(h, steps)
    assert h.spoken == []
    assert any("newer lines" in d.get("decision", "") for d in h.guest.decisions)


def test_known_names_from_roster_and_speakers() -> None:
    c = GhostRoomClient()
    c.guest_name = "Howard"
    c.on_ui_event({"type": "panel_roster", "members": [{"role": "host", "name": "Claire"}, {"role": "guest", "name": "Amy"}]})
    c.on_ui_event({"type": "transcript", "role": "commentator", "step": "x", "speaker": "Ryan", "text": "Hi."})
    c.record_own_line("Howard", "Hello.")
    assert c.known_names() == {"Claire", "Amy", "Ryan"}


def test_seed_turn_waits_for_the_person_the_floor_was_handed_to(monkeypatch) -> None:
    import eval.ghost_session.client as client_mod
    from eval.ghost_session.client import ScriptedTurn

    monkeypatch.setattr(client_mod, "LINE_END_QUIET_S", 0.05)

    async def scenario():
        c = GhostRoomClient()
        c.guest_name = "Howard"
        events: list[str] = []
        turn = ScriptedTurn(text="", after_lines=1, when="line_end", max_wait_sec=5, generate=True)
        task = asyncio.create_task(c.wait_for_turn(turn, lines_before=0, on_target=lambda: events.append("target")))
        tr = lambda role, step, spk, text: c.on_ui_event(
            {"type": "transcript", "role": role, "step": step, "speaker": spk, "text": text})
        tr("host", "o", "Claire", "Welcome! Tonight: AI and learning. Ryan, is AI a crutch?")
        c.on_ui_event({"type": "role_idle"})
        c.on_ui_event({"type": "line_end", "role": "host", "step": "o"})
        await _tick(0.3)
        events.append("host line over")
        tr("commentator", "r", "Ryan", "Claire, I'd call it a power tool.")
        c.on_ui_event({"type": "role_active", "role": "commentator"})
        await _tick(0.2)
        events.append("ryan still talking")
        c.on_ui_event({"type": "role_idle"})
        c.on_ui_event({"type": "line_end", "role": "commentator", "step": "r"})
        events.append(await asyncio.wait_for(task, 2))
        return events

    assert asyncio.run(scenario()) == ["host line over", "target", "ryan still talking", "line_end"]


# --- ending ----------------------------------------------------------------------------


def test_gap_rules(monkeypatch) -> None:
    import eval.ghost_session.ending as ending_mod

    c = GhostRoomClient()
    c.guest_name = "Howard"
    c.on_ui_event({"type": "transcript", "role": "host", "step": "a", "speaker": "Claire", "text": "Hello. Amy, what do you think?"})
    c.on_ui_event({"type": "role_active", "role": "host"})
    assert not ending_mod.in_gap(c, pause=0)  # playing
    c.on_ui_event({"type": "role_idle"})
    assert not ending_mod.in_gap(c, pause=0)  # between sentences: the line has not ended
    c.on_ui_event({"type": "line_end", "role": "host", "step": "a"})
    assert not ending_mod.in_gap(c, pause=0)  # the gap is Amy's
    assert ending_mod.in_gap(c, pause=0, strict=False)
    c.on_ui_event({"type": "transcript", "role": "guest", "step": "b", "speaker": "Amy", "text": "I think it's fine."})
    c.on_ui_event({"type": "role_active", "role": "guest"})
    c.on_ui_event({"type": "role_idle"})
    c.on_ui_event({"type": "line_end", "role": "guest", "step": "b"})
    assert ending_mod.in_gap(c, pause=0)


def test_show_over_rules(monkeypatch) -> None:
    import eval.ghost_session.ending as ending_mod

    monkeypatch.setattr(ending_mod, "CLOSE_QUIET_S", 0.1)
    monkeypatch.setattr(ending_mod, "CLOSE_FALLBACK_S", 0.2)
    monkeypatch.setattr(ending_mod, "FALLBACK_QUIET_S", 0.1)
    c = GhostRoomClient()
    wrap_t = time.monotonic()
    c.on_ui_event({"type": "transcript", "role": "host", "step": "close_round", "text": "Goodnight!"})
    c.on_ui_event({"type": "role_active", "role": "host"})
    assert not ending_mod.show_over(c, wrap_t)
    c.on_ui_event({"type": "role_idle"})
    c.on_ui_event({"type": "line_end", "role": "host", "step": "close_round"})
    assert not ending_mod.show_over(c, wrap_t)  # not quiet long enough yet
    time.sleep(0.12)
    assert ending_mod.show_over(c, wrap_t)
    c.on_ui_event({"type": "role_active", "role": "host"})
    assert not ending_mod.show_over(c, wrap_t)  # never while something plays

    # Fallback: no close_round at all.
    d = GhostRoomClient()
    wrap_t = time.monotonic()
    assert not ending_mod.show_over(d, wrap_t)
    time.sleep(0.25)
    assert ending_mod.show_over(d, wrap_t)


def test_goodbye_prompt_and_fixed_text() -> None:
    from eval.ghost_session.client import GhostScript, prepare_goodbye
    from eval.ghost_session.live import build_goodbye_messages

    msgs = build_goodbye_messages(name="Howard", persona="p", topic="t", transcript=[])
    assert "head out" in msgs[1]["content"] and "fifteen to thirty" in msgs[1]["content"]
    script = GhostScript(name="Howard", goodbye="Bye all, thanks!")
    text, speech = asyncio.run(prepare_goodbye(GhostRoomClient(), None, script, voice=False))
    assert (text, speech) == ("Bye all, thanks!", None)


# --- nudge -------------------------------------------------------------------------------


def test_nudge_after_quiet_lines() -> None:
    h = Harness({}, limits=ReactiveLimits(max_turns=8, min_lines_between_turns=2, max_minutes=1, nudge_after_lines=4))

    async def steps(h):
        for i in range(5):
            text = "Amy, what do you think?" if i == 3 else f"Point {i}, plain and simple."
            role = "host" if i == 3 else "guest"
            s = h.announce(role, text)
            h.play(role)
            await _tick(0.03)
            h.end(role, s)
            await _tick(0.1)

    _drive(h, steps)
    # Lines 2-3: the LLM says silent; line 4 hands the floor to Amy (skipped even though he
    # is due); line 5 is the first eligible line after 4 quiet ones: forced, after the line.
    assert h.prepared == [(2, False), (3, False), (5, True)]
    assert [t for t, _ in h.spoken] == ["reply to 5"]
    assert h.guest.decisions[-1].get("nudge") and h.guest.decisions[-1]["decision"] == "respond"


def test_reaction_prompt_leans_to_respond() -> None:
    line = TranscriptLine("guest", "Amy", "Exactly, Howard — that line you draw is the whole ballgame.")
    mentioned = build_reaction_messages(name="Howard", persona="p", topic="t", transcript=[], line=line,
                                        addressed="mentioned")[1]["content"]
    assert "usually respond" in mentioned
    nudged = build_reaction_messages(name="Howard", persona="p", topic="t", transcript=[], line=line,
                                     must_respond=True, allow_cut_in=False)[1]["content"]
    assert 'action "respond"' in nudged and "quiet for a while" in nudged and "cut_in" not in nudged


def test_load_nudge_key(tmp_path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text("reactive: true\nnudge_after_lines: 6\n", encoding="utf-8")
    assert load_script(p).nudge_after_lines == 6
    p.write_text("reactive: true\n", encoding="utf-8")
    assert load_script(p).nudge_after_lines == 5
