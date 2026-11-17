from types import SimpleNamespace as NS

from agent.panel.show_frame import show_frame, topic_of, with_frame


H = NS(lines=[NS(text="Welcome, I am Claire."),
              NS(text="You wanted to talk about Has tipping culture gone too far? \u2014 the floor is yours, go ahead.")])


def test_topic_from_host_line(monkeypatch):
    monkeypatch.delenv("TALKSHOW_TOPIC", raising=False)
    monkeypatch.delenv("TALKSHOW_TEXT_TOPIC", raising=False)
    assert topic_of(H) == "Has tipping culture gone too far?"
    monkeypatch.setenv("TALKSHOW_TOPIC", "Four-day week")
    assert topic_of(H) == "Four-day week"


def test_frame_off_is_unchanged(monkeypatch):
    monkeypatch.delenv("TALKSHOW_SHOW_FRAME", raising=False)
    assert with_frame("You are Ryan.", H) == "You are Ryan."


def test_frame_on(monkeypatch):
    monkeypatch.delenv("TALKSHOW_TOPIC", raising=False)
    monkeypatch.delenv("TALKSHOW_TEXT_TOPIC", raising=False)
    monkeypatch.setenv("TALKSHOW_SHOW_FRAME", "1")
    s = with_frame("You are Ryan.", H)
    assert s.endswith("You are Ryan.") and "not a debate" in s and "Today's topic: Has tipping" in s
    assert "Claire hosts" in show_frame(H)


def test_host_identity_wording(monkeypatch):
    from agent.floor.floor_rules import host_identity

    monkeypatch.delenv("TALKSHOW_SHOW_FRAME", raising=False)
    assert "debaters" in host_identity("Claire")
    monkeypatch.setenv("TALKSHOW_SHOW_FRAME", "1")
    assert "debate" not in host_identity("Claire")
