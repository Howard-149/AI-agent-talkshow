from agent.panel.line_length import cap_sentences, split_sentences


def test_split_keeps_abbreviations_and_numbers():
    t = "The U.S. spent 4.5 billion. Dr. Smith disagreed! Was it worth it? Yes."
    assert split_sentences(t) == ["The U.S. spent 4.5 billion.", "Dr. Smith disagreed!", "Was it worth it?", "Yes."]


def test_cap():
    t = "One point. Two points. Three points."
    assert cap_sentences(t, 2) == "One point. Two points."
    assert cap_sentences(t, 0) == t
    assert cap_sentences("Short line.", 2) == "Short line."
    assert cap_sentences(None, 2) is None


def test_flag(monkeypatch):
    monkeypatch.delenv("TALKSHOW_MAX_SENTENCES", raising=False)
    assert cap_sentences("A. B. C.") == "A. B. C."
    monkeypatch.setenv("TALKSHOW_MAX_SENTENCES", "2")
    assert cap_sentences("Alpha one. Beta two. Gamma three.") == "Alpha one. Beta two."


def test_parse_caps_content(monkeypatch):
    import json

    from agent.interaction.think_action import parse_think_action

    monkeypatch.setenv("TALKSHOW_MAX_SENTENCES", "1")
    raw = json.dumps({"think": "", "action": {"decision": "speak", "intent": {"type": "challenge", "goal": "g",
                                                                              "target": None},
                                              "content": "First. Second.", "backchannels": []}})
    ta = parse_think_action(raw, floor_state="free", interval_text="")
    assert ta.action.content == "First."
