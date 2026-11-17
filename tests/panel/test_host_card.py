from agent.floor.floor_rules import host_identity


def test_host_card_flag(monkeypatch):
    monkeypatch.delenv("TALKSHOW_HOST_CARD", raising=False)
    monkeypatch.setenv("TALKSHOW_SHOW_FRAME", "1")
    assert "restate" not in host_identity("Lessac")
    monkeypatch.setenv("TALKSHOW_HOST_CARD", "1")
    card = host_identity("Lessac")
    assert card.startswith("You are Lessac") and "OTHER panelist" in card and "never take a side" in card


def test_demo_uses_host_card():
    from eval.quality_ruler.arms import ARMS

    assert ARMS["demo"]["live_flags"]["TALKSHOW_HOST_CARD"] == "1"
    assert "TALKSHOW_HOST_CARD" not in ARMS["demo_host_v1"]["live_flags"]
