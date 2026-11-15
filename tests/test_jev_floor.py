import asyncio

from agent.floor import jev_floor


CANDS = [{"role": "commentator", "speaker": "Ryan", "content": "Then why did they end it?"},
         {"role": "guest", "speaker": "Amy", "content": "My neighbour lost her job."}]


def test_request_has_candidates_and_host():
    body = jev_floor.build_request("Audience: why?", CANDS, {"guest": 2}, {"host": "Lessac"}, "host")
    crit = body["questions"]["next"]["criteria"]
    assert set(crit) == {"commentator", "guest", "host"}
    assert "2 turns so far" in body["state"] and "Then why did they end it?" in body["state"]


def test_pick_without_key_is_skipped(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    role, conf, info = asyncio.run(jev_floor.jev_pick("t", CANDS, {}, {}, "host"))
    assert role is None and info["jev"] == "skipped"


def test_pick_parses_answer_and_rejects_unknown(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(jev_floor, "_post", lambda body, key: {
        "answers": {"next": {"choice": "guest", "confidence": 0.9, "probabilities": {}}}})
    role, conf, info = asyncio.run(jev_floor.jev_pick("t", CANDS, {}, {}, "host"))
    assert (role, conf, info["jev"]) == ("guest", 0.9, "ok")
    monkeypatch.setattr(jev_floor, "_post", lambda body, key: {
        "answers": {"next": {"choice": "human", "confidence": 1.0}}})
    assert asyncio.run(jev_floor.jev_pick("t", CANDS, {}, {}, "host"))[0] is None


def test_pick_error_falls_back(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")

    def boom(body, key):
        raise TimeoutError

    monkeypatch.setattr(jev_floor, "_post", boom)
    role, _, info = asyncio.run(jev_floor.jev_pick("t", CANDS, {}, {}, "host"))
    assert role is None and info["jev_error"] == "TimeoutError"
