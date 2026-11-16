"""Candidate selection (TALKSHOW_PANEL_SELECT, agent/panel/select.py): helpers, the live hooks in
ThinkActionInteraction, and the T6 harness."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent.interaction import ThinkActionInteraction
from agent.interaction.think_action import parse_think_action
from agent.panel import select
from agent.show.show_history import ShowHistory

FLAGS = ("TALKSHOW_PANEL_SELECT", "TALKSHOW_MAX_SENTENCES", "TALKSHOW_PERSONA_LIBRARY",
         "TYPESAFE_API_KEY", "TALKSHOW_THINK_LEAN", "TALKSHOW_THINK_PERSONA_NEAR", "TALKSHOW_FACT_PACK",
         "TALKSHOW_DIALOGUE_LIBRARY", "TALKSHOW_PANEL_CONCRETE", "TALKSHOW_UPTAKE_PLANNER",
         "TALKSHOW_PANEL_SCAFFOLD_FIX", "TALKSHOW_ADDRESSEE_FLOOR", "TALKSHOW_OPENER_VARIETY", "TALKSHOW_FLOOR_JEV")
FLOOR_ROLES = ["host", "commentator", "guest"]


def think_json(decision: str, content: str | None) -> str:
    intent = {"type": "contribute", "goal": "g", "target": None} if decision == "speak" else None
    return json.dumps({"think": "", "action": {"decision": decision, "intent": intent, "content": content,
                                               "backchannels": []}})


def act(decision: str, content: str | None):
    return parse_think_action(think_json(decision, content), floor_state="free", interval_text="x",
                              allow_backchannels=True).action


class _Env(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {"TALKSHOW_TEXT_TOKENIZER": "word"})
        env.start()
        self.addCleanup(env.stop)
        for name in FLAGS:
            os.environ.pop(name, None)


class SelectHelperTests(_Env):
    def test_select_n(self) -> None:
        self.assertEqual(select.select_n(), 1)
        self.assertFalse(select.select_enabled())
        for value, n in (("1", 1), ("0", 1), ("x", 1), ("3", 3), ("99", select.MAX_N)):
            os.environ["TALKSHOW_PANEL_SELECT"] = value
            self.assertEqual(select.select_n(), n, value)

    def test_candidates_follow_the_first_sample(self) -> None:
        extras = [act("speak", "Two."), act("silent", None), act("speak", "One."), act("speak", "Three.")]
        self.assertEqual(select.candidate_contents(act("speak", "One."), extras), ["One.", "Two.", "Three."])
        self.assertEqual(select.candidate_contents(act("silent", None), extras), [])
        self.assertEqual(select.candidate_contents(None, extras), [])

    def test_request_is_one_choice_question(self) -> None:
        body = select.build_request("Amy: rents are up.", "Ryan", "Ryan: a skeptic.", ["A.", "B."])
        q = body["questions"]["best"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(set(q["criteria"]), {"line_1", "line_2"})
        self.assertIn('"B."', q["criteria"]["line_2"])
        self.assertIn("Ryan", q["instructions"])
        self.assertIn("rents are up", body["state"])
        self.assertIn("a skeptic", body["state"])

    def test_pick_best(self) -> None:
        from agent.floor import jev_floor

        cands = ["A.", "B.", "C."]
        self.assertEqual(asyncio.run(select.pick_best("t", "commentator", ["A."]))[0], 0)
        idx, info = asyncio.run(select.pick_best("t", "commentator", cands))
        self.assertEqual((idx, info["select"]), (0, "skipped"))
        os.environ["TYPESAFE_API_KEY"] = "test"
        with mock.patch.object(jev_floor, "_post", lambda body, key: {
                "answers": {"best": {"choice": "line_3", "confidence": 0.8}}}):
            idx, info = asyncio.run(select.pick_best("t", "commentator", cands))
        self.assertEqual((idx, info["select"], info["select_conf"]), (2, "ok", 0.8))
        with mock.patch.object(jev_floor, "_post", lambda body, key: {
                "answers": {"best": {"choice": "line_9", "confidence": 0.8}}}):
            self.assertEqual(asyncio.run(select.pick_best("t", "commentator", cands))[1]["select"], "invalid")

        def boom(body, key):
            raise TimeoutError

        with mock.patch.object(jev_floor, "_post", boom):
            idx, info = asyncio.run(select.pick_best("t", "commentator", cands))
        self.assertEqual((idx, info["select_error"]), (0, "TimeoutError"))


def make_data(responses: dict[str, list[str]] | None = None) -> SimpleNamespace:
    """A show data stub whose client answers think calls by temperature (0.2: the usual sample) and logs calls."""
    calls: list[dict] = []
    queue = {k: list(v) for k, v in (responses or {}).items()}

    async def complete_text(prompt, **kw):
        calls.append({"prompt": prompt, **kw})
        key = "first" if kw.get("temperature") == 0.2 else "extra"
        return queue[key].pop(0) if queue.get(key) else think_json("speak", "Default line.")

    events = mock.Mock()
    return SimpleNamespace(
        show_history=ShowHistory(),
        scenario=SimpleNamespace(turn_control=SimpleNamespace(listen_role="host", order=["human", *FLOOR_ROLES])),
        runtime=SimpleNamespace(gemma_client=SimpleNamespace(log_text_only_event=events, complete_text=complete_text,
                                                             calls=calls)),
        room_name="test", turn_log=None)


class _Null:
    def disclose(self, *a, **k):
        return None


class LiveHookTests(_Env, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        roles = mock.patch.object(ThinkActionInteraction, "_floor_roles", staticmethod(lambda data: FLOOR_ROLES))
        roles.start()
        self.addCleanup(roles.stop)

    async def think(self, data, role: str) -> ThinkActionInteraction:
        inter = ThinkActionInteraction(_Null())
        data.show_history.append("human", "Is tipping out of hand?")
        await inter._think_and_act(data, role, history=ShowHistory(lines=list(data.show_history.lines)),
                                   speaker="human", interval_text="Is tipping out of hand?",
                                   visible_text="Is tipping out of hand?", checkpoint=1, floor_state="free")
        return inter

    async def test_flags_off_one_call_and_unchanged_reply(self) -> None:
        data = make_data({"first": [think_json("speak", "Yes, it is.")]})
        inter = await self.think(data, "commentator")
        self.assertEqual(len(data.runtime.gemma_client.calls), 1)
        self.assertEqual(inter.speak_candidates, {})
        events_before = data.runtime.gemma_client.log_text_only_event.call_count
        self.assertEqual(await inter.finish_proposed_reply(data, "commentator"), "Yes, it is.")
        self.assertEqual(data.runtime.gemma_client.log_text_only_event.call_count, events_before)
        self.assertEqual(len(data.runtime.gemma_client.calls), 1)
        self.assertNotIn("commentator", inter.actions)

    async def test_flag_samples_panelists_only(self) -> None:
        os.environ["TALKSHOW_PANEL_SELECT"] = "3"
        data = make_data({"first": [think_json("speak", "One.")],
                          "extra": [think_json("speak", "Two."), think_json("silent", None)]})
        inter = await self.think(data, "commentator")
        temps = sorted(c["temperature"] for c in data.runtime.gemma_client.calls)
        self.assertEqual(temps, [0.2, select.SAMPLE_TEMPERATURE, select.SAMPLE_TEMPERATURE])
        self.assertEqual(inter.speak_candidates["commentator"], ["One.", "Two."])
        self.assertEqual(inter.actions["commentator"].content, "One.")  # the floor decision is the first sample's
        host = make_data()
        await self.think(host, "host")
        self.assertEqual(len(host.runtime.gemma_client.calls), 1)

    async def test_first_sample_silent_gives_no_candidates(self) -> None:
        os.environ["TALKSHOW_PANEL_SELECT"] = "3"
        data = make_data({"first": [think_json("silent", None)]})
        inter = await self.think(data, "guest")
        self.assertEqual(inter.actions["guest"].decision, "silent")
        self.assertEqual(inter.speak_candidates["guest"], [])

    async def test_finish_picks(self) -> None:
        os.environ.update(TALKSHOW_PANEL_SELECT="3")
        data = make_data({"first": [think_json("speak", "One long line about tips.")],
                          "extra": [think_json("speak", "Two lines about tips here."),
                                    think_json("speak", "Three.")]})
        inter = await self.think(data, "commentator")
        with mock.patch("agent.panel.select.pick_best", mock.AsyncMock(return_value=(1, {"select": "ok"}))):
            text = await inter.finish_proposed_reply(data, "commentator")
        self.assertEqual(text, "Two lines about tips here.")
        kinds = [c.args[1] for c in data.runtime.gemma_client.log_text_only_event.call_args_list]
        self.assertIn("line select", kinds)
        self.assertNotIn("commentator", inter.speak_candidates)


class ArmTests(unittest.TestCase):
    def test_demo_arms(self) -> None:
        from eval.quality_ruler.arms import ARMS

        v1 = ARMS["demo_v1"]["live_flags"]
        want = {"demo_select": {"TALKSHOW_PANEL_SELECT": "3"}, "demo_frame": {"TALKSHOW_SHOW_FRAME": "1"},
                "demo_select_frame": {"TALKSHOW_PANEL_SELECT": "3", "TALKSHOW_SHOW_FRAME": "1"}}
        for arm, extra in want.items():
            self.assertEqual(ARMS[arm]["live_flags"], {**v1, **extra})
            self.assertTrue(ARMS[arm]["fact_pack"])
        self.assertEqual(ARMS["demo"], ARMS["demo_select_frame"])
        self.assertNotIn("TALKSHOW_PANEL_SELECT", v1)

    def test_flags_are_registered(self) -> None:
        from agent.session.ghost_replay import REPLAY_FLAGS
        from agent.telemetry.run_meta import _LATENCY_ENV
        from eval.quality_ruler.arms import LIVE_FLAG_NAMES

        for name in ("TALKSHOW_PANEL_SELECT", "TALKSHOW_SHOW_FRAME"):
            self.assertIn(name, LIVE_FLAG_NAMES)
            self.assertIn(name, REPLAY_FLAGS)
            self.assertIn(name, _LATENCY_ENV)


class ShowHarnessTests(_Env):
    def run_show(self, env: dict[str, str]) -> tuple[dict, list]:
        from eval.bench import show, systems

        seen: list[systems.Request] = []

        def respond(r: systems.Request) -> str:
            seen.append(r)
            if r.kind.startswith("think_extra"):
                return think_json("speak", "Extra line from a sample.")
            if r.kind.startswith("think:host"):
                return think_json("silent", None)
            if r.kind.startswith("think"):
                return think_json("speak", "First sample line here.")
            return systems.FakeBackend().default(r)

        backend = systems.FakeBackend(respond)
        system = systems.make_system("think_action", backend, "fast_v2_lib_force")
        os.environ.update(env)
        eps = [show.new_episode(s, "x", 0) for s in show.load_scripts(topics="tipping")]
        with contextlib.redirect_stdout(io.StringIO()):
            show.run_think_action(system, backend, eps, slots=2)
        return show.finish(eps[0]), seen

    def test_flags_off_no_extra_calls_or_fields(self) -> None:
        e, seen = self.run_show({})
        self.assertFalse(any(r.kind.startswith("think_extra") for r in seen))
        think = [l for l in e["lines"] if l.get("source") == "think"]
        self.assertTrue(think)
        for l in think:
            self.assertFalse({"candidates", "select"} & set(l))
            self.assertEqual(l["text"], "First sample line here.")

    def test_select(self) -> None:
        os.environ["TYPESAFE_API_KEY"] = "test"
        from agent.floor import jev_floor

        with mock.patch.object(jev_floor, "_post", lambda body, key: {
                "answers": {"best": {"choice": "line_2", "confidence": 0.7}}}):
            e, seen = self.run_show({"TALKSHOW_PANEL_SELECT": "3"})
        extra = [r for r in seen if r.kind.startswith("think_extra")]
        self.assertTrue(extra and all(r.kind != "think_extra:host" for r in extra))
        panel = [l for l in e["lines"] if l.get("source") == "think" and l["role"] in ("commentator", "guest")]
        self.assertTrue(panel)
        for l in panel:
            self.assertEqual(l["candidates"][0], "First sample line here.")
            self.assertEqual(l["select"]["select"], "ok")
            self.assertEqual(l["text"], l["candidates"][1])


if __name__ == "__main__":
    unittest.main()
