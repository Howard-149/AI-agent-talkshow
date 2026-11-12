"""Fact pack on Think & Action panel turns: the think prompt gets the worker's block (flag default off)."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest

import agent.agents  # noqa: F401  (import order the app uses; avoids a pre-existing circular import)
from agent.interaction import ThinkActionInteraction
from agent.panel import fact_pack
from agent.panel.concrete import CONCRETE
from agent.panel.think_prompt import HEADER
from agent.show.show_history import ShowHistory
from tests.panel.test_concrete import FakeGemma, live_data
from tests.panel.test_fact_pack import clean_env, draft, write_pack

THINK_JSON = '{"think": "", "action": {"decision": "silent", "intent": null, "content": null}}'


class _Gemma(FakeGemma):
    def log_text_only_event(self, *args, **kwargs) -> None:
        return None


class _Consumer:
    def stop(self):
        return None, False


def think_prompt(env: dict, role: str = "guest") -> tuple[str, object]:
    gemma = _Gemma([THINK_JSON])
    data = live_data(gemma)
    p = clean_env(env)
    try:
        heard = data.show_history.lines[-1].text
        asyncio.run(
            ThinkActionInteraction(_Consumer())._think_and_act(
                data,
                role,
                history=ShowHistory(lines=list(data.show_history.lines)),
                speaker="human",
                interval_text=heard,
                visible_text=heard,
                checkpoint=1,
                floor_state="free",
            )
        )
        block = fact_pack.pack_block(data)[0]
    finally:
        p.stop()
    return gemma.calls[-1][1], block


class ThinkActionFactPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pack = write_pack(self.tmp.name)

    def test_flag_off_adds_nothing(self) -> None:
        prompt, _ = think_prompt({})
        self.assertNotIn(HEADER, prompt)
        self.assertNotIn(fact_pack.HEADER, prompt)

    def test_think_prompt_gets_the_same_block_as_the_worker(self) -> None:
        env = {"TALKSHOW_FACT_PACK": self.pack}
        prompt, block = think_prompt(env)
        self.assertTrue(block.strip().startswith(fact_pack.HEADER))
        self.assertIn(block.strip(), prompt)
        self.assertTrue(prompt.rstrip().endswith(fact_pack.RULE))
        _d, worker_call, _g = draft(env)
        self.assertIn(block.strip(), worker_call[1])

    def test_concrete_instruction_points_at_the_facts(self) -> None:
        prompt, _ = think_prompt({"TALKSHOW_FACT_PACK": self.pack, "TALKSHOW_PANEL_CONCRETE": "1"})
        self.assertTrue(prompt.endswith(fact_pack.CONCRETE_WITH_PACK))
        self.assertNotIn(CONCRETE, prompt)
        self.assertLess(prompt.index(fact_pack.HEADER), prompt.index(fact_pack.CONCRETE_WITH_PACK))

    def test_host_think_gets_no_facts(self) -> None:
        prompt, _ = think_prompt({"TALKSHOW_FACT_PACK": self.pack}, role="host")
        self.assertNotIn(fact_pack.HEADER, prompt)

    def test_offline_think_request_uses_the_topic_id(self) -> None:
        from eval.think_action_prompt import think_request

        p = clean_env({"TALKSHOW_FACT_PACK": self.pack})
        try:
            prefix = {"lines": [{"role_id": "host", "speaker": "Lessac", "text": "Welcome.", "procedural": False}]}
            req = think_request(prefix, "guest", human_text="Is tipping out of hand?", topic_id="tipping")
        finally:
            p.stop()
        self.assertIn("$2.13 an hour", req["messages"][-1]["content"])
        self.assertEqual(req["fact_pack"]["fact_pack_topic"], "tipping")
        self.assertEqual(req["fact_pack"]["fact_pack_match"], "topic_id")


if __name__ == "__main__":
    unittest.main()
