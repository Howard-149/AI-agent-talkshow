"""LLM host decision: tool schema, prompt and tool-call parsing (no model)."""

from __future__ import annotations

import unittest

from agent.floor.host_decide import (
    CALL_ON,
    CLOSE,
    TO_HUMAN,
    Raise,
    build_tools,
    parse_decision,
    render_decide_prompt,
)

ROLES = {"ryan": "commentator", "amy": "guest"}


def names(tools):
    return [t["function"]["name"] for t in tools]


class HostDecideTests(unittest.TestCase):
    def test_tools(self) -> None:
        tools = build_tools(["Ryan", "Amy"], human_must_go=False)
        self.assertEqual(names(tools), [CALL_ON, TO_HUMAN, CLOSE])
        enum = tools[0]["function"]["parameters"]["properties"]["panelist"]["enum"]
        self.assertEqual(enum, ["Ryan", "Amy"])
        # A raised human hand: handing back is the only option.
        self.assertEqual(names(build_tools(["Ryan", "Amy"], human_must_go=True)), [TO_HUMAN])

    def test_prompt(self) -> None:
        p = render_decide_prompt(
            host="Lessac",
            raises=[Raise("Ryan", True, "HR's role"), Raise("Amy", False)],
            human=Raise("Human guest", True),
            turns={"Ryan": 2},
            last_speaker="Ryan",
            human_must_go=False,
        )
        self.assertIn('- Ryan: RAISED — wants to talk about "HR\'s role"', p)
        self.assertIn("- Amy: not raised", p)
        self.assertIn("- Human guest: RAISED", p)
        self.assertIn("Ryan ×2", p)
        must = render_decide_prompt(
            host="Lessac", raises=[], human=Raise("Human guest", True), turns={},
            last_speaker="Amy", human_must_go=True,
        )
        self.assertIn("Give them the floor now", must)

    def test_parse_call_on(self) -> None:
        d, why = parse_decision(
            [(CALL_ON, {"panelist": "Amy", "line": "is that fair to the team?"})],
            name_to_role=ROLES, human_must_go=False,
        )
        self.assertEqual((d.role, why), ("guest", ""))
        self.assertTrue(d.line.startswith("Amy, "))  # name added when missing

    def test_parse_rejects(self) -> None:
        cases = [
            ([], False, "no_tool_call"),
            ([(CALL_ON, {"panelist": "Bob", "line": "Bob, go."})], False, "unknown_panelist:Bob"),
            ([(CALL_ON, {"panelist": "Amy", "line": ""})], False, "empty_line"),
            ([(CALL_ON, {"panelist": "Amy", "line": "Amy, go on."})], True, "human_must_go_but_" + CALL_ON),
            ([("dance", {"line": "Let's dance."})], False, "unknown_tool:dance"),
        ]
        for calls, must, why in cases:
            d, got = parse_decision(calls, name_to_role=ROLES, human_must_go=must)
            self.assertIsNone(d)
            self.assertEqual(got, why)

    def test_parse_human_and_close(self) -> None:
        d, _ = parse_decision([(TO_HUMAN, {"line": "[next:human] Back to you."})],
                              name_to_role=ROLES, human_must_go=True)
        self.assertEqual((d.role, d.line), ("human", "Back to you."))
        d, _ = parse_decision([(CLOSE, {"line": "That's our time."})],
                              name_to_role=ROLES, human_must_go=False)
        self.assertEqual(d.role, "close")


if __name__ == "__main__":
    unittest.main()
