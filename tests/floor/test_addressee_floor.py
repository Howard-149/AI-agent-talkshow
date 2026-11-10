"""Addressee-aware floor (TALKSHOW_ADDRESSEE_FLOOR) — pure logic, no LiveKit."""

from __future__ import annotations

import os
import unittest
from unittest import mock


from agent.floor.addressee import addressed_role, addressee_floor_enabled

NAMES = {"guest": "Amy", "commentator": "Ryan"}


class AddressedRoleTests(unittest.TestCase):
    def test_vocative_opening(self) -> None:
        self.assertEqual(addressed_role("Amy, but the averages still say…", NAMES), "guest")
        self.assertEqual(addressed_role("Ryan — is that fair?", NAMES), "commentator")
        self.assertEqual(addressed_role("But Amy: what about rent?", NAMES), "guest")

    def test_vocative_closing_question(self) -> None:
        self.assertEqual(addressed_role("What do you make of that, Ryan?", NAMES), "commentator")

    def test_mention_is_not_address(self) -> None:
        self.assertIsNone(addressed_role("I agree with what Ryan said about cost.", NAMES))
        self.assertIsNone(addressed_role("Nobody addressed anyone here.", NAMES))

    def test_two_addressees_is_ambiguous(self) -> None:
        self.assertIsNone(addressed_role("Amy, and what do you think, Ryan?", NAMES))


class FlagTests(unittest.TestCase):
    def test_default_off(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TALKSHOW_ADDRESSEE_FLOOR", None)
            self.assertFalse(addressee_floor_enabled())

    def test_on(self) -> None:
        with mock.patch.dict(os.environ, {"TALKSHOW_ADDRESSEE_FLOOR": "1"}):
            self.assertTrue(addressee_floor_enabled())



if __name__ == "__main__":
    unittest.main()
