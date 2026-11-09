"""Addressee-aware floor (TALKSHOW_ADDRESSEE_FLOOR) — pure logic, no LiveKit."""

from __future__ import annotations

import os
import unittest
from unittest import mock


from agent.floor.addressee import addressed_role, addressee_floor_enabled, apply_addressee
from agent.floor.floor_parser import HandRaiseResult
from agent.floor.hand_raise_queue import HandRaiseQueue

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


class ApplyAddresseeTests(unittest.TestCase):
    def _poll(self, guest: bool, commentator: bool) -> dict[str, HandRaiseResult]:
        return {"guest": HandRaiseResult("guest", guest), "commentator": HandRaiseResult("commentator", commentator)}

    def test_tie_goes_to_addressee(self) -> None:
        # Pilot failure mode: both raise, priority favours the commentator.
        poll, prio = apply_addressee(self._poll(True, True), ["commentator", "guest"], "guest")
        winner, yes, tie = HandRaiseQueue().apply_poll_batch(poll, prio)
        self.assertEqual(winner, "guest")
        self.assertTrue(tie)

    def test_addressee_who_did_not_raise_still_gets_floor(self) -> None:
        poll, prio = apply_addressee(self._poll(False, True), ["commentator", "guest"], "guest")
        winner, _yes, _tie = HandRaiseQueue().apply_poll_batch(poll, prio)
        self.assertEqual(winner, "guest")

    def test_no_addressee_is_unchanged(self) -> None:
        raw = self._poll(True, True)
        poll, prio = apply_addressee(raw, ["commentator", "guest"], None)
        self.assertIs(poll, raw)
        winner, _yes, _tie = HandRaiseQueue().apply_poll_batch(poll, prio)
        self.assertEqual(winner, "commentator")


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


class RotationTests(unittest.TestCase):
    """Regression: after the addressee speaks, the OTHER panelist must win the next tie."""

    def _poll(self) -> dict:
        return {"guest": HandRaiseResult("guest", True), "commentator": HandRaiseResult("commentator", True)}

    def test_next_tie_goes_to_the_other_panelist(self) -> None:
        from agent.floor.addressee import pick_with_addressee

        prio = ["commentator", "guest"]
        w1, _y, _t, prio = pick_with_addressee(HandRaiseQueue(), self._poll(), prio, "commentator")
        self.assertEqual(w1, "commentator")
        # next round: a panelist has spoken → no addressee → plain tie-break on the rotated priority
        w2, _y, _t, prio = pick_with_addressee(HandRaiseQueue(), self._poll(), prio, None)
        self.assertEqual(w2, "guest")

    def test_addressee_not_first_in_priority(self) -> None:
        from agent.floor.addressee import pick_with_addressee

        w1, _y, _t, prio = pick_with_addressee(HandRaiseQueue(), self._poll(), ["commentator", "guest"], "guest")
        self.assertEqual(w1, "guest")
        w2, *_ = pick_with_addressee(HandRaiseQueue(), self._poll(), prio, None)
        self.assertEqual(w2, "commentator")
