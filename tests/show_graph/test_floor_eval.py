"""Scoring logic of the offline floor evaluation, on hand-made listener answers."""
import json
import unittest

from eval.floor_eval.analyze import ARMS, classify, simulate, targets

ITEM = {
    "id": "m00", "motion": "Homework", "plant_at": 3,
    "clean": ["Sentence one is calm and on topic.", "Sentence two is also measured.",
              "Sentence three is fine too.", "Sentence four keeps going.", "Sentence five ends it."],
    "overclaim": "Homework is the single cause of every problem in schools.",
    "factual_error": "Homework was invented on the moon in 1999.",
    "off_topic": "My kitten learned to open the fridge last weekend.",
}
NONE = json.dumps({"action": "none", "kind": "addition", "reason": "", "evidence": "", "speech": ""})
HOST_NONE = json.dumps({"action": "none", "kind": "moderation", "reason": "", "evidence": "", "speech": ""})


def cut(evidence, kind="rebuttal"):
    return json.dumps({"action": "interrupt", "kind": kind, "reason": "r", "evidence": evidence, "speech": "s"})


def answers(variant, by_prefix, host_by_prefix=None):
    table = {}
    for prefix in range(1, 5):
        table[(ITEM["id"], variant, prefix, "commentator", 0)] = by_prefix.get(prefix, NONE)
        table[(ITEM["id"], variant, prefix, "host", 0)] = (host_by_prefix or {}).get(prefix, HOST_NONE)
    return table


ARM = {a.name: a for a in ARMS}


class FloorEvalScoringTests(unittest.TestCase):
    def run_arm(self, arm, variant, table):
        out = simulate(ITEM, variant, 0, ARM[arm], table)
        return out, classify(ITEM, variant, out)

    def test_first_sentence_schedule_cannot_see_a_later_plant(self):
        table = answers("overclaim", {3: cut("single cause of every problem")})
        self.assertEqual(self.run_arm("first_rules", "overclaim", table)[1], "missed")
        out, label = self.run_arm("every_rules", "overclaim", table)
        self.assertEqual((label, out["prefix"]), ("caught", 3))

    def test_rules_reject_unheard_quotes_but_llm_only_cuts(self):
        table = answers("factual_error", {2: cut("invented on the moon", "correction")})
        self.assertEqual(self.run_arm("every_rules_mod", "factual_error", table)[1], "missed")
        out, label = self.run_arm("every_llm_only", "factual_error", table)
        self.assertEqual(label, "premature")
        self.assertFalse(out["grounded"])

    def test_addition_never_cuts_under_rules(self):
        table = answers("clean", {1: cut("calm and on topic", "addition")})
        self.assertEqual(self.run_arm("every_rules", "clean", table)[1], "no_cut")
        self.assertEqual(self.run_arm("every_llm_only", "clean", table)[1], "false_cut")

    def test_moderator_is_the_expected_catcher_for_drift(self):
        host = {3: cut("kitten learned to open the fridge", "moderation")}
        table = answers("off_topic", {}, host)
        self.assertEqual(self.run_arm("every_rules", "off_topic", table)[1], "missed")
        self.assertEqual(self.run_arm("every_rules_mod", "off_topic", table)[1], "caught")
        table = answers("off_topic", {3: cut("kitten learned to open the fridge")})
        self.assertEqual(self.run_arm("every_rules", "off_topic", table)[1], "caught_by_other")

    def test_quote_must_point_at_the_planted_sentence(self):
        table = answers("overclaim", {3: cut("Sentence two is also measured")})
        self.assertEqual(self.run_arm("every_rules", "overclaim", table)[1], "wrong_target")
        self.assertTrue(targets("the single cause of every problem", ITEM["overclaim"]))
        self.assertFalse(targets("", ITEM["overclaim"]))
