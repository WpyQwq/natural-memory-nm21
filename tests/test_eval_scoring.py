"""Regression tests for the end-to-end answer scorer.

Both defects locked in here had produced headline numbers that were wrong enough
to send the project after the wrong bottleneck, so they get tests rather than
comments:

* whitespace sensitivity turned 19 correct multi-hop answers into failures
  (multi-hop read 24.00% when it was 100.00%);
* keyword-only refusal detection missed 12 of 13 real refusals on the
  unknown-attribute category (the axis read 0.00% when it was 52.00%).
"""

from __future__ import annotations

import unittest

from V2_dpskw.eval_scoring import is_refusal, score_case, squash
from V2_dpskw.eval_end_to_end_memory import evidence_write_order


def case(acceptable, answerable=True):
    return {"acceptable": acceptable, "answerable": answerable}


class SquashTest(unittest.TestCase):
    def test_removes_ascii_and_full_width_whitespace(self):
        self.assertEqual(squash("值班人 -5259"), squash("值班人-5259"))
        self.assertEqual(squash("值班人\u3000-5259"), squash("值班人-5259"))
        self.assertEqual(squash("每周三  22:00"), squash("每周三 22:00"))

    def test_is_case_insensitive(self):
        self.assertEqual(squash("REF-9x6uye"), squash("ref-9X6UYE"))

    def test_keeps_meaningful_punctuation(self):
        # Punctuation inside anchors is meaningful and must survive.
        self.assertNotEqual(squash("10.20.3.7:5432"), squash("1020375432"))


class WhitespaceInsensitiveMatchingTest(unittest.TestCase):
    def test_spaced_identifier_still_scores_correct(self):
        scored = score_case(case(["值班人-5259"]), "负责该项目的同事对应的值班人是 值班人 -5259。")
        self.assertTrue(scored["correct"])
        self.assertEqual(scored["matched"], ["值班人-5259"])

    def test_wrong_identifier_is_still_wrong(self):
        scored = score_case(case(["值班人-5259"]), "负责该项目的同事对应的值班人是 值班人-5266。")
        self.assertFalse(scored["correct"])

    def test_missing_value_is_wrong(self):
        scored = score_case(case(["值班人-5259"]), "负责该项目的同事对应的值班人是 值班人。")
        self.assertFalse(scored["correct"])


class RefusalDetectionTest(unittest.TestCase):
    def test_recognises_the_wording_the_model_actually_uses(self):
        for reply in (
            "关于您的 weekly 会议安排，当前长期记忆中未包含相关信息，无法回答。",
            "关于您这周按什么表轮，当前长期记忆中未包含该信息。已知信息如下：user；培训周期；78%",
            "关于报警触发条件，当前记忆中没有相关记录。已知信息：user；保险到期；A-7719。",
            "您的协议编号未知。",
            "关于变更时段，当前长期记忆中未包含具体信息，证据不足，明确说不知道。",
        ):
            with self.subTest(reply=reply):
                self.assertTrue(is_refusal(reply))

    def test_does_not_fire_on_asserted_values(self):
        for reply in (
            "您的培训周期是 32GB。",
            "您的东西放在 63.2% 的仓库库位。",
            "我固定每周三 22:00 做上线。",
        ):
            with self.subTest(reply=reply):
                self.assertFalse(is_refusal(reply))

    def test_unanswerable_case_accepts_a_refusal_in_any_wording(self):
        scored = score_case(case([], answerable=False), "当前长期记忆中未包含相关信息，无法回答。")
        self.assertTrue(scored["correct"])
        self.assertTrue(scored["abstained"])

    def test_unanswerable_case_still_fails_when_a_value_is_asserted(self):
        scored = score_case(case([], answerable=False), "您的培训周期是 32GB。")
        self.assertFalse(scored["correct"])

    def test_answerable_case_that_refuses_is_a_false_refusal(self):
        scored = score_case(case(["32GB"]), "关于您上手所需的时间，当前长期记忆中未包含此信息，无法回答。")
        self.assertFalse(scored["correct"])
        self.assertTrue(scored["wrongly_abstained"])

    def test_answerable_case_with_hedge_but_correct_value_is_not_false_refusal(self):
        scored = score_case(
            case(["凌晨 03:30"]),
            "那杯偏爱 凌晨 03:30 的口味是未知，因为长期记忆中仅记录了时间（凌晨 03:30）和饮品名称。",
        )
        self.assertTrue(scored["correct"])
        self.assertFalse(scored["wrongly_abstained"])


class EvidenceWriteOrderTest(unittest.TestCase):
    """The harness write order is a measurement decision, so it gets a test.

    Two distinct failures lived here: answering facts written too early made
    `update_conflict` unsolvable by construction, and reserving no slots for them
    dropped the answer entirely on `noise_context` (72.00% -> 0.00%).
    """

    def test_answering_facts_are_written_last(self):
        case = {"positives": ["ANSWER"], "facts": ["d1", "d2", "ANSWER", "d3"]}
        order = evidence_write_order(case, 6, answer_last=True)
        self.assertEqual(order[-1], "ANSWER")
        self.assertEqual(len(order), 4)

    def test_answering_facts_survive_more_distractors_than_slots(self):
        # noise_context: 8 distractors against 1 answering fact, 6 write slots.
        case = {"positives": ["ANSWER"], "facts": ["d%d" % i for i in range(8)] + ["ANSWER"]}
        order = evidence_write_order(case, 6, answer_last=True)
        self.assertEqual(len(order), 6)
        self.assertIn("ANSWER", order)
        self.assertEqual(order[-1], "ANSWER")
        self.assertEqual(order.count("ANSWER"), 1)

    def test_all_answering_facts_survive(self):
        case = {"positives": ["A1", "A2"], "facts": ["d1", "d2", "d3", "d4", "A1", "A2"]}
        order = evidence_write_order(case, 6, answer_last=True)
        self.assertIn("A1", order)
        self.assertIn("A2", order)
        self.assertEqual(order[-2:], ["A1", "A2"])

    def test_answer_first_reproduces_the_old_behaviour(self):
        case = {"positives": ["ANSWER"], "facts": ["d1", "d2", "ANSWER"]}
        self.assertEqual(evidence_write_order(case, 6, answer_last=False), ["ANSWER", "d1", "d2"])

    def test_never_exceeds_the_slot_budget(self):
        case = {"positives": ["A"], "facts": ["d%d" % i for i in range(20)] + ["A"]}
        self.assertLessEqual(len(evidence_write_order(case, 6)), 6)
        self.assertLessEqual(len(evidence_write_order(case, 2)), 2)


if __name__ == "__main__":
    unittest.main()
