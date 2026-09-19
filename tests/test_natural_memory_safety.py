import unittest

from V2_dpskw.qwen_integration import infer_memory_metadata


class NaturalMemorySafetyTest(unittest.TestCase):
    def test_explicit_fact_gets_structured_version_metadata(self) -> None:
        item = infer_memory_metadata("请记住：我的常用时区是Asia/Shanghai。")
        self.assertEqual(item["kind"], "fact")
        self.assertEqual(item["entity"], "user")
        self.assertEqual(item["attribute"], "常用时区")
        self.assertEqual(item["value"], "Asia/Shanghai")
        self.assertTrue(item["should_write"])

    def test_correction_is_structured_but_not_collapsed_into_a_question(self) -> None:
        item = infer_memory_metadata("更正一下：我的工作地点改为深圳，旧值不再有效。")
        self.assertEqual(item["kind"], "correction")
        self.assertEqual(item["attribute"], "工作地点")
        self.assertEqual(item["value"], "深圳")

    def test_explicit_forget_never_becomes_a_new_record(self) -> None:
        item = infer_memory_metadata("请删除关于我的水果偏好的记忆。")
        self.assertEqual(item["kind"], "forget")
        self.assertFalse(item["should_write"])
        self.assertEqual(item["attribute"], "水果偏好")

    def test_question_and_hypothetical_are_not_facts(self) -> None:
        self.assertEqual(infer_memory_metadata("我的常住城市是什么？")["kind"], "query")
        self.assertEqual(infer_memory_metadata("如果我的常住城市改成成都，会怎样？")["kind"], "query")


if __name__ == "__main__":
    unittest.main()
