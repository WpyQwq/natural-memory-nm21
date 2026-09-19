from __future__ import annotations

import unittest

import torch

from V2_dpskw.benchmark_dirty_real_corpus_4b import (
    _failure_stage,
    _is_refusal_strict,
)
from V2_dpskw.memory_os_v2 import MemoryRouterV2, PagedMemoryBankV2
from V2_dpskw.qwen_integration import (
    _looks_like_grounded_memory_query,
    format_memory_evidence,
)


class FailureDrivenV2Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.router = MemoryRouterV2(16, router_dim=8, num_heads=2, max_hops=3)

    def test_grounding_guard_is_scoped_to_personal_and_project_facts(self) -> None:
        self.assertTrue(_looks_like_grounded_memory_query("当前项目的模型叫什么"))
        self.assertTrue(_looks_like_grounded_memory_query("README 是否记录了重启方式"))
        self.assertFalse(_looks_like_grounded_memory_query("北京今天会下雨吗"))

    def test_unknown_wording_is_a_hard_refusal_signal(self) -> None:
        self.assertTrue(_is_refusal_strict("仓库中没有列出任何具体的部署区域。"))
        self.assertTrue(_is_refusal_strict("这个字段没有出现在保存的记录里，我无法确认。"))

    def test_structured_evidence_keeps_raw_fragment_first(self) -> None:
        evidence = format_memory_evidence(
            "用户给当前模型定名：Natural Memory v1。",
            entity="user",
            attribute="模型名称",
            value="Natural Memory v1",
        )
        self.assertLess(evidence.index("事实"), evidence.index("已确认值"))
        self.assertIn("已确认值：Natural Memory v1", evidence)

    def test_failure_stage_classification_is_observable(self) -> None:
        self.assertEqual(
            _failure_stage(
                {
                    "correct": False,
                    "answerable": True,
                    "retrieval_target_found": False,
                    "response": "不知道",
                }
            ),
            "routing_miss",
        )
        self.assertEqual(
            _failure_stage(
                {
                    "correct": False,
                    "answerable": True,
                    "retrieval_target_found": True,
                    "expected_anchors": ["H:\\Memory", "Natural Memory"],
                    "response": "项目在 H:\\Memory。",
                }
            ),
            "evidence_fusion",
        )
        self.assertEqual(
            _failure_stage(
                {
                    "correct": False,
                    "answerable": True,
                    "retrieval_target_found": True,
                    "expected_anchors": ["H:\\Memory"],
                    "response": "没有记录。",
                }
            ),
            "generation_control",
        )
        self.assertEqual(
            _failure_stage(
                {
                    "correct": False,
                    "answerable": False,
                    "retrieval_target_found": False,
                    "response": "我无法确认。",
                }
            ),
            "unknown_refusal_control",
        )

    def test_attribute_alias_routes_without_exact_source_wording(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=2,
            top_k_pages=2,
            top_k_records=2,
            max_hops=2,
        )
        record, _ = bank.write(
            text="记忆主体优先放在进程内存，显存只保存热点记录。",
            key=torch.randn(16),
            entity="user",
            attribute="记忆存储优先级",
            value="DRAM",
            confidence=0.98,
        )
        selected, _ = bank.query(
            query_key=torch.randn(16),
            query_text="容量不够时才降级，平时优先使用哪层内存？",
            top_k_pages=2,
            top_k_records=2,
        )
        self.assertIn(record.record_id, {item.record_id for item in selected})

    def test_cjk_phrase_alias_reaches_operational_record(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=2,
            top_k_pages=2,
            top_k_records=2,
        )
        record, _ = bank.write(
            text="当前 token 不会对一百万个 slot 做全量注意力。",
            key=torch.randn(16),
            semantic_key=torch.randn(16),
            entity="README_NATURAL_MEMORY_V2.md",
            attribute="operational:1",
            value="Top-K；全量注意力",
            confidence=0.98,
        )
        selected, _ = bank.query(
            query_key=torch.randn(16),
            query_text="百万 slot 会不会进行全量注意力？",
            top_k_pages=2,
            top_k_records=2,
        )
        self.assertIn(record.record_id, {item.record_id for item in selected})

    def test_ambiguous_symbol_fans_out_but_stays_bounded(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=1,
            top_k_pages=8,
            top_k_records=1,
            max_pages=64,
        )
        expected = set()
        for index in range(6):
            record, _ = bank.write(
                text=f"第 {index} 个文件定义了 run 函数。",
                key=torch.randn(16),
                entity=f"module_{index}.py",
                attribute="symbol:run",
                value="run",
                confidence=0.9,
            )
            expected.add(record.record_id)
        selected, _ = bank.query(
            query_key=torch.randn(16),
            query_text="run 函数分别在哪些文件中定义？",
            top_k_pages=8,
            top_k_records=1,
        )
        selected_ids = {item.record_id for item in selected}
        self.assertEqual(selected_ids, expected)
        self.assertLessEqual(len(selected), 8)


if __name__ == "__main__":
    unittest.main()
