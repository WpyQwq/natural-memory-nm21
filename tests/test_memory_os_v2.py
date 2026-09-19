from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import torch

from V2_dpskw.memory_os_v2 import (
    KVBudgetManagerV2,
    MemoryOSV2,
    MemoryRouterV2,
    PagedMemoryBankV2,
    STATUS_ACTIVE,
    STATUS_QUARANTINED,
    STATUS_SUPERSEDED,
)
from V2_dpskw.tiered_memory_store_v2 import TieredMemoryStoreV2


class MemoryOSV2Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.router = MemoryRouterV2(16, router_dim=8, num_heads=2, max_hops=3)
        self.bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=2,
            max_pages=512,
            hot_pages=2,
            top_k_pages=2,
            top_k_records=4,
            max_hops=3,
            coarse_index_bits=8,
        )

    def test_router_scores_and_compressed_address(self) -> None:
        query = torch.randn(4, 16)
        candidates = torch.randn(4, 5, 16)
        output = self.router(query, candidates)
        self.assertEqual(tuple(output["scores"].shape), (4, 5))
        self.assertEqual(tuple(output["head_scores"].shape), (4, 5, 2))
        self.assertEqual(tuple(self.router.encode_key(query).shape), (4, 8))

    def test_write_version_and_conflict_resolution(self) -> None:
        first, first_action = self.bank.write(
            text="我住在上海",
            key=torch.randn(16),
            entity="user",
            attribute="city",
            value="上海",
            confidence=0.9,
        )
        second, second_action = self.bank.write(
            text="我搬到了杭州",
            key=torch.randn(16),
            entity="user",
            attribute="city",
            value="杭州",
            confidence=0.95,
        )
        self.assertEqual(first_action, "inserted")
        self.assertEqual(second_action, "updated")
        self.assertEqual(first.status, STATUS_SUPERSEDED)
        self.assertEqual(second.status, STATUS_ACTIVE)
        self.assertEqual(second.version, 1)
        self.assertEqual(self.bank.active_by_conflict["user::city"], second.record_id)

    def test_quarantine_and_approval(self) -> None:
        record, action = self.bank.write(
            text="未经确认的推断",
            key=torch.randn(16),
            trusted=False,
            confidence=0.1,
        )
        self.assertEqual(action, "quarantined")
        self.assertEqual(record.status, STATUS_QUARANTINED)
        self.assertNotIn(record.record_id, self.bank.records)
        approved = self.bank.approve(record.record_id)
        self.assertEqual(approved.status, STATUS_ACTIVE)
        self.assertIn(approved.record_id, self.bank.records)

    def test_multi_hop_and_slot_replacement(self) -> None:
        second, _ = self.bank.write(text="项目的第二个节点", key=torch.randn(16), slot_index=2)
        third, _ = self.bank.write(text="项目的第三个节点", key=torch.randn(16), slot_index=3)
        first, _ = self.bank.write(
            text="项目的第一个节点",
            key=torch.randn(16),
            related_ids=[second.record_id, third.record_id],
            slot_index=1,
        )
        replacement, action = self.bank.write(
            text="项目的第一个节点修正版",
            key=torch.randn(16),
            related_ids=[second.record_id],
            slot_index=1,
        )
        self.assertEqual(action, "updated")
        self.assertEqual(first.status, STATUS_SUPERSEDED)
        self.assertEqual(replacement.status, STATUS_ACTIVE)
        records, decision = self.bank.query(
            query_key=replacement.key,
            top_k_pages=2,
            top_k_records=4,
            max_hops=3,
        )
        ids = {record.record_id for record in records}
        self.assertIn(replacement.record_id, ids)
        self.assertGreaterEqual(decision.hop_count, 1)

    def test_coarse_index_bounds_candidate_pages(self) -> None:
        for index in range(300):
            key = torch.zeros(16)
            key[index % 16] = 1.0
            key[(index * 7 + 3) % 16] += 0.05
            self.bank.write(text=f"memory-{index}", key=key, importance=0.2)
        query_key = torch.zeros(16)
        query_key[3] = 1.0
        self.bank.query(query_key=query_key, top_k_pages=2, top_k_records=2)
        stats = self.bank.stats()
        self.assertGreater(stats["pages"], 128)
        self.assertLess(stats["last_coarse_candidates"], stats["pages"])

    def test_export_and_restore(self) -> None:
        record, _ = self.bank.write(
            text="可持久化事实",
            key=torch.randn(16),
            token_ids=torch.tensor([4, 5, 6]),
            token_mask=torch.tensor([True, True, True]),
        )
        payload = self.bank.export_payload()
        restored = PagedMemoryBankV2.from_payload(payload, router=self.router)
        self.assertEqual(restored.stats()["active_records"], 1)
        self.assertTrue(torch.equal(restored.records[record.record_id].token_ids, torch.tensor([4, 5, 6])))
        self.assertEqual(restored.records[record.record_id].page_id, record.page_id)

    def test_lazy_capacity_is_bounded(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=1,
            max_pages=2,
            hot_pages=0,
            coarse_index_bits=8,
        )
        bank.write(text="容量一", key=torch.randn(16))
        bank.write(text="容量二", key=torch.randn(16))
        self.assertEqual(bank.stats()["pages"], 2)
        with self.assertRaises(RuntimeError):
            bank.write(text="容量三", key=torch.randn(16))

    def test_memory_os_and_kv_budget(self) -> None:
        os_v2 = MemoryOSV2(16, router=self.router)
        record, action = os_v2.write(
            text="可靠事实",
            key=torch.randn(16),
            importance=0.9,
            confidence=0.9,
        )
        self.assertEqual(action, "inserted")
        self.assertIn(record.record_id, os_v2.bank.records)
        budget = KVBudgetManagerV2(max_tokens=128, hard_max_tokens=512, keep_recent_tokens=32)
        self.assertFalse(budget.needs_compaction(100))
        self.assertTrue(budget.needs_compaction(120))
        self.assertEqual(budget.overflow(140), 12)

    def test_batch_context_records_keep_all_chunks_active(self) -> None:
        os_v2 = MemoryOSV2(16, router=self.router)
        output = os_v2.write_batch(
            [
                {
                    "text": "context_chunk:0:0:4",
                    "key": torch.randn(16),
                    "memory_type": "context_chunk",
                    "importance": 0.55,
                    "confidence": 0.8,
                    "trusted": True,
                    "force": True,
                },
                {
                    "text": "context_chunk:0:4:8",
                    "key": torch.randn(16),
                    "memory_type": "context_chunk",
                    "importance": 0.55,
                    "confidence": 0.8,
                    "trusted": True,
                    "force": True,
                },
            ]
        )
        self.assertEqual(len(output), 2)
        self.assertEqual(os_v2.stats()["active_records"], 2)

    def test_independent_fragments_keep_semantic_keys_across_export(self) -> None:
        first, _ = self.bank.write(
            text="项目负责人是成员A",
            key=torch.randn(16),
            semantic_key=torch.randn(16),
            slot_index=-1,
            token_ids=torch.tensor([1, 2, 3]),
        )
        second, _ = self.bank.write(
            text="成员A的工作代号是H7",
            key=torch.randn(16),
            semantic_key=torch.randn(16),
            slot_index=-1,
            token_ids=torch.tensor([4, 5, 6]),
        )
        self.assertEqual(self.bank.stats()["active_records"], 2)
        payload = self.bank.export_payload()
        restored = PagedMemoryBankV2.from_payload(payload, router=self.router)
        self.assertEqual(restored.stats()["active_records"], 2)
        self.assertIsNotNone(restored.records[first.record_id].semantic_key)
        self.assertIsNotNone(restored.records[second.record_id].semantic_key)

    def test_bounded_record_reranker_runs_inside_selected_pages(self) -> None:
        first, _ = self.bank.write(
            text="候选一",
            key=torch.tensor([1.0] + [0.0] * 15),
            semantic_key=torch.ones(16),
        )
        second, _ = self.bank.write(
            text="候选二",
            key=torch.tensor([1.0] + [0.0] * 15),
            semantic_key=torch.ones(16) * 2,
        )

        def scorer(query, candidates):
            # The callback receives only the two records in the selected page.
            return torch.tensor([0.1, 0.9], device=query.device)

        self.bank.record_scorer = scorer
        records, _ = self.bank.query(
            query_key=torch.tensor([1.0] + [0.0] * 15),
            query_text="候选",
            top_k_pages=1,
            top_k_records=1,
        )
        self.assertEqual(records[0].record_id, second.record_id)

    def test_explicit_entity_address_beats_semantic_collision(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=1,
            hot_pages=0,
            top_k_pages=1,
            top_k_records=1,
            coarse_index_bits=8,
        )
        distractor, _ = bank.write(
            text="评估用户00056的档案代号为x",
            key=torch.ones(16),
            entity="评估用户00056",
            attribute="档案代号",
            value="x",
        )
        target, _ = bank.write(
            text="评估用户00062的档案代号为w",
            key=torch.ones(16),
            entity="评估用户00062",
            attribute="档案代号",
            value="w",
        )
        records, _ = bank.query(
            query_key=torch.ones(16),
            query_text="只查询评估用户00062的档案代号",
            top_k_pages=1,
            top_k_records=1,
        )
        self.assertEqual(records[0].record_id, target.record_id)
        self.assertNotEqual(records[0].record_id, distractor.record_id)

    def test_symbol_address_returns_bounded_ambiguous_candidates(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=1,
            hot_pages=0,
            top_k_pages=4,
            top_k_records=1,
            coarse_index_bits=8,
        )
        records = []
        for path in ("first.py", "second.py", "third.py"):
            record, _ = bank.write(
                text=f"def run in {path}",
                key=torch.ones(16),
                semantic_key=torch.ones(16),
                entity=path,
                attribute="symbol:run",
                value=path,
                token_ids=torch.tensor([1, 2, 3]),
            )
            records.append(record)
        selected, _ = bank.query(
            query_key=torch.ones(16),
            query_text="帮我定位项目里的 run 函数",
            top_k_pages=4,
            top_k_records=1,
        )
        self.assertEqual(
            {record.record_id for record in selected},
            {record.record_id for record in records},
        )

    def test_explicit_file_address_filters_same_named_symbol(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=1,
            hot_pages=0,
            top_k_pages=8,
            top_k_records=1,
            coarse_index_bits=8,
        )
        target, _ = bank.write(
            text="def main in target.py",
            key=torch.ones(16),
            semantic_key=torch.ones(16),
            entity="target.py",
            attribute="symbol:main",
            value="target.py",
            token_ids=torch.tensor([1, 2, 3]),
        )
        distractor, _ = bank.write(
            text="def main in other.py",
            key=torch.ones(16),
            semantic_key=torch.ones(16),
            entity="other.py",
            attribute="symbol:main",
            value="other.py",
            token_ids=torch.tensor([1, 2, 3]),
        )
        selected, _ = bank.query(
            query_key=torch.ones(16),
            query_text="请定位 target.py 里的 main 函数",
            top_k_pages=8,
            top_k_records=1,
        )
        self.assertEqual([record.record_id for record in selected], [target.record_id])
        self.assertNotIn(distractor.record_id, {record.record_id for record in selected})

    def test_identifier_subtokens_reach_operational_evidence(self) -> None:
        target, _ = self.bank.write(
            text="代码定义 DEFAULT_MEMORY_RESET_TOKEN，用于清空记忆",
            key=torch.ones(16),
            semantic_key=torch.ones(16),
            entity="qwen_integration.py",
            attribute="operational:reset",
            value="qwen_integration.py",
            token_ids=torch.tensor([1, 2, 3]),
        )
        selected, _ = self.bank.query(
            query_key=torch.ones(16),
            query_text="模型代码里的默认 reset token 是什么",
            top_k_pages=2,
            top_k_records=1,
        )
        self.assertTrue(selected)
        self.assertEqual(selected[0].record_id, target.record_id)

    def test_exact_entity_attribute_filters_injected_distractor(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=8,
            hot_pages=0,
            top_k_pages=1,
            top_k_records=2,
            coarse_index_bits=8,
        )
        target, _ = bank.write(
            text="评估用户00119的常用语言为X",
            key=torch.ones(16),
            entity="评估用户00119",
            attribute="常用语言",
            value="X",
        )
        distractor, _ = bank.write(
            text="评估用户00009的常用语言为T",
            key=torch.ones(16),
            entity="评估用户00009",
            attribute="常用语言",
            value="T",
        )
        records, _ = bank.query(
            query_key=torch.ones(16),
            query_text="请查询评估用户00119的常用语言",
            top_k_pages=1,
            top_k_records=2,
        )
        self.assertEqual([record.record_id for record in records], [target.record_id])
        self.assertNotIn(distractor.record_id, [record.record_id for record in records])

    def test_distinctive_address_reaches_target_beyond_coarse_page_limit(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=1,
            max_pages=512,
            hot_pages=0,
            top_k_pages=1,
            top_k_records=1,
            coarse_index_bits=8,
        )
        target = None
        for index in range(180):
            record, _ = bank.write(
                text=f"代码文件 file_{index}.py 的函数 fn_{index}",
                key=torch.ones(16),
                entity=f"file_{index}.py",
                attribute=f"symbol:fn_{index}",
                value=f"file_{index}.py",
            )
            if index == 179:
                target = record
        assert target is not None
        records, _ = bank.query(
            query_key=torch.ones(16),
            query_text="请查找 file_179.py 中的 fn_179 定义",
            top_k_pages=1,
            top_k_records=1,
        )
        self.assertTrue(records)
        self.assertEqual(records[0].record_id, target.record_id)

    def test_numeric_suffix_does_not_cross_entity_namespace(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=1,
            hot_pages=0,
            top_k_pages=1,
            top_k_records=2,
            coarse_index_bits=8,
        )
        training, _ = bank.write(
            text="训练用户00006的档案代号为F",
            key=torch.ones(16),
            entity="训练用户00006",
            attribute="档案代号",
            value="F",
        )
        evaluation, _ = bank.write(
            text="评估用户00006的档案代号为L",
            key=torch.ones(16),
            entity="评估用户00006",
            attribute="档案代号",
            value="L",
        )
        records, _ = bank.query(
            query_key=torch.ones(16),
            query_text="请读取评估用户00006的档案代号",
            top_k_pages=1,
            top_k_records=2,
        )
        self.assertIn(evaluation.record_id, [record.record_id for record in records])
        self.assertNotIn(training.record_id, [record.record_id for record in records])

    def test_explicit_address_is_marked_for_fast_route(self) -> None:
        bank = PagedMemoryBankV2(
            16,
            router=self.router,
            page_capacity=2,
            max_pages=16,
            coarse_index_bits=8,
        )
        bank.write(
            text="代码文件 symbol:parse_args 的状态为enabled",
            key=torch.randn(16),
            entity="symbol:parse_args",
            attribute="状态",
            value="enabled",
        )
        self.assertTrue(bank.has_explicit_address("symbol:parse_args 当前状态是什么？"))
        self.assertFalse(bank.has_explicit_address("当前状态是什么？"))

    def test_management_list_edit_retract_and_audit(self) -> None:
        os_v2 = MemoryOSV2(16, router=self.router)
        record, _ = os_v2.write(
            text="用户喜欢蓝色",
            key=torch.randn(16),
            entity="user",
            attribute="color",
            value="蓝色",
            confidence=0.95,
            importance=0.9,
        )
        listed = os_v2.list_records(query_text="蓝色", status="active", limit=10)
        self.assertEqual([item.record_id for item in listed], [record.record_id])
        edited = os_v2.edit_record(
            record.record_id,
            text="用户喜欢绿色",
            entity="user",
            attribute="color",
            value="绿色",
            evidence=["user_correction"],
        )
        self.assertEqual(edited.version, 1)
        self.assertEqual(edited.supersedes, record.record_id)
        self.assertEqual(os_v2.bank.records[record.record_id].status, STATUS_SUPERSEDED)
        self.assertEqual(os_v2.list_records(query_text="绿色")[0].record_id, edited.record_id)
        os_v2.retract_record(edited.record_id)
        self.assertEqual(os_v2.bank.records[edited.record_id].status, "retracted")
        audit = os_v2.audit()
        self.assertTrue(audit["healthy"], audit)
        all_records = os_v2.list_records(status="all", limit=10)
        self.assertEqual(len(all_records), 2)

    def test_tiered_storage_restarts_and_evicts_cold_records(self) -> None:
        with TemporaryDirectory() as directory:
            path = f"{directory}/memory.sqlite"
            store = TieredMemoryStoreV2(path, key_dim=8, page_capacity=2)
            bank = PagedMemoryBankV2(
                16,
                router=self.router,
                page_capacity=2,
                max_pages=64,
                hot_pages=1,
                top_k_pages=2,
                top_k_records=2,
                tier_store=store,
                max_resident_pages=1,
                coarse_index_bits=8,
            )
            for index in range(8):
                key = torch.zeros(16)
                key[index % 8] = 1.0
                bank.write(
                    text=f"tiered-memory-{index}",
                    key=key,
                    entity="user",
                    attribute=f"attr-{index}",
                    value=f"value-{index}",
                    importance=0.1 if index < 7 else 1.0,
                    confidence=0.95,
                )
            stats = bank.stats()
            self.assertEqual(stats["storage_mode"], "tiered")
            self.assertGreaterEqual(stats["pages"], 4)
            self.assertGreater(stats["cold_pages"], 0)
            self.assertLess(stats["resident_records"], stats["records"])
            store.close()

            reopened_store = TieredMemoryStoreV2(path, key_dim=8, page_capacity=2)
            reopened = PagedMemoryBankV2(
                16,
                router=self.router,
                page_capacity=2,
                max_pages=64,
                hot_pages=1,
                top_k_pages=2,
                top_k_records=2,
                tier_store=reopened_store,
                max_resident_pages=1,
                coarse_index_bits=8,
            )
            records, decision = reopened.query(
                query_key=torch.nn.functional.one_hot(torch.tensor(3), num_classes=16).float(),
                query_text="tiered-memory-3",
                top_k_pages=2,
                top_k_records=2,
            )
            self.assertTrue(records)
            self.assertTrue(any(item.text == "tiered-memory-3" for item in records))
            self.assertGreaterEqual(decision.hop_count, 1)
            quarantined, action = reopened.write(
                text="待审批事实",
                key=torch.randn(16),
                trusted=False,
                confidence=0.1,
            )
            self.assertEqual(action, "quarantined")
            reopened_store.close()

            final_store = TieredMemoryStoreV2(path, key_dim=8, page_capacity=2)
            final_bank = PagedMemoryBankV2(
                16,
                router=self.router,
                page_capacity=2,
                max_pages=64,
                hot_pages=1,
                tier_store=final_store,
                max_resident_pages=2,
                coarse_index_bits=8,
            )
            self.assertIn(quarantined.record_id, final_bank.quarantine)
            approved = final_bank.approve(quarantined.record_id)
            self.assertEqual(approved.status, STATUS_ACTIVE)
            final_store.close()


if __name__ == "__main__":
    unittest.main()
