from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from V2_dpskw.memory_os_v2 import MemoryRouterV2
from V2_dpskw.prepare_memory_router_dataset import Corpus, _add_generic_qa_row, _add_normalized_policy_file, _source_candidates
from V2_dpskw.train_memory_router_large import _episode_tensors, _router_loss


class RouterTrainingContractTests(unittest.TestCase):
    def test_policy_builder_keeps_latest_fact_and_excludes_query_candidates(self) -> None:
        corpus = Corpus()
        rows = [
            (1, {"id": "g:fact", "group_id": "g", "text": "我的项目是旧项目。", "kind": "fact", "write_label": 1.0, "attribute": "项目", "subject": "u"}),
            (2, {"id": "g:replacement", "group_id": "g", "text": "更正：我的项目改为新项目。", "kind": "replacement", "write_label": 1.0, "attribute": "项目", "subject": "u"}),
            (3, {"id": "g:query", "group_id": "g", "text": "我的项目是什么？", "kind": "query", "write_label": 0.0, "attribute": "项目", "subject": "u"}),
            (4, {"id": "g:forget", "group_id": "g", "text": "忘掉我的项目。", "kind": "forget", "write_label": 0.0, "attribute": "项目", "subject": "u"}),
        ]
        _add_normalized_policy_file(corpus, "train", "local.jsonl", rows)
        self.assertEqual(len(corpus.episodes["train"]), 1)
        episode = corpus.episodes["train"][0]
        row = _source_candidates(
            "train",
            episode,
            corpus.items["train"],
            candidate_count=4,
            seed=7,
            eligible_items=[item for item in corpus.items["train"].values() if item.kind not in {"query", "forget"}],
        )
        self.assertIsNotNone(row)
        assert row is not None
        candidates = row["candidates"]
        self.assertTrue(all(item["kind"] not in {"query", "forget"} for item in candidates))
        positive = candidates[row["positive_index"]]
        self.assertIn("新项目", positive["text"])

    def test_episode_tensors_support_variable_candidates_and_unknowns(self) -> None:
        episodes = [
            {
                "query": "q1",
                "candidates": [{"text": "a"}, {"text": "b"}],
                "positive_indices": [1],
                "need_memory": 1.0,
                "hop": 1,
                "family": "qa",
            },
            {
                "query": "q2",
                "candidates": [{"text": "c"}],
                "positive_indices": [],
                "need_memory": 0.0,
                "hop": 0,
                "family": "qa",
            },
        ]
        from V2_dpskw.train_memory_router_large import _text_key

        lookup = {_text_key(text): index for index, text in enumerate(("q1", "a", "b", "q2", "c"))}
        data = _episode_tensors(episodes, lookup)
        self.assertEqual(tuple(data["candidate_indices"].shape), (2, 2))
        self.assertEqual(data["candidate_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(data["positive_mask"].tolist(), [[False, True], [False, False]])

    def test_router_loss_is_finite_for_multi_positive_and_abstention_batch(self) -> None:
        torch.manual_seed(4)
        router = MemoryRouterV2(16, router_dim=8, num_heads=2, max_hops=3)
        batch = {
            "query": torch.randn(2, 16),
            "candidates": torch.randn(2, 3, 16),
            "candidate_mask": torch.tensor([[True, True, True], [True, True, False]]),
            "positive_mask": torch.tensor([[True, True, False], [False, False, False]]),
            "need": torch.tensor([1.0, 0.0]),
            "hops": torch.tensor([2, 0]),
        }
        args = type(
            "Args",
            (),
            {
                "margin": 0.10,
                "need_loss_weight": 0.75,
                "hop_loss_weight": 0.35,
                "margin_loss_weight": 0.25,
            },
        )()
        loss, parts = _router_loss(router, batch, args)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(torch.isfinite(torch.tensor(value)).item() for value in parts.values()))

    def test_codesearchnet_style_documentation_becomes_query(self) -> None:
        corpus = Corpus()
        _add_generic_qa_row(
            corpus,
            "train",
            "code_search_net|train|python|train",
            "hf:code_search_net",
            1,
            {
                "id": "code-1",
                "func_documentation_string": "load the memory page by id",
                "whole_func_string": "def load_page(page_id): return pages[page_id]",
                "repository_name": "example/repo",
            },
            "code",
        )
        self.assertEqual(len(corpus.episodes["train"]), 1)
        self.assertEqual(corpus.episodes["train"][0].query, "load the memory page by id")
        self.assertEqual(len(corpus.episodes["train"][0].positive_ids), 1)

    def test_generated_manifest_and_eval_hash_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1] / "data" / "router_training"
        manifest_path = root / "manifest.json"
        eval_path = root / "eval.jsonl"
        hash_path = root / "eval.sha256"
        if not (manifest_path.exists() and eval_path.exists() and hash_path.exists()):
            self.skipTest("router_training data has not been generated")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = __import__("hashlib").sha256(eval_path.read_bytes()).hexdigest()
        self.assertEqual(actual, manifest["files"]["eval"]["sha256"])
        self.assertEqual(actual, hash_path.read_text(encoding="ascii").strip())
        self.assertEqual(manifest["leakage_check"]["group_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
