"""Contract tests for the new MemoryRouterXL architecture.

The XL router must satisfy two things at once:

* be a genuinely larger, separate model (not a tweak of MemoryRouterV2), and
* expose the exact runtime contract ``PagedMemoryBankV2`` already drives, so a
  trained XL router can be dropped into the existing memory OS.

These tests run on tiny tensors only; no Qwen model is loaded.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from V2_dpskw.memory_os_v2 import MemoryRouterV2, PagedMemoryBankV2
from V2_dpskw.router_xl import ARCH_NAME, MemoryRouterXL, load_router_xl


def _tiny_xl(**overrides: object) -> MemoryRouterXL:
    kwargs: dict[str, object] = dict(
        router_dim=8,
        num_heads=2,
        max_hops=3,
        encoder_layers=2,
        encoder_hidden=8,
        pair_blocks=1,
        pair_hidden=8,
        policy_layers=2,
        policy_hidden=8,
    )
    kwargs.update(overrides)
    torch.manual_seed(11)
    return MemoryRouterXL(16, **kwargs)


class RouterXLContractTest(unittest.TestCase):
    def test_forward_matches_the_runtime_contract(self) -> None:
        router = _tiny_xl()
        query = torch.randn(4, 16)
        candidates = torch.randn(4, 5, 16)
        output = router(query, candidates)
        self.assertEqual(tuple(output["scores"].shape), (4, 5))
        self.assertEqual(tuple(output["head_scores"].shape), (4, 5, 2))
        self.assertEqual(tuple(output["need_memory_logits"].shape), (4,))
        self.assertEqual(tuple(output["hop_logits"].shape), (4, 4))
        self.assertEqual(tuple(router.encode_key(query).shape), (4, 8))
        self.assertEqual(tuple(router.encode_query(query).shape), (4, 8))
        self.assertAlmostEqual(float(router.encode_query(query).norm(dim=-1).mean()), 1.0, places=5)

    def test_projected_scores_accepts_a_shared_candidate_bank(self) -> None:
        router = _tiny_xl()
        query = torch.randn(2, 16)
        projected = router.encode_key(torch.randn(7, 16))
        scores, head_scores = router.projected_scores(query, projected)
        self.assertEqual(tuple(scores.shape), (2, 7))
        self.assertEqual(tuple(head_scores.shape), (2, 7, 2))
        # A bank-shaped [N, router_dim] key must broadcast like the V2 router.
        scores2, _ = router.projected_scores(query, router.encode_key(torch.randn(7, 16)))
        self.assertEqual(tuple(scores2.shape), (2, 7))

    def test_wrong_shapes_are_rejected(self) -> None:
        router = _tiny_xl()
        with self.assertRaises(ValueError):
            router.encode_key(torch.randn(3, 15))
        with self.assertRaises(ValueError):
            router.projected_scores(torch.randn(3, 16), torch.randn(3, 5, 7))

    def test_arch_config_round_trip_is_exact(self) -> None:
        router = _tiny_xl()
        config = router.arch_config()
        self.assertEqual(config["arch"], ARCH_NAME)
        clone = MemoryRouterXL.from_arch_config(config)
        self.assertEqual(clone.parameter_count(), router.parameter_count())
        self.assertEqual(set(clone.state_dict()), set(router.state_dict()))
        clone.load_state_dict(router.state_dict(), strict=True)
        x = torch.randn(2, 16)
        candidates = torch.randn(2, 6, 16)
        router.eval()
        clone.eval()  # pair dropout is active in train mode; compare deterministically
        self.assertTrue(torch.allclose(router(x, candidates)["scores"], clone(x, candidates)["scores"]))

    def test_checkpoint_round_trip_via_load_router_xl(self) -> None:
        router = _tiny_xl()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory_router_xl.pt"
            torch.save({key: value.detach().cpu() for key, value in router.state_dict().items()}, path)
            (Path(directory) / "router_arch.json").write_text(
                __import__("json").dumps(router.arch_config()), encoding="utf-8"
            )
            restored = load_router_xl(path)
            self.assertEqual(restored.router_dim, router.router_dim)
            self.assertEqual(restored.parameter_count(), router.parameter_count())

    def test_paged_memory_bank_accepts_the_xl_router(self) -> None:
        """Runtime integration: the fork's bank must drive the new router."""

        torch.manual_seed(5)
        router = _tiny_xl()
        bank = PagedMemoryBankV2(
            16,
            router=router,
            page_capacity=2,
            max_pages=64,
            hot_pages=2,
            top_k_pages=2,
            top_k_records=3,
            max_hops=3,
            coarse_index_bits=8,
        )
        self.assertEqual(bank.key_dim, router.router_dim)
        for index in range(3):
            bank.write(
                text=f"fact number {index}",
                key=torch.randn(16),
                entity=f"entity-{index}",
                attribute="value",
                value=str(index),
                confidence=0.9,
            )
        records, decision = bank.query(query_key=torch.randn(16), query_text="fact number 1")
        self.assertTrue(records, "XL router must route at least one active record")
        self.assertTrue(decision.record_ids)
        self.assertIsInstance(decision.need_memory, bool)

    def test_capacity_scales_with_router_dim(self) -> None:
        """Documented capacity jump: XL is a different weight class, not a rename."""

        v2 = MemoryRouterV2(2560, router_dim=512, num_heads=8)
        v2_params = sum(p.numel() for p in v2.parameters())
        xl_small = MemoryRouterXL(2560, router_dim=512, num_heads=16, encoder_hidden=512, pair_hidden=512, policy_hidden=512)
        xl_1024 = MemoryRouterXL(2560, router_dim=1024, num_heads=16, encoder_hidden=1024, pair_hidden=1024, policy_hidden=512)
        xl_2048 = MemoryRouterXL(2560, router_dim=2048, num_heads=16, encoder_hidden=2048, pair_hidden=2048, policy_hidden=512)
        small_params = xl_small.parameter_count()["total"]
        mid_params = xl_1024.parameter_count()["total"]
        large_params = xl_2048.parameter_count()["total"]
        self.assertGreater(small_params, v2_params)
        self.assertLess(small_params, mid_params)
        self.assertLess(mid_params, large_params)
        self.assertEqual(xl_1024.router_dim, 1024)
        self.assertEqual(xl_2048.parameter_count()["trainable"], large_params)


if __name__ == "__main__":
    unittest.main()
