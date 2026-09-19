"""Comprehensive, hardware-independent evaluation for Natural Memory v2."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import (
    KVBudgetManagerV2,
    MemoryRouterV2,
    PagedMemoryBankV2,
    STATUS_ACTIVE,
    STATUS_QUARANTINED,
    STATUS_RETRACTED,
    STATUS_SUPERSEDED,
)
from V2_dpskw.train_memory_router_v2 import _latent_to_hidden, _make_basis, evaluate


def _load_router(args: argparse.Namespace, device: torch.device) -> tuple[MemoryRouterV2, torch.Tensor, str]:
    router = MemoryRouterV2(
        args.hidden_size,
        router_dim=args.router_dim,
        num_heads=args.num_heads,
        max_hops=args.max_hops,
    ).to(device)
    checkpoint = Path(args.router_checkpoint)
    basis_path = checkpoint.with_name("memory_router_v2_basis.pt")
    if checkpoint.exists() and basis_path.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        router.load_state_dict(state, strict=True)
        basis = torch.load(basis_path, map_location=device, weights_only=True).to(device)
        return router, basis, "trained_checkpoint"
    basis = _make_basis(args.hidden_size, args.latent_size, device)
    return router, basis, "untrained_router"


def _metric(value: bool) -> float:
    return 1.0 if value else 0.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    router, basis, router_source = _load_router(args, device)
    router.eval()
    bank = PagedMemoryBankV2(
        args.hidden_size,
        router=router,
        page_capacity=args.page_capacity,
        max_pages=args.max_pages,
        hot_pages=args.hot_pages,
        top_k_pages=args.top_k_pages,
        top_k_records=args.top_k_records,
        max_hops=args.max_hops,
        coarse_index_bits=args.coarse_index_bits,
    )

    # Route quality on held-out samples from the same latent factor space.
    route_metrics = evaluate(
        router,
        basis=basis,
        device=device,
        batches=args.router_eval_batches,
        batch_size=args.router_eval_batch_size,
        candidate_count=args.candidate_count,
    )

    # Populate a large enough store to activate the coarse index.  Keys are
    # low-entropy semantic points, while their positions are deliberately
    # unrelated to their topic ids.
    records: list[Any] = []
    started = time.perf_counter()
    for index in range(args.records):
        latent = torch.randn(basis.shape[1], device=device)
        key = _latent_to_hidden(latent.unsqueeze(0), basis, 0.04)[0]
        record, action = bank.write(
            text=f"长期事实 {index}",
            key=key,
            token_ids=torch.tensor([index % 997, 17, 23]),
            token_mask=torch.tensor([True, True, True]),
            importance=0.5 + 0.5 * (index % 7 == 0),
            confidence=0.95,
            source="synthetic_episode",
        )
        records.append(record)
    write_seconds = time.perf_counter() - started

    recall_hits = 0
    page_hits = 0
    candidate_counts: list[int] = []
    query_count = min(args.query_count, len(records))
    for index in torch.randperm(len(records), device=device)[:query_count].tolist():
        target = records[index]
        query_key = target.key.to(device) if router_source == "untrained_router" else (
            target.key.to(device)
        )
        # ``target.key`` is already in compact address space.  This tests the
        # same storage-space path used after a Qwen hidden state is projected.
        found, decision = bank.query(
            query_key=query_key,
            top_k_pages=args.top_k_pages,
            top_k_records=args.top_k_records,
        )
        found_ids = {record.record_id for record in found}
        recall_hits += int(target.record_id in found_ids)
        page_hits += int(target.page_id in decision.page_ids)
        candidate_counts.append(bank.stats()["last_coarse_candidates"])

    # Conflict/versioning and explicit correction.
    conflict_key = torch.randn(args.hidden_size, device=device)
    first, _ = bank.write(
        text="用户当前工作地点是上海",
        key=conflict_key,
        entity="user",
        attribute="work_city",
        value="上海",
        confidence=0.90,
    )
    second, conflict_action = bank.write(
        text="用户当前工作地点是杭州",
        key=conflict_key,
        entity="user",
        attribute="work_city",
        value="杭州",
        confidence=0.98,
    )
    corrected, correction_action = bank.correct(
        text="纠正：用户当前工作地点是苏州",
        key=conflict_key,
        entity="user",
        attribute="work_city",
        value="苏州",
        confidence=1.0,
    )

    # Pollution protection: untrusted write stays out of the searchable bank.
    quarantined, quarantine_action = bank.write(
        text="模型猜测的生日",
        key=torch.randn(args.hidden_size, device=device),
        confidence=0.05,
        trusted=False,
    )
    quarantine_before_approval = (
        quarantined.status == STATUS_QUARANTINED
        and quarantined.record_id not in bank.records
        and quarantine_action == "quarantined"
    )
    approved = bank.approve(quarantined.record_id)
    approved_active = approved.status == STATUS_ACTIVE

    # Multi-hop: source page contains only the anchor; related evidence lives
    # in other pages.  Restrict first-hop page selection to force expansion.
    hop_bank = PagedMemoryBankV2(
        args.hidden_size,
        router=router,
        page_capacity=1,
        max_pages=64,
        hot_pages=1,
        top_k_pages=1,
        top_k_records=3,
        max_hops=args.max_hops,
        coarse_index_bits=args.coarse_index_bits,
    )
    hop_b, _ = hop_bank.write(text="链路证据 B", key=torch.randn(args.hidden_size, device=device), slot_index=20001)
    hop_c, _ = hop_bank.write(text="链路证据 C", key=torch.randn(args.hidden_size, device=device), slot_index=20002)
    hop_a, _ = hop_bank.write(
        text="链路锚点 A",
        key=torch.randn(args.hidden_size, device=device),
        related_ids=[hop_b.record_id, hop_c.record_id],
        slot_index=20000,
    )
    hop_records, hop_decision = hop_bank.query(
        query_key=hop_a.key,
        top_k_pages=1,
        top_k_records=3,
        max_hops=args.max_hops,
    )
    hop_ids = {record.record_id for record in hop_records}
    multi_hop_success = hop_b.record_id in hop_ids or hop_c.record_id in hop_ids

    # Idempotence, retraction and restart serialization.
    duplicate, duplicate_action = bank.write(
        text="长期事实 0",
        key=records[0].key,
        token_ids=records[0].token_ids,
        token_mask=records[0].token_mask,
        confidence=0.99,
    )
    bank.retract(approved.record_id)
    restart_payload = bank.export_payload()
    restored = PagedMemoryBankV2.from_payload(restart_payload, router=router)
    restored_records, restored_decision = restored.query(
        query_key=records[0].key,
        top_k_pages=args.top_k_pages,
        top_k_records=args.top_k_records,
    )

    budget = KVBudgetManagerV2(
        max_tokens=args.kv_budget,
        hard_max_tokens=args.kv_hard_max,
        keep_recent_tokens=args.kv_keep_recent,
    )
    budget_checks = {
        "below_trigger": not budget.needs_compaction(int(args.kv_budget * 0.5)),
        "at_trigger": budget.needs_compaction(budget.trigger_tokens),
        "overflow": budget.overflow(args.kv_budget + 123),
    }

    stats = bank.stats()
    summary = {
        "format_version": 2,
        "seed": args.seed,
        "device": str(device),
        "router_source": router_source,
        "router": route_metrics,
        "storage": {
            "records_requested": args.records,
            "records_stored_before_scenarios": len(records),
            "write_seconds": write_seconds,
            "pages": stats["pages"],
            "coarse_index_buckets": stats["coarse_index_buckets"],
            "coarse_candidate_mean": sum(candidate_counts) / max(1, len(candidate_counts)),
            "coarse_candidate_max": max(candidate_counts, default=0),
            "coarse_candidate_ratio": (
                sum(candidate_counts) / max(1, len(candidate_counts)) / max(1, stats["pages"])
            ),
        },
        "retrieval": {
            "query_count": query_count,
            "record_recall_at_k": recall_hits / max(1, query_count),
            "page_recall_at_k": page_hits / max(1, query_count),
            "multi_hop_success": _metric(multi_hop_success),
            "multi_hop_hops": hop_decision.hop_count,
            "restart_record_recall": _metric(bool(restored_records)),
            "restart_page_count": restored.stats()["pages"],
        },
        "integrity": {
            "conflict_action": conflict_action,
            "correction_action": correction_action,
            "old_conflict_superseded": _metric(first.status == STATUS_SUPERSEDED),
            "latest_correction_active": _metric(corrected.status == STATUS_ACTIVE),
            "active_conflict_value": corrected.value,
            "quarantine_action": quarantine_action,
            "quarantine_isolation": _metric(quarantine_before_approval),
            "approved_active": _metric(approved_active),
            "retracted_status": bank.records[approved.record_id].status,
            "retraction_isolated": _metric(bank.records[approved.record_id].status == STATUS_RETRACTED),
            "duplicate_action": duplicate_action,
            "duplicate_idempotent": _metric(duplicate.record_id == records[0].record_id),
        },
        "kv_budget": budget_checks,
        "final_stats": stats,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="W:/Flash/model/V2_dpskw/natural_memory_v2_benchmark.json")
    parser.add_argument("--router-checkpoint", default="W:/Flash/model/V2_dpskw/checkpoints/natural_memory_v2_router/memory_router_v2.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--router-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--router-eval-batches", type=int, default=40)
    parser.add_argument("--router-eval-batch-size", type=int, default=64)
    parser.add_argument("--records", type=int, default=512)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--page-capacity", type=int, default=32)
    parser.add_argument("--max-pages", type=int, default=32768)
    parser.add_argument("--hot-pages", type=int, default=8)
    parser.add_argument("--top-k-pages", type=int, default=4)
    parser.add_argument("--top-k-records", type=int, default=8)
    parser.add_argument("--coarse-index-bits", type=int, default=20)
    parser.add_argument("--kv-budget", type=int, default=32768)
    parser.add_argument("--kv-hard-max", type=int, default=131072)
    parser.add_argument("--kv-keep-recent", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
