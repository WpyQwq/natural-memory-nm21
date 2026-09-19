"""Stress the durable Natural Memory v2 page tier without loading Qwen."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import MemoryRouterV2, PagedMemoryBankV2
from V2_dpskw.tiered_memory_store_v2 import TieredMemoryStoreV2


def run(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="natural-memory-v2-tiered-", dir=str(report_path.parent)) as temp_dir:
        store_path = Path(temp_dir) / "memory.sqlite"
        router = MemoryRouterV2(args.hidden_size, router_dim=args.key_dim, num_heads=args.heads)
        store = TieredMemoryStoreV2(
            store_path,
            key_dim=args.key_dim,
            page_capacity=args.page_capacity,
        )
        bank = PagedMemoryBankV2(
            args.hidden_size,
            page_capacity=args.page_capacity,
            max_pages=max(1, (args.records + args.page_capacity - 1) // args.page_capacity + 8),
            hot_pages=args.hot_pages,
            top_k_pages=args.top_k_pages,
            top_k_records=args.top_k_records,
            router=router,
            key_dim=args.key_dim,
            coarse_index_bits=args.coarse_index_bits,
            tier_store=store,
            max_resident_pages=args.resident_pages,
        )
        target_key = None
        target_text = ""
        started = time.perf_counter()
        for start in range(0, args.records, args.batch_size):
            batch: list[dict[str, object]] = []
            for index in range(start, min(args.records, start + args.batch_size)):
                key = torch.randn(args.hidden_size)
                if index == args.target_index:
                    target_key = key.clone()
                    target_text = f"tiered-record-{index}"
                batch.append(
                    {
                        "text": f"tiered-record-{index}",
                        "key": key,
                        "summary": key,
                        "entity": "benchmark",
                        "attribute": f"attribute-{index}",
                        "value": f"value-{index}",
                        "importance": 0.2 if index != args.target_index else 1.0,
                        "confidence": 0.95,
                        "source": "tiered-benchmark",
                        "trusted": True,
                    }
                )
            bank.write_batch(batch)
        write_seconds = time.perf_counter() - started
        before = bank.stats()
        store.close()

        reopen_started = time.perf_counter()
        reopened_store = TieredMemoryStoreV2(
            store_path,
            key_dim=args.key_dim,
            page_capacity=args.page_capacity,
        )
        reopened = PagedMemoryBankV2(
            args.hidden_size,
            page_capacity=args.page_capacity,
            max_pages=max(1, (args.records + args.page_capacity - 1) // args.page_capacity + 8),
            hot_pages=args.hot_pages,
            top_k_pages=args.top_k_pages,
            top_k_records=args.top_k_records,
            router=router,
            key_dim=args.key_dim,
            coarse_index_bits=args.coarse_index_bits,
            tier_store=reopened_store,
            max_resident_pages=args.resident_pages,
        )
        reopen_seconds = time.perf_counter() - reopen_started
        if target_key is None:
            raise RuntimeError("target index was not generated")
        records, decision = reopened.query(
            query_key=target_key,
            query_text=target_text,
            top_k_pages=args.top_k_pages,
            top_k_records=args.top_k_records,
        )
        after = reopened.stats()
        found = any(record.text == target_text for record in records)
        target_row = reopened_store.find_by_text(target_text, active_status="active")
        target_page_id = target_row["page_id"] if target_row is not None else None
        candidate_pages = reopened._candidate_page_ids(target_key)
        reopened_store.close()

    report = {
        "format_version": 2,
        "records_requested": args.records,
        "target_index": args.target_index,
        "page_capacity": args.page_capacity,
        "coarse_index_bits": args.coarse_index_bits,
        "resident_pages": args.resident_pages,
        "write_seconds": write_seconds,
        "reopen_seconds": reopen_seconds,
        "before_close": before,
        "after_reopen": after,
        "target_recalled_after_restart": found,
        "target_page_id": target_page_id,
        "target_page_in_coarse_candidates": target_page_id in candidate_pages if target_page_id else False,
        "coarse_candidate_count": len(candidate_pages),
        "decision": {
            "page_ids": decision.page_ids,
            "record_ids": decision.record_ids,
            "hop_count": decision.hop_count,
            "confidence": decision.confidence,
            "stop_reason": decision.stop_reason,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--target-index", type=int, default=99_999)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--page-capacity", type=int, default=32)
    parser.add_argument("--resident-pages", type=int, default=64)
    parser.add_argument("--hot-pages", type=int, default=8)
    parser.add_argument("--top-k-pages", type=int, default=4)
    parser.add_argument("--top-k-records", type=int, default=8)
    parser.add_argument("--coarse-index-bits", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--output",
        default="W:/Flash/model/V2_dpskw/tiered_memory_v2_benchmark.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
