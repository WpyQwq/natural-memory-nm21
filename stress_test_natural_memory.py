"""Run a bounded, restart-like Natural Memory v2 endurance test.

The test deliberately clears only the in-process working copy.  It never
calls ``save_embedded_memory_weights`` and therefore does not modify the
shipped model package.  Defaults are conservative for a 12 GB GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from .qwen_integration import load_qwen_dynamic, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _gpu_stats() -> dict[str, float | int | None]:
    if not torch.cuda.is_available():
        return {"allocated_mb": None, "reserved_mb": None, "free_mb": None, "total_mb": None}
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "free_mb": round(free / (1024 * 1024), 2),
        "total_mb": round(total / (1024 * 1024), 2),
    }


def _long_input(tokenizer, target_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    seed_text = (
        "这是 Natural Memory v2 的长上下文压力测试。它应当把旧上下文归档到有界的记忆记录，"
        "保留最近工作窗口，并且不让当前 token 对全部页面做注意力。"
    )
    seed = tokenizer(seed_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    repeats = max(1, (target_tokens + seed.numel() - 1) // seed.numel())
    ids = seed.repeat(repeats)[:target_tokens].unsqueeze(0)
    return ids, torch.ones_like(ids)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--records-per-round", type=int, default=8)
    parser.add_argument("--long-context-every", type=int, default=10)
    parser.add_argument("--long-context-tokens", type=int, default=2048)
    parser.add_argument("--kv-budget", type=int, default=1024)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--duration-minutes", type=float, default=0.0)
    parser.add_argument("--output", default="natural_memory_v2_stress_report.json")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.records_per_round < 1:
        raise SystemExit("--rounds and --records-per-round must be positive")
    if args.kv_budget < 1 or args.long_context_tokens < args.kv_budget:
        raise SystemExit("--long-context-tokens must be >= --kv-budget >= 1")

    model_path = _project_path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    model = load_qwen_dynamic(model_path, load_in_4bit=not args.no_4bit)
    model.eval()
    device = model._find_layer_device()
    if model.memory_os_v2 is None:
        raise RuntimeError("the selected model does not contain hierarchical memory")

    # Work on an empty ephemeral memory instance.  The embedded package on
    # disk is never changed by this script.
    model.clear_hierarchical_memory()
    model.memory_os_v2.read_threshold = 0.0
    model.memory_os_v2.kv_budget.max_tokens = min(
        int(args.kv_budget), int(model.memory_os_v2.kv_budget.hard_max_tokens)
    )
    model.reset_runtime_memory(batch_size=1, device=device)

    latencies: list[float] = []
    writes = 0
    read_hits = 0
    read_attempts = 0
    compactions = 0
    archived_records = 0
    errors: list[str] = []
    start = time.perf_counter()
    next_record = 0
    try:
        for round_index in range(args.rounds):
            if args.duration_minutes > 0 and (time.perf_counter() - start) >= args.duration_minutes * 60:
                break
            round_start = time.perf_counter()
            try:
                for _ in range(args.records_per_round):
                    record_index = next_record
                    next_record += 1
                    key = torch.randn(model.memory.hidden_size)
                    token_ids = torch.tensor([1000 + (record_index % 10000), 2000 + (record_index % 10000)])
                    text = f"stress record {record_index} belongs to Natural Memory endurance test"
                    record, _ = model.memory_os_v2.write(
                        text=text,
                        key=key,
                        summary=key,
                        token_ids=token_ids,
                        token_mask=torch.ones_like(token_ids, dtype=torch.bool),
                        memory_type="stress_test",
                        entity=f"stress-user-{record_index % 7}",
                        attribute=f"attribute-{record_index}",
                        value=str(record_index),
                        importance=0.9,
                        confidence=0.99,
                        source="stress_test",
                        trusted=True,
                        force=True,
                    )
                    writes += 1
                    read_attempts += 1
                    found, _ = model.memory_os_v2.read(
                        query_key=key,
                        query_text=text,
                        query_token_ids=token_ids,
                        top_k_pages=4,
                        top_k_records=8,
                        max_hops=3,
                    )
                    if any(item.record_id == record.record_id for item in found):
                        read_hits += 1

                if args.long_context_every > 0 and (round_index + 1) % args.long_context_every == 0:
                    ids, mask = _long_input(tokenizer, args.long_context_tokens)
                    _, _, plan = model.compact_context_for_kv(
                        ids.to(device),
                        mask.to(device),
                        archive=True,
                        chunk_tokens=args.chunk_tokens,
                    )
                    if plan.get("compacted"):
                        compactions += 1
                        archived_records += int(plan.get("archived_records", 0))
            except Exception as error:  # keep the report useful after one bad round
                errors.append(f"round {round_index}: {type(error).__name__}: {error}")
            latencies.append(time.perf_counter() - round_start)
    finally:
        final_stats = model.memory_v2_stats()
        audit = model.audit_memory()
        gpu_peak = _gpu_stats()
        model.close_memory_storage()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    latencies_sorted = sorted(latencies)
    report = {
        "format_version": 1,
        "model_path": str(model_path),
        "rounds_completed": len(latencies),
        "records_per_round": args.records_per_round,
        "writes": writes,
        "read_attempts": read_attempts,
        "read_hits": read_hits,
        "read_hit_rate": read_hits / max(1, read_attempts),
        "compactions": compactions,
        "archived_records": archived_records,
        "round_latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0,
            "p95": latencies_sorted[min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95))] if latencies_sorted else 0.0,
            "max": max(latencies, default=0.0),
        },
        "max_gpu": gpu_peak,
        "final_memory_stats": final_stats,
        "audit": audit,
        "errors": errors,
        "package_mutated": False,
        "warning": "This is an endurance/safety smoke test, not a quality benchmark.",
    }
    output = _project_path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
