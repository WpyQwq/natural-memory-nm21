"""Stress-test Natural Memory retrieval on a large in-memory 4B library.

This is a routing/reader benchmark, not a language-generation benchmark.  It
builds a multi-million-token address space with sparse project and personal
facts, then evaluates paraphrased answerable queries, explicit unknown queries,
and versioned conflict updates.  Filler records stay in process RAM and only
the selected records are promoted to the model's bounded hot cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .qwen_integration import load_qwen_dynamic, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() or path.exists() else PROJECT_ROOT / path


def _max_memory(gpu_memory_gb: float) -> dict[Any, str] | None:
    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return None
    return {0: f"{gpu_memory_gb:.1f}GiB", "cpu": "64GiB"}


def _set_cuda_process_cap(gpu_memory_gb: float) -> None:
    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(0.95, max(0.05, gpu_memory_gb * 1024**3 / total))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_facts(count: int) -> list[dict[str, str]]:
    facts: list[dict[str, str]] = []
    for index in range(count):
        code = f"PROJ-{index:04d}"
        if index % 4 == 0:
            text = f"项目 {code} 的部署区域是 region-{index % 7}，责任服务是 service_{index:04d}。"
            query = f"请查项目 {code} 的部署区域，直接给出 region。"
        elif index % 4 == 1:
            text = f"项目 {code} 的回滚命令是 rollback_{index:04d}，发布窗口为周三。"
            query = f"项目 {code} 出问题时应该执行哪条回滚命令？"
        elif index % 4 == 2:
            text = f"个人偏好 {code}：通知时间设为 {8 + index % 5}:30，提醒渠道为 email。"
            query = f"我在 {code} 里设置的通知时间是多少？"
        else:
            text = f"仓库 {code} 的关键文件是 src/module_{index:04d}.py，入口函数为 run_{index:04d}。"
            query = f"仓库 {code} 的关键入口函数叫什么？"
        facts.append({"code": code, "text": text, "query": query})
    return facts


def _encode_texts(model: Any, tokenizer: Any, texts: list[str], device: torch.device) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    return model._encode_model_key(input_ids, mask).detach().cpu()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=r"H:\Memory")
    parser.add_argument("--memory-model", default=r"H:\Memory\V2_dpskw\qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--output", default=r"H:\Memory\V2_dpskw\large_memory_4b.json")
    parser.add_argument("--library-tokens", type=int, default=4_194_304)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--unknown-count", type=int, default=16)
    parser.add_argument("--gpu-memory-gb", type=float, default=10.0)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    _set_cuda_process_cap(args.gpu_memory_gb)

    tokenizer = load_tokenizer(_path(args.base_model))
    use_4bit = not args.no_4bit
    max_memory = _max_memory(args.gpu_memory_gb)
    print("loading Natural Memory 4B")
    model = load_qwen_dynamic(
        _path(args.memory_model),
        load_in_4bit=use_4bit,
        max_memory=max_memory,
    )
    model.eval()
    model.memory_config.memory_top_k_records = 2
    model.memory_os_v2.bank.top_k_records = 2
    device = model._find_layer_device()
    model.reset_memory(batch_size=1, device=device)

    target_count = max(1, min(int(args.target_count), 256))
    unknown_count = max(1, min(int(args.unknown_count), 128))
    facts = _make_facts(target_count)
    chunk_count = max(1, int(args.library_tokens) // max(1, int(args.chunk_tokens)))
    positions = random.Random(20260905).sample(range(chunk_count), min(target_count, chunk_count))
    position_to_fact = dict(zip(positions, facts))

    target_keys: dict[str, torch.Tensor] = {}
    for start in range(0, len(facts), 16):
        batch = facts[start : start + 16]
        for fact, key in zip(batch, _encode_texts(model, tokenizer, [item["text"] for item in batch], device)):
            target_keys[fact["code"]] = F.normalize(key.float(), dim=0)

    generator = torch.Generator(device="cpu").manual_seed(20260905)
    records: list[dict[str, Any]] = []
    for chunk_index in range(chunk_count):
        fact = position_to_fact.get(chunk_index)
        if fact is None:
            key = F.normalize(torch.randn(model.memory.hidden_size, generator=generator), dim=0)
            records.append(
                {
                    "text": f"library_chunk:{chunk_index}",
                    "key": key,
                    "summary": key,
                    "semantic_key": key,
                    "memory_type": "context_chunk",
                    "importance": 0.4,
                    "confidence": 0.8,
                    "source": "large_memory_benchmark",
                    "evidence": [f"library_chunk:{chunk_index}"],
                    "trusted": True,
                    "force": True,
                }
            )
            continue
        key = target_keys[fact["code"]]
        token_ids = torch.tensor(tokenizer(fact["text"], add_special_tokens=False)["input_ids"], dtype=torch.long)
        records.append(
            {
                "text": fact["text"],
                "key": key,
                "summary": key,
                "semantic_key": key,
                "memory_type": "fact",
                "entity": fact["code"],
                "attribute": "benchmark_fact",
                "value": fact["text"],
                "importance": 0.9,
                "confidence": 0.95,
                "source": "large_memory_benchmark",
                "evidence": [f"library_chunk:{chunk_index}"],
                "token_ids": token_ids,
                "token_mask": torch.ones_like(token_ids, dtype=torch.bool),
                "trusted": True,
                "force": True,
            }
        )

    _sync(device)
    write_started = time.perf_counter()
    model.memory_os_v2.write_batch(records)
    _sync(device)
    write_ms = (time.perf_counter() - write_started) * 1000.0
    target_record_ids: dict[str, str] = {}
    # Resolve target IDs by text after the batch write.  This avoids depending
    # on the bank's stable-ID timestamp while preserving an exact recall test.
    for fact in facts:
        matches = [record.record_id for record in model.memory_os_v2.bank.records.values() if record.text == fact["text"]]
        if matches:
            target_record_ids[fact["code"]] = matches[0]

    answerable_queries = [fact["query"] for fact in facts]
    unknown_queries = [f"我的不存在的字段 UNKNOWN-{index:04d} 是什么？如果没有登记就说不知道。" for index in range(unknown_count)]
    all_queries = answerable_queries + unknown_queries
    query_keys = _encode_texts(model, tokenizer, all_queries, device)
    answerable_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []
    query_started = time.perf_counter()
    for index, query in enumerate(all_queries):
        encoded = tokenizer(query, add_special_tokens=False, return_tensors="pt")
        query_ids = encoded["input_ids"][0]
        if index < len(answerable_queries):
            fact = facts[index]
            expected_id = target_record_ids.get(fact["code"], "")
            records_out, decision = model.memory_os_v2.read(
                query_key=query_keys[index],
                query_text=query,
                query_token_ids=query_ids,
                top_k_pages=4,
                top_k_records=2,
                max_hops=3,
            )
            returned_ids = [record.record_id for record in records_out]
            answerable_rows.append(
                {
                    "code": fact["code"],
                    "expected_record_id": expected_id,
                    "returned_record_ids": returned_ids,
                    "hit": expected_id in returned_ids,
                    "stop_reason": decision.stop_reason,
                    "coarse_candidates": model.memory_os_v2.bank._last_coarse_candidates,
                }
            )
        else:
            records_out, decision = model.memory_os_v2.read(
                query_key=query_keys[index],
                query_text=query,
                query_token_ids=query_ids,
                top_k_pages=4,
                top_k_records=2,
                max_hops=3,
            )
            unknown_rows.append(
                {
                    "query": query,
                    "returned_records": len(records_out),
                    "abstained": not records_out,
                    "stop_reason": decision.stop_reason,
                }
            )
    _sync(device)
    query_ms = (time.perf_counter() - query_started) * 1000.0

    conflict_entity = "PROJ-CONFLICT"
    conflict_old = model.memory_os_v2.write(
        text="项目 PROJ-CONFLICT 的负责人是 Alice。",
        key=F.normalize(torch.randn(model.memory.hidden_size, generator=generator), dim=0),
        entity=conflict_entity,
        attribute="owner",
        value="Alice",
        source="large_memory_benchmark",
        trusted=True,
        force=True,
    )[0]
    conflict_new = model.memory_os_v2.write(
        text="项目 PROJ-CONFLICT 的负责人是 Bob。",
        key=F.normalize(torch.randn(model.memory.hidden_size, generator=generator), dim=0),
        entity=conflict_entity,
        attribute="owner",
        value="Bob",
        source="large_memory_benchmark",
        trusted=True,
        force=True,
    )[0]
    conflict_query = "项目 PROJ-CONFLICT 当前负责人是谁？"
    conflict_key = _encode_texts(model, tokenizer, [conflict_query], device)[0]
    conflict_out, conflict_decision = model.memory_os_v2.read(
        query_key=conflict_key,
        query_text=conflict_query,
        query_token_ids=torch.tensor(tokenizer(conflict_query, add_special_tokens=False)["input_ids"]),
        top_k_pages=4,
        top_k_records=2,
        max_hops=3,
    )
    conflict_pass = conflict_new.record_id in [record.record_id for record in conflict_out] and conflict_old.status != "active"

    report = {
        "benchmark": "large_memory_recall_4b",
        "quantization": "4bit_nf4" if use_4bit else "none",
        "library_tokens": int(args.library_tokens),
        "chunk_tokens": int(args.chunk_tokens),
        "chunk_count": chunk_count,
        "stored_records": len(model.memory_os_v2.bank.records),
        "answerable_count": len(answerable_rows),
        "answerable_hits": sum(int(row["hit"]) for row in answerable_rows),
        "answerable_recall": sum(int(row["hit"]) for row in answerable_rows) / max(1, len(answerable_rows)),
        "unknown_count": len(unknown_rows),
        "unknown_abstentions": sum(int(row["abstained"]) for row in unknown_rows),
        "unknown_abstention_rate": sum(int(row["abstained"]) for row in unknown_rows) / max(1, len(unknown_rows)),
        "conflict_update_pass": conflict_pass,
        "conflict_stop_reason": conflict_decision.stop_reason,
        "write_ms": write_ms,
        "query_total_ms": query_ms,
        "query_mean_ms": query_ms / max(1, len(all_queries)),
        "max_coarse_candidates": max((row["coarse_candidates"] for row in answerable_rows), default=0),
        "max_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
        "conflict_old_status": conflict_old.status,
        "conflict_new_status": conflict_new.status,
        "conflict_returned_ids": [record.record_id for record in conflict_out],
        "answerable_rows": answerable_rows,
        "unknown_rows": unknown_rows,
        "warning": "Routing test uses a sparse lexical address side-index plus v9 neural reranking; generation quality still needs real chat/repository workloads.",
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={output}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
