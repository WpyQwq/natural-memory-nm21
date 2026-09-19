"""Run a same-protocol 4B comparison: base Qwen, Chunk RAG, and Natural Memory.

The benchmark uses the same general-memory and real-repository query sets for
all systems.  It reports answer correctness, refusal correctness, prompt/read
overhead, generation speed, and peak VRAM.  Memory is kept in process RAM and
the GPU placement cap is shared by every model load.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .benchmark_real_scale_memory_4b import (
    _build_general_records,
    _build_project_records,
    _chat_generate,
    _contains_answer,
    _is_refusal,
    _max_memory,
    _native_cases,
    _path,
    _prepare_records,
    _set_cuda_process_cap,
    _source_files,
    _sync,
)
from .qwen_integration import load_qwen_base, load_qwen_dynamic, load_tokenizer
from .stream_chat_qwen_memory import _chat_tensor


PROJECT_ROOT = Path(__file__).resolve().parent
TERM_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_\-]+")


def _terms(text: str) -> set[str]:
    return set(TERM_PATTERN.findall(str(text).lower()))


def _vram_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"allocated_gb": None, "reserved_gb": None}
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
    }


def _record_entry(record: dict[str, Any]) -> dict[str, Any]:
    text = str(record.get("text", ""))
    return {
        "text": text,
        "terms": _terms(text),
        "entity": str(record.get("entity", "")),
        "attribute": str(record.get("attribute", "")),
        "value": str(record.get("value", "")),
    }


def _build_rag_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_record_entry(record) for record in records if str(record.get("text", "")).strip()]


def _rag_retrieve(
    index: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    query_terms = _terms(query)
    query_lower = query.strip().lower()
    started = time.perf_counter()
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, item in enumerate(index):
        shared = len(query_terms.intersection(item["terms"]))
        score = shared / math.sqrt(max(1, len(query_terms) * len(item["terms"])))
        entity = item["entity"].strip().lower()
        attribute = item["attribute"].strip().lower()
        if len(entity) >= 4 and entity in query_lower:
            score += 5.0
        if attribute and attribute in query_lower:
            score += 1.0
        scored.append((score, -position, item))
    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return [item for _, _, item in scored[: max(1, int(top_k))]], elapsed_ms


def _rag_prompt(query: str, records: list[dict[str, Any]]) -> str:
    evidence = "\n".join(
        f"[证据 {index}] {record['text']}"
        for index, record in enumerate(records, 1)
    )
    return (
        "以下是检索器返回的记忆证据。只能使用证据中明确出现的事实，"
        "先核对实体、属性和已确认值；多个候选并存时不要把候选拼成一个事实；"
        "新旧冲突时优先最新且来源更可靠的证据；找不到目标时请明确说不知道，"
        "不要用相似用户的信息代替。\n"
        "---记忆证据开始---\n"
        + evidence
        + "\n---记忆证据结束---\n问题："
        + query
    )


@torch.inference_mode()
def _base_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    encoded = _chat_tensor(tokenizer, prompt)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    _sync(device)
    elapsed = time.perf_counter() - started
    response_ids = output[0, encoded["input_ids"].shape[1] :]
    response = tokenizer.decode(
        response_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
    ).strip()
    generated_tokens = int(response_ids.numel())
    row = {
        "status": "ok",
        "response": response,
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "generated_tokens": generated_tokens,
        "total_latency_s": elapsed,
        "decode_tok_s": generated_tokens / max(elapsed, 1e-9),
    }
    row.update(_vram_snapshot(device))
    del output
    return row


def _quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    unknown = [row for row in rows if not row["answerable"]]
    return {
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "answerable_correct": sum(int(row["correct"]) for row in answerable),
        "answerable_accuracy": sum(int(row["correct"]) for row in answerable) / max(1, len(answerable)),
        "unknown_cases": len(unknown),
        "unknown_correct": sum(int(row["correct"]) for row in unknown),
        "unknown_refusal_accuracy": sum(int(row["correct"]) for row in unknown) / max(1, len(unknown)),
        "mean_prompt_tokens": mean(row["prompt_tokens"] for row in rows) if rows else 0.0,
        "mean_total_latency_ms": mean(row["total_latency_s"] for row in rows) * 1000.0 if rows else 0.0,
        "mean_decode_tok_s": mean(row["decode_tok_s"] for row in rows) if rows else 0.0,
        "peak_allocated_gb": max((row.get("allocated_gb") or 0.0 for row in rows), default=0.0),
        "peak_reserved_gb": max((row.get("reserved_gb") or 0.0 for row in rows), default=0.0),
    }


def _run_base_system(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    index: list[dict[str, Any]],
    device: torch.device,
    *,
    mode: str,
    rag_top_k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        query = str(row["query"])
        retrieve_ms = 0.0
        retrieved: list[dict[str, Any]] = []
        if mode == "chunk_rag":
            retrieved, retrieve_ms = _rag_retrieve(index, query, top_k=rag_top_k)
            prompt = _rag_prompt(query, retrieved)
        else:
            prompt = query
        generated = _base_generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=max_new_tokens,
        )
        expected = str(row.get("expected", ""))
        answerable = bool(row.get("answerable", True))
        correct = (
            _contains_answer(generated["response"], expected)
            if answerable
            else _is_refusal(generated["response"])
        )
        retrieved_target = answerable and any(
            _contains_answer(item["value"], expected)
            or _contains_answer(item["text"], expected)
            for item in retrieved
        )
        output_rows.append(
            {
                "id": row.get("id", ""),
                "query": query,
                "expected": expected,
                "answerable": answerable,
                "correct": bool(correct),
                "retrieved_target": bool(retrieved_target),
                "retrieved_count": len(retrieved),
                "retriever_ms": retrieve_ms,
                "retrieved_values": [item["value"] for item in retrieved],
                **generated,
            }
        )
    summary = _quality_summary(output_rows)
    summary["mean_retriever_ms"] = mean(row["retriever_ms"] for row in output_rows) if output_rows else 0.0
    reader_values = [float(row["reader_ms"]) for row in output_rows if "reader_ms" in row]
    if reader_values:
        summary["mean_reader_ms"] = mean(reader_values)
    summary["answerable_retrieval_recall"] = (
        sum(int(row["retrieved_target"]) for row in output_rows if row["answerable"])
        / max(1, sum(int(row["answerable"]) for row in output_rows))
    )
    summary["rows"] = output_rows
    return summary


def _run_natural_memory(
    model: Any,
    tokenizer: Any,
    general_records: list[dict[str, Any]],
    general_rows: list[dict[str, Any]],
    project_records: list[dict[str, Any]],
    project_rows: list[dict[str, Any]],
    device: torch.device,
    *,
    generation_general: int,
    generation_project: int,
    max_new_tokens: int,
    encode_batch_size: int,
) -> dict[str, Any]:
    model.eval()
    model.memory_config.memory_top_k_records = 2
    model.memory_os_v2.bank.top_k_records = 2
    model.reset_memory(batch_size=1, device=device)

    output: dict[str, Any] = {}
    prepared_general = _prepare_records(
        model,
        tokenizer,
        general_records,
        device,
        batch_size=max(1, encode_batch_size),
    )
    model.memory_os_v2.write_batch(prepared_general)
    general_rows = general_rows[: max(1, generation_general)]
    general_output: list[dict[str, Any]] = []
    for row in general_rows:
        generated = _chat_generate(
            model,
            tokenizer,
            str(row["query"]),
            device,
            max_new_tokens,
        )
        expected = str(row.get("expected", ""))
        answerable = bool(row.get("answerable", True))
        correct = (
            _contains_answer(generated["response"], expected)
            if answerable
            else _is_refusal(generated["response"])
        )
        general_output.append(
            {
                "id": row.get("id", ""),
                "query": row["query"],
                "expected": expected,
                "answerable": answerable,
                "correct": bool(correct),
                "prompt_tokens": int(generated["public_prompt_tokens"] + generated["prefix_tokens"]),
                "generated_tokens": int(generated["generated_tokens"]),
                "total_latency_s": float(generated["generation_seconds"]),
                "decode_tok_s": int(generated["generated_tokens"]) / max(float(generated["generation_seconds"]), 1e-9),
                "reader_ms": float(generated["memory_read_seconds"]) * 1000.0,
                "prefix_used": bool(generated["prefix_used"]),
                "prefix_tokens": int(generated["prefix_tokens"]),
                "retrieved_values": list(generated["selected_values"]),
                "retrieved_ids": list(generated["selected_record_ids"]),
                "allocated_gb": _vram_snapshot(device)["allocated_gb"],
                "reserved_gb": _vram_snapshot(device)["reserved_gb"],
                "response": generated["response"],
            }
        )
    output["general"] = _quality_summary(general_output)
    output["general"]["rows"] = general_output

    model.memory_os_v2 = model._new_memory_os_v2(model.memory.hidden_size)
    model.reset_memory(batch_size=1, device=device)
    prepared_project = _prepare_records(
        model,
        tokenizer,
        project_records,
        device,
        batch_size=max(1, encode_batch_size),
    )
    model.memory_os_v2.write_batch(prepared_project)
    project_rows = project_rows[: max(1, generation_project)]
    project_output: list[dict[str, Any]] = []
    for row in project_rows:
        generated = _chat_generate(
            model,
            tokenizer,
            str(row["query"]),
            device,
            max_new_tokens,
        )
        expected = str(row.get("expected", ""))
        correct = _contains_answer(generated["response"], expected)
        snapshot = _vram_snapshot(device)
        project_output.append(
            {
                "id": row.get("id", ""),
                "query": row["query"],
                "expected": expected,
                "answerable": True,
                "correct": bool(correct),
                "prompt_tokens": int(generated["public_prompt_tokens"] + generated["prefix_tokens"]),
                "generated_tokens": int(generated["generated_tokens"]),
                "total_latency_s": float(generated["generation_seconds"]),
                "decode_tok_s": int(generated["generated_tokens"]) / max(float(generated["generation_seconds"]), 1e-9),
                "reader_ms": float(generated["memory_read_seconds"]) * 1000.0,
                "prefix_used": bool(generated["prefix_used"]),
                "prefix_tokens": int(generated["prefix_tokens"]),
                "retrieved_values": list(generated["selected_values"]),
                "retrieved_ids": list(generated["selected_record_ids"]),
                "allocated_gb": snapshot["allocated_gb"],
                "reserved_gb": snapshot["reserved_gb"],
                "response": generated["response"],
            }
        )
    output["project"] = _quality_summary(project_output)
    output["project"]["rows"] = project_output
    return output


def _release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=r"H:\Memory")
    parser.add_argument(
        "--memory-model",
        default=r"H:\Memory\V2_dpskw\qwen3_5_4b_natural_memory_v2",
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "baseline_compare_4b.json"))
    parser.add_argument("--general-cases", type=int, default=640)
    parser.add_argument("--project-records", type=int, default=8192)
    parser.add_argument("--project-targets", type=int, default=256)
    parser.add_argument("--generation-general", type=int, default=128)
    parser.add_argument("--generation-project", type=int, default=64)
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-gb", type=float, default=10.0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--reuse-original-report",
        default=None,
        help="reuse original_qwen results from an earlier report and rerun only Natural Memory",
    )
    args = parser.parse_args()
    _set_cuda_process_cap(args.gpu_memory_gb)

    tokenizer = load_tokenizer(_path(args.base_model))
    general_cases = _native_cases(_path(args.data_root), max(1, int(args.general_cases)))
    general_raw_records, general_queries = _build_general_records(general_cases)
    general_generation_rows = [
        row for row in general_queries if row["split"] == "eval"
    ][: max(1, int(args.generation_general))]
    project_files = _source_files()
    project_raw_records, project_queries, project_meta = _build_project_records(
        project_files,
        record_count=max(128, int(args.project_records)),
        target_count=max(1, int(args.project_targets)),
        chunk_tokens=512,
    )
    project_generation_rows = [
        {"id": f"project-{index}", "query": row["query"], "expected": row["expected"], "answerable": True}
        for index, row in enumerate(project_queries[: max(1, int(args.generation_project))])
    ]
    general_index = _build_rag_index(general_raw_records)
    project_index = _build_rag_index(project_raw_records)
    use_4bit = not args.no_4bit
    max_memory = _max_memory(args.gpu_memory_gb)

    reused_original_report = None
    if args.reuse_original_report:
        reused_original_report = json.loads(
            _path(args.reuse_original_report).read_text(encoding="utf-8")
        )
        original_system = reused_original_report.get("systems", {}).get("original_qwen")
        if not isinstance(original_system, dict):
            raise ValueError("reuse report does not contain systems.original_qwen")
        base_load_vram = original_system.get("load_vram")
        base_general = original_system["general"]
        base_project = original_system["project"]
        print("reusing original Qwen results")
    else:
        print("loading original Qwen 4B")
        base = load_qwen_base(
            _path(args.base_model),
            load_in_4bit=use_4bit,
            max_memory=max_memory,
        )
        base.eval()
        base_device = base.get_input_embeddings().weight.device
        base_load_vram = _vram_snapshot(base_device)
        base_general = {
            "no_memory": _run_base_system(
                base,
                tokenizer,
                general_generation_rows,
                general_index,
                base_device,
                mode="no_memory",
                rag_top_k=args.rag_top_k,
                max_new_tokens=max(1, args.max_new_tokens),
            ),
            "chunk_rag": _run_base_system(
                base,
                tokenizer,
                general_generation_rows,
                general_index,
                base_device,
                mode="chunk_rag",
                rag_top_k=max(1, args.rag_top_k),
                max_new_tokens=max(1, args.max_new_tokens),
            ),
        }
        base_project = {
            "no_memory": _run_base_system(
                base,
                tokenizer,
                project_generation_rows,
                project_index,
                base_device,
                mode="no_memory",
                rag_top_k=args.rag_top_k,
                max_new_tokens=max(1, args.max_new_tokens),
            ),
            "chunk_rag": _run_base_system(
                base,
                tokenizer,
                project_generation_rows,
                project_index,
                base_device,
                mode="chunk_rag",
                rag_top_k=max(1, args.rag_top_k),
                max_new_tokens=max(1, args.max_new_tokens),
            ),
        }
        _release(base)
        # Drop the caller's reference as well.  Otherwise the original Qwen
        # remains resident while Natural Memory is loaded below, making the
        # latter's VRAM measurement include two complete 4B models.
        base = None

    print("loading Natural Memory 4B")
    memory = load_qwen_dynamic(
        _path(args.memory_model),
        load_in_4bit=use_4bit,
        max_memory=max_memory,
    )
    memory_device = memory._find_layer_device()
    natural = _run_natural_memory(
        memory,
        tokenizer,
        general_raw_records,
        general_generation_rows,
        project_raw_records,
        project_generation_rows,
        memory_device,
        generation_general=args.generation_general,
        generation_project=args.generation_project,
        max_new_tokens=max(1, args.max_new_tokens),
        encode_batch_size=max(1, args.encode_batch_size),
    )
    natural_load_vram = _vram_snapshot(memory_device)
    _release(memory)

    report = {
        "benchmark": "baseline_compare_4b",
        "base_model": str(_path(args.base_model)),
        "memory_model": str(_path(args.memory_model)),
        "quantization": "4bit_nf4" if use_4bit else "none",
        "gpu_memory_cap_gb": float(args.gpu_memory_gb),
        "general_cases": len(general_cases),
        "general_records": len(general_raw_records),
        "project": project_meta,
        "protocol": {
            "generation_general": len(general_generation_rows),
            "generation_project": len(project_generation_rows),
            "rag_top_k": int(args.rag_top_k),
            "max_new_tokens": int(args.max_new_tokens),
            "same_tokenizer": True,
            "same_sampling": "greedy",
            "memory_storage": "process_ram; selected Natural Memory records promoted to bounded GPU cache",
            "reused_original_report": str(_path(args.reuse_original_report)) if args.reuse_original_report else None,
        },
        "systems": {
            "original_qwen": {
                "load_vram": base_load_vram,
                "general": base_general,
                "project": base_project,
            },
            "natural_memory": {
                "load_vram": natural_load_vram,
                "general": natural["general"],
                "project": natural["project"],
            },
        },
        "limitations": [
            "Chunk RAG uses a CPU lexical candidate index with exact entity/attribute bonuses; it is a transparent baseline, not a hosted embedding service.",
            "The general corpus is the local native-memory benchmark; the project corpus is the current repository source and documentation.",
            "Latency for Natural Memory is split into reader_ms and generation is measured by the existing model wrapper; this first comparison prioritizes correctness and VRAM.",
        ],
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        name: {
            "general": {key: value for key, value in system["general"].items() if key != "rows"},
            "project": {key: value for key, value in system["project"].items() if key != "rows"},
        }
        for name, system in report["systems"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
