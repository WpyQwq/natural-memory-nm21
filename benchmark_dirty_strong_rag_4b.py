"""Evaluate the production-style local dense+rereank baseline on dirty data.

This is the companion baseline for ``benchmark_dirty_real_corpus_4b``.  It
uses the same real repository corpus, the same user-dialogue slice, the same
queries, tokenizer, greedy decoding and GPU cap.  Retrieval is CPU-only:
local BERT embeddings provide recall and a bounded transparent reranker uses
lexical/entity/attribute signals.  It is stronger than lexical Chunk RAG,
but is explicitly not presented as a trained public cross-encoder.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .benchmark_compare_baselines_4b import (
    _base_generate,
    _rag_prompt,
    _vram_snapshot,
)
from .benchmark_dirty_real_corpus_4b import (
    _build_all_records,
    _dirty_source_files,
    _is_refusal_strict,
    _path,
    _quality_pass,
    _retrieval_hit,
    _summarize,
)
from .benchmark_real_scale_memory_4b import _max_memory, _set_cuda_process_cap
from .strong_rag_baseline import LocalEmbeddingReranker, _resolve_local_encoder
from .qwen_integration import load_qwen_base, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent


def _run_dirty(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    index: LocalEmbeddingReranker,
    device: torch.device,
    *,
    top_k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, 1):
        if row_index == 1 or row_index % 32 == 0 or row_index == len(rows):
            print(f"Strong RAG generation: {row_index}/{len(rows)}", flush=True)
        query = str(row["query"])
        retrieved, retrieve_ms = index.retrieve(query, top_k=top_k)
        prompt = _rag_prompt(query, retrieved)
        generated = _base_generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=max_new_tokens,
        )
        selected = [
            {
                "value": item.get("value", ""),
                "text_preview": item.get("text", ""),
                "entity": item.get("entity", ""),
                "attribute": item.get("attribute", ""),
            }
            for item in retrieved
        ]
        response = str(generated.get("response", ""))
        output_rows.append(
            {
                **row,
                "response": response,
                "correct": _quality_pass(response, row),
                "retrieval_target_found": _retrieval_hit(selected, row),
                "retrieved_values": [item.get("value", "") for item in retrieved],
                "retrieved_ids": [],
                "selected_records": selected,
                "prefix_used": bool(retrieved),
                "prefix_tokens": int(generated.get("prompt_tokens", 0)),
                "reader_ms": float(retrieve_ms),
                "retriever_ms": float(retrieve_ms),
                "prompt_tokens": int(generated.get("prompt_tokens", 0)),
                "generated_tokens": int(generated.get("generated_tokens", 0)),
                "total_latency_s": float(generated.get("total_latency_s", 0.0)),
                "decode_tok_s": float(generated.get("decode_tok_s", 0.0)),
                "allocated_gb": generated.get("allocated_gb"),
                "reserved_gb": generated.get("reserved_gb"),
            }
        )
    return _summarize(output_rows, include_reader=False)


def _release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=r"H:\Memory")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "dirty_strong_rag_compare_4b.json"))
    parser.add_argument("--project-records", type=int, default=8192)
    parser.add_argument("--project-targets", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=384)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-gb", type=float, default=10.0)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    _set_cuda_process_cap(args.gpu_memory_gb)

    tokenizer = load_tokenizer(_path(args.base_model))
    files = _dirty_source_files()
    user_records, user_rows, code_records, code_rows, corpus_meta = _build_all_records(
        files,
        project_records=max(128, int(args.project_records)),
        project_targets=max(1, int(args.project_targets)),
    )
    all_records = user_records + code_records
    all_rows = user_rows + code_rows
    print(
        f"dirty corpus: files={len(files)} real_chunks={corpus_meta['stored_real_chunk_records']} "
        f"semantic_targets={corpus_meta['semantic_target_records']} user_records={len(user_records)} "
        f"queries={len(all_rows)}"
    )

    encoder_path = _resolve_local_encoder(args.embedding_model)
    print(f"loading local embedding model on CPU: {encoder_path}")
    embedder = LocalEmbeddingReranker(
        encoder_path,
        max_length=args.embedding_max_length,
        batch_size=args.embedding_batch_size,
        candidate_k=args.candidate_k,
    )
    started = time.perf_counter()
    embedder.add(all_records)
    index_seconds = time.perf_counter() - started
    print(f"dense_index_seconds={index_seconds:.3f}")

    print("loading original Qwen3.5 4B for strong-RAG generation")
    base = load_qwen_base(
        _path(args.base_model),
        load_in_4bit=not args.no_4bit,
        max_memory=_max_memory(args.gpu_memory_gb),
    )
    base.eval()
    device = base.get_input_embeddings().weight.device
    load_vram = _vram_snapshot(device)
    result = _run_dirty(
        base,
        tokenizer,
        all_rows,
        embedder,
        device,
        top_k=max(1, args.top_k),
        max_new_tokens=max(1, args.max_new_tokens),
    )
    _release(base)

    report = {
        "benchmark": "dirty_strong_rag_compare_4b",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_model": str(_path(args.base_model)),
        "embedding_model": str(encoder_path),
        "quantization": "4bit_nf4" if not args.no_4bit else "none",
        "gpu_memory_cap_gb": float(args.gpu_memory_gb),
        "corpus": corpus_meta,
        "query_counts": {
            "all": len(all_rows),
            "user_dialogue": len(user_rows),
            "real_repository": len(code_rows),
            "answerable": sum(int(row["answerable"]) for row in all_rows),
            "unknown": sum(int(not row["answerable"]) for row in all_rows),
            "by_category": dict(Counter(str(row["category"]) for row in all_rows)),
        },
        "protocol": {
            "same_tokenizer": True,
            "same_sampling": "greedy",
            "top_k": int(args.top_k),
            "candidate_k": int(args.candidate_k),
            "embedding_device": "cpu",
            "embedding_max_length": int(args.embedding_max_length),
            "embedding_batch_size": int(args.embedding_batch_size),
            "max_new_tokens": int(args.max_new_tokens),
            "baseline_note": "local BERT dense retrieval plus fixed transparent reranker; not a trained public cross-encoder",
        },
        "index_build": {
            "records": len(all_records),
            "indexed_records": len(embedder.records),
            "seconds": index_seconds,
        },
        "system": {
            "load_vram": load_vram,
            "all": result,
        },
        "limitations": [
            "The embedding encoder is local bert-base-chinese and CPU-only.",
            "The reranker is a fixed transparent feature reranker, not a trained public cross-encoder.",
            "The same strict answer-anchor evaluator is used for Natural Memory comparison.",
            "Embedding/index build time is reported separately from per-query retrieval and generation.",
        ],
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"strong_rag": _compact(result), "index_build": report["index_build"]}, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
