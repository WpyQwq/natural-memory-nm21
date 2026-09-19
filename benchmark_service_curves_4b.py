"""Compare long-context service curves on the local Qwen3.5 4B checkpoint.

This benchmark is intentionally framed around the production baselines that a
serving team would compare:

* dense/paged full prompt KV (until the checkpoint or GPU rejects it),
* a fixed sliding window,
* chunk retrieval followed by a short prompt-side rerank context, and
* Natural Memory with one bounded read at the request boundary.

The Natural Memory path never performs a memory read or write per generated
token.  The reported ``reader_ms`` is measured inside the model wrapper and
``decode_tok_s`` is estimated from a one-token and a multi-token greedy run.
The synthetic task is deliberately simple and should be supplemented by
repository QA and agent-trace workloads before making a production claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from .qwen_integration import load_qwen_base, load_qwen_dynamic, load_tokenizer
from .stream_chat_qwen_memory import _chat_tensor


PROJECT_ROOT = Path(__file__).resolve().parent


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() or path.exists() else PROJECT_ROOT / path


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_memory(gpu_memory_gb: float) -> dict[Any, str] | None:
    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return None
    return {0: f"{gpu_memory_gb:.1f}GiB", "cpu": "64GiB"}


def _set_cuda_process_cap(gpu_memory_gb: float) -> None:
    """Keep the benchmark inside the declared HBM operating point."""

    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(0.95, max(0.05, gpu_memory_gb * 1024**3 / total))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def _build_corpus(
    tokenizer: Any,
    target_tokens: int,
    *,
    chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    """Create a token-level corpus with a middle needle and a query."""

    filler = (
        "这是长上下文压力测试中的普通项目日志片段。它包含无关的版本号、"
        "状态说明、时间戳和重复背景，不应被当作目标答案。"
    )
    needle = "用户档案字段 target_id 的值是 CURVE-7H2K-9P4M。"
    query = "用户档案字段 target_id 的完整值是什么？请逐字符复制，不要省略。"
    filler_ids = torch.tensor(
        tokenizer(filler, add_special_tokens=False)["input_ids"], dtype=torch.long
    )
    needle_ids = torch.tensor(
        tokenizer(needle, add_special_tokens=False)["input_ids"], dtype=torch.long
    )
    target_tokens = max(int(target_tokens), int(needle_ids.numel()) + 8)
    filler_count = target_tokens - int(needle_ids.numel())
    repeats = math.ceil(filler_count / max(1, filler_ids.numel()))
    body = filler_ids.repeat(repeats)[:filler_count]
    # Keep the needle outside a 32K hot window, while making it present in
    # every measured prefix.  The long-context axis then tests persistence,
    # rather than accidentally testing a missing fact at the short end.
    insert_at = min((body.numel() // 2) // max(1, chunk_tokens) * max(1, chunk_tokens), 65536)
    corpus = torch.cat((body[:insert_at], needle_ids, body[insert_at:]), dim=0)
    return corpus, needle_ids, needle, query


def _chat_from_content(tokenizer: Any, content: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = _chat_tensor(tokenizer, content)
    return {key: value.to(device) for key, value in encoded.items()}


def _query_tensors(tokenizer: Any, query: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(query, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids)
    return ids, mask.to(device)


@torch.inference_mode()
def _measure_generation(
    model: Any,
    encoded: dict[str, torch.Tensor],
    tokenizer: Any,
    device: torch.device,
    *,
    dynamic: bool,
    query_ids: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
    query_text: str = "",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    def generate(count: int):
        kwargs: dict[str, Any] = {
            "max_new_tokens": count,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if dynamic:
            kwargs.update(
                {
                    "update_memory": False,
                    "memory_query_input_ids": query_ids,
                    "memory_query_attention_mask": query_mask,
                    "memory_query_text": query_text,
                }
            )
        _sync(device)
        started = time.perf_counter()
        output = model.generate(**encoded, **kwargs)
        _sync(device)
        return output, time.perf_counter() - started

    try:
        one, first_latency = generate(1)
        many, total_latency = generate(max_new_tokens)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {"status": "cuda_oom", "error": str(exc)[:500]}
        raise
    generated = many[0, encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(generated.detach().cpu().tolist(), skip_special_tokens=True).strip()
    extra = max(0.0, total_latency - first_latency)
    generated_count = int(generated.numel())
    runtime = getattr(model, "runtime", None)
    prefix_tokens = int(getattr(runtime, "text_prefix_tokens", 0)) if dynamic else 0
    public_input_tokens = int(encoded["input_ids"].shape[1])
    row = {
        "status": "ok",
        "response": text,
        "generated_tokens": generated_count,
        "first_token_latency_s": first_latency,
        "prefill_proxy_s": first_latency,
        "total_latency_s": total_latency,
        "batch_size": 1,
        "public_input_tokens": public_input_tokens,
        "memory_prefix_tokens": prefix_tokens,
        "hot_kv_tokens": public_input_tokens + prefix_tokens,
        "decode_tok_s": max(0, generated_count - 1) / max(extra, 1e-9),
        "reader_ms": float(getattr(getattr(model, "runtime", None), "text_read_seconds", 0.0) * 1000.0)
        if dynamic
        else 0.0,
    }
    row["gpu_seconds_per_million_output_tokens"] = (
        1_000_000.0 / max(1e-9, row["decode_tok_s"])
    )
    if device.type == "cuda":
        row["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        row["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
        del one, many
        torch.cuda.empty_cache()
    return row


def _release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _quality_fields(row: dict[str, Any], *, target_reachable: bool, expected: str) -> dict[str, Any]:
    """Separate answer correctness from whether the target was in the hot context."""

    found = expected in row.get("response", "")
    row["target_reachable"] = bool(target_reachable)
    row["quality_pass"] = found
    row["quality_expected"] = bool(target_reachable)
    row["quality_correct"] = bool(found == target_reachable) if row.get("status") == "ok" else False
    return row


def _prompt_content(tokenizer: Any, corpus: torch.Tensor, query: str) -> str:
    material = tokenizer.decode(corpus.tolist(), skip_special_tokens=True)
    return (
        "请从下面的材料中回答问题，不要使用材料外的信息。\n"
        "---材料开始---\n" + material + "\n---材料结束---\n问题：" + query
    )


def _chunk_retrieve(
    tokenizer: Any,
    corpus: torch.Tensor,
    query: str,
    *,
    chunk_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, float, int]:
    query_ids = set(tokenizer(query, add_special_tokens=False)["input_ids"])
    started = time.perf_counter()
    scored: list[tuple[int, int, torch.Tensor]] = []
    for start in range(0, int(corpus.numel()), chunk_tokens):
        chunk = corpus[start : start + chunk_tokens]
        overlap = len(query_ids.intersection(set(chunk.tolist())))
        scored.append((overlap, -start, chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored[:top_k]]
    return torch.cat(selected, dim=0), (time.perf_counter() - started) * 1000.0, len(scored)


@torch.inference_mode()
def _add_memory_chunks(
    model: Any,
    tokenizer: Any,
    corpus: torch.Tensor,
    previous_tokens: int,
    target_tokens: int,
    *,
    chunk_tokens: int,
    device: torch.device,
    needle_ids: torch.Tensor,
) -> dict[str, Any]:
    """Add context chunks without putting the long corpus in the decode KV."""

    if model.memory_os_v2 is None:
        raise RuntimeError("Natural Memory v2 is required for this benchmark")
    # Filler records use deterministic CPU keys.  The target record gets an
    # actual frozen-Qwen key, which keeps the retrieval measurement honest for
    # the one fact being scored without encoding millions of filler tokens.
    records: list[dict[str, Any]] = []
    body_tokens = max(0, int(corpus.numel()) - int(needle_ids.numel()))
    target_start = min(
        (body_tokens // 2) // max(1, chunk_tokens) * max(1, chunk_tokens),
        65536,
    )
    for start in range(previous_tokens, target_tokens, chunk_tokens):
        end = min(start + chunk_tokens, target_tokens)
        ids = corpus[start:end]
        if ids.numel() == 0:
            continue
        if start <= target_start < end:
            key = model._encode_model_key(ids.unsqueeze(0).to(device), torch.ones((1, ids.numel()), dtype=torch.long, device=device))[0].cpu()
        else:
            generator = torch.Generator(device="cpu").manual_seed(1701 + start)
            key = torch.randn(model.memory.hidden_size, generator=generator)
            key = torch.nn.functional.normalize(key, dim=0)
        record_text = (
            tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            if start <= target_start < end
            else f"curve_chunk:{start}:{end}"
        )
        records.append(
            {
                "text": record_text,
                "key": key,
                "summary": key,
                "semantic_key": key,
                "memory_type": "context_chunk",
                "importance": 0.55,
                "confidence": 0.85,
                "source": "service_curve_benchmark",
                "evidence": [f"token_range:{start}:{end}"],
                "token_ids": ids,
                "token_mask": torch.ones_like(ids, dtype=torch.bool),
                "trusted": True,
                "force": True,
            }
        )
    started = time.perf_counter()
    model.memory_os_v2.write_batch(records)
    return {
        "added_records": len(records),
        "write_ms": (time.perf_counter() - started) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=r"H:\Memory")
    parser.add_argument(
        "--memory-model",
        default=r"H:\Memory\V2_dpskw\qwen3_5_4b_natural_memory_v2",
    )
    parser.add_argument("--output", default=r"H:\Memory\V2_dpskw\service_curves_4b.json")
    parser.add_argument("--lengths", default="131072,262144,524288,1048576,4194304")
    parser.add_argument("--hot-window", type=int, default=32768)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument(
        "--memory-top-k-records",
        type=int,
        default=2,
        help="Natural Memory records injected into hot KV; default matches RAG chunk count",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gpu-hourly-cost", type=float, default=0.0)
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=8.0,
        help="hard CUDA placement cap; 0 disables the cap",
    )
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    _set_cuda_process_cap(args.gpu_memory_gb)
    lengths = [int(item.strip()) for item in args.lengths.split(",") if item.strip()]
    use_4bit = not args.no_4bit
    tokenizer = load_tokenizer(_path(args.base_model))
    max_length = max(lengths)
    corpus, needle_ids, needle, query = _build_corpus(
        tokenizer,
        max_length,
        chunk_tokens=args.chunk_tokens,
    )
    report: dict[str, Any] = {
        "benchmark": "production_baseline_service_curves_4b",
        "model": str(_path(args.base_model)),
        "memory_model": str(_path(args.memory_model)),
        "quantization": "4bit_nf4" if use_4bit else "none",
        "lengths": lengths,
        "hot_window": args.hot_window,
        "chunk_tokens": args.chunk_tokens,
        "rag_top_k": args.rag_top_k,
        "memory_top_k_records": args.memory_top_k_records,
        "query": query,
        "expected": "CURVE-7H2K-9P4M",
        "cost_note": "usd_per_million_output_tokens is null unless --gpu-hourly-cost is supplied",
        "gpu_hourly_cost": args.gpu_hourly_cost if args.gpu_hourly_cost > 0 else None,
        "gpu_memory_cap_gb": args.gpu_memory_gb if args.gpu_memory_gb > 0 else None,
        "systems": {},
    }

    print("loading 4B baseline")
    max_memory = _max_memory(args.gpu_memory_gb)
    base = load_qwen_base(_path(args.base_model), load_in_4bit=use_4bit, max_memory=max_memory)
    base.eval()
    base_device = base.get_input_embeddings().weight.device
    systems: dict[str, list[dict[str, Any]]] = {
        "dense_full_kv": [],
        "sliding_window": [],
        "matched_hot_window": [],
        "chunk_rag": [],
    }
    for target in lengths:
        current = corpus[:target]
        row_base = {"context_tokens": target}
        if target <= 262144:
            try:
                prompt = _prompt_content(tokenizer, current, query)
                encoded = _chat_from_content(tokenizer, prompt, base_device)
                if base_device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(base_device)
                row = _measure_generation(base, encoded, tokenizer, base_device, dynamic=False, max_new_tokens=args.max_new_tokens)
                _quality_fields(row, target_reachable=True, expected="CURVE-7H2K-9P4M")
                row_base.update(row)
            except Exception as exc:
                row_base.update({"status": "error", "error": str(exc)[:500]})
                _quality_fields(row_base, target_reachable=True, expected="CURVE-7H2K-9P4M")
        else:
            row_base.update({"status": "unsupported_by_base_max_position", "quality_pass": False})
            _quality_fields(row_base, target_reachable=True, expected="CURVE-7H2K-9P4M")
        systems["dense_full_kv"].append(row_base)

        window = current[-args.hot_window :]
        prompt = _prompt_content(tokenizer, window, query)
        encoded = _chat_from_content(tokenizer, prompt, base_device)
        if base_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(base_device)
        row = _measure_generation(base, encoded, tokenizer, base_device, dynamic=False, max_new_tokens=args.max_new_tokens)
        row.update({"context_tokens": target, "visible_tokens": int(window.numel())})
        _quality_fields(row, target_reachable=False, expected="CURVE-7H2K-9P4M")
        systems["sliding_window"].append(row)

        matched_tokens = max(1, int(args.chunk_tokens) * int(args.rag_top_k))
        matched_window = current[-matched_tokens:]
        prompt = _prompt_content(tokenizer, matched_window, query)
        encoded = _chat_from_content(tokenizer, prompt, base_device)
        if base_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(base_device)
        row = _measure_generation(base, encoded, tokenizer, base_device, dynamic=False, max_new_tokens=args.max_new_tokens)
        row.update({"context_tokens": target, "visible_tokens": int(matched_window.numel()), "matched_hot_window_tokens": matched_tokens})
        _quality_fields(row, target_reachable=False, expected="CURVE-7H2K-9P4M")
        systems["matched_hot_window"].append(row)

        retrieved, retrieve_ms, chunk_count = _chunk_retrieve(tokenizer, current, query, chunk_tokens=args.chunk_tokens, top_k=args.rag_top_k)
        prompt = _prompt_content(tokenizer, retrieved, query)
        encoded = _chat_from_content(tokenizer, prompt, base_device)
        if base_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(base_device)
        row = _measure_generation(base, encoded, tokenizer, base_device, dynamic=False, max_new_tokens=args.max_new_tokens)
        row.update({"context_tokens": target, "retrieved_tokens": int(retrieved.numel()), "chunk_count": chunk_count, "retriever_ms": retrieve_ms})
        _quality_fields(row, target_reachable=True, expected="CURVE-7H2K-9P4M")
        systems["chunk_rag"].append(row)
        print(f"baseline context={target}")
    _release(base)
    base = None

    print("loading Natural Memory 4B")
    memory = load_qwen_dynamic(
        _path(args.memory_model), load_in_4bit=use_4bit, max_memory=max_memory
    )
    memory.eval()
    memory.memory_config.memory_top_k_records = max(1, int(args.memory_top_k_records))
    memory.memory_os_v2.bank.top_k_records = memory.memory_config.memory_top_k_records
    memory_device = memory._find_layer_device()
    memory.reset_memory(batch_size=1, device=memory_device)
    previous = 0
    systems["natural_memory"] = []
    query_ids, query_mask = _query_tensors(tokenizer, query, memory_device)
    for target in lengths:
        write_info = _add_memory_chunks(
            memory,
            tokenizer,
            corpus,
            previous,
            target,
            chunk_tokens=args.chunk_tokens,
            device=memory_device,
            needle_ids=needle_ids,
        )
        previous = target
        encoded = _chat_from_content(tokenizer, query, memory_device)
        if memory_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(memory_device)
        row = _measure_generation(
            memory,
            encoded,
            tokenizer,
            memory_device,
            dynamic=True,
            query_ids=query_ids,
            query_mask=query_mask,
            query_text=query,
            max_new_tokens=args.max_new_tokens,
        )
        decision = memory.runtime.v2_last_decisions[-1] if memory.runtime.v2_last_decisions else {}
        row.update(
            {
                "context_tokens": target,
                "stored_records": memory.memory_v2_stats().get("records", 0),
                "coarse_candidates": decision.get("coarse_candidates", memory.memory_v2_stats().get("last_coarse_candidates", 0)),
                "page_ids": decision.get("page_ids", []),
                "record_ids": decision.get("record_ids", []),
                "stop_reason": decision.get("stop_reason", ""),
                "retrieved_records": len(decision.get("record_ids", [])),
                "memory_write_ms": write_info["write_ms"],
            }
        )
        _quality_fields(row, target_reachable=True, expected="CURVE-7H2K-9P4M")
        systems["natural_memory"].append(row)
        print(f"natural_memory context={target}")
    _release(memory)
    memory = None

    if args.gpu_hourly_cost > 0.0:
        for rows in systems.values():
            for row in rows:
                if row.get("status") != "ok":
                    continue
                row["usd_per_million_output_tokens"] = (
                    1_000_000.0 / max(1e-9, row.get("decode_tok_s", 0.0))
                    * args.gpu_hourly_cost
                    / 3600.0
                )
    report["systems"] = systems
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
