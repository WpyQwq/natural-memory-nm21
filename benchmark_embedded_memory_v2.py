"""Low-pressure end-to-end test for the embedded Natural Memory v2 path.

This benchmark intentionally uses only the model package's third safetensors
memory shard.  It does not create SQLite files or exercise disk paging.  The
long-context cases lower the temporary KV budget so the test measures the
model-owned archive/read path without asking a 12 GiB GPU to hold a huge KV.
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

from .qwen_integration import load_qwen_dynamic, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=r"H:\Memory\V2_dpskw\qwen3_5_4b_natural_memory_v2",
    )
    parser.add_argument("--lengths", default="4096,8192,16384,32768")
    parser.add_argument("--kv-budget", type=int, default=2048)
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--output",
        default=r"H:\Memory\V2_dpskw\embedded_memory_v2_long_benchmark.json",
    )
    return parser.parse_args()


def build_prompt(tokenizer: Any, target_tokens: int, seed: int) -> tuple[str, str, int]:
    rng = random.Random(seed + target_tokens)
    answer = f"EMBEDDED-LONG-{target_tokens}-{rng.randrange(100000, 999999)}"
    needle = f"长期记忆锚点：唯一编号是 {answer}。"
    filler = (
        "这是长文本记忆压力测试中的普通背景段落，包含项目说明、日期、日志和无关备注。"
        "这些内容不是问题答案，读取时应保留原文但忽略干扰。"
    )
    chunks: list[str] = []
    while len(tokenizer(" ".join(chunks + [filler, needle]), add_special_tokens=False)["input_ids"]) < target_tokens:
        chunks.append(filler)
    # Keep the needle safely inside the archived prefix even in the smallest
    # case, so a pass must come from memory rather than the retained window.
    pivot = max(1, len(chunks) // 3)
    material = " ".join(chunks[:pivot] + [needle] + chunks[pivot:])
    prompt = (
        "请阅读下面的长材料，回答末尾问题，只输出编号，不要解释。\n"
        "---开始材料---\n"
        f"{material}\n"
        "---结束材料---\n"
        "问题：长期记忆锚点的唯一编号是什么？"
    )
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt, answer, prompt_tokens


def chat_inputs(tokenizer: Any, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def run_case(model: Any, tokenizer: Any, target_tokens: int, args: argparse.Namespace) -> dict[str, Any]:
    device = model._find_layer_device()
    model.reset_memory(batch_size=1, device=device)
    assert model.memory_os_v2 is not None
    model.memory_os_v2.kv_budget.max_tokens = int(args.kv_budget)
    prompt, answer, prompt_tokens = build_prompt(tokenizer, target_tokens, 20260904)
    encoded = chat_inputs(tokenizer, prompt, device)
    query_text = "长期记忆锚点的唯一编号"
    query = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
    query_ids = query["input_ids"].to(device)
    query_mask = query.get("attention_mask")
    if query_mask is None:
        query_mask = torch.ones_like(query_ids)
    query_mask = query_mask.to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            update_memory=False,
            pad_token_id=tokenizer.pad_token_id,
            memory_query_input_ids=query_ids,
            memory_query_attention_mask=query_mask,
            memory_query_text=query_text,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    response_ids = output[0, encoded["input_ids"].shape[1] :]
    response = tokenizer.decode(response_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()
    stats = model.memory_v2_stats()
    records = model.memory_os_v2.bank.records
    archived_text_hit = any(
        answer in tokenizer.decode(record.token_ids.tolist(), skip_special_tokens=True)
        for record in records.values()
        if record.memory_type == "context_chunk" and isinstance(record.token_ids, torch.Tensor)
    )
    query_key = model._encode_model_key(query_ids, query_mask)[0]
    retrieved, decision = model.read_hierarchical_memory(
        query_key,
        query_text=query_text,
        query_token_ids=query_ids[0],
        top_k_pages=model.memory_config.memory_top_k_pages,
        top_k_records=model.memory_config.memory_top_k_records,
        max_hops=model.memory_config.memory_max_hops,
    )
    retrieved_hit = any(
        answer in tokenizer.decode(record.token_ids.tolist(), skip_special_tokens=True)
        for record in retrieved
        if isinstance(record.token_ids, torch.Tensor)
    )
    return {
        "target_tokens": target_tokens,
        "prompt_tokens": prompt_tokens,
        "kv_budget_tokens": args.kv_budget,
        "chunk_tokens": args.chunk_tokens,
        "response": response,
        "expected": answer,
        "generation_hit": answer in response,
        "archived_text_hit": archived_text_hit,
        "retrieved_text_hit": retrieved_hit,
        "retrieved_records": len(retrieved),
        "router_stop_reason": decision.stop_reason,
        "router_hop_count": decision.hop_count,
        "seconds": elapsed,
        "records": stats.get("records", 0),
        "pages": stats.get("pages", 0),
        "gpu_cache_records": stats.get("gpu_cache_records", 0),
        "gpu_cache_tokens": stats.get("gpu_cache_tokens", 0),
        "gpu_cache_device": stats.get("gpu_cache_device", "none"),
        "passed": bool(archived_text_hit and retrieved_hit),
    }


def main() -> None:
    args = parse_args()
    lengths = [int(item.strip()) for item in args.lengths.split(",") if item.strip()]
    tokenizer = load_tokenizer(args.model_path)
    model = load_qwen_dynamic(args.model_path, load_in_4bit=not args.no_4bit)
    model.eval()
    model.memory_config.context_chunk_tokens = int(args.chunk_tokens)
    report: dict[str, Any] = {
        "benchmark": "Natural Memory v2 embedded third-shard long-memory test",
        "storage_mode": model.memory_config.memory_storage_mode,
        "tier_store_enabled": bool(model.memory_os_v2 and model.memory_os_v2.bank.tier_store is not None),
        "model_path": str(Path(args.model_path).resolve()),
        "lengths": lengths,
        "quantization": "4bit_nf4" if not args.no_4bit else "none",
        "kv_budget_tokens": args.kv_budget,
        "chunk_tokens": args.chunk_tokens,
        "rows": [],
    }
    try:
        if report["storage_mode"] != "embedded" or report["tier_store_enabled"]:
            raise RuntimeError("embedded benchmark requires memory_storage_mode=embedded and no tier store")
        for target_tokens in lengths:
            row = run_case(model, tokenizer, target_tokens, args)
            report["rows"].append(row)
            print(json.dumps(row, ensure_ascii=False))
    finally:
        model.reset_memory(batch_size=1, device=model._find_layer_device())
        model.close_memory_storage()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report["passed_cases"] = sum(bool(row["passed"]) for row in report["rows"])
    report["total_cases"] = len(report["rows"])
    report["all_passed"] = report["passed_cases"] == report["total_cases"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
