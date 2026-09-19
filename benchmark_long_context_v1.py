"""Practical long-context stress probe for the local Qwen/Natural Memory setup."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from .qwen_integration import load_qwen_base, load_qwen_dynamic, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=r"H:\Memory")
    parser.add_argument(
        "--memory-model",
        default=r"H:\Memory\V2_dpskw\qwen3_5_4b_memory_merged_v13",
    )
    parser.add_argument(
        "--output",
        default=r"H:\Memory\V2_dpskw\long_context_v1_stress.json",
    )
    parser.add_argument("--lengths", default="8192,16384,32768")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def build_prompt(tokenizer: Any, target_tokens: int, seed: int) -> tuple[str, str, int]:
    rng = random.Random(seed + target_tokens)
    answer = f"LONGCTX-{target_tokens}-{rng.randrange(100000, 999999)}"
    needle = f"唯一目标记录：检索编码为 {answer}。"
    filler = (
        "这段材料是用于压力测试的背景文本。它包含版本、日志、普通备注和无关的项目描述，"
        "不包含目标编码。阅读时应忽略这些干扰内容，只寻找唯一目标记录。"
    )
    chunks: list[str] = []
    while len(tokenizer(" ".join(chunks + [filler, needle]), add_special_tokens=False)["input_ids"]) < target_tokens:
        chunks.append(filler)
    half = len(chunks) // 2
    material = " ".join(chunks[:half] + [needle] + chunks[half:])
    prompt = (
        "请阅读下面的长材料，只输出唯一目标记录中的检索编码，不要解释。\n"
        "---开始---\n"
        f"{material}\n"
        "---结束---\n"
        "问题：唯一目标记录中的检索编码是什么？"
    )
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt, answer, prompt_tokens


def make_inputs(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
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


def probe_model(model: Any, tokenizer: Any, lengths: list[int], *, dynamic: bool, max_new_tokens: int) -> dict[str, Any]:
    device = model._find_layer_device() if dynamic else model.get_input_embeddings().weight.device
    rows = []
    for length in lengths:
        prompt, answer, prompt_tokens = build_prompt(tokenizer, length, 20260904)
        if dynamic:
            model.reset_memory(batch_size=1, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        row: dict[str, Any] = {
            "target_tokens": length,
            "prompt_tokens": prompt_tokens,
            "expected": answer,
        }
        try:
            encoded = make_inputs(tokenizer, prompt, device)
            query = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            query_ids = query["input_ids"].to(device)
            query_mask = query.get("attention_mask")
            if query_mask is None:
                query_mask = torch.ones_like(query_ids)
            query_mask = query_mask.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
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
                        }
                    )
                output = model.generate(**encoded, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            response_ids = output[0, encoded["input_ids"].shape[1] :]
            response = tokenizer.decode(response_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()
            row.update(
                {
                    "status": "ok",
                    "response": response,
                    "passed": answer in response,
                    "generated_tokens": int(response_ids.numel()),
                    "seconds": elapsed,
                    "tokens_per_second": int(response_ids.numel()) / max(elapsed, 1e-9),
                }
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            message = str(exc)
            if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower():
                row.update({"status": "cuda_oom", "error": message[:500]})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise
        if device.type == "cuda":
            row["peak_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
            row["peak_memory_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
        rows.append(row)
    return {"rows": rows}


def release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    lengths = [int(value.strip()) for value in args.lengths.split(",") if value.strip()]
    use_4bit = not args.no_4bit
    tokenizer = load_tokenizer(args.base_model)
    report: dict[str, Any] = {
        "benchmark": "Natural Memory v1 practical long-context stress probe",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": str(Path(args.base_model).resolve()),
        "memory_model": str(Path(args.memory_model).resolve()),
        "lengths": lengths,
        "quantization": "4bit_nf4" if use_4bit else "none",
        "max_new_tokens": args.max_new_tokens,
    }
    print("loading baseline")
    model = load_qwen_base(args.base_model, load_in_4bit=use_4bit)
    model.eval()
    report["baseline"] = probe_model(
        model, tokenizer, lengths, dynamic=False, max_new_tokens=args.max_new_tokens
    )
    release(model)
    print("loading Natural Memory v1")
    model = load_qwen_dynamic(args.memory_model, load_in_4bit=use_4bit)
    model.eval()
    report["natural_memory_v1"] = probe_model(
        model, tokenizer, lengths, dynamic=True, max_new_tokens=args.max_new_tokens
    )
    release(model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
