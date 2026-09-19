"""Smoke-test the model-owned KV offload and long-context compaction paths.

The test intentionally uses the real Qwen package.  It verifies that
Transformers' CPU-backed DynamicCache can generate through Qwen3.5's hybrid
linear/full-attention stack and that an over-budget prompt is archived into
Natural Memory before only the recent hot window is passed to generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.qwen_integration import load_qwen_dynamic, load_tokenizer
from V2_dpskw.stream_chat_qwen_memory import _chat_tensor


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=r"H:\Memory\V2_dpskw\qwen3_5_4b_natural_memory_v2",
    )
    parser.add_argument(
        "--report",
        default=r"H:\Memory\V2_dpskw\kv_offload_compaction_test.json",
    )
    parser.add_argument("--compact-hot-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    model = load_qwen_dynamic(model_path, load_in_4bit=not args.no_4bit)
    model.eval()
    model.memory_config.kv_offload = True
    device = model._find_layer_device()

    answer_inputs = {
        key: value.to(device)
        for key, value in _chat_tensor(tokenizer, "请用一句话说明你支持什么。").items()
    }
    generated = model.generate(
        **answer_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        update_memory=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    answer = tokenizer.decode(generated[0].detach().cpu().tolist(), skip_special_tokens=True).strip()

    if model.memory_os_v2 is None:
        raise RuntimeError("the selected package does not contain Natural Memory v2")
    model.memory_os_v2.kv_budget.max_tokens = int(args.compact_hot_tokens)
    model.memory_os_v2.kv_budget.keep_recent_tokens = min(
        model.memory_os_v2.kv_budget.keep_recent_tokens,
        model.memory_os_v2.kv_budget.max_tokens,
    )
    model.memory_config.context_chunk_tokens = max(4, min(16, args.compact_hot_tokens // 2))
    long_text = " ".join(
        [
            "历史上下文片段用于验证 Natural Memory 的自动分页压缩。",
            "这段内容应该被写入长期上下文页面，而不是继续占用当前热 KV。",
        ]
        * 12
    )
    encoded = tokenizer(long_text, add_special_tokens=False, return_tensors="pt")
    long_ids = encoded["input_ids"].to(device)
    long_mask = encoded.get("attention_mask")
    if long_mask is None:
        long_mask = torch.ones_like(long_ids)
    long_mask = long_mask.to(device)
    before = model.memory_v2_stats()
    generated_long = model.generate(
        input_ids=long_ids,
        attention_mask=long_mask,
        max_new_tokens=1,
        do_sample=False,
        update_memory=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    compaction = dict(model.runtime.context_compaction or {})
    after = model.memory_v2_stats()
    compaction_pass = bool(
        compaction["compacted"]
        and compaction["archived_records"] > 0
        and max(compaction["retained_tokens"]) <= args.compact_hot_tokens
        and int(compaction["retained_tokens"][0]) <= args.compact_hot_tokens
        and after["active_records"] >= before["active_records"] + compaction["archived_records"]
    )

    report = {
        "model_path": str(model_path),
        "transformers_cache": "DynamicCache(offloading=True)",
        "kv_offload_pass": True,
        "answer": answer,
        "compaction": compaction,
        "compaction_pass": compaction_pass,
        "long_generation_tokens": int(generated_long.shape[1]),
        "memory_before": before,
        "memory_after": after,
        "device": str(device),
        "cuda_memory_allocated_bytes": (
            int(torch.cuda.memory_allocated(device)) if torch.cuda.is_available() else 0
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not compaction_pass:
        raise SystemExit("context compaction verification failed")


if __name__ == "__main__":
    main()
