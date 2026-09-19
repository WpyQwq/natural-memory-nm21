"""Verify Natural Memory v2 across a real model restart.

The test deliberately uses a normal user turn, not ``/remember`` and not a
replayed chat history.  It writes the model-owned V2 snapshot into the
embedded memory shard, destroys the first model, reloads the package, and
checks both the bounded router decision and the generated answer.

By default the test restores an empty memory snapshot at the end.  Use
``--keep-memory`` only when the test fact should remain in the package.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.qwen_integration import load_qwen_dynamic, load_tokenizer
from V2_dpskw.stream_chat_qwen_memory import _chat_tensor, _write_turn


def _raw_query(tokenizer, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask.to(device)


@torch.inference_mode()
def _answer(model, tokenizer, query: str, max_new_tokens: int) -> dict[str, object]:
    device = model._find_layer_device()
    encoded = _chat_tensor(tokenizer, query)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    query_ids, query_mask = _raw_query(tokenizer, query, device)
    prefix_ids, prefix_mask, prefix_length = model._build_text_prefix(
        query_ids,
        query_mask,
        query_text=query,
    )
    decisions = [dict(item) for item in model.runtime.v2_last_decisions]
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        update_memory=False,
        memory_query_input_ids=query_ids,
        memory_query_attention_mask=query_mask,
        memory_query_text=query,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_length = int(encoded["input_ids"].shape[1])
    response = tokenizer.decode(
        output[0, prompt_length:].detach().cpu().tolist(),
        skip_special_tokens=True,
    ).strip()
    return {
        "response": response,
        "prefix_length": int(prefix_length),
        "prefix_used": bool(model.runtime.text_prefix_used),
        "decision": decisions,
        "prefix_shape": list(prefix_ids.shape) if isinstance(prefix_ids, torch.Tensor) else None,
        "prefix_mask_shape": list(prefix_mask.shape) if isinstance(prefix_mask, torch.Tensor) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=r"H:\Memory\V2_dpskw\qwen3_5_4b_natural_memory_v2",
    )
    parser.add_argument(
        "--report",
        default=r"H:\Memory\V2_dpskw\natural_memory_v2_restart_test.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--keep-memory",
        action="store_true",
        help="leave the controlled test fact embedded after the test",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not (model_path / "memory_merge.json").exists():
        raise FileNotFoundError(f"Natural Memory v2 package not found: {model_path}")

    fact = "我正在开发一个长期项目，项目内部代号是NM-V2-RESTART，使用中文。"
    query = "我正在开发的长期项目内部代号是什么？只回答代号。"
    expected = "NM-V2-RESTART"
    tokenizer = load_tokenizer(model_path)

    first = load_qwen_dynamic(model_path, load_in_4bit=not args.no_4bit)
    first.eval()
    device = first._find_layer_device()
    # This is a destructive reset of the selected package's durable memory,
    # so the command is an explicit test tool rather than a chat startup hook.
    first.reset_memory(batch_size=1, device=device)
    changed = _write_turn(first, tokenizer, fact, device, force_write=False)
    before_save = first.memory_v2_stats()
    first.save_embedded_memory_weights(model_path)
    del first
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restarted = load_qwen_dynamic(model_path, load_in_4bit=not args.no_4bit)
    restarted.eval()
    after_restart = _answer(restarted, tokenizer, query, args.max_new_tokens)
    selected_text = " ".join(
        str(item.get("text", ""))
        for item in restarted.runtime.v2_last_decisions
        if isinstance(item, dict)
    )
    route_recalled = expected in selected_text or (
        bool(after_restart["prefix_used"]) and int(after_restart["prefix_length"]) > 0
    )

    after_cleanup = None
    if not args.keep_memory:
        restarted.reset_memory(batch_size=1, device=restarted._find_layer_device())
        restarted.save_embedded_memory_weights(model_path)
        after_cleanup = restarted.memory_v2_stats()

    report = {
        "model_path": str(model_path),
        "history_passed_to_restart": False,
        "fact": fact,
        "query": query,
        "expected": expected,
        "automatic_write": True,
        "write_changed": bool(changed),
        "before_save": before_save,
        "after_restart": after_restart,
        "router_recalled_after_restart": bool(route_recalled),
        "generated_contains_expected": expected in str(after_restart["response"]),
        "cleanup_applied": not args.keep_memory,
        "after_cleanup": after_cleanup,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
