"""Production-oriented natural-language memory acceptance benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from .qwen_integration import (
    DEFAULT_MEMORY_RESET_TOKEN,
    load_memory_config,
    load_qwen_dynamic,
    load_tokenizer,
    resolve_memory_reset_token,
)


def _chat_tensor(tokenizer, messages, *, add_generation_prompt: bool):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    return {
        key: value
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def _memory_system_prefix(tokenizer, content: str):
    full = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": content},
            {"role": "user", "content": "__memory_query_boundary__"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    input_ids = full["input_ids"]
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    positions = (input_ids[0] == int(im_start)).nonzero(as_tuple=False).flatten()
    if positions.numel() < 2:
        raise RuntimeError("could not locate the system/user memory boundary")
    end = int(positions[1].item())
    return {
        "input_ids": input_ids[:, :end],
        "attention_mask": torch.ones((1, end), dtype=torch.long),
    }


@torch.inference_mode()
def _generate(model, tokenizer, user_text: str, max_new_tokens: int) -> str:
    encoded = _chat_tensor(
        tokenizer,
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
    )
    device = model._find_layer_device()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    query = tokenizer(user_text, add_special_tokens=False, return_tensors="pt")
    query_ids = query["input_ids"].to(device)
    query_mask = query.get("attention_mask")
    if query_mask is None:
        query_mask = torch.ones_like(query_ids)
    query_mask = query_mask.to(device)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        update_memory=False,
        memory_query_input_ids=query_ids,
        memory_query_attention_mask=query_mask,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    response_ids = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(response_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()


@torch.inference_mode()
def _write(model, tokenizer, fact: str, acknowledgement: str) -> dict:
    device = model._find_layer_device()
    dialogue = [
        {"role": "user", "content": fact},
        {"role": "assistant", "content": acknowledgement},
    ]
    encoded = _chat_tensor(tokenizer, dialogue, add_generation_prompt=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    text_prefix = _memory_system_prefix(
        tokenizer,
        "以下是与当前用户相关的已保存长期记忆。仅在问题相关时使用，不要编造：\n" + fact,
    )
    text_ids = text_prefix["input_ids"].to(device)
    text_mask = text_prefix["attention_mask"].to(device)
    key = tokenizer(fact, add_special_tokens=False, return_tensors="pt")
    key_ids = key["input_ids"].to(device)
    key_mask = key.get("attention_mask")
    if key_mask is None:
        key_mask = torch.ones_like(key_ids)
    key_mask = key_mask.to(device)
    storage = tokenizer(fact, add_special_tokens=False, return_tensors="pt")
    storage_ids = storage["input_ids"].to(device)
    storage_mask = storage.get("attention_mask")
    if storage_mask is None:
        storage_mask = torch.ones_like(storage_ids)
    storage_mask = storage_mask.to(device)
    model(
        **encoded,
        read_memory=False,
        update_memory=True,
        return_memory=True,
        use_cache=False,
        memory_text_input_ids=text_ids,
        memory_text_attention_mask=text_mask,
        memory_key_input_ids=key_ids,
        memory_key_attention_mask=key_mask,
        memory_storage_input_ids=storage_ids,
        memory_storage_attention_mask=storage_mask,
    )
    address = model.memory.last_write_address
    probability = model.memory.last_write_probability
    stored_slot = model.runtime.text_last_written_slot
    return {
        "fact": fact,
        "write_probability": float(probability.detach().mean()) if probability is not None else None,
        "selected_slot": int(address.argmax(dim=-1)[0].item()) if address is not None else None,
        "stored_slot": int(stored_slot[0].item()) if stored_slot is not None else None,
        "valid_slots_after_write": int(model.runtime.text_slot_valid.sum())
        if model.runtime.text_slot_valid is not None
        else 0,
    }


def _load_persistent_checkpoint(model_path: str, adapter_path: str, *, no_4bit: bool):
    config = load_memory_config(adapter_path)
    restarted = load_qwen_dynamic(
        model_path,
        memory_config=config,
        load_in_4bit=not no_4bit,
    )
    restarted.load_memory_adapter(adapter_path)
    restarted.eval()
    return restarted, config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--adapter", default="V2_dpskw/qwen_memory_adapter_native_v3")
    parser.add_argument(
        "--output-adapter",
        default="V2_dpskw/qwen_memory_adapter_natural_production_v1",
    )
    parser.add_argument(
        "--report",
        default="V2_dpskw/benchmark_natural_language_memory.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--text-memory-threshold", type=float, default=0.35)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    config = load_memory_config(args.adapter)
    config.persistent_memory = True
    config.natural_language_memory = True
    config.text_memory_threshold = args.text_memory_threshold
    config.reset_token_id = resolve_memory_reset_token(tokenizer, DEFAULT_MEMORY_RESET_TOKEN)
    model = load_qwen_dynamic(
        args.model_path,
        memory_config=config,
        load_in_4bit=not args.no_4bit,
    )
    model.load_memory_adapter(args.adapter)
    model.reset_memory()

    writes = [
        _write(model, tokenizer, "请记住：我的工作地点代号是R7。", "好的，我会记住。"),
        _write(model, tokenizer, "请记住：我最喜欢的水果是红富士苹果。", "好的，我会记住。"),
        _write(model, tokenizer, "更新一下：我的工作地点代号改为K9。", "好的，已更新。"),
    ]
    model.save_persistent_memory_checkpoint(args.output_adapter)
    saved_valid_slots = int(model.runtime.text_slot_valid.sum())
    saved_norm = float(model.runtime.state.detach().float().norm())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restarted, restart_config = _load_persistent_checkpoint(
        args.model_path,
        args.output_adapter,
        no_4bit=args.no_4bit,
    )
    loaded_norm = float(restarted.runtime.state.detach().float().norm())

    queries = [
        {"name": "replaced_work_code", "query": "我的工作地点代号是什么？", "expected": "K9"},
        {"name": "favorite_fruit", "query": "我最喜欢吃什么水果？", "expected": "红富士苹果"},
        {"name": "unknown_blood_type", "query": "我的血型是什么？如果没有记录，请明确说不知道。", "expected": "不知道"},
    ]
    query_results = []
    for item in queries:
        response = _generate(restarted, tokenizer, item["query"], args.max_new_tokens)
        relevance = restarted.runtime.text_read_relevance
        overlap = restarted.runtime.text_read_overlap
        selected_slots = restarted.runtime.text_read_slots
        query_results.append(
            {
                **item,
                "response": response,
                "text_prefix_used": restarted.runtime.text_prefix_used,
                "retrieval_relevance": float(relevance[0].item()) if relevance is not None else None,
                "retrieval_overlap": overlap[0].detach().cpu().tolist()
                if overlap is not None
                else None,
                "retrieved_slots": selected_slots[0].detach().cpu().tolist()
                if selected_slots is not None
                else None,
                "expected_found": item["expected"] in response,
                "refused_unknown": item["name"] != "unknown_blood_type"
                or any(marker in response for marker in ("不知道", "没有记录", "无相关", "未找到", "不清楚")),
            }
        )

    del restarted
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restarted, second_restart_config = _load_persistent_checkpoint(
        args.model_path,
        args.output_adapter,
        no_4bit=args.no_4bit,
    )
    second_loaded_norm = float(restarted.runtime.state.detach().float().norm())
    second_restart_results = []
    for item in queries[:2]:
        response = _generate(restarted, tokenizer, item["query"], args.max_new_tokens)
        second_restart_results.append(
            {
                "name": item["name"],
                "response": response,
                "expected": item["expected"],
                "expected_found": item["expected"] in response,
                "text_prefix_used": restarted.runtime.text_prefix_used,
            }
        )

    reset_inputs = _chat_tensor(
        tokenizer,
        [{"role": "user", "content": DEFAULT_MEMORY_RESET_TOKEN}],
        add_generation_prompt=True,
    )
    device = restarted._find_layer_device()
    reset_inputs = {key: value.to(device) for key, value in reset_inputs.items()}
    restarted.generate(
        **reset_inputs,
        max_new_tokens=1,
        do_sample=False,
        update_memory=False,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    reset_norm = float(restarted.runtime.state.detach().float().norm())
    reset_valid_slots = int(restarted.runtime.text_slot_valid.sum())
    after_reset = _generate(restarted, tokenizer, "我的工作地点代号是什么？", args.max_new_tokens)

    report = {
        "writes": writes,
        "history_passed_to_restart": False,
        "saved_valid_slots": saved_valid_slots,
        "saved_memory_norm": saved_norm,
        "loaded_memory_norm": loaded_norm,
        "restart_state_equal_norm": abs(saved_norm - loaded_norm) < 1e-5,
        "second_loaded_memory_norm": second_loaded_norm,
        "second_restart_state_equal_norm": abs(saved_norm - second_loaded_norm) < 1e-5,
        "queries": query_results,
        "second_restart_queries": second_restart_results,
        "second_restart_pass": all(row["expected_found"] for row in second_restart_results),
        "all_known_queries_pass": all(row["expected_found"] for row in query_results[:2]),
        "unknown_refusal_pass": query_results[2]["refused_unknown"],
        "reset_token": DEFAULT_MEMORY_RESET_TOKEN,
        "reset_token_id": restart_config.reset_token_id,
        "reset_memory_norm": reset_norm,
        "reset_valid_slots": reset_valid_slots,
        "reset_cleared_pass": reset_norm < 1e-5 and reset_valid_slots == 0,
        "response_after_reset": after_reset,
    }
    report["production_gate_pass"] = bool(
        report["history_passed_to_restart"] is False
        and report["restart_state_equal_norm"]
        and report["second_restart_state_equal_norm"]
        and report["all_known_queries_pass"]
        and report["second_restart_pass"]
        and report["unknown_refusal_pass"]
        and report["reset_cleared_pass"]
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
