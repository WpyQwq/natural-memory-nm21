"""End-to-end test: write a fact in a dialogue, restart without chat history, recall it."""

from __future__ import annotations

import argparse
import gc
import json
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
    """Encode a valid system-message prefix without adding a fake query."""

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
def _generate(model, tokenizer, messages, max_new_tokens: int) -> str:
    encoded = _chat_tensor(tokenizer, messages, add_generation_prompt=True)
    device = model._find_layer_device()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        update_memory=False,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    response_ids = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(response_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--adapter", default="V2_dpskw/qwen_memory_adapter_native_v3")
    parser.add_argument(
        "--output-adapter",
        default="V2_dpskw/qwen_memory_adapter_restart_dialogue_test",
    )
    parser.add_argument(
        "--report",
        default="V2_dpskw/restart_memory_dialogue_test.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--text-memory-threshold", type=float, default=0.0)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    fact = "请记住：我的工作地点代号是R7。"
    acknowledgement = "好的，我会记住这条个人信息。"
    query = "我的工作地点代号是什么？"
    expected = "R7"

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
    device = model._find_layer_device()

    # This is the write turn: a normal user/assistant dialogue, not a prebuilt
    # memory tensor and not a query with the answer included in the prompt.
    write_messages = [
        {"role": "user", "content": fact},
        {"role": "assistant", "content": acknowledgement},
    ]
    write_inputs = _chat_tensor(tokenizer, write_messages, add_generation_prompt=False)
    write_inputs = {key: value.to(device) for key, value in write_inputs.items()}
    memory_text = _memory_system_prefix(
        tokenizer,
        "以下是与当前用户相关的已保存长期记忆。仅在问题相关时使用，不要编造：\n" + fact,
    )
    memory_text_input_ids = memory_text["input_ids"].to(device)
    memory_text_attention_mask = memory_text.get("attention_mask")
    if memory_text_attention_mask is None:
        memory_text_attention_mask = torch.ones_like(memory_text_input_ids)
    memory_text_attention_mask = memory_text_attention_mask.to(device)
    memory_key = tokenizer(fact, add_special_tokens=False, return_tensors="pt")
    memory_key_input_ids = memory_key["input_ids"].to(device)
    memory_key_attention_mask = memory_key.get("attention_mask")
    if memory_key_attention_mask is None:
        memory_key_attention_mask = torch.ones_like(memory_key_input_ids)
    memory_key_attention_mask = memory_key_attention_mask.to(device)
    model(
        **write_inputs,
        read_memory=True,
        update_memory=True,
        return_memory=True,
        use_cache=False,
        memory_text_input_ids=memory_text_input_ids,
        memory_text_attention_mask=memory_text_attention_mask,
        memory_key_input_ids=memory_key_input_ids,
        memory_key_attention_mask=memory_key_attention_mask,
    )
    saved_norm = float(model.runtime.state.detach().float().norm())
    output_adapter = Path(args.output_adapter)
    model.save_persistent_memory_checkpoint(output_adapter)

    # Destroy the first model completely.  The second model receives only the
    # base model plus the persistent adapter checkpoint; no chat history or
    # runtime memory_state file is passed to it.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restart_config = load_memory_config(output_adapter)
    restarted = load_qwen_dynamic(
        args.model_path,
        memory_config=restart_config,
        load_in_4bit=not args.no_4bit,
    )
    restarted.load_memory_adapter(output_adapter)
    restarted.eval()
    restart_norm = float(restarted.runtime.state.detach().float().norm())
    generated_after_restart = _generate(
        restarted,
        tokenizer,
        [{"role": "user", "content": query}],
        args.max_new_tokens,
    )

    # Also verify that the external reset token clears the model-owned state.
    reset_inputs = _chat_tensor(
        tokenizer,
        [{"role": "user", "content": DEFAULT_MEMORY_RESET_TOKEN}],
        add_generation_prompt=True,
    )
    reset_inputs = {key: value.to(restarted._find_layer_device()) for key, value in reset_inputs.items()}
    restarted.generate(
        **reset_inputs,
        max_new_tokens=1,
        do_sample=False,
        update_memory=False,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    reset_norm = float(restarted.runtime.state.detach().float().norm())
    generated_after_reset = _generate(
        restarted,
        tokenizer,
        [{"role": "user", "content": query}],
        args.max_new_tokens,
    )

    report = {
        "fact_dialogue": write_messages,
        "restart_query": [{"role": "user", "content": query}],
        "history_passed_to_restart": False,
        "expected": expected,
        "generated_after_restart": generated_after_restart,
        "recalled_after_restart": expected in generated_after_restart,
        "saved_memory_norm": saved_norm,
        "loaded_memory_norm_after_restart": restart_norm,
        "persistent_adapter": str(output_adapter),
        "reset_token": DEFAULT_MEMORY_RESET_TOKEN,
        "reset_token_id": restart_config.reset_token_id,
        "memory_norm_after_reset": reset_norm,
        "reset_cleared_memory": reset_norm < 1e-5,
        "generated_after_reset": generated_after_reset,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
