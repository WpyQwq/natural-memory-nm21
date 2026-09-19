"""Check whether teacher-forcing and cached greedy generation agree on token 1."""

from __future__ import annotations

import json

import torch

from .benchmark_qwen import _generation_prompt
from .qwen_integration import QwenMemoryConfig, load_qwen_dynamic, load_tokenizer
from .train_qwen_memory import encode_messages, pad_batch


def main() -> None:
    model = load_qwen_dynamic(
        ".",
        memory_config=QwenMemoryConfig(mode="blend", blend_init=0.1),
        load_in_4bit=True,
    )
    model.load_memory_adapter("V2_dpskw/qwen_memory_adapter_full")
    model.eval()
    tokenizer = load_tokenizer(".")
    record = json.loads(
        next(open("V2_dpskw/data/benchmark_eval.jsonl", encoding="utf-8"))
    )
    device = model._find_layer_device()
    memory = encode_messages(tokenizer, record["memory"], 128)
    memory_input, memory_mask, _ = pad_batch([memory], int(tokenizer.pad_token_id))
    prompt = _generation_prompt(tokenizer, record["query"][:-1], device)

    with torch.inference_mode():
        memory_state = model(
            input_ids=memory_input.to(device),
            attention_mask=memory_mask.to(device),
            read_memory=False,
            update_memory=True,
            return_memory=True,
            use_cache=False,
        ).memory
        teacher_forcing = model(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            memory_state=memory_state,
            read_memory=True,
            update_memory=False,
            return_memory=True,
            use_cache=False,
        )
        generated = model.generate(
            **prompt,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    teacher_id = int(teacher_forcing.logits[0, -1].argmax())
    generated_id = int(generated[0, -1])
    print(json.dumps({"expected": record["answer"]}, ensure_ascii=True))
    print(json.dumps({"teacher_id": teacher_id, "teacher_text": tokenizer.decode([teacher_id])}, ensure_ascii=True))
    print(json.dumps({"generated_id": generated_id, "generated_text": tokenizer.decode([generated_id])}, ensure_ascii=True))
    print(f"same={teacher_id == generated_id}")


if __name__ == "__main__":
    main()
