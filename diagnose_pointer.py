"""Debug the raw token pointer memory path on one benchmark record."""

from __future__ import annotations

import json

import torch

from .benchmark_qwen import _generation_prompt
from .qwen_integration import QwenMemoryConfig, load_qwen_dynamic, load_tokenizer
from .train_qwen_memory import encode_messages, pad_batch


def main() -> None:
    config = QwenMemoryConfig(
        mode="blend",
        blend_init=0.1,
        write_token_offset=4,
        broadcast_write=True,
        raw_token_write=True,
        raw_logit_scale=30.0,
    )
    model = load_qwen_dynamic(".", memory_config=config, load_in_4bit=True)
    model.eval()
    tokenizer = load_tokenizer(".")
    record = json.loads(next(open("V2_dpskw/data/benchmark_eval.jsonl", encoding="utf-8")))
    device = model._find_layer_device()
    memory = encode_messages(tokenizer, record["memory"], 128)
    memory_input, memory_mask, _ = pad_batch([memory], int(tokenizer.pad_token_id))
    print("memory_ids", memory_input.tolist())
    print("memory_mask", memory_mask.tolist())
    prompt = _generation_prompt(tokenizer, record["query"][:-1], device)
    print("prompt_ids", prompt["input_ids"].tolist())
    with torch.inference_mode():
        output = model(
            input_ids=memory_input.to(device),
            attention_mask=memory_mask.to(device),
            read_memory=False,
            update_memory=True,
            return_memory=True,
            use_cache=False,
        )
        print("raw_memory_shape", tuple(model.runtime.raw_memory.shape))
        print("raw_memory_norm", float(model.runtime.raw_memory.float().norm()))
        generated = model.generate(
            **prompt,
            max_new_tokens=1,
            do_sample=False,
            update_memory=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("generated", generated.tolist())


if __name__ == "__main__":
    main()
