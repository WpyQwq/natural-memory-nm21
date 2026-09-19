"""Verify answering after a model restart using only a saved user memory state."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .benchmark_qwen import _generation_prompt
from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer
from .train_qwen_memory import encode_messages, pad_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument(
        "--adapter",
        default="V2_dpskw/qwen_memory_adapter_pointer",
    )
    parser.add_argument(
        "--data",
        default="V2_dpskw/data/benchmark_eval.jsonl",
    )
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument(
        "--memory-state",
        default="V2_dpskw/data/user_memory_demo.pt",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[args.record_index]
    tokenizer = load_tokenizer(args.model_path)
    memory_config = load_memory_config(args.adapter)

    model = load_qwen_dynamic(
        args.model_path,
        memory_config=memory_config,
        load_in_4bit=not args.no_4bit,
    )
    model.load_memory_adapter(args.adapter)
    model.eval()
    device = model._find_layer_device()
    memory = encode_messages(tokenizer, record["memory"], args.max_length)
    memory_input, memory_mask, _ = pad_batch([memory], int(tokenizer.pad_token_id))
    state_path = Path(args.memory_state)

    with torch.inference_mode():
        model(
            input_ids=memory_input.to(device),
            attention_mask=memory_mask.to(device),
            read_memory=False,
            update_memory=True,
            return_memory=True,
            use_cache=False,
        )
        model.save_runtime_memory(state_path)
    print(f"saved_memory_state={state_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restarted = load_qwen_dynamic(
        args.model_path,
        memory_config=memory_config,
        load_in_4bit=not args.no_4bit,
    )
    restarted.load_memory_adapter(args.adapter)
    restarted.eval()
    device = restarted._find_layer_device()
    restarted.load_runtime_memory(state_path, device=device)
    prompt = _generation_prompt(tokenizer, record["query"][:-1], device)
    with torch.inference_mode():
        output = restarted.generate(
            **prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            update_memory=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(
        output[0, prompt["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).replace(" ", "").replace("\r", "").replace("\n", "").strip()
    expected = str(record["answer"])
    print(f"query_contains_history=false")
    print(f"expected={expected}")
    print(f"restarted_generated={generated}")
    print(f"correct={generated.startswith(expected)}")


if __name__ == "__main__":
    main()
