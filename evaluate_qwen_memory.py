"""Evaluate a Qwen dynamic-memory adapter on streaming JSONL records."""

from __future__ import annotations

import argparse

import torch

from .qwen_integration import load_qwen_dynamic, load_tokenizer
from .train_qwen_memory import encode_messages, load_records, pad_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--data", default="V2_dpskw/data/demo_stream.jsonl")
    parser.add_argument("--adapter", default="V2_dpskw/qwen_memory_adapter")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    model = load_qwen_dynamic(args.model_path, load_in_4bit=not args.no_4bit)
    model.load_memory_adapter(args.adapter)
    model.eval()
    device = model._find_layer_device()
    pad_id = int(tokenizer.pad_token_id)
    records = load_records(args.data)
    correct = 0

    for record in records:
        memory_input, memory_mask, _ = pad_batch(
            [encode_messages(tokenizer, record["memory"], args.max_length)], pad_id
        )
        query_input, query_mask, query_labels = pad_batch(
            [encode_messages(tokenizer, record["query"], args.max_length)], pad_id
        )
        with torch.no_grad():
            memory_output = model(
                input_ids=memory_input.to(device),
                attention_mask=memory_mask.to(device),
                read_memory=False,
                update_memory=True,
            )
            output = model(
                input_ids=query_input.to(device),
                attention_mask=query_mask.to(device),
                memory_state=memory_output.memory,
                read_memory=True,
                update_memory=False,
            )
        labels = query_labels.to(device)
        shifted_labels = labels[..., 1:]
        predictions = output.logits[..., :-1, :].argmax(dim=-1)
        target_positions = shifted_labels != -100
        sequence_ok = bool((predictions[target_positions] == shifted_labels[target_positions]).all())
        correct += int(sequence_ok)
        print(f"sequence_ok={sequence_ok}")

    print(f"exact_sequence_accuracy={correct / len(records):.3f}")


if __name__ == "__main__":
    main()
