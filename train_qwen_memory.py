"""Train only the dynamic memory adapter on streaming SFT records.

Each JSONL record must contain ``memory`` and ``query`` message lists. The
memory turn is observed first; the query turn is evaluated afterwards using
the updated state. The loss is therefore downstream of a differentiable write.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from .qwen_integration import QwenMemoryConfig, load_qwen_dynamic, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--data", default="V2_dpskw/data/demo_stream.jsonl")
    parser.add_argument("--output-dir", default="V2_dpskw/qwen_memory_adapter")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-adapter", default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--surgery-mode",
        choices=("residual", "blend", "replace"),
        default="residual",
        help="residual adds memory, blend learns a gradual token-mixer replacement, replace removes the original token mixer",
    )
    parser.add_argument(
        "--blend-init",
        type=float,
        default=0.0,
        help="initial fraction of the token mixer supplied by memory in blend mode",
    )
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="+",
        default=None,
        help="zero-based Qwen layer indices to adapt; defaults to four full-attention layers",
    )
    parser.add_argument(
        "--direct-logit-scale",
        type=float,
        default=0.0,
        help="add the final memory readout directly to vocabulary logits; useful for exact recall experiments",
    )
    parser.add_argument(
        "--write-token-offset",
        type=int,
        default=None,
        help="write a fixed token counted from the end of the memory sequence instead of the final token",
    )
    parser.add_argument(
        "--broadcast-write",
        action="store_true",
        help="write the proposal to every memory slot; useful for one-fact recall ablations",
    )
    parser.add_argument(
        "--raw-token-write",
        action="store_true",
        help="write the selected token's output-projection row into runtime memory",
    )
    parser.add_argument(
        "--raw-logit-scale",
        type=float,
        default=0.0,
        help="scale the raw token memory logits during training and generation",
    )
    parser.add_argument(
        "--native-mode",
        action="store_true",
        help="use the learned write/forget controller instead of the legacy memory rule",
    )
    parser.add_argument(
        "--persistent-memory",
        action="store_true",
        help="keep the learned runtime memory as part of the model instance",
    )
    parser.add_argument("--reset-token-id", type=int, default=None)
    parser.add_argument(
        "--no-summary-pooling",
        action="store_true",
        help="use the final hidden state instead of learned summary pooling",
    )
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("memory"), list) or not isinstance(record.get("query"), list):
            raise ValueError(f"line {line_number}: expected memory/query message lists")
        records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def encode_messages(tokenizer: Any, messages: list[dict[str, Any]], max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_encoding = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full = full_encoding["input_ids"] if hasattr(full_encoding, "__getitem__") and "input_ids" in full_encoding else full_encoding
    prompt = prompt_encoding["input_ids"] if hasattr(prompt_encoding, "__getitem__") and "input_ids" in prompt_encoding else prompt_encoding
    if full and isinstance(full[0], list):
        full = full[0]
    if prompt and isinstance(prompt[0], list):
        prompt = prompt[0]
    original_full_length = len(full)
    truncated_prefix = max(0, original_full_length - max_length)
    if len(full) > max_length:
        full = full[-max_length:]
    input_ids = torch.tensor(full, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    # The prompt may have been truncated from the left. Keep only the target
    # part visible to the loss.
    prompt_tokens = max(0, min(len(prompt) - truncated_prefix, len(full)))
    labels = input_ids.clone()
    labels[:prompt_tokens] = -100
    return input_ids, attention_mask, labels


def pad_batch(items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(item[0].numel() for item in items)
    input_ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (ids, mask, row_labels) in enumerate(items):
        input_ids[row, : ids.numel()] = ids
        attention_mask[row, : mask.numel()] = mask
        labels[row, : row_labels.numel()] = row_labels
    return input_ids, attention_mask, labels


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.data)
    tokenizer = load_tokenizer(args.model_path)
    memory_config = QwenMemoryConfig(
        mode=args.surgery_mode,
        blend_init=args.blend_init,
        layer_indices=tuple(args.layer_indices) if args.layer_indices else None,
        direct_logit_scale=args.direct_logit_scale,
        write_token_offset=args.write_token_offset,
        broadcast_write=args.broadcast_write,
        raw_token_write=args.raw_token_write,
        raw_logit_scale=args.raw_logit_scale,
        native_mode=args.native_mode,
        persistent_memory=args.persistent_memory,
        reset_token_id=args.reset_token_id,
        summary_pooling=not args.no_summary_pooling,
    )
    model = load_qwen_dynamic(
        args.model_path,
        memory_config=memory_config,
        load_in_4bit=not args.no_4bit,
    )
    if args.resume_adapter:
        model.load_memory_adapter(args.resume_adapter)
    model.train()

    parameters = list(model.trainable_parameters)
    if not parameters:
        raise RuntimeError("no trainable memory parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    device = model._find_layer_device()
    pad_id = int(tokenizer.pad_token_id)
    output_dir = Path(args.output_dir)

    print(
        f"device={device} records={len(records)} memory_layers={model.layer_indices} "
        f"surgery_mode={memory_config.mode} blend_init={memory_config.blend_init}"
    )
    for step in range(1, args.steps + 1):
        if memory_config.persistent_memory:
            model.reset_memory(batch_size=args.batch_size, device=model._find_layer_device())
        chosen = [records[(step * args.batch_size + i) % len(records)] for i in range(args.batch_size)]
        memory_items = [encode_messages(tokenizer, item["memory"], args.max_length) for item in chosen]
        query_items = [encode_messages(tokenizer, item["query"], args.max_length) for item in chosen]
        memory_input, memory_mask, _ = pad_batch(memory_items, pad_id)
        query_input, query_mask, query_labels = pad_batch(query_items, pad_id)
        memory_input = memory_input.to(device)
        memory_mask = memory_mask.to(device)
        query_input = query_input.to(device)
        query_mask = query_mask.to(device)
        query_labels = query_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        memory_output = model(
            input_ids=memory_input,
            attention_mask=memory_mask,
            update_memory=True,
            read_memory=False,
            detach_memory=False,
            return_memory=True,
        )
        query_output = model(
            input_ids=query_input,
            attention_mask=query_mask,
            labels=query_labels,
            memory_state=memory_output.memory,
            update_memory=False,
            read_memory=True,
            return_memory=True,
        )
        if query_output.loss is None:
            raise RuntimeError("Qwen did not return an SFT loss")
        query_output.loss.backward()
        clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f"step={step:4d} loss={query_output.loss.detach().item():.4f}")
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_memory_adapter(output_dir)
            (output_dir / "training_state.json").write_text(
                json.dumps({"step": step, "data": str(args.data), "model_path": str(args.model_path)}, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
