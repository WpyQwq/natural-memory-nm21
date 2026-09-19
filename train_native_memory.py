"""Train the learned persistent-memory controller with policy auxiliary losses."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .qwen_integration import QwenMemoryConfig, load_qwen_dynamic, load_tokenizer
from .train_qwen_memory import encode_messages, pad_batch


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--data", default="V2_dpskw/data/native_memory/train.jsonl")
    parser.add_argument("--output-dir", default="V2_dpskw/qwen_memory_adapter_native")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--direct-logit-scale", type=float, default=4.0)
    parser.add_argument("--write-loss-weight", type=float, default=0.25)
    parser.add_argument("--forget-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="weight for aligning each labeled fact's write representation to its value token",
    )
    parser.add_argument(
        "--value-cosine-weight",
        type=float,
        default=1.0,
        help="additional cosine alignment weight against the frozen output embedding row",
    )
    parser.add_argument(
        "--forget-positive-weight",
        type=float,
        default=4.0,
        help="extra BCE weight for positive replacement/forget examples",
    )
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.data)
    tokenizer = load_tokenizer(args.model_path)
    config = QwenMemoryConfig(
        mode="blend",
        blend_init=0.1,
        direct_logit_scale=args.direct_logit_scale,
        native_mode=True,
        persistent_memory=False,
        summary_pooling=True,
    )
    model = load_qwen_dynamic(
        args.model_path,
        memory_config=config,
        load_in_4bit=not args.no_4bit,
    )
    model.train()
    parameters = list(model.trainable_parameters)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    device = model._find_layer_device()
    pad_id = int(tokenizer.pad_token_id)
    output_dir = Path(args.output_dir)

    print(
        f"device={device} records={len(records)} layers={model.layer_indices} "
        f"direct_logit_scale={args.direct_logit_scale}"
    )
    for step in range(1, args.steps + 1):
        record = records[(step - 1) % len(records)]
        model.reset_memory()
        optimizer.zero_grad(set_to_none=True)
        write_losses: list[torch.Tensor] = []
        forget_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        value_cosine_losses: list[torch.Tensor] = []
        chunks = record.get("memory_chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("each native-memory record needs a non-empty memory_chunks list")

        for chunk in chunks:
            memory_item = encode_messages(tokenizer, chunk["messages"], args.max_length)
            memory_input, memory_mask, _ = pad_batch([memory_item], pad_id)
            memory_output = model(
                input_ids=memory_input.to(device),
                attention_mask=memory_mask.to(device),
                read_memory=False,
                update_memory=True,
                return_memory=True,
                use_cache=False,
            )
            del memory_output
            write_probability = model.memory.last_write_probability
            forget_probability = model.memory.last_forget_probability
            if write_probability is None or forget_probability is None:
                raise RuntimeError("native memory controller did not expose write statistics")
            write_target = torch.full_like(write_probability, float(chunk.get("write_label", 1.0)))
            forget_target = torch.full_like(
                forget_probability,
                float(chunk.get("forget_label", 0.0)),
            )
            write_losses.append(F.binary_cross_entropy(write_probability, write_target))
            forget_weight = 1.0 + (args.forget_positive_weight - 1.0) * forget_target
            forget_losses.append(
                F.binary_cross_entropy(forget_probability, forget_target, weight=forget_weight)
            )
            value = chunk.get("value")
            write_representation = model.memory.last_write_representation
            if value and write_representation is not None:
                value_tokens = tokenizer(
                    str(value),
                    add_special_tokens=False,
                )["input_ids"]
                if value_tokens and isinstance(value_tokens[0], list):
                    value_tokens = value_tokens[0]
                if value_tokens:
                    target_id = torch.tensor(
                        [int(value_tokens[0])],
                        dtype=torch.long,
                        device=model.base_model.get_output_embeddings().weight.device,
                    )
                    output_embeddings = model.base_model.get_output_embeddings()
                    write_logits = output_embeddings(
                        write_representation.to(
                            device=output_embeddings.weight.device,
                            dtype=output_embeddings.weight.dtype,
                        )
                    ).float()
                    value_losses.append(F.cross_entropy(write_logits, target_id))
                    target_embedding = output_embeddings.weight[target_id].detach().float()
                    predicted_embedding = write_representation.float()
                    value_cosine_losses.append(
                        1.0
                        - F.cosine_similarity(predicted_embedding, target_embedding, dim=-1).mean()
                    )

        query_item = encode_messages(tokenizer, record["query"], args.max_length)
        query_input, query_mask, query_labels = pad_batch([query_item], pad_id)
        query_output = model(
            input_ids=query_input.to(device),
            attention_mask=query_mask.to(device),
            labels=query_labels.to(device),
            read_memory=True,
            update_memory=False,
            return_memory=True,
            use_cache=False,
        )
        if query_output.loss is None:
            raise RuntimeError("native Qwen query returned no loss")
        write_loss = torch.stack(write_losses).mean()
        forget_loss = torch.stack(forget_losses).mean()
        value_loss = torch.stack(value_losses).mean() if value_losses else query_output.loss.new_zeros(())
        value_cosine_loss = (
            torch.stack(value_cosine_losses).mean()
            if value_cosine_losses
            else query_output.loss.new_zeros(())
        )
        loss = (
            query_output.loss
            + args.write_loss_weight * write_loss
            + args.forget_loss_weight * forget_loss
            + args.value_loss_weight * value_loss
            + args.value_cosine_weight * value_cosine_loss
        )
        loss.backward()
        clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == args.steps:
            write_mean = float(torch.stack(write_losses).detach().mean())
            forget_mean = float(torch.stack(forget_losses).detach().mean())
            print(
                f"step={step:4d} loss={loss.detach().item():.4f} "
                f"query={query_output.loss.detach().item():.4f} "
                f"write_bce={write_mean:.4f} forget_bce={forget_mean:.4f} "
                f"value={value_loss.detach().item():.4f} "
                f"value_cos={value_cosine_loss.detach().item():.4f}"
            )
        if step % args.save_every == 0 or step == args.steps:
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_memory_adapter(output_dir)
            (output_dir / "training_state.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "data": str(args.data),
                        "model_path": str(args.model_path),
                        "controller": "native_learned_write_forget_summary",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
