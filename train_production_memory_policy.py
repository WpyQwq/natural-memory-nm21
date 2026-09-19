"""Train the automatic write/forget policy from normalized conversation JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _read_jsonl(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            rows.append(json.loads(raw))
            if max_examples is not None and len(rows) >= max_examples:
                break
    if not rows:
        raise ValueError(f"no examples found in {path}")
    return rows


def _encode_batch(tokenizer, texts: list[str], device: torch.device):
    encoded_rows = []
    for text in texts:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        encoded_rows.append(
            (
                encoded["input_ids"][0],
                encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))[0],
            )
        )
    max_length = max(row.numel() for row, _ in encoded_rows)
    input_ids = torch.zeros(len(encoded_rows), max_length, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, (row, mask) in enumerate(encoded_rows):
        input_ids[index, : row.numel()] = row.to(device)
        attention_mask[index, : mask.numel()] = mask.to(device)
    return input_ids, attention_mask


@torch.inference_mode()
def _collect(model, tokenizer, rows: list[dict[str, Any]], *, batch_size: int, device: torch.device):
    vectors: list[torch.Tensor] = []
    write_labels: list[float] = []
    forget_labels: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids, attention_mask = _encode_batch(
            tokenizer,
            [str(row["text"]) for row in batch],
            device,
        )
        model.reset_memory(batch_size=len(batch), device=device)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            read_memory=False,
            update_memory=True,
            return_memory=True,
            use_cache=False,
        )
        representation = getattr(
            model.memory,
            "last_write_summary",
            getattr(model.memory, "last_write_representation", None),
        )
        if representation is None:
            raise RuntimeError("the loaded memory controller did not expose last_write_representation")
        vectors.append(representation.detach().float().cpu())
        write_labels.extend(float(row.get("write_label", 0.0)) for row in batch)
        forget_labels.extend(float(row.get("forget_label", 0.0)) for row in batch)
    return (
        torch.cat(vectors),
        torch.tensor(write_labels, dtype=torch.float32),
        torch.tensor(forget_labels, dtype=torch.float32),
    )


def _metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    probabilities = torch.sigmoid(logits.detach()).reshape(-1).cpu()
    labels = labels.reshape(-1).cpu() >= 0.5
    predictions = probabilities >= threshold
    positive = labels
    negative = ~labels
    tp = int((predictions & positive).sum())
    fn = int((~predictions & positive).sum())
    fp = int((predictions & negative).sum())
    tn = int((~predictions & negative).sum())
    return {
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).float().mean()),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "false_positive_rate": fp / max(1, fp + tn),
        "f1": (2.0 * tp) / max(1, 2 * tp + fp + fn),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
    }


def _choose_threshold(logits: torch.Tensor, labels: torch.Tensor, max_fpr: float) -> dict[str, float]:
    candidates = [index / 100.0 for index in range(10, 91, 2)]
    reports = [_metrics(logits, labels, threshold) for threshold in candidates]
    acceptable = [item for item in reports if item["false_positive_rate"] <= max_fpr]
    return max(acceptable or reports, key=lambda item: (item["recall"], item["specificity"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--base-adapter", default="qwen_memory_adapter_natural_auto_v13")
    parser.add_argument("--dataset-dir", default="data/production_memory")
    parser.add_argument("--output-adapter", default="qwen_memory_adapter_natural_production_candidate")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = _project_path(args.dataset_dir)
    train_rows = _read_jsonl(dataset_dir / "train.jsonl", args.max_examples)
    eval_rows = _read_jsonl(dataset_dir / "eval.jsonl", args.max_examples)
    model_path = _project_path(args.model_path)
    base_adapter = _project_path(args.base_adapter)
    tokenizer = load_tokenizer(model_path)
    # A merged v2 package carries the router architecture metadata.  Loading
    # the older v1 adapter config here would disable hierarchical memory before
    # the embedded shard is read, so the package config is authoritative.
    config_source = model_path if (model_path / "memory_merge.json").exists() else base_adapter
    config = load_memory_config(config_source)
    config.natural_language_memory = True
    config.automatic_memory = True
    config.automatic_memory_policy_version = 2
    config.auto_forget_threshold = 0.50
    config.persistent_memory = False
    model = load_qwen_dynamic(model_path, memory_config=config, load_in_4bit=not args.no_4bit)
    # The base adapter has the old one-logit policy.  Its write head is a
    # useful initialization; the new forget head starts trainable and is
    # intentionally loaded with strict=False.
    model.load_memory_adapter(base_adapter, strict=False)
    if model.memory_policy is None:
        raise RuntimeError("automatic memory policy is disabled by the selected configuration")
    # The policy is now trained on the frozen Qwen semantic summary rather
    # than the value-path projection used by the bootstrap adapter.  Reset
    # only this small controller so stale input-space weights cannot poison
    # the new feature space; the main model, memory bank, and retriever stay
    # untouched.
    model.memory_policy.apply(model.memory_policy._init_weights)
    model.eval()
    model.memory_policy.train()
    device = model._find_layer_device()

    train_vectors, train_labels, train_forget_labels = _collect(
        model, tokenizer, train_rows, batch_size=args.batch_size, device=device
    )
    eval_vectors, eval_labels, eval_forget_labels = _collect(
        model, tokenizer, eval_rows, batch_size=args.batch_size, device=device
    )
    train_vectors = train_vectors.to(device)
    train_labels = train_labels.to(device)
    train_forget_labels = train_forget_labels.to(device)
    eval_vectors = eval_vectors.to(device)
    eval_labels = eval_labels.to(device)
    eval_forget_labels = eval_forget_labels.to(device)
    optimizer = torch.optim.AdamW(model.memory_policy.parameters(), lr=args.lr, weight_decay=0.01)
    positive_weight = torch.tensor([1.5], device=device)
    forget_positive_count = max(1, int(train_forget_labels.sum().item()))
    forget_negative_count = max(1, int(train_forget_labels.numel() - forget_positive_count))
    forget_positive_weight = max(4.0, 0.75 * forget_negative_count / forget_positive_count)
    rng = random.Random(args.seed + 1)
    steps = max(1, int(args.steps))
    for step in range(1, steps + 1):
        indices = torch.tensor(
            [rng.randrange(train_vectors.shape[0]) for _ in range(min(args.batch_size, train_vectors.shape[0]))],
            dtype=torch.long,
            device=device,
        )
        logits = model.memory_policy(train_vectors[indices]).reshape(-1)
        labels = train_labels[indices]
        weights = torch.where(labels >= 0.5, positive_weight.expand_as(labels), torch.ones_like(labels))
        write_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        forget_logits = model.memory_policy.forget_logits(train_vectors[indices]).reshape(-1)
        forget_labels = train_forget_labels[indices]
        forget_weights = torch.where(
            forget_labels >= 0.5,
            torch.full_like(forget_labels, forget_positive_weight),
            torch.ones_like(forget_labels),
        )
        forget_loss = F.binary_cross_entropy_with_logits(
            forget_logits, forget_labels, weight=forget_weights
        )
        loss = write_loss + forget_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.memory_policy.parameters(), 1.0)
        optimizer.step()

    model.memory_policy.eval()
    model._memory_policy_ready = True
    train_logits = model.memory_policy(train_vectors).reshape(-1)
    eval_logits = model.memory_policy(eval_vectors).reshape(-1)
    train_forget_logits = model.memory_policy.forget_logits(train_vectors).reshape(-1)
    eval_forget_logits = model.memory_policy.forget_logits(eval_vectors).reshape(-1)
    selected = _choose_threshold(eval_logits, eval_labels, args.max_fpr)
    selected_forget = _choose_threshold(eval_forget_logits, eval_forget_labels, args.max_fpr)
    threshold = float(args.threshold if args.threshold is not None else selected["threshold"])
    forget_threshold = float(selected_forget["threshold"])
    # The selected thresholds are part of the adapter contract.  Keeping the
    # write threshold local to the report would silently revert to the config
    # default when the adapter is loaded by the benchmark or service.
    model.memory_config.auto_memory_threshold = threshold
    model.memory_config.auto_forget_threshold = forget_threshold
    output_dir = _project_path(args.output_adapter)
    # The embedded package loader temporarily restores its user snapshot and
    # marks the config persistent.  A policy adapter must be stateless: never
    # ship the source user's memory with a training candidate.
    model.memory_config.persistent_memory = False
    stale_user_state = output_dir / "persistent_memory.pt"
    if stale_user_state.exists():
        stale_user_state.rename(output_dir / "persistent_memory.pt.disabled")
    model.save_memory_adapter(output_dir)
    report = {
        "format_version": 1,
        "dataset_dir": str(dataset_dir),
        "base_adapter": str(base_adapter),
        "steps": steps,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "selected_threshold": selected,
        "threshold": threshold,
        "selected_forget_threshold": selected_forget,
        "forget_threshold": forget_threshold,
        "forget_positive_weight": forget_positive_weight,
        "train": _metrics(train_logits, train_labels, threshold),
        "eval": _metrics(eval_logits, eval_labels, threshold),
        "train_forget": _metrics(
            train_forget_logits,
            train_forget_labels,
            forget_threshold,
        ),
        "eval_forget": _metrics(
            eval_forget_logits,
            eval_forget_labels,
            forget_threshold,
        ),
        "forget_label_count": int(sum(float(row.get("forget_label", 0.0)) >= 0.5 for row in train_rows + eval_rows)),
        "warning": "This candidate must pass the full benchmark before replacing the production adapter.",
    }
    (output_dir / "production_policy_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
