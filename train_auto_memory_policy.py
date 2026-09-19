"""Train the internal high-recall automatic memory write policy.

The frozen Qwen backbone and the existing native memory controller provide
the representation.  Only a small policy head is trained.  Positive examples
cover durable personal facts, preferences, plans, project constraints and
corrections; negative examples cover questions, requests, hypotheticals and
casual conversation.  The runtime still stores the exact user token sequence,
so this head decides *whether* to remember rather than compressing the fact.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer


POSITIVE = (
    "我叫{value}。",
    "我的常住城市是{value}。",
    "我最喜欢的水果是{value}。",
    "我正在开发{value}项目。",
    "以后请把代码默认写成{value}。",
    "我计划在{value}完成这个任务。",
    "我的工作地点是{value}。",
    "我的常用时区是{value}。",
    "请记住这条信息：{value}。",
    "更正一下，刚才的内容应该是{value}。",
    "这是我的长期偏好：{value}。",
    "这个项目的重要约束是{value}。",
)

NEGATIVE = (
    "帮我写一段关于{value}的代码。",
    "请解释{value}是什么意思。",
    "{value}是什么？",
    "你觉得{value}怎么样？",
    "如果以后遇到{value}，应该怎么办？",
    "今天天气不错，随便聊聊{value}。",
    "请把{value}翻译成英文。",
    "计算一下{value}。",
    "给我介绍一下{value}。",
    "哈哈，{value}真有意思。",
    "我想知道之前有没有提到{value}。",
    "假设我选择{value}，会发生什么？",
    "我叫什么？",
    "我的名字是什么？",
    "你还记得我叫什么吗？",
    "我正在开发什么项目？",
    "我的项目叫什么？",
    "请问我的项目叫什么？",
    "我的工作地点是什么？",
    "我的工作地点代号是什么？",
    "工作地点代号是多少？",
    "你记得我的工作地点吗？",
    "请告诉我之前有没有说过{value}。",
    "我之前有没有告诉过你{value}？",
    "能不能帮我完成{value}？",
    "如何处理{value}？",
    "请给我一个{value}的方案。",
)

VALUES = (
    "小明",
    "上海",
    "红富士苹果",
    "个人记忆系统",
    "Python",
    "下周五",
    "R7",
    "Asia/Shanghai",
    "不要删除用户数据",
    "使用简洁中文",
    "每天晚上八点",
    "蓝鲸-47",
)


def make_examples(seed: int, count: int) -> list[tuple[str, float]]:
    rng = random.Random(seed)
    examples: list[tuple[str, float]] = []
    for _ in range(count):
        examples.append((rng.choice(POSITIVE).format(value=rng.choice(VALUES)), 1.0))
        examples.append((rng.choice(NEGATIVE).format(value=rng.choice(VALUES)), 0.0))
    rng.shuffle(examples)
    return examples


def encode_batch(tokenizer, texts: list[str], device: torch.device):
    rows = []
    masks = []
    for text in texts:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        rows.append(encoded["input_ids"][0])
        masks.append(encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))[0])
    max_length = max(row.numel() for row in rows)
    input_ids = torch.zeros(len(rows), max_length, dtype=torch.long, device=device)
    attention_mask = torch.zeros(len(rows), max_length, dtype=torch.long, device=device)
    for index, (row, mask) in enumerate(zip(rows, masks)):
        input_ids[index, : row.numel()] = row.to(device)
        attention_mask[index, : mask.numel()] = mask.to(device)
    return input_ids, attention_mask


@torch.inference_mode()
def collect_representations(model, tokenizer, examples, *, batch_size: int, device):
    vectors = []
    labels = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, attention_mask = encode_batch(
            tokenizer,
            [item[0] for item in batch],
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
        representation = getattr(model.memory, "last_write_representation", None)
        if representation is None:
            raise RuntimeError("native controller did not expose a write representation")
        vectors.append(representation.detach().float().cpu())
        labels.extend(item[1] for item in batch)
    return torch.cat(vectors), torch.tensor(labels, dtype=torch.float32)


def evaluate(policy, vectors, labels, threshold: float) -> dict[str, float]:
    with torch.inference_mode():
        probabilities = torch.sigmoid(policy(vectors)).cpu()
        labels = labels.cpu()
        predictions = probabilities >= threshold
        positive = labels >= 0.5
        negative = ~positive
        true_positive = (predictions & positive).sum().item()
        false_negative = ((~predictions) & positive).sum().item()
        false_positive = (predictions & negative).sum().item()
        true_negative = ((~predictions) & negative).sum().item()
    return {
        "threshold": threshold,
        "accuracy": float((predictions == positive).float().mean()),
        "positive_recall": true_positive / max(1, true_positive + false_negative),
        "negative_specificity": true_negative / max(1, true_negative + false_positive),
        "false_positive_rate": false_positive / max(1, false_positive + true_negative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument(
        "--base-adapter",
        default="V2_dpskw/qwen_memory_adapter_natural_controller_v3",
    )
    parser.add_argument(
        "--output-adapter",
        default="V2_dpskw/qwen_memory_adapter_natural_auto_v3",
    )
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--example-count", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument(
        "--text-memory-threshold",
        type=float,
        default=0.30,
        help="retrieval threshold used by the automatic-memory adapter",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    config = load_memory_config(args.base_adapter)
    config.natural_language_memory = True
    config.automatic_memory = True
    config.auto_memory_threshold = args.threshold
    config.text_memory_threshold = args.text_memory_threshold
    config.persistent_memory = False
    model = load_qwen_dynamic(
        args.model_path,
        memory_config=config,
        load_in_4bit=not args.no_4bit,
    )
    model.load_memory_adapter(args.base_adapter, strict=True)
    if model.memory_policy is None:
        raise RuntimeError("automatic memory policy was not created")
    model.eval()
    model.memory_policy.train()
    device = model._find_layer_device()

    examples = make_examples(args.seed, args.example_count)
    split = int(len(examples) * 0.8)
    train_examples = examples[:split]
    eval_examples = examples[split:]
    train_vectors, train_labels = collect_representations(
        model,
        tokenizer,
        train_examples,
        batch_size=args.batch_size,
        device=device,
    )
    eval_vectors, eval_labels = collect_representations(
        model,
        tokenizer,
        eval_examples,
        batch_size=args.batch_size,
        device=device,
    )
    train_vectors = train_vectors.to(device)
    train_labels = train_labels.to(device)
    eval_vectors = eval_vectors.to(device)
    eval_labels = eval_labels.to(device)

    optimizer = torch.optim.AdamW(model.memory_policy.parameters(), lr=args.lr, weight_decay=0.01)
    positive_weight = torch.tensor([1.5], device=device)
    rng = random.Random(args.seed + 1)
    for step in range(1, args.steps + 1):
        indices = torch.tensor(
            [rng.randrange(train_vectors.shape[0]) for _ in range(args.batch_size)],
            dtype=torch.long,
            device=device,
        )
        logits = model.memory_policy(train_vectors[indices])
        weights = torch.where(train_labels[indices] >= 0.5, positive_weight, torch.ones_like(logits))
        loss = F.binary_cross_entropy_with_logits(logits, train_labels[indices], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.memory_policy.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            stats = evaluate(model.memory_policy, train_vectors, train_labels, args.threshold)
            print(
                f"step={step} loss={float(loss.detach()):.5f} "
                f"train_accuracy={stats['accuracy']:.3f} "
                f"positive_recall={stats['positive_recall']:.3f} "
                f"false_positive_rate={stats['false_positive_rate']:.3f}"
            )

    model.memory_policy.eval()
    model._memory_policy_ready = True
    output_dir = Path(args.output_adapter)
    model.save_memory_adapter(output_dir)
    report = {
        "steps": args.steps,
        "example_count_per_class": args.example_count,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "source_adapter": str(args.base_adapter),
        "policy": "high_recall_automatic_memory_importance",
        "threshold": args.threshold,
        "train": evaluate(model.memory_policy, train_vectors, train_labels, args.threshold),
        "eval": evaluate(model.memory_policy, eval_vectors, eval_labels, args.threshold),
    }
    (output_dir / "auto_policy_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
