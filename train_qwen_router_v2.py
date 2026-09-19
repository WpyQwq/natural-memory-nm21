"""Train the V2 router on real Qwen hidden representations.

This is intentionally separate from the fast synthetic router pre-training.
The production checkpoint must see the same representation distribution that
the memory adapter will use at runtime.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import MemoryRouterV2
from V2_dpskw.qwen_integration import QwenMemoryConfig, load_qwen_dynamic, load_tokenizer


ATTRIBUTES = [
    "姓名",
    "常住城市",
    "工作地点",
    "项目代号",
    "喜欢的水果",
    "宠物名字",
    "生日月份",
    "最常用的编辑器",
    "长期目标",
    "周末习惯",
    "学习方向",
    "重要联系人",
]
ENTITIES = [f"用户档案{index:02d}" for index in range(64)]
VALUES = [
    "林浩",
    "上海",
    "杭州",
    "NM-V2",
    "青提",
    "小灰",
    "十月",
    "Neovim",
    "做出新的记忆架构",
    "阅读论文",
    "稀疏路由",
    "陈老师",
    "苏州",
    "Natural Memory",
    "星河项目",
    "午夜跑步",
]


def _device(name: str, fallback: torch.device | None = None) -> torch.device:
    if name == "auto":
        return fallback or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _encode_texts(model, tokenizer, texts: list[str], device: torch.device, batch_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        key = model._encode_model_key(input_ids, attention_mask.to(device))
        chunks.append(key.detach().cpu())
    return torch.cat(chunks, dim=0)


def _make_fact_set(count: int, seed: int) -> tuple[list[str], list[str], list[int]]:
    random.seed(seed)
    facts: list[str] = []
    queries: list[str] = []
    hops: list[int] = []
    for index in range(count):
        attribute = ATTRIBUTES[index % len(ATTRIBUTES)]
        value = VALUES[(index * 7 + 3) % len(VALUES)]
        entity = ENTITIES[index // len(ATTRIBUTES)]
        fact = f"{entity}的{attribute}是{value}。"
        query_templates = [
            (f"请问{entity}的{attribute}是什么？", 1),
            (f"我之前告诉过你的{entity}{attribute}，答案是什么？", 1),
            (f"先找出{entity}的{attribute}，再结合关联记忆回答。", 2),
            (f"回忆一下，{entity}在{attribute}这一项的信息。", 1),
        ]
        facts.append(fact)
        query, hop = query_templates[index % len(query_templates)]
        queries.append(query)
        hops.append(hop)
    return facts, queries, hops


def _make_episodes(
    fact_keys: torch.Tensor,
    query_keys: torch.Tensor,
    fact_hops: list[int],
    *,
    candidate_count: int,
    seed: int,
    no_memory_keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    random.seed(seed)
    sample_count = fact_keys.shape[0]
    queries: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    positives: list[int] = []
    need: list[float] = []
    hops: list[int] = []
    for index in range(sample_count):
        candidate_indices = [index]
        # Prefer same-entity/nearby attribute negatives before random ones.
        for offset in range(1, sample_count):
            candidate_indices.append((index + offset) % sample_count)
            if len(candidate_indices) >= candidate_count:
                break
        candidate_tensor = fact_keys[candidate_indices]
        queries.append(query_keys[index])
        candidates.append(candidate_tensor)
        positives.append(0)
        need.append(1.0)
        hops.append(int(fact_hops[index]))
    for index in range(min(sample_count // 3, no_memory_keys.shape[0])):
        queries.append(no_memory_keys[index])
        candidates.append(fact_keys[torch.randperm(sample_count)[:candidate_count]])
        positives.append(0)
        need.append(0.0)
        hops.append(0)
    return (
        torch.stack(queries),
        torch.stack(candidates),
        torch.tensor(positives, dtype=torch.long),
        torch.tensor(need, dtype=torch.float32),
        torch.tensor(hops, dtype=torch.long),
    )


@torch.no_grad()
def _evaluate(
    router: MemoryRouterV2,
    query: torch.Tensor,
    candidates: torch.Tensor,
    positive: torch.Tensor,
    need: torch.Tensor,
    hops: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    router.eval()
    output = router(query.to(device), candidates.to(device))
    need_mask = need.to(device).bool()
    route_pred = output["scores"].argmax(dim=-1)
    route_correct = ((route_pred == positive.to(device)) & need_mask).sum().item()
    route_total = need_mask.sum().item()
    need_pred = (torch.sigmoid(output["need_memory_logits"]) >= 0.5).float()
    need_device = need.to(device)
    tp = ((need_pred == 1) & (need_device == 1)).sum().item()
    tn = ((need_pred == 0) & (need_device == 0)).sum().item()
    fp = ((need_pred == 1) & (need_device == 0)).sum().item()
    fn = ((need_pred == 0) & (need_device == 1)).sum().item()
    hop_accuracy = (output["hop_logits"].argmax(dim=-1) == hops.to(device)).float().mean().item()
    return {
        "route_accuracy": route_correct / max(1, route_total),
        "need_memory_precision": tp / max(1, tp + fp),
        "need_memory_recall": tp / max(1, tp + fn),
        "need_memory_specificity": tn / max(1, tn + fp),
        "hop_accuracy": hop_accuracy,
        "need_tp": float(tp),
        "need_tn": float(tn),
        "need_fp": float(fp),
        "need_fn": float(fn),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = QwenMemoryConfig(
        memory_slots=16,
        memory_dim=512,
        layer_indices=(7, 15, 23, 31),
        mode="blend",
        blend_init=0.1,
        native_mode=True,
        persistent_memory=True,
        natural_language_memory=True,
        automatic_memory=True,
        memory_version=2,
        hierarchical_memory=True,
        memory_router_dim=args.router_dim,
        memory_router_heads=args.num_heads,
        memory_max_hops=args.max_hops,
    )
    model = load_qwen_dynamic(
        args.model_path,
        memory_config=config,
        load_in_4bit=not args.no_4bit,
        device_map="auto",
    )
    model.eval()
    model_device = model._find_layer_device()
    device = _device(args.device, model_device)
    tokenizer = load_tokenizer(args.model_path)
    fact_texts, query_texts, fact_hops = _make_fact_set(args.fact_count, args.seed)
    no_memory_texts = [
        "请写一首关于春天的短诗。",
        "解释一下二分查找的时间复杂度。",
        "帮我规划一个周末旅行。",
        "什么是矩阵乘法？",
        "把这句话翻译成英文。",
        "今天适合做什么运动？",
    ]
    fact_keys = _encode_texts(model, tokenizer, fact_texts, model_device, args.encode_batch_size)
    query_keys = _encode_texts(model, tokenizer, query_texts, model_device, args.encode_batch_size)
    no_memory_keys = _encode_texts(model, tokenizer, no_memory_texts, model_device, args.encode_batch_size)
    train_count = max(1, int(fact_keys.shape[0] * 0.8))
    train_query, train_candidates, train_positive, train_need, train_hops = _make_episodes(
        fact_keys[:train_count],
        query_keys[:train_count],
        fact_hops[:train_count],
        candidate_count=args.candidate_count,
        seed=args.seed,
        no_memory_keys=no_memory_keys,
    )
    heldout_query, heldout_candidates, heldout_positive, heldout_need, heldout_hops = _make_episodes(
        fact_keys[train_count:],
        query_keys[train_count:],
        fact_hops[train_count:],
        candidate_count=args.candidate_count,
        seed=args.seed + 1,
        no_memory_keys=no_memory_keys,
    )
    router = MemoryRouterV2(
        args.hidden_size,
        router_dim=args.router_dim,
        num_heads=args.num_heads,
        max_hops=args.max_hops,
    ).to(device)
    if args.init_checkpoint:
        initial_state = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        router.load_state_dict(initial_state, strict=True)
    if args.freeze_retrieval:
        for name, parameter in router.named_parameters():
            parameter.requires_grad = name.startswith("hop_controller.")
    trainable_parameters = [parameter for parameter in router.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    router.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(0, train_query.shape[0], (args.batch_size,))
        query = train_query[indices].to(device)
        candidates = train_candidates[indices].to(device)
        positive = train_positive[indices].to(device)
        need = train_need[indices].to(device)
        hops = train_hops[indices].clamp(0, args.max_hops).to(device)
        output = router(query, candidates)
        need_mask = need.bool()
        candidate_loss = (
            F.cross_entropy(output["scores"][need_mask], positive[need_mask])
            if bool(need_mask.any())
            else output["scores"].sum() * 0.0
        )
        need_loss = F.binary_cross_entropy_with_logits(output["need_memory_logits"], need)
        hop_loss = F.cross_entropy(output["hop_logits"], hops)
        loss = candidate_loss + args.need_loss_weight * need_loss + args.hop_loss_weight * hop_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "candidate_loss": float(candidate_loss.detach().cpu()),
                    "need_loss": float(need_loss.detach().cpu()),
                    "hop_loss": float(hop_loss.detach().cpu()),
                }
            )
    validation = _evaluate(
        router,
        heldout_query,
        heldout_candidates,
        heldout_positive,
        heldout_need,
        heldout_hops,
        device,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), output_dir / "memory_router_v2.pt")
    torch.save(
        {
            "fact_keys": fact_keys,
            "query_keys": query_keys,
            "no_memory_keys": no_memory_keys,
            "fact_texts": fact_texts,
            "query_texts": query_texts,
        },
        output_dir / "qwen_router_v2_encoded_dataset.pt",
    )
    summary = {
        "format_version": 2,
        "representation": "qwen3.5_hidden_state",
        "model_path": args.model_path,
        "device": str(device),
        "model_device": str(model_device),
        "hidden_size": args.hidden_size,
        "router_dim": args.router_dim,
        "num_heads": args.num_heads,
        "max_hops": args.max_hops,
        "fact_count": args.fact_count,
        "train_count": train_count,
        "heldout_count": int(fact_keys.shape[0] - train_count),
        "steps": args.steps,
        "training_history": history,
        "validation": validation,
    }
    (output_dir / "qwen_router_v2_training.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="W:/Flash/model/V2_dpskw/qwen3_5_4b_memory_merged_v13")
    parser.add_argument("--output-dir", default="W:/Flash/model/V2_dpskw/checkpoints/natural_memory_v2_qwen_router")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--router-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--fact-count", type=int, default=160)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--need-loss-weight", type=float, default=0.75)
    parser.add_argument("--hop-loss-weight", type=float, default=0.35)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--freeze-retrieval", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))
