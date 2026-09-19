"""Train the hierarchical page router on real frozen Qwen text states.

The old router smoke test used an abstract latent basis.  That is useful for
checking tensor shapes, but it does not prove that a router can address the
actual Chinese natural-language memory keys emitted by Qwen.  This script
keeps the backbone frozen, encodes randomized fact/query episodes with the
same runtime key path, and trains only the small MemoryRouterV2 controller.
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
from V2_dpskw.qwen_integration import load_qwen_dynamic, load_tokenizer


ATTRIBUTES = (
    "常用语言",
    "工作地点",
    "最喜欢的水果",
    "项目代号",
    "常用时区",
    "提醒时间",
    "默认输出风格",
    "备用联系人",
    "编辑器",
    "常住城市",
)

FACT_TEMPLATES = (
    "我的{attribute}是{value}。",
    "请记住，我的{attribute}为{value}。",
    "个人资料更新：{attribute}={value}。",
    "以后涉及{attribute}时，请使用{value}。",
)

READ_TEMPLATES = (
    "跨对话后，请告诉我已经保存的{attribute}。",
    "不要猜测，读取我的{attribute}资料。",
    "之前登记的{attribute}是哪一个？",
    "长期记忆中，我的{attribute}是什么？",
    "请从个人资料里查找{attribute}。",
)

UNKNOWN_TEMPLATES = (
    "请解释一下这个概念，不要查询个人资料。",
    "帮我写一段代码，不需要读取记忆。",
    "如果我的{attribute}改成另一个值，会有什么影响？",
    "别人说我的{attribute}是某个值，但那不是我的资料。",
    "今天的临时编号是什么？如果没有记录就说不知道。",
    "我的不存在的{attribute}是什么？如果没有登记就说不知道。",
    "从来没有登记过的{attribute}是什么？不要从别的字段推断。",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _plain_ids(tokenizer: Any, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def _build_episodes(seed: int, count: int, candidate_count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    episodes: list[dict[str, Any]] = []
    for index in range(count):
        attribute = ATTRIBUTES[index % len(ATTRIBUTES)]
        value = f"ROUTE-{seed}-{index:06d}"
        target = rng.choice(FACT_TEMPLATES).format(attribute=attribute, value=value)
        candidates = [target]
        while len(candidates) < candidate_count:
            other_attribute = rng.choice(ATTRIBUTES)
            other_value = f"DIST-{seed}-{index:06d}-{len(candidates):02d}"
            candidates.append(
                rng.choice(FACT_TEMPLATES).format(
                    attribute=other_attribute,
                    value=other_value,
                )
            )
        rng.shuffle(candidates)
        episodes.append(
            {
                "query": rng.choice(READ_TEMPLATES).format(attribute=attribute),
                "candidates": candidates,
                "positive_index": candidates.index(target),
                "need_memory": 1.0,
                "hop": 1,
            }
        )
        if index % 2 == 0:
            unknown_query = rng.choice(UNKNOWN_TEMPLATES).format(attribute=attribute)
            unknown_candidates = [
                rng.choice(FACT_TEMPLATES).format(
                    attribute=rng.choice(ATTRIBUTES),
                    value=f"UNKNOWN-{seed}-{index:06d}-{candidate:02d}",
                )
                for candidate in range(candidate_count)
            ]
            episodes.append(
                {
                    "query": unknown_query,
                    "candidates": unknown_candidates,
                    "positive_index": 0,
                    "need_memory": 0.0,
                    "hop": 0,
                }
            )
    rng.shuffle(episodes)
    return episodes


@torch.inference_mode()
def _encode_texts(model: Any, tokenizer: Any, texts: list[str], *, batch_size: int, device: torch.device) -> torch.Tensor:
    rows = [_plain_ids(tokenizer, text) for text in texts]
    vectors: list[torch.Tensor | None] = [None] * len(rows)
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(int(row.numel()), []).append(index)
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            length = rows[selected[0]].numel()
            ids = torch.stack([rows[index] for index in selected]).to(device)
            mask = torch.ones((len(selected), length), dtype=torch.long, device=device)
            encoded = model._encode_model_key(ids, mask).detach().float().cpu()
            for row_index, vector in zip(selected, encoded):
                vectors[row_index] = vector
    if any(vector is None for vector in vectors):
        raise RuntimeError("failed to encode router text states")
    return torch.stack([vector for vector in vectors if vector is not None])


def _tensorize_episodes(
    model: Any,
    tokenizer: Any,
    episodes: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    all_texts: list[str] = []
    query_index: list[int] = []
    candidate_indices: list[list[int]] = []
    lookup: dict[str, int] = {}
    for episode in episodes:
        query = str(episode["query"])
        if query not in lookup:
            lookup[query] = len(all_texts)
            all_texts.append(query)
        query_index.append(lookup[query])
        row_indices: list[int] = []
        for candidate in episode["candidates"]:
            candidate = str(candidate)
            if candidate not in lookup:
                lookup[candidate] = len(all_texts)
                all_texts.append(candidate)
            row_indices.append(lookup[candidate])
        candidate_indices.append(row_indices)
    encoded = _encode_texts(model, tokenizer, all_texts, batch_size=batch_size, device=device)
    queries = encoded[torch.tensor(query_index, dtype=torch.long)]
    candidates = torch.stack(
        [encoded[torch.tensor(indices, dtype=torch.long)] for indices in candidate_indices]
    )
    positives = torch.tensor(
        [int(episode["positive_index"]) for episode in episodes],
        dtype=torch.long,
    )
    need = torch.tensor([float(episode["need_memory"]) for episode in episodes], dtype=torch.float32)
    hops = torch.tensor([int(episode["hop"]) for episode in episodes], dtype=torch.long)
    return queries, candidates, positives, need, hops


@torch.inference_mode()
def _evaluate(router: MemoryRouterV2, data: tuple[torch.Tensor, ...], *, device: torch.device) -> dict[str, float]:
    queries, candidates, positives, need, hops = data
    router.eval()
    output = router(queries.to(device), candidates.to(device))
    predicted = output["scores"].argmax(dim=-1).cpu()
    need_probability = torch.sigmoid(output["need_memory_logits"]).cpu()
    need_pred = need_probability >= 0.5
    required = need >= 0.5
    route_mask = required
    route_accuracy = float((predicted[route_mask] == positives[route_mask]).float().mean()) if bool(route_mask.any()) else 0.0
    tp = int((need_pred & required).sum())
    tn = int((~need_pred & ~required).sum())
    fp = int((need_pred & ~required).sum())
    fn = int((~need_pred & required).sum())
    score_sorted = output["scores"].detach().cpu().topk(min(2, output["scores"].shape[-1]), dim=-1).values
    margin = score_sorted[:, 0] - score_sorted[:, 1] if score_sorted.shape[-1] > 1 else score_sorted[:, 0]
    return {
        "episodes": float(len(queries)),
        "route_accuracy": route_accuracy,
        "need_precision": tp / max(1, tp + fp),
        "need_recall": tp / max(1, tp + fn),
        "need_specificity": tn / max(1, tn + fp),
        "abstention_accuracy": float((need_pred == required).float().mean()),
        "hop_accuracy": float((output["hop_logits"].argmax(dim=-1).cpu() == hops).float().mean()),
        "positive_margin_mean": float(margin[route_mask].mean()) if bool(route_mask.any()) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="W:/Flash/model/V2_dpskw/qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--output-dir", default="W:/Flash/model/V2_dpskw/checkpoints/natural_memory_v2_router_text_v8")
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--eval-episodes", type=int, default=320)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    model = load_qwen_dynamic(model_path, load_in_4bit=not args.no_4bit)
    model.eval()
    hidden_size = int(model.memory.hidden_size)
    train_episodes = _build_episodes(args.seed, args.train_episodes, args.candidate_count)
    eval_episodes = _build_episodes(args.seed + 1, args.eval_episodes, args.candidate_count)
    train_data = _tensorize_episodes(
        model, tokenizer, train_episodes,
        batch_size=args.encode_batch_size, device=device,
    )
    eval_data = _tensorize_episodes(
        model, tokenizer, eval_episodes,
        batch_size=args.encode_batch_size, device=device,
    )
    router = MemoryRouterV2(
        hidden_size,
        router_dim=128,
        num_heads=8,
        max_hops=3,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = random.Random(args.seed + 2)
    query_cpu, candidates_cpu, positives_cpu, need_cpu, hops_cpu = train_data
    history: list[dict[str, float]] = []
    router.train()
    for step in range(1, args.steps + 1):
        indices = torch.tensor(
            [rng.randrange(query_cpu.shape[0]) for _ in range(min(args.batch_size, query_cpu.shape[0]))],
            dtype=torch.long,
        )
        query = query_cpu[indices].to(device)
        candidates = candidates_cpu[indices].to(device)
        positives = positives_cpu[indices].to(device)
        need = need_cpu[indices].to(device)
        hops = hops_cpu[indices].to(device)
        output = router(query, candidates)
        route_mask = need >= 0.5
        candidate_loss = (
            F.cross_entropy(output["scores"][route_mask], positives[route_mask])
            if bool(route_mask.any()) else output["scores"].sum() * 0.0
        )
        need_loss = F.binary_cross_entropy_with_logits(output["need_memory_logits"], need)
        hop_loss = F.cross_entropy(output["hop_logits"], hops)
        loss = candidate_loss + 1.0 * need_loss + 0.35 * hop_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            stats = _evaluate(router, eval_data, device=device)
            stats["step"] = float(step)
            stats["loss"] = float(loss.detach().cpu())
            history.append(stats)
            print(json.dumps(stats, ensure_ascii=False))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), output_dir / "memory_router_v2.pt")
    report = {
        "format_version": 1,
        "training_protocol": "frozen_qwen_hidden_states_with_randomized_paraphrase_distractor_and_unknown_episodes",
        "model_path": str(model_path),
        "hidden_size": hidden_size,
        "router_dim": 128,
        "num_heads": 8,
        "candidate_count": args.candidate_count,
        "train_episodes": len(train_episodes),
        "eval_episodes": len(eval_episodes),
        "steps": args.steps,
        "device": str(device),
        "history": history,
        "eval": _evaluate(router, eval_data, device=device),
        "warning": "Synthetic red-team router training; require redacted real traces before production certification.",
    }
    (output_dir / "memory_router_text_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
