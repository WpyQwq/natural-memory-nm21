"""Train the Natural Memory v2 sparse router on hard-negative episodes.

The default dataset is generated from a shared latent factor space.  Train
and validation topics are sampled independently from that same space, so the
validation numbers measure generalization rather than memorization of topic
ids.  A JSONL collector can be added later without changing the router loss;
the important unit is still a query, candidate addresses, a positive index,
and a write/read policy label.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import MemoryRouterV2


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_basis(hidden_size: int, latent_size: int, device: torch.device) -> torch.Tensor:
    basis = torch.randn(hidden_size, latent_size, device=device)
    return F.normalize(basis, dim=0)


def _latent_to_hidden(latent: torch.Tensor, basis: torch.Tensor, noise: float) -> torch.Tensor:
    hidden = latent @ basis.T
    if noise > 0:
        hidden = hidden + noise * torch.randn_like(hidden)
    return hidden


def sample_episode(
    *,
    batch_size: int,
    candidate_count: int,
    basis: torch.Tensor,
    device: torch.device,
    no_memory_rate: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create semantic positives, hard negatives and no-memory episodes."""

    latent_size = basis.shape[1]
    query_latent = torch.randn(batch_size, latent_size, device=device)
    no_memory = torch.rand(batch_size, device=device) < no_memory_rate
    # A no-memory query is deliberately drawn from a low-information region
    # instead of being another random topic.  This gives the policy head a
    # learnable abstention signal while still leaving unrelated hard negatives
    # in the candidate set.
    query_latent[no_memory] = 0.0
    query = _latent_to_hidden(query_latent, basis, 0.04)
    candidates_latent = torch.randn(batch_size, candidate_count, latent_size, device=device)
    candidates_latent[:, 0] = query_latent + 0.04 * torch.randn_like(query_latent)
    if candidate_count > 1:
        # A hard negative shares most factors but changes two dimensions.
        hard = query_latent + 0.16 * torch.randn_like(query_latent)
        hard[:, :2] = -hard[:, :2]
        candidates_latent[:, 1] = hard
    candidates_latent[no_memory] = torch.randn(
        int(no_memory.sum().item()), candidate_count, latent_size, device=device
    )
    candidates = _latent_to_hidden(candidates_latent, basis, 0.04)
    need_memory = (~no_memory).float()
    hop_label = torch.where(
        query_latent.norm(dim=-1) > math.sqrt(latent_size),
        torch.full((batch_size,), 2, device=device, dtype=torch.long),
        torch.where(
        query_latent[:, 0] > 0,
            torch.ones(batch_size, device=device, dtype=torch.long),
            torch.zeros(batch_size, device=device, dtype=torch.long),
        ),
    )
    positive_index = torch.zeros(batch_size, device=device, dtype=torch.long)
    return query, candidates, positive_index, need_memory, hop_label


@torch.no_grad()
def evaluate(
    router: MemoryRouterV2,
    *,
    basis: torch.Tensor,
    device: torch.device,
    batches: int,
    batch_size: int,
    candidate_count: int,
) -> dict[str, float]:
    router.eval()
    route_correct = 0
    route_total = 0
    need_tp = need_tn = need_fp = need_fn = 0
    hop_correct = 0
    hop_total = 0
    for _ in range(batches):
        query, candidates, positive, need, hop = sample_episode(
            batch_size=batch_size,
            candidate_count=candidate_count,
            basis=basis,
            device=device,
            no_memory_rate=0.25,
        )
        output = router(query, candidates)
        predicted = output["scores"].argmax(dim=-1)
        route_correct += int(((predicted == positive) & need.bool()).sum().item())
        route_total += int(need.sum().item())
        need_pred = (torch.sigmoid(output["need_memory_logits"]) >= 0.5).float()
        need_tp += int(((need_pred == 1) & (need == 1)).sum().item())
        need_tn += int(((need_pred == 0) & (need == 0)).sum().item())
        need_fp += int(((need_pred == 1) & (need == 0)).sum().item())
        need_fn += int(((need_pred == 0) & (need == 1)).sum().item())
        hop_correct += int((output["hop_logits"].argmax(dim=-1) == hop).sum().item())
        hop_total += batch_size
    precision = need_tp / max(1, need_tp + need_fp)
    recall = need_tp / max(1, need_tp + need_fn)
    return {
        "route_accuracy": route_correct / max(1, route_total),
        "need_memory_precision": precision,
        "need_memory_recall": recall,
        "need_memory_specificity": need_tn / max(1, need_tn + need_fp),
        "hop_accuracy": hop_correct / max(1, hop_total),
        "need_tp": float(need_tp),
        "need_tn": float(need_tn),
        "need_fp": float(need_fp),
        "need_fn": float(need_fn),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    basis = _make_basis(args.hidden_size, args.latent_size, device)
    router = MemoryRouterV2(
        args.hidden_size,
        router_dim=args.router_dim,
        num_heads=args.num_heads,
        max_hops=args.max_hops,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    router.train()
    for step in range(1, args.steps + 1):
        query, candidates, positive, need, hop = sample_episode(
            batch_size=args.batch_size,
            candidate_count=args.candidate_count,
            basis=basis,
            device=device,
        )
        output = router(query, candidates)
        route_mask = need.bool()
        candidate_loss = (
            F.cross_entropy(output["scores"][route_mask], positive[route_mask])
            if bool(route_mask.any())
            else output["scores"].sum() * 0.0
        )
        need_loss = F.binary_cross_entropy_with_logits(output["need_memory_logits"], need)
        hop_loss = F.cross_entropy(output["hop_logits"], hop)
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
    validation = evaluate(
        router,
        basis=basis,
        device=device,
        batches=args.eval_batches,
        batch_size=args.batch_size,
        candidate_count=args.candidate_count,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), output_dir / "memory_router_v2.pt")
    torch.save(basis.detach().cpu(), output_dir / "memory_router_v2_basis.pt")
    summary = {
        "format_version": 2,
        "seed": args.seed,
        "device": str(device),
        "hidden_size": args.hidden_size,
        "router_dim": args.router_dim,
        "num_heads": args.num_heads,
        "max_hops": args.max_hops,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "candidate_count": args.candidate_count,
        "training_history": history,
        "validation": validation,
    }
    (output_dir / "memory_router_v2_training.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="W:/Flash/model/V2_dpskw/checkpoints/natural_memory_v2_router")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--router-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--need-loss-weight", type=float, default=0.75)
    parser.add_argument("--hop-loss-weight", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result["validation"], ensure_ascii=False, indent=2))
