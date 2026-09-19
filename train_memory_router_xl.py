"""Train MemoryRouterXL on the frozen routing episodes used by the 512-dim run.

Design choice: the *measurement* code is imported from
``V2_dpskw.train_memory_router_large`` (frozen data reader, feature-cache
validation, batch assembly, listwise + margin loss, evaluation metrics,
selection score and sampling).  Only the model changes -- ``MemoryRouterXL``
instead of ``MemoryRouterV2`` -- so every number this script prints is directly
comparable with the existing 512-dim baseline on the same frozen ``eval.jsonl``.

The backbone is never updated and (when the feature cache already matches the
frozen data hashes) the 4B Qwen model is never even loaded.

Example::

    python -m V2_dpskw.train_memory_router_xl ^
        --train-file data/router_training_v3/train.jsonl ^
        --eval-file  data/router_training_v3/eval.jsonl ^
        --model-path qwen3_5_4b_natural_memory_v2 ^
        --feature-cache-dir checkpoints/router_shared/feature_cache ^
        --output-dir checkpoints/router_xl_1024 ^
        --router-dim 1024 --num-heads 16 --encoder-layers 2 ^
        --pair-blocks 1 --pair-hidden 1024 --policy-layers 2 --policy-hidden 512 ^
        --steps 100000 --batch-size 64 --eval-interval 500 --checkpoint-interval 5000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.router_xl import ARCH_NAME, ARCH_VERSION, MemoryRouterXL
from V2_dpskw.train_memory_router_large import (
    _batch_from_indices,
    _device,
    _episode_tensors,
    _evaluate,
    _load_checkpoint,
    _prepare_feature_cache,
    _read_episodes,
    _resolve_path,
    _router_loss,
    _sample_indices,
    _set_cuda_cap,
    _sha256,
)


def _arch_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "router_dim": args.router_dim,
        "num_heads": args.num_heads,
        "max_hops": args.max_hops,
        "encoder_layers": args.encoder_layers,
        "encoder_hidden": args.encoder_hidden,
        "pair_blocks": args.pair_blocks,
        "pair_hidden": args.pair_hidden,
        "pair_expansion": args.pair_expansion,
        "pair_dropout": args.pair_dropout,
        "use_interaction": args.use_interaction,
        "policy_layers": args.policy_layers,
        "policy_hidden": args.policy_hidden,
        "policy_dropout": args.policy_dropout,
        "learnable_cosine_scale": args.cosine_scale,
    }


def _save_checkpoint(path: Path, router: MemoryRouterXL, optimizer: Any, scheduler: Any, step: int, best_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 3,
        "arch_config": router.arch_config(),
        "step": step,
        "best_score": best_score,
        "router_state_dict": {key: value.detach().cpu() for key, value in router.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _selection_score(metrics: dict[str, float]) -> float:
    """Exactly the baseline protocol: 0.5*top1 + 0.3*need_f1 + 0.2*mrr."""

    return 0.5 * metrics["route_top1"] + 0.3 * metrics["need_f1"] + 0.2 * metrics["route_mrr"]


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = _device(args.device)
    train_path = _resolve_path(args.train_file)
    eval_path = _resolve_path(args.eval_file)
    train = _read_episodes(train_path, max_rows=args.max_train_episodes)
    evaluation = _read_episodes(eval_path, max_rows=args.max_eval_episodes)
    eval_sha256 = _sha256(eval_path)
    expected_hash = str(args.expected_eval_sha256 or "").strip().lower()
    adjacent_hash = eval_path.with_name("eval.sha256")
    if not expected_hash and adjacent_hash.exists():
        expected_hash = adjacent_hash.read_text(encoding="ascii").strip().split()[0].lower()
    if expected_hash and expected_hash != eval_sha256:
        raise RuntimeError(f"frozen eval hash mismatch: expected {expected_hash}, got {eval_sha256}")

    vectors, lookup, feature_meta = _prepare_feature_cache(args, train, evaluation, train_path, eval_path)
    hidden_size = int(vectors.shape[-1])
    train_data = _episode_tensors(train, lookup)
    eval_data = _episode_tensors(evaluation, lookup)

    arch_kwargs = _arch_kwargs(args)
    if hidden_size != int(args.hidden_size):
        raise ValueError(
            f"feature bank hidden size {hidden_size} does not match --hidden-size {args.hidden_size}"
        )
    router = MemoryRouterXL(hidden_size, **arch_kwargs).to(device)
    parameter_count = router.parameter_count()
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup = max(0, int(args.warmup_steps))
    total_steps = max(1, int(args.steps))

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-6, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    best_score = -float("inf")
    if args.resume:
        resume_path = _resolve_path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        start_step, best_score = _load_checkpoint(resume_path, router, optimizer, scheduler)
        if start_step >= total_steps:
            raise ValueError(f"resume checkpoint is already at step {start_step}; --steps must be greater")

    by_family: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(train_data["families"]):
        by_family[str(family)].append(index)
    rng = random.Random(args.seed + 17)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "router_arch.json").write_text(
        json.dumps(router.arch_config(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    if args.overwrite_metrics and metrics_path.exists():
        metrics_path.unlink()

    startup = {
        "event": "startup",
        "label": args.label,
        "arch": ARCH_NAME,
        "arch_version": ARCH_VERSION,
        "arch_config": router.arch_config(),
        "parameters": parameter_count,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_episodes": len(train),
        "eval_episodes": len(evaluation),
        "eval_sha256": eval_sha256,
        "feature_bank": feature_meta,
        "steps": total_steps,
        "batch_size": args.batch_size,
        "protocol": "identical to train_memory_router_large (selection=0.5*top1+0.3*need_f1+0.2*mrr)",
    }
    print(json.dumps(startup, ensure_ascii=False), flush=True)

    history: list[dict[str, Any]] = []
    best_step = start_step if args.resume else 0
    eval_events = 0
    router.train()
    with metrics_path.open("a", encoding="utf-8") as metrics_handle:
        metrics_handle.write(json.dumps(startup, ensure_ascii=False) + "\n")
        metrics_handle.flush()
        initial_eval = _evaluate(
            router, eval_data, vectors, device=device, batch_size=args.eval_batch_size,
            threshold=args.need_threshold, max_batches=args.eval_max_batches,
        )
        initial_eval.update({"event": "eval_resume" if start_step else "eval", "step": start_step})
        initial_eval["selection_score"] = _selection_score(initial_eval)
        metrics_handle.write(json.dumps(initial_eval, ensure_ascii=False) + "\n")
        metrics_handle.flush()
        print(json.dumps(initial_eval, ensure_ascii=False), flush=True)
        if not start_step:
            history.append(initial_eval)
            best_score = initial_eval["selection_score"]
            best_step = 0

        for step in range(start_step + 1, total_steps + 1):
            step_started = time.perf_counter()
            indices = _sample_indices(
                train_data, batch_size=args.batch_size, rng=rng,
                mode=args.sampling_mode, by_family=by_family,
            )
            batch = _batch_from_indices(train_data, vectors, indices, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.amp and device.type == "cuda")):
                loss, parts = _router_loss(router, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(router.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            loss_row = {
                "event": "train",
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                "step_seconds": time.perf_counter() - step_started,
                **parts,
            }
            metrics_handle.write(json.dumps(loss_row, ensure_ascii=False) + "\n")
            metrics_handle.flush()
            if step % args.log_every == 0:
                print(json.dumps(loss_row, ensure_ascii=False), flush=True)

            if step % args.eval_interval == 0 or step == total_steps:
                evaluation_metrics = _evaluate(
                    router, eval_data, vectors, device=device, batch_size=args.eval_batch_size,
                    threshold=args.need_threshold, max_batches=args.eval_max_batches,
                )
                score = _selection_score(evaluation_metrics)
                event = {"event": "eval", "step": step, "selection_score": score, **evaluation_metrics}
                metrics_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                metrics_handle.flush()
                print(json.dumps(event, ensure_ascii=False), flush=True)
                eval_events += 1
                history.append(event)
                if step % max(1, args.checkpoint_interval) == 0 or step == total_steps:
                    _save_checkpoint(output_dir / f"router_step_{step:08d}.pt", router, optimizer, scheduler, step, best_score)
                if score > best_score:
                    best_score = score
                    best_step = step
                    _save_checkpoint(output_dir / "router_best.pt", router, optimizer, scheduler, step, best_score)
                router.train()

    torch.save(
        {key: value.detach().cpu() for key, value in router.state_dict().items()},
        output_dir / "memory_router_xl.pt",
    )
    _save_checkpoint(output_dir / "router_final.pt", router, optimizer, scheduler, total_steps, best_score)
    final_eval = _evaluate(
        router, eval_data, vectors, device=device, batch_size=args.eval_batch_size,
        threshold=args.need_threshold, max_batches=args.eval_max_batches,
    )
    best_metrics = max(history, key=lambda item: item["selection_score"]) if history else final_eval
    summary = {
        "format_version": 3,
        "label": args.label,
        "arch": ARCH_NAME,
        "arch_version": ARCH_VERSION,
        "arch_config": router.arch_config(),
        "parameters": parameter_count,
        "model_path": str(_resolve_path(args.model_path)),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_sha256": _sha256(train_path),
        "eval_sha256": eval_sha256,
        "eval_frozen": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hidden_size": hidden_size,
        "train_episodes": len(train),
        "eval_episodes": len(evaluation),
        "feature_bank": feature_meta,
        "steps": total_steps,
        "start_step": start_step,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eval_interval": args.eval_interval,
        "eval_events": eval_events,
        "best_step": best_step,
        "best_selection_score": best_score,
        "best_eval": best_metrics,
        "final_eval": final_eval,
        "training_history": history,
        "safety": {
            "gpu_memory_cap_gb": args.gpu_memory_gb,
            "feature_dtype": "float16_cpu",
            "qwen_backbone_updated": False,
            "router_only_updated": True,
        },
    }
    (output_dir / "router_xl_training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="router_xl")
    parser.add_argument("--train-file", default="data/router_training_v3/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v3/eval.jsonl")
    parser.add_argument("--expected-eval-sha256", default="")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--output-dir", default="checkpoints/router_xl")
    parser.add_argument("--feature-cache-dir", default="checkpoints/router_shared/feature_cache")
    parser.add_argument("--resume", default="", help="checkpoint from router_step_*.pt/router_best.pt; --steps is the final global step")
    parser.add_argument("--precompute-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-memory-gb", type=float, default=9.0)
    parser.add_argument("--max-key-tokens", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--precompute-log-every", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2560)

    # ---- MemoryRouterXL capacity knobs -------------------------------------
    parser.add_argument("--router-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-hidden", type=int, default=0, help="0 -> router_dim")
    parser.add_argument("--pair-blocks", type=int, default=1)
    parser.add_argument("--pair-hidden", type=int, default=0, help="0 -> router_dim")
    parser.add_argument("--pair-expansion", type=int, default=2)
    parser.add_argument("--pair-dropout", type=float, default=0.05)
    parser.add_argument("--use-interaction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-layers", type=int, default=2)
    parser.add_argument("--policy-hidden", type=int, default=512)
    parser.add_argument("--policy-dropout", type=float, default=0.0)
    parser.add_argument("--cosine-scale", action=argparse.BooleanOptionalAction, default=True)

    # ---- optimisation (same defaults as the 512-dim baseline) --------------
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-eval-episodes", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--need-threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--need-loss-weight", type=float, default=0.75)
    parser.add_argument("--hop-loss-weight", type=float, default=0.35)
    parser.add_argument("--margin-loss-weight", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sampling-mode", choices=("uniform", "source_balanced"), default="source_balanced")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    if args.checkpoint_interval < 1:
        raise SystemExit("--checkpoint-interval must be positive")
    if args.eval_interval < 1:
        raise SystemExit("--eval-interval must be positive")
    if args.steps < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise SystemExit("steps and batch sizes must be positive")
    if not 0.0 <= args.need_threshold <= 1.0:
        raise SystemExit("--need-threshold must be in [0, 1]")
    if args.router_dim % args.num_heads != 0:
        raise SystemExit("--router-dim must be divisible by --num-heads")
    return args


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps({
        "label": result["label"],
        "parameters": result["parameters"],
        "best_step": result["best_step"],
        "best_selection_score": result["best_selection_score"],
        "best_eval": result["best_eval"],
        "final_eval": result["final_eval"],
    }, ensure_ascii=False, indent=2))
