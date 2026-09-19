"""Train a 512-dim router on the full v5 dataset with streaming data loading.

Two problems made the original trainer unusable for the 87k-episode v5 dataset:

* ``_read_episodes`` parses every episode into Python objects (~1.15 GB of JSONL
  becomes several GB of RAM);
* ``_prepare_feature_cache`` loads the whole feature bank as a resident tensor,
  and the v5 bank is 2.12M x 2560 fp16 = 10.86 GB.

This trainer therefore:

* **streams** ``train.jsonl`` / ``eval.jsonl`` line by line, filling preallocated
  arrays (two passes: one to count lines, one to build indices);
* **memory-maps** the ``features.f16.npy`` bank so only the rows a batch touches
  are read (the OS keeps hot pages cached);
* verifies the bank manifest against the frozen dataset hashes *and* the model
  path, so a stale bank can never be trained against silently;
* imports the loss, evaluation, sampling and selection-score code from
  ``train_memory_router_large`` so metrics stay comparable with the 512 baseline.

Usage::

    python -m V2_dpskw.train_router_v5 ^
        --arch xl --label xl512_v5 ^
        --train-file data/router_training_v5/train.jsonl ^
        --eval-file data/router_training_v5/eval.jsonl ^
        --feature-cache H:\\Memory\\nm_cache\\nm_router_v5\\feature_cache ^
        --output-dir checkpoints/router_v5_xl512 ^
        --router-dim 512 --num-heads 8 ^
        --steps 100000 --batch-size 64 --eval-interval 500
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import MemoryRouterV2
from V2_dpskw.router_xl import MemoryRouterXL
from V2_dpskw.train_memory_router_large import (
    _batch_from_indices,
    _evaluate,
    _router_loss,
    _sha256,
    _set_cuda_cap,
)

BANK_NAME = "features.f16.npy"
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent / path


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            count += block.count(b"\n")
    return count


def load_feature_bank(cache_dir: Path, *, train_path: Path, eval_path: Path, model_path: str, max_key_tokens: int) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """Open the memory-mapped bank after validating it against the frozen data."""

    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"feature bank manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "train_sha256": _sha256(train_path),
        "eval_sha256": _sha256(eval_path),
        "max_key_tokens": int(max_key_tokens),
        "model_path": str(model_path),
    }
    mismatched = {key: (manifest.get(key), value) for key, value in checks.items() if manifest.get(key) != value}
    if mismatched:
        raise RuntimeError(f"feature bank does not match the frozen inputs: {mismatched}")
    # An interrupted encode once still wrote a manifest that looked complete, and a
    # training run was started on the ~14% of rows that were still zero.  Refuse an
    # explicitly incomplete bank before touching the data.
    if manifest.get("complete") is False:
        raise RuntimeError(
            "feature bank is marked incomplete "
            f"(encoded_rows={manifest.get('encoded_rows')}, text_count={manifest.get('text_count')}); "
            "finish or gap-fill the encode first"
        )
    started = time.perf_counter()
    lookup = json.loads((cache_dir / INDEX_NAME).read_text(encoding="utf-8"))
    bank = np.load(cache_dir / BANK_NAME, mmap_mode="r")
    if int(bank.shape[0]) != len(lookup):
        raise RuntimeError(f"bank rows {bank.shape[0]} != index entries {len(lookup)}")
    print(json.dumps({
        "event": "feature_bank",
        "path": str(cache_dir / BANK_NAME),
        "shape": list(bank.shape),
        "dtype": str(bank.dtype),
        "unique_texts": len(lookup),
        "manifest_complete": manifest.get("complete", "legacy(no flag)"),
        "load_seconds": round(time.perf_counter() - started, 1),
        "resident_mode": "mmap",
    }, ensure_ascii=False), flush=True)
    return bank, lookup, manifest


def assert_bank_rows_present(bank: np.ndarray, data: dict[str, Any], *, label: str, sample: int = 400) -> None:
    """Fail loudly if any sampled referenced row is still all-zero.

    Zero rows mean "this text was never encoded"; they would silently train the
    router on a constant feature and quietly depress every metric.
    """

    query_rows = data["query_indices"].numpy()
    candidate_rows = data["candidate_indices"].numpy()[data["candidate_mask"].numpy()]
    total = len(query_rows) + len(candidate_rows)
    if total == 0:
        return
    step = max(1, total // max(1, sample))
    picks = np.concatenate([query_rows, candidate_rows])[::step][:sample]
    empty = [int(row) for row in picks if not np.any(np.asarray(bank[int(row)]))]
    if empty:
        raise RuntimeError(
            f"{label}: {len(empty)}/{len(picks)} sampled rows of the feature bank are all-zero "
            f"(e.g. rows {empty[:5]}); the encode is incomplete"
        )
    print(json.dumps({"event": "bank_rows_ok", "slice": label, "sampled": int(len(picks))}, ensure_ascii=False), flush=True)


def stream_episode_tensors(
    path: Path,
    lookup: dict[str, int],
    *,
    max_candidates: int = 32,
    max_episodes: int = 0,
    progress_every: int = 50000,
) -> dict[str, Any]:
    """Build the index tensors without ever holding the parsed episodes.

    Memory is bounded by the arrays themselves (a few tens of MB) plus a
    per-line JSON object that is released immediately.
    """

    import hashlib

    def text_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()

    total = _count_lines(path)
    if max_episodes:
        total = min(total, max_episodes)
    query_indices = np.zeros(total, dtype=np.int64)
    candidate_indices = np.zeros((total, max_candidates), dtype=np.int64)
    candidate_mask = np.zeros((total, max_candidates), dtype=bool)
    positive_mask = np.zeros((total, max_candidates), dtype=bool)
    need = np.zeros(total, dtype=np.float32)
    hops = np.zeros(total, dtype=np.int64)
    families: list[str] = []
    categories: list[str] = []
    seen = 0
    missing = 0
    started = time.perf_counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if seen >= total:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_text = str(row.get("query", "")).strip()
            query_row = lookup.get(text_key(query_text))
            if query_row is None:
                missing += 1
                continue
            query_indices[seen] = query_row
            candidates = row.get("candidates") or []
            positives = {int(index) for index in (row.get("positive_indices") or [])}
            width = min(len(candidates), max_candidates)
            for position in range(width):
                candidate = candidates[position]
                text = str(candidate.get("text", "")).strip() if isinstance(candidate, dict) else ""
                text_row = lookup.get(text_key(text)) if text else None
                if text_row is None:
                    missing += 1
                    continue
                candidate_indices[seen, position] = text_row
                candidate_mask[seen, position] = True
                positive_mask[seen, position] = position in positives
            if width < max_candidates:
                # Pad with the first real candidate so indexing never reads garbage.
                for position in range(width, max_candidates):
                    candidate_indices[seen, position] = candidate_indices[seen, 0] if width else 0
            need[seen] = float(row.get("need_memory", bool(positives)))
            hops[seen] = int(row.get("hop", min(3, len(positives)) if positives else 0))
            families.append(str(row.get("family", "unknown")))
            metadata = row.get("metadata")
            categories.append(str((metadata or {}).get("category", "")) if isinstance(metadata, dict) else "")
            seen += 1
            if seen % progress_every == 0:
                print(json.dumps({"event": "stream", "file": path.name, "episodes": seen,
                                  "total": total, "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if seen == 0:
        raise ValueError(f"no usable episodes streamed from {path}")
    if missing:
        print(json.dumps({"event": "stream_warning", "file": path.name,
                          "candidates_missing_from_bank": missing}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "stream_done", "file": path.name, "episodes": seen,
                      "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    data: dict[str, Any] = {
        "query_indices": torch.from_numpy(query_indices[:seen]),
        "candidate_indices": torch.from_numpy(candidate_indices[:seen]),
        "candidate_mask": torch.from_numpy(candidate_mask[:seen]),
        "positive_mask": torch.from_numpy(positive_mask[:seen]),
        "need": torch.from_numpy(need[:seen]),
        "hops": torch.from_numpy(hops[:seen]),
        "families": families[:seen],
        "categories": categories[:seen],
    }
    return data


def sample_indices(
    data: dict[str, Any],
    *,
    batch_size: int,
    rng: random.Random,
    mode: str,
    by_family: dict[str, list[int]],
) -> torch.Tensor:
    """Uniform, per-family balanced, or sqrt-weighted family sampling."""

    n = int(data["query_indices"].shape[0])
    if mode == "uniform" or not by_family:
        return torch.tensor([rng.randrange(n) for _ in range(min(batch_size, n))], dtype=torch.long)
    if mode == "source_balanced":
        families = list(by_family)
        values = [rng.choice(by_family[rng.choice(families)]) for _ in range(min(batch_size, n))]
        return torch.tensor(values, dtype=torch.long)
    # family_sqrt: weight families by sqrt(size) so small families stay visible
    # without dominating (which is what equal-per-family sampling does).
    families = list(by_family)
    sizes = np.array([len(by_family[name]) for name in families], dtype=np.float64)
    weights = np.sqrt(sizes)
    cumulative = np.cumsum(weights / weights.sum())
    values = []
    for _ in range(min(batch_size, n)):
        pick = bisect.bisect_left(cumulative.tolist(), rng.random())
        family = families[min(pick, len(families) - 1)]
        values.append(rng.choice(by_family[family]))
    return torch.tensor(values, dtype=torch.long)


def build_router(args: argparse.Namespace, hidden_size: int) -> torch.nn.Module:
    if args.arch == "v2":
        return MemoryRouterV2(hidden_size, router_dim=args.router_dim, num_heads=args.num_heads, max_hops=args.max_hops)
    return MemoryRouterXL(
        hidden_size,
        router_dim=args.router_dim,
        num_heads=args.num_heads,
        max_hops=args.max_hops,
        encoder_layers=args.encoder_layers,
        encoder_hidden=args.encoder_hidden,
        pair_blocks=args.pair_blocks,
        pair_hidden=args.pair_hidden,
        pair_expansion=args.pair_expansion,
        pair_dropout=args.pair_dropout,
        use_interaction=args.use_interaction,
        policy_layers=args.policy_layers,
        policy_hidden=args.policy_hidden,
        policy_dropout=args.policy_dropout,
        learnable_cosine_scale=args.cosine_scale,
    )


def arch_config_of(router: torch.nn.Module) -> dict[str, Any]:
    if isinstance(router, MemoryRouterXL):
        return router.arch_config()
    return {
        "arch": "router_v2",
        "hidden_size": router.hidden_size,
        "router_dim": router.router_dim,
        "num_heads": router.num_heads,
        "max_hops": router.max_hops,
    }


def save_checkpoint(path: Path, router: torch.nn.Module, optimizer: Any, scheduler: Any, step: int, best_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 3,
        "arch_config": arch_config_of(router),
        "step": step,
        "best_score": best_score,
        "router_state_dict": {key: value.detach().cpu() for key, value in router.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def selection_score(metrics: dict[str, float]) -> float:
    return 0.5 * metrics["route_top1"] + 0.3 * metrics["need_f1"] + 0.2 * metrics["route_mrr"]


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    if args.gpu_memory_gb > 0:
        _set_cuda_cap(args.gpu_memory_gb)
    device = _device(args.device)
    train_path = _resolve(args.train_file)
    eval_path = _resolve(args.eval_file)
    cache_dir = _resolve(args.feature_cache)

    bank, lookup, manifest = load_feature_bank(
        cache_dir, train_path=train_path, eval_path=eval_path,
        model_path=args.model_path, max_key_tokens=args.max_key_tokens,
    )
    hidden_size = int(bank.shape[1])
    vectors = torch.from_numpy(bank)
    if vectors.dtype != torch.float16:
        vectors = vectors.to(torch.float16)

    train_data = stream_episode_tensors(train_path, lookup, max_candidates=args.candidate_count,
                                       max_episodes=args.max_train_episodes)
    eval_data = stream_episode_tensors(eval_path, lookup, max_candidates=args.candidate_count,
                                      max_episodes=args.max_eval_episodes)
    assert_bank_rows_present(bank, train_data, label="train")
    assert_bank_rows_present(bank, eval_data, label="eval")

    router = build_router(args, hidden_size).to(device)
    parameters = sum(parameter.numel() for parameter in router.parameters())
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, int(args.steps))
    warmup = max(0, int(args.warmup_steps))

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-6, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    best_score = -float("inf")
    if args.resume:
        resume_path = _resolve(args.resume)
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        router.load_state_dict(payload["router_state_dict"], strict=True)
        if isinstance(payload.get("optimizer_state_dict"), dict):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if isinstance(payload.get("scheduler_state_dict"), dict):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_step = int(payload.get("step", 0))
        best_score = float(payload.get("best_score", -float("inf")))

    by_family: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(train_data["families"]):
        by_family[str(family)].append(index)
    by_family = {name: values for name, values in by_family.items() if values}

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "router_arch.json").write_text(json.dumps(arch_config_of(router), ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    if args.overwrite_metrics and metrics_path.exists():
        metrics_path.unlink()
    startup = {
        "event": "startup",
        "label": args.label,
        "arch_config": arch_config_of(router),
        "parameters": parameters,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_episodes": len(train_data["families"]),
        "eval_episodes": len(eval_data["families"]),
        "family_counts": {name: len(values) for name, values in by_family.items()},
        "sampling_mode": args.sampling_mode,
        "steps": total_steps,
        "batch_size": args.batch_size,
        "feature_bank": {"path": str(cache_dir / BANK_NAME), "texts": manifest.get("text_count"),
                         "dtype": manifest.get("dtype"), "resident_mode": "mmap"},
        "protocol": "loss/eval/selection imported from train_memory_router_large",
    }
    rng = random.Random(args.seed + 17)
    print(json.dumps(startup, ensure_ascii=False), flush=True)
    history: list[dict[str, Any]] = []
    best_step = 0
    router.train()
    with metrics_path.open("a", encoding="utf-8") as metrics_handle:
        metrics_handle.write(json.dumps(startup, ensure_ascii=False) + "\n")
        metrics_handle.flush()
        initial = _evaluate(router, eval_data, vectors, device=device, batch_size=args.eval_batch_size, threshold=args.need_threshold)
        initial.update({"event": "eval_resume" if start_step else "eval", "step": start_step})
        initial["selection_score"] = selection_score(initial)
        metrics_handle.write(json.dumps(initial, ensure_ascii=False) + "\n")
        metrics_handle.flush()
        print(json.dumps(initial, ensure_ascii=False), flush=True)
        history.append(initial)
        if not start_step and initial["selection_score"] > best_score:
            best_score, best_step = initial["selection_score"], 0

        for step in range(start_step + 1, total_steps + 1):
            started = time.perf_counter()
            indices = sample_indices(train_data, batch_size=args.batch_size, rng=rng,
                                     mode=args.sampling_mode, by_family=by_family)
            batch = _batch_from_indices(train_data, vectors, indices, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.amp and device.type == "cuda")):
                loss, parts = _router_loss(router, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(router.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            row = {
                "event": "train", "step": step, "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                "step_seconds": time.perf_counter() - started, **parts,
            }
            metrics_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_handle.flush()
            if step % args.log_every == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if step % args.eval_interval == 0 or step == total_steps:
                metrics = _evaluate(router, eval_data, vectors, device=device,
                                    batch_size=args.eval_batch_size, threshold=args.need_threshold)
                score = selection_score(metrics)
                event = {"event": "eval", "step": step, "selection_score": score, **metrics}
                metrics_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                metrics_handle.flush()
                print(json.dumps(event, ensure_ascii=False), flush=True)
                history.append(event)
                if step % max(1, args.checkpoint_interval) == 0 or step == total_steps:
                    save_checkpoint(output_dir / f"router_step_{step:08d}.pt", router, optimizer, scheduler, step, best_score)
                if score > best_score:
                    best_score, best_step = score, step
                    save_checkpoint(output_dir / "router_best.pt", router, optimizer, scheduler, step, best_score)
                router.train()

    final_name = "memory_router_xl.pt" if args.arch == "xl" else "memory_router_v2.pt"
    torch.save({key: value.detach().cpu() for key, value in router.state_dict().items()}, output_dir / final_name)
    final_eval = _evaluate(router, eval_data, vectors, device=device, batch_size=args.eval_batch_size, threshold=args.need_threshold)
    best_eval = max(history, key=lambda item: item["selection_score"])
    summary = {
        "format_version": 3,
        "label": args.label,
        "arch_config": arch_config_of(router),
        "parameters": parameters,
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_sha256": _sha256(train_path),
        "eval_sha256": _sha256(eval_path),
        "feature_bank": {"path": str(cache_dir / BANK_NAME), "texts": manifest.get("text_count")},
        "train_episodes": len(train_data["families"]),
        "eval_episodes": len(eval_data["families"]),
        "sampling_mode": args.sampling_mode,
        "steps": total_steps,
        "start_step": start_step,
        "best_step": best_step,
        "best_selection_score": best_score,
        "best_eval": best_eval,
        "final_eval": final_eval,
        "training_history": history,
        "safety": {"qwen_backbone_updated": False, "router_only_updated": True, "resident_mode": "mmap"},
    }
    (output_dir / "router_v5_training.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="router_v5")
    parser.add_argument("--arch", choices=("v2", "xl"), default="xl")
    parser.add_argument("--train-file", default="data/router_training_v5/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v5/eval.jsonl")
    parser.add_argument("--feature-cache", default=r"H:\Memory\nm_cache\nm_router_v5\feature_cache")
    parser.add_argument("--output-dir", default="checkpoints/router_v5")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--max-key-tokens", type=int, default=256)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-memory-gb", type=float, default=9.0)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-eval-episodes", type=int, default=0)

    parser.add_argument("--router-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-hidden", type=int, default=512)
    parser.add_argument("--pair-blocks", type=int, default=1)
    parser.add_argument("--pair-hidden", type=int, default=512)
    parser.add_argument("--pair-expansion", type=int, default=2)
    parser.add_argument("--pair-dropout", type=float, default=0.05)
    parser.add_argument("--use-interaction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-layers", type=int, default=2)
    parser.add_argument("--policy-hidden", type=int, default=512)
    parser.add_argument("--policy-dropout", type=float, default=0.0)
    parser.add_argument("--cosine-scale", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--need-threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--need-loss-weight", type=float, default=0.75)
    parser.add_argument("--hop-loss-weight", type=float, default=0.35)
    parser.add_argument("--margin-loss-weight", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sampling-mode", choices=("uniform", "source_balanced", "family_sqrt"), default="family_sqrt")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.router_dim % args.num_heads != 0:
        raise SystemExit("--router-dim must be divisible by --num-heads")
    return args


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps({
        "label": result["label"],
        "parameters": result["parameters"],
        "best_step": result["best_step"],
        "best_eval": result["best_eval"],
        "final_eval": result["final_eval"],
    }, ensure_ascii=False, indent=2))
