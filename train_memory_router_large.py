"""Train MemoryRouterV2 on frozen, mixed-domain routing episodes.

The command has two phases:

1. encode the unique query/candidate texts once with the frozen Qwen
   representation and store a CPU feature bank;
2. train only ``MemoryRouterV2`` from that bank.

This keeps the 5070/12GB path safe: Qwen is loaded only during feature
generation with a conservative GPU cap, while the actual router update uses a
small batch of hidden states.  Loss is printed once per optimizer step.  The
pre-generated eval file is hashed at startup and evaluated exactly every
``--eval-interval`` steps (500 by default).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import MemoryRouterV2
from V2_dpskw.qwen_integration import load_qwen_dynamic, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    # Accept both forms when launched from H:\\Memory or from the package
    # directory itself: ``data/...`` and ``V2_dpskw/data/...``.
    if path.parts and path.parts[0].lower() == PROJECT_ROOT.name.lower():
        path = Path(*path.parts[1:])
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="replace")).hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _max_memory(gpu_memory_gb: float) -> dict[Any, str] | None:
    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return None
    return {0: f"{gpu_memory_gb:.1f}GiB", "cpu": "64GiB"}


def _set_cuda_cap(gpu_memory_gb: float) -> None:
    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(0.90, max(0.05, gpu_memory_gb * 1024**3 / total))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def _read_episodes(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if max_rows and len(rows) >= max_rows:
                break
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            query = str(row.get("query", "")).strip()
            candidates = row.get("candidates")
            if not query or not isinstance(candidates, list) or not candidates:
                continue
            positive = row.get("positive_indices")
            if positive is None:
                index = int(row.get("positive_index", -1))
                positive = [] if index < 0 else [index]
            positive = [int(index) for index in positive if 0 <= int(index) < len(candidates)]
            row["positive_indices"] = sorted(set(positive))
            row["need_memory"] = float(row.get("need_memory", bool(positive)))
            row["hop"] = int(row.get("hop", min(3, len(positive)) if positive else 0))
            rows.append(row)
    if not rows:
        raise ValueError(f"no valid routing episodes found in {path}")
    return rows


def _collect_texts(episodes: Iterable[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    texts: list[str] = []
    lookup: dict[str, int] = {}
    for row in episodes:
        values = [str(row["query"])]
        values.extend(str(item.get("text", "")) for item in row["candidates"] if isinstance(item, dict))
        for text in values:
            text = text.strip()
            if not text:
                continue
            key = _text_key(text)
            if key not in lookup:
                lookup[key] = len(texts)
                texts.append(text)
    return texts, lookup


def _token_ids(tokenizer: Any, text: str, max_tokens: int) -> torch.Tensor:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    ids = encoded["input_ids"][0].long()
    if ids.numel() == 0:
        fallback = tokenizer.eos_token_id
        ids = torch.tensor([int(fallback or 0)], dtype=torch.long)
    return ids


@torch.inference_mode()
def _encode_feature_bank(
    *,
    texts: list[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    max_tokens: int,
    output_path: Path,
    progress_every: int,
) -> tuple[int, int]:
    rows = [_token_ids(tokenizer, text, max_tokens) for text in texts]
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row.numel())].append(index)
    hidden_size = int(model.memory.hidden_size)
    vectors = torch.empty((len(texts), hidden_size), dtype=torch.float16, device="cpu")
    completed = 0
    for length in sorted(groups):
        indices = groups[length]
        for start in range(0, len(indices), max(1, batch_size)):
            selected = indices[start : start + max(1, batch_size)]
            input_ids = torch.stack([rows[index] for index in selected])
            attention_mask = torch.ones_like(input_ids)
            encoded = model._encode_model_key(input_ids, attention_mask).detach().float().cpu()
            vectors[selected] = encoded.to(dtype=torch.float16)
            completed += len(selected)
            if completed == len(selected) or completed % max(1, progress_every) == 0 or completed == len(texts):
                print(json.dumps({"phase": "feature_encode", "completed": completed, "total": len(texts)}, ensure_ascii=False), flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(vectors, temp_path)
    temp_path.replace(output_path)
    return len(texts), hidden_size


def _cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _cache_is_compatible(cache_dir: Path, expected: dict[str, Any]) -> bool:
    features = cache_dir / "features.pt"
    index = cache_dir / "index.json"
    manifest = _cache_meta_path(cache_dir)
    if not features.exists() or not index.exists() or not manifest.exists():
        return False
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(saved.get(key) == value for key, value in expected.items())


def _prepare_feature_cache(args: argparse.Namespace, train: list[dict[str, Any]], evaluation: list[dict[str, Any]], train_path: Path, eval_path: Path) -> tuple[torch.Tensor, dict[str, int], dict[str, Any]]:
    all_episodes = train + evaluation
    texts, lookup = _collect_texts(all_episodes)
    cache_dir = _resolve_path(args.feature_cache_dir)
    model_path = _resolve_path(args.model_path)
    expected = {
        "format_version": 1,
        "train_sha256": _sha256(train_path),
        "eval_sha256": _sha256(eval_path),
        "model_path": str(model_path),
        "max_key_tokens": int(args.max_key_tokens),
        "hidden_size": int(args.hidden_size),
        "text_count": len(texts),
        "dtype": "float16_cpu",
    }
    if args.rebuild_features or not _cache_is_compatible(cache_dir, expected):
        if not args.precompute_features:
            raise RuntimeError(
                f"feature cache is absent or stale: {cache_dir}. Run again with --precompute-features."
            )
        _set_cuda_cap(args.gpu_memory_gb)
        tokenizer = load_tokenizer(model_path)
        model = load_qwen_dynamic(
            model_path,
            load_in_4bit=not args.no_4bit,
            max_memory=_max_memory(args.gpu_memory_gb),
        )
        model.eval()
        actual_hidden = int(model.memory.hidden_size)
        if actual_hidden != int(args.hidden_size):
            expected["hidden_size"] = actual_hidden
        _encode_feature_bank(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            batch_size=args.encode_batch_size,
            max_tokens=args.max_key_tokens,
            output_path=cache_dir / "features.pt",
            progress_every=args.precompute_log_every,
        )
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    vectors = torch.load(cache_dir / "features.pt", map_location="cpu", weights_only=True)
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
        raise ValueError(f"invalid feature bank at {cache_dir / 'features.pt'}")
    if vectors.shape[0] != len(texts):
        raise ValueError("feature bank text count does not match frozen dataset")
    (cache_dir / "index.json").write_text(json.dumps(lookup, ensure_ascii=False), encoding="utf-8")
    expected["hidden_size"] = int(vectors.shape[-1])
    expected["text_count"] = int(vectors.shape[0])
    _cache_meta_path(cache_dir).write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return vectors, lookup, expected


def _episode_tensors(episodes: list[dict[str, Any]], lookup: dict[str, int]) -> dict[str, torch.Tensor | list[str]]:
    max_candidates = max(len(row["candidates"]) for row in episodes)
    query_indices: list[int] = []
    candidate_indices: list[list[int]] = []
    candidate_mask: list[list[bool]] = []
    positive_mask: list[list[bool]] = []
    need: list[float] = []
    hops: list[int] = []
    families: list[str] = []
    for row in episodes:
        query = str(row["query"])
        query_indices.append(lookup[_text_key(query)])
        row_candidates: list[int] = []
        row_positive = set(int(index) for index in row.get("positive_indices", []))
        row_mask: list[bool] = []
        row_pos: list[bool] = []
        for index, candidate in enumerate(row["candidates"]):
            text = str(candidate.get("text", ""))
            row_candidates.append(lookup[_text_key(text)])
            row_mask.append(True)
            row_pos.append(index in row_positive)
        while len(row_candidates) < max_candidates:
            row_candidates.append(row_candidates[0])
            row_mask.append(False)
            row_pos.append(False)
        candidate_indices.append(row_candidates)
        candidate_mask.append(row_mask)
        positive_mask.append(row_pos)
        need.append(float(row.get("need_memory", bool(row_positive))))
        hops.append(int(row.get("hop", min(3, len(row_positive)) if row_positive else 0)))
        families.append(str(row.get("family", "unknown")))
    return {
        "query_indices": torch.tensor(query_indices, dtype=torch.long),
        "candidate_indices": torch.tensor(candidate_indices, dtype=torch.long),
        "candidate_mask": torch.tensor(candidate_mask, dtype=torch.bool),
        "positive_mask": torch.tensor(positive_mask, dtype=torch.bool),
        "need": torch.tensor(need, dtype=torch.float32),
        "hops": torch.tensor(hops, dtype=torch.long),
        "families": families,
    }


def _batch_from_indices(data: dict[str, Any], vectors: torch.Tensor, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    q = vectors[data["query_indices"][indices]].to(device=device, dtype=torch.float32, non_blocking=True)
    c = vectors[data["candidate_indices"][indices]].to(device=device, dtype=torch.float32, non_blocking=True)
    return {
        "query": q,
        "candidates": c,
        "candidate_mask": data["candidate_mask"][indices].to(device),
        "positive_mask": data["positive_mask"][indices].to(device),
        "need": data["need"][indices].to(device),
        "hops": data["hops"][indices].to(device),
    }


def _masked_scores(scores: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)


def _router_loss(router: MemoryRouterV2, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    output = router(batch["query"], batch["candidates"])
    scores = _masked_scores(output["scores"], batch["candidate_mask"])
    positive_mask = batch["positive_mask"] & batch["candidate_mask"]
    valid = positive_mask.any(dim=-1)
    if bool(valid.any()):
        all_logsumexp = torch.logsumexp(scores[valid], dim=-1)
        positive_scores = scores[valid].masked_fill(~positive_mask[valid], torch.finfo(scores.dtype).min)
        positive_logsumexp = torch.logsumexp(positive_scores, dim=-1)
        route_loss = (all_logsumexp - positive_logsumexp).mean()
        best_positive = positive_scores.max(dim=-1).values
        negative_scores = scores[valid].masked_fill(positive_mask[valid], torch.finfo(scores.dtype).min)
        best_negative = negative_scores.max(dim=-1).values
        margin_loss = F.relu(args.margin - best_positive + best_negative).mean()
    else:
        route_loss = scores.sum() * 0.0
        margin_loss = scores.sum() * 0.0
    need_loss = F.binary_cross_entropy_with_logits(output["need_memory_logits"], batch["need"])
    hops = batch["hops"].clamp(min=0, max=router.max_hops)
    hop_loss = F.cross_entropy(output["hop_logits"], hops)
    total = route_loss + args.need_loss_weight * need_loss + args.hop_loss_weight * hop_loss + args.margin_loss_weight * margin_loss
    return total, {
        "route_loss": float(route_loss.detach().cpu()),
        "need_loss": float(need_loss.detach().cpu()),
        "hop_loss": float(hop_loss.detach().cpu()),
        "margin_loss": float(margin_loss.detach().cpu()),
    }


@torch.inference_mode()
def _evaluate(router: MemoryRouterV2, data: dict[str, Any], vectors: torch.Tensor, *, device: torch.device, batch_size: int, threshold: float, max_batches: int = 0) -> dict[str, float]:
    router.eval()
    n = int(data["query_indices"].shape[0])
    route_total = route_top1 = route_top3 = 0
    reciprocal_sum = 0.0
    need_tp = need_tn = need_fp = need_fn = 0
    hop_correct = 0
    hop_total = 0
    batches_seen = 0
    for start in range(0, n, max(1, batch_size)):
        if max_batches and batches_seen >= max_batches:
            break
        indices = torch.arange(start, min(n, start + max(1, batch_size)), dtype=torch.long)
        batch = _batch_from_indices(data, vectors, indices, device)
        output = router(batch["query"], batch["candidates"])
        scores = _masked_scores(output["scores"], batch["candidate_mask"])
        prediction = scores.argmax(dim=-1)
        pos = batch["positive_mask"] & batch["candidate_mask"]
        valid = pos.any(dim=-1)
        if bool(valid.any()):
            valid_pos = pos[valid]
            valid_scores = scores[valid]
            order = valid_scores.argsort(dim=-1, descending=True)
            ranked_pos = valid_pos.gather(1, order)
            ranks = ranked_pos.float().argmax(dim=-1) + 1
            route_total += int(valid.sum().item())
            route_top1 += int(valid_pos.gather(1, prediction[valid].unsqueeze(1)).sum().item())
            top_k = min(3, valid_scores.shape[-1])
            route_top3 += int(valid_pos.gather(1, order[:, :top_k]).any(dim=-1).sum().item())
            reciprocal_sum += float((1.0 / ranks.float()).sum().cpu())
        need_pred = torch.sigmoid(output["need_memory_logits"]) >= threshold
        required = batch["need"] >= 0.5
        need_tp += int((need_pred & required).sum().item())
        need_tn += int((~need_pred & ~required).sum().item())
        need_fp += int((need_pred & ~required).sum().item())
        need_fn += int((~need_pred & required).sum().item())
        hop_correct += int((output["hop_logits"].argmax(dim=-1) == batch["hops"].clamp(0, router.max_hops)).sum().item())
        hop_total += int(batch["hops"].numel())
        batches_seen += 1
    precision = need_tp / max(1, need_tp + need_fp)
    recall = need_tp / max(1, need_tp + need_fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "episodes": float(n if not max_batches else min(n, batches_seen * batch_size)),
        "route_top1": route_top1 / max(1, route_total),
        "route_recall_at3": route_top3 / max(1, route_total),
        "route_mrr": reciprocal_sum / max(1, route_total),
        "need_precision": precision,
        "need_recall": recall,
        "need_f1": f1,
        "need_specificity": need_tn / max(1, need_tn + need_fp),
        "abstention_accuracy": (need_tp + need_tn) / max(1, need_tp + need_tn + need_fp + need_fn),
        "hop_accuracy": hop_correct / max(1, hop_total),
        "route_positive_episodes": float(route_total),
        "unknown_episodes": float(need_tn + need_fp),
    }


def _sample_indices(data: dict[str, Any], *, batch_size: int, rng: random.Random, mode: str, by_family: dict[str, list[int]]) -> torch.Tensor:
    n = int(data["query_indices"].shape[0])
    if mode == "source_balanced" and by_family:
        families = list(by_family)
        values: list[int] = []
        for _ in range(min(batch_size, n)):
            family = rng.choice(families)
            values.append(rng.choice(by_family[family]))
        return torch.tensor(values, dtype=torch.long)
    return torch.tensor([rng.randrange(n) for _ in range(min(batch_size, n))], dtype=torch.long)


def _save_checkpoint(path: Path, router: MemoryRouterV2, optimizer: torch.optim.Optimizer, scheduler: Any, step: int, best_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "step": step,
        "best_score": best_score,
        "router_state_dict": {key: value.detach().cpu() for key, value in router.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _load_checkpoint(path: Path, router: MemoryRouterV2, optimizer: torch.optim.Optimizer, scheduler: Any) -> tuple[int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid router checkpoint: {path}")
    state = payload.get("router_state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no router state dict: {path}")
    router.load_state_dict(state, strict=True)
    if "optimizer_state_dict" in payload and isinstance(payload["optimizer_state_dict"], dict):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and isinstance(payload.get("scheduler_state_dict"), dict):
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload.get("step", 0)), float(payload.get("best_score", -float("inf")))


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
    router = MemoryRouterV2(
        hidden_size,
        router_dim=args.router_dim,
        num_heads=args.num_heads,
        max_hops=args.max_hops,
    ).to(device)
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
    metrics_path = output_dir / "metrics.jsonl"
    if args.overwrite_metrics and metrics_path.exists():
        metrics_path.unlink()
    history: list[dict[str, Any]] = []
    best_step = start_step if args.resume else 0
    eval_events = 0
    router.train()
    with metrics_path.open("a", encoding="utf-8") as metrics_handle:
        initial_eval = _evaluate(router, eval_data, vectors, device=device, batch_size=args.eval_batch_size, threshold=args.need_threshold, max_batches=args.eval_max_batches)
        initial_eval.update({"event": "eval_resume" if start_step else "eval", "step": start_step})
        metrics_handle.write(json.dumps(initial_eval, ensure_ascii=False) + "\n")
        metrics_handle.flush()
        print(json.dumps(initial_eval, ensure_ascii=False), flush=True)
        for step in range(start_step + 1, total_steps + 1):
            indices = _sample_indices(
                train_data,
                batch_size=args.batch_size,
                rng=rng,
                mode=args.sampling_mode,
                by_family=by_family,
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
                **parts,
            }
            metrics_handle.write(json.dumps(loss_row, ensure_ascii=False) + "\n")
            metrics_handle.flush()
            print(json.dumps(loss_row, ensure_ascii=False), flush=True)
            if step % args.eval_interval == 0 or step == total_steps:
                evaluation_metrics = _evaluate(
                    router,
                    eval_data,
                    vectors,
                    device=device,
                    batch_size=args.eval_batch_size,
                    threshold=args.need_threshold,
                    max_batches=args.eval_max_batches,
                )
                score = 0.5 * evaluation_metrics["route_top1"] + 0.3 * evaluation_metrics["need_f1"] + 0.2 * evaluation_metrics["route_mrr"]
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
    torch.save({key: value.detach().cpu() for key, value in router.state_dict().items()}, output_dir / "memory_router_v2.pt")
    final_eval = _evaluate(router, eval_data, vectors, device=device, batch_size=args.eval_batch_size, threshold=args.need_threshold, max_batches=args.eval_max_batches)
    summary = {
        "format_version": 2,
        "model_path": str(_resolve_path(args.model_path)),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_sha256": _sha256(train_path),
        "eval_sha256": eval_sha256,
        "eval_frozen": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hidden_size": hidden_size,
        "router_dim": args.router_dim,
        "num_heads": args.num_heads,
        "max_hops": args.max_hops,
        "train_episodes": len(train),
        "eval_episodes": len(evaluation),
        "feature_bank": feature_meta,
        "steps": total_steps,
        "start_step": start_step,
        "eval_interval": args.eval_interval,
        "eval_events": eval_events,
        "best_step": best_step,
        "best_selection_score": best_score,
        "final_eval": final_eval,
        "training_history": history,
        "safety": {
            "gpu_memory_cap_gb": args.gpu_memory_gb,
            "feature_dtype": "float16_cpu",
            "qwen_backbone_updated": False,
            "router_only_updated": True,
        },
    }
    (output_dir / "memory_router_large_training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/router_training/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training/eval.jsonl")
    parser.add_argument("--expected-eval-sha256", default="")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--output-dir", default="checkpoints/natural_memory_v2_router_large")
    parser.add_argument("--feature-cache-dir", default="checkpoints/natural_memory_v2_router_large/feature_cache")
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
    parser.add_argument("--router-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-eval-episodes", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="save resumable step checkpoints at this interval; eval still runs every --eval-interval")
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
    if args.eval_interval != 500:
        raise SystemExit("this production protocol requires --eval-interval 500")
    if args.checkpoint_interval < 1:
        raise SystemExit("--checkpoint-interval must be positive")
    if args.steps < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise SystemExit("steps and batch sizes must be positive")
    if not 0.0 <= args.need_threshold <= 1.0:
        raise SystemExit("--need-threshold must be in [0, 1]")
    return args


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result["final_eval"], ensure_ascii=False, indent=2))
