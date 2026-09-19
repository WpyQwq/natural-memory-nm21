"""Multi-axis scorecard for Natural Memory routers (V2 and XL).

The 512-dim baseline run was judged by a single composite ``selection_score``.
That hides the trade-offs a router actually has to make, so this tool reports a
full scorecard on the frozen ``router_training_v3`` eval set:

* retrieval quality   —— Top-1, Recall@1/3/5/8, MRR, nDCG@3, mean rank
* multi-hop quality   —— hop accuracy, hop confusion, all-evidence-in-Top-K
* read/abstain policy —— need precision/recall/F1, specificity, abstention
                         accuracy, **known-question refusal rate**
                         (answerable question wrongly refused) and
                         **unknown-question read rate** (unknown wrongly read)
* calibration         —— score margin between the best positive and the best
                         negative, and the need-memory probability split
* speed               —— training step time, batch routing throughput and
                         single-query routing latency (GPU and CPU)
* storage             —— parameters, checkpoint bytes, address bytes per record
                         and the projected address cost for 1M records
* breakdowns          —— per source family and per mega-validation category

Qwen is never loaded: the tool reads the frozen feature cache, so every router
is scored on exactly the same frozen representations.

Usage::

    python -m V2_dpskw.eval_router_scorecard ^
        --run "V2-512 baseline best=H:\\...\\natural_memory_v2_router_512\\router_best.pt" ^
        --run "XL-512 best=H:\\...\\checkpoints\\router_xl_512\\router_best.pt" ^
        --output router_scorecard.json --markdown router_scorecard.md
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import MemoryRouterV2
from V2_dpskw.router_xl import MemoryRouterXL
from V2_dpskw.train_memory_router_large import (
    _batch_from_indices,
    _evaluate,
    _episode_tensors,
    _prepare_feature_cache,
    _read_episodes,
    _resolve_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# checkpoint loading (works for the V2 baseline and for XL routers)
# --------------------------------------------------------------------------- #
def _infer_v2_config(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    query = state["query_projection.weight"]
    hop_key = "hop_controller.2.weight" if "hop_controller.2.weight" in state else "hop_controller.4.weight"
    return {
        "hidden_size": int(query.shape[1]),
        "router_dim": int(query.shape[0]),
        "num_heads": int(state["head_gate.weight"].shape[0]),
        "max_hops": max(1, int(state[hop_key].shape[0]) - 1),
    }


def _linear_indices(state: dict[str, torch.Tensor], prefix: str) -> list[int]:
    return sorted(
        {
            int(key.split(".")[1])
            for key in state
            if key.startswith(prefix + ".") and key.endswith(".weight")
        }
    )


def _infer_xl_config(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Recover an XL architecture config from tensor shapes alone."""

    query_indices = _linear_indices(state, "query_projection")
    hidden_size = int(state[f"query_projection.{query_indices[0]}.weight"].shape[1])
    router_dim = int(state[f"query_projection.{query_indices[-1]}.weight"].shape[0])
    encoder_layers = len(query_indices)
    encoder_hidden = int(state[f"query_projection.{query_indices[0]}.weight"].shape[0])
    pair_input_dim = int(state["pair_in.weight"].shape[1])
    pair_hidden = int(state["pair_in.weight"].shape[0])
    pair_blocks = len({int(key.split(".")[1]) for key in state if key.startswith("pair_blocks_module.")})
    expansion = 1
    if pair_blocks:
        expansion = int(state["pair_blocks_module.0.fc1.weight"].shape[0]) // max(1, pair_hidden)
    policy_indices = _linear_indices(state, "need_memory")
    hop_indices = _linear_indices(state, "hop_controller")
    return {
        "hidden_size": hidden_size,
        "router_dim": router_dim,
        "num_heads": int(state["head_gate.weight"].shape[0]),
        "max_hops": max(1, int(state[f"hop_controller.{hop_indices[-1]}.weight"].shape[0]) - 1),
        "encoder_layers": max(1, encoder_layers),
        "encoder_hidden": encoder_hidden if encoder_layers > 1 else 0,
        "pair_blocks": pair_blocks,
        "pair_hidden": pair_hidden,
        "pair_expansion": max(1, expansion),
        "use_interaction": pair_input_dim == 4 * router_dim,
        "policy_layers": max(1, len(policy_indices)),
        "policy_hidden": int(state[f"need_memory.{policy_indices[0]}.weight"].shape[0]),
        "learnable_cosine_scale": "cosine_scale" in state,
    }


def load_router_any(path: Path) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Rebuild any router checkpoint and report what was found."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    arch_config = None
    step = None
    if isinstance(payload, dict) and "router_state_dict" in payload:
        state = payload["router_state_dict"]
        arch_config = payload.get("arch_config")
        step = payload.get("step")
    else:
        state = payload
    is_xl = any(key.startswith("pair_in.") or key.startswith("pair_norm.") for key in state)
    if is_xl:
        if arch_config is None:
            sidecar = path.with_name("router_arch.json")
            arch_config = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else _infer_xl_config(state)
        router = MemoryRouterXL.from_arch_config(arch_config)
        kind = "router_xl"
    else:
        config = _infer_v2_config(state)
        arch_config = {"arch": "router_v2", **config}
        router = MemoryRouterV2(**config)
        kind = "router_v2"
    router.load_state_dict(state, strict=True)
    router.eval()
    info = {
        "kind": kind,
        "step": step,
        "arch_config": arch_config,
        "parameters": sum(parameter.numel() for parameter in router.parameters()),
        "checkpoint_bytes": path.stat().st_size,
    }
    return router, arch_config, info


# --------------------------------------------------------------------------- #
# scorecard math
# --------------------------------------------------------------------------- #
def _ndcg_at_k(ranked_positive: list[bool], k: int) -> float:
    dcg = 0.0
    for index, hit in enumerate(ranked_positive[:k], start=1):
        if hit:
            dcg += 1.0 / math.log2(index + 1)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(k, sum(ranked_positive)) + 1))
    return dcg / ideal if ideal > 0 else 0.0


@torch.inference_mode()
def score_router(
    router: torch.nn.Module,
    data: dict[str, Any],
    vectors: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    thresholds: tuple[float, ...] = (0.5,),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return aggregate metrics plus per-episode rows for breakdowns."""

    n = int(data["query_indices"].shape[0])
    per_episode: list[dict[str, Any]] = []
    for start in range(0, n, max(1, batch_size)):
        indices = torch.arange(start, min(n, start + max(1, batch_size)), dtype=torch.long)
        batch = _batch_from_indices(data, vectors, indices, device)
        output = router(batch["query"], batch["candidates"])
        scores = output["scores"].masked_fill(~batch["candidate_mask"], torch.finfo(output["scores"].dtype).min)
        positive = batch["positive_mask"] & batch["candidate_mask"]
        need_prob = torch.sigmoid(output["need_memory_logits"])
        hop_pred = output["hop_logits"].argmax(dim=-1)
        for row in range(scores.shape[0]):
            row_positive = positive[row]
            row_valid = batch["candidate_mask"][row]
            row_negative = row_valid & ~row_positive
            row_scores = scores[row]
            order = row_scores.argsort(dim=-1, descending=True)
            ranked_positive = row_positive.gather(0, order).tolist()
            has_positive = bool(row_positive.any())
            ranks = [index + 1 for index, hit in enumerate(ranked_positive) if hit]
            first_rank = ranks[0] if ranks else 0
            best_positive = float(row_scores[row_positive].max()) if has_positive else float("nan")
            # Padding slots are masked to finfo.min and must not count as negatives.
            best_negative = float(row_scores[row_negative].max()) if bool(row_negative.any()) else float("nan")
            per_episode.append(
                {
                    "index": start + row,
                    "family": str(data["families"][start + row]),
                    "has_positive": has_positive,
                    "first_rank": first_rank,
                    "top1": bool(has_positive and ranked_positive[0]),
                    "recall1": sum(ranked_positive[:1]),
                    "recall3": sum(ranked_positive[:3]),
                    "recall5": sum(ranked_positive[:5]),
                    "recall8": sum(ranked_positive[:8]),
                    # The trainer's ``route_recall_at3`` is "at least one positive
                    # in the top 3" (a fraction, not a count); keep it under its own
                    # name so the two definitions cannot be confused again.
                    "hit3": bool(any(ranked_positive[:3])),
                    "positive_count": int(row_positive.sum().item()),
                    "all_positive_in_top3": bool(has_positive and sum(ranked_positive[:3]) == int(row_positive.sum().item())),
                    "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
                    "ndcg3": _ndcg_at_k(ranked_positive, 3),
                    "need_prob": float(need_prob[row]),
                    "need_true": float(batch["need"][row]) >= 0.5,
                    "hop_true": int(batch["hops"][row].clamp(0, router.max_hops).item()),
                    "hop_pred": int(hop_pred[row]),
                    "score_margin": (best_positive - best_negative) if has_positive else float("nan"),
                    "category": str((data.get("categories") or [""] * n)[start + row]),
                }
            )
    metrics = _summarize(per_episode, thresholds=thresholds)
    return metrics, per_episode


def _summarize(rows: list[dict[str, Any]], *, thresholds: tuple[float, ...]) -> dict[str, Any]:
    answerable = [row for row in rows if row["has_positive"]]
    unknown = [row for row in rows if not row["has_positive"]]
    total = len(rows)

    def mean(values: list[float]) -> float | None:
        """Mean, or ``None`` when the slice has no data (never a fake 0%)."""

        return (sum(values) / len(values)) if values else None

    summary: dict[str, Any] = {
        "episodes": total,
        "answerable_episodes": len(answerable),
        "unknown_episodes": len(unknown),
        # --- retrieval quality, answerable only -----------------------------
        # Recall@k is the FRACTION of an episode's evidence retrieved, not a raw
        # count: multi-hop episodes carry two positives, and counting them produced
        # "recall" values above 100%, which is not a recall at all.
        "top1": mean([1.0 if row["top1"] else 0.0 for row in answerable]),
        "recall1": mean([float(row["recall1"]) / max(1, row["positive_count"]) for row in answerable]),
        "recall3": mean([float(row["recall3"]) / max(1, row["positive_count"]) for row in answerable]),
        "recall5": mean([float(row["recall5"]) / max(1, row["positive_count"]) for row in answerable]),
        "recall8": mean([float(row["recall8"]) / max(1, row["positive_count"]) for row in answerable]),
        "evidence_count_in_top3": mean([float(row["recall3"]) for row in answerable]),
        "hit3": mean([1.0 if row["hit3"] else 0.0 for row in answerable]),
        "mrr": mean([row["reciprocal_rank"] for row in answerable]),
        "ndcg3": mean([row["ndcg3"] for row in answerable]),
        "mean_first_positive_rank": mean([float(row["first_rank"]) for row in answerable if row["first_rank"]]),
        # --- multi-hop evidence completeness --------------------------------
        "all_evidence_in_top3": mean([1.0 if row["all_positive_in_top3"] else 0.0 for row in answerable]),
        "multi_positive_episodes": sum(1 for row in answerable if row["positive_count"] > 1),
        "all_evidence_in_top3_multi": mean(
            [1.0 if row["all_positive_in_top3"] else 0.0 for row in answerable if row["positive_count"] > 1]
        ),
        # --- hop controller --------------------------------------------------
        "hop_accuracy": mean([1.0 if row["hop_pred"] == row["hop_true"] else 0.0 for row in rows]),
        "hop_accuracy_answerable": mean(
            [1.0 if row["hop_pred"] == row["hop_true"] else 0.0 for row in answerable]
        ),
        "hop_under_prediction": mean(
            [1.0 if row["hop_pred"] < row["hop_true"] else 0.0 for row in answerable]
        ),
        "hop_over_prediction": mean(
            [1.0 if row["hop_pred"] > row["hop_true"] else 0.0 for row in answerable]
        ),
        # --- calibration ------------------------------------------------------
        "mean_score_margin": mean([row["score_margin"] for row in answerable]),
        "median_score_margin": statistics.median([row["score_margin"] for row in answerable]) if answerable else None,
        "negative_margin_rate": mean([1.0 if row["score_margin"] < 0 else 0.0 for row in answerable]),
    }
    hypothesis = [row["hop_pred"] for row in rows]
    reference = [row["hop_true"] for row in rows]
    summary["hop_confusion"] = {
        f"true_{true}_pred_{pred}": sum(
            1 for row in rows if row["hop_true"] == true and row["hop_pred"] == pred
        )
        for true in sorted(set(reference))
        for pred in sorted(set(hypothesis) | set(reference))
    }
    summary["known_mean_need_prob"] = mean([row["need_prob"] for row in answerable])
    summary["unknown_mean_need_prob"] = mean([row["need_prob"] for row in unknown])

    for threshold in thresholds:
        tag = f"thr{threshold:.2f}"
        tp = sum(1 for row in rows if row["need_true"] and row["need_prob"] >= threshold)
        tn = sum(1 for row in rows if not row["need_true"] and row["need_prob"] < threshold)
        fp = sum(1 for row in rows if not row["need_true"] and row["need_prob"] >= threshold)
        fn = sum(1 for row in rows if row["need_true"] and row["need_prob"] < threshold)
        positive_denominator = tp + fp
        required_denominator = tp + fn
        negative_denominator = tn + fp
        precision = (tp / positive_denominator) if positive_denominator else None
        recall = (tp / required_denominator) if required_denominator else None
        summary[tag] = {
            # read/abstain policy; None means "not applicable for this slice"
            "need_precision": precision,
            "need_recall": recall,
            "need_f1": (2 * precision * recall / (precision + recall)) if precision and recall else None,
            "specificity_unknown_refusal": (tn / negative_denominator) if negative_denominator else None,
            "abstention_accuracy": ((tp + tn) / total) if total else None,
            "known_question_refusal_rate": (fn / required_denominator) if required_denominator else None,
            "unknown_question_read_rate": (fp / negative_denominator) if negative_denominator else None,
            "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }
    return summary


def _breakdown(rows: list[dict[str, Any]], key: str, *, thresholds: tuple[float, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "") or "unknown")].append(row)
    return {name: _summarize(group, thresholds=thresholds) for name, group in sorted(groups.items())}


@torch.inference_mode()
def measure_latency(router: torch.nn.Module, data: dict[str, Any], vectors: torch.Tensor, *, devices: list[str], samples: int, warmup: int) -> dict[str, Any]:
    n = int(data["query_indices"].shape[0])
    out: dict[str, Any] = {}
    for name in devices:
        device = torch.device(name if name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = router.to(device).eval()
        timings: list[float] = []
        for step in range(warmup + min(samples, n)):
            index = step % n
            query = vectors[data["query_indices"][index]].to(device=device, dtype=torch.float32).unsqueeze(0)
            candidates = vectors[data["candidate_indices"][index]].to(device=device, dtype=torch.float32).unsqueeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(query, candidates)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            if step >= warmup:
                timings.append(elapsed)
        timings.sort()
        entry: dict[str, Any] = {
            "single_query_latency_ms_mean": sum(timings) / len(timings),
            "single_query_latency_ms_p50": timings[len(timings) // 2],
            "single_query_latency_ms_p95": timings[int(len(timings) * 0.95)],
            "queries_per_second": 1000.0 / (sum(timings) / len(timings)),
            "samples": len(timings),
        }
        # Batched throughput: a single query only measures fixed per-call overhead,
        # so also report what a server sees when routing many turns at once.
        for batch in (64, 256):
            if n < batch:
                continue
            indices = torch.arange(batch, dtype=torch.long)
            query = vectors[data["query_indices"][indices]].to(device=device, dtype=torch.float32)
            candidates = vectors[data["candidate_indices"][indices]].to(device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(query, candidates)
            if device.type == "cuda":
                torch.cuda.synchronize()
            repeats = 5
            started = time.perf_counter()
            for _ in range(repeats):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    model(query, candidates)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            entry[f"batched_qps_{batch}"] = (batch * repeats) / max(1e-9, elapsed)
            entry[f"batched_latency_ms_{batch}"] = 1000.0 * elapsed / (batch * repeats)
        out[str(device)] = entry
        router.to(torch.device("cpu"))
    return out


# --------------------------------------------------------------------------- #
def _pct(value: Any) -> str:
    """Rates are always reported as percentages, never as bare decimals."""

    if isinstance(value, float):
        return f"{value * 100:.2f}%"
    return "-" if value is None else str(value)


def _format_table(scorecards: dict[str, dict[str, Any]]) -> str:
    rows = []
    headers = ["metric"] + list(scorecards)
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "---|" * len(headers))

    def add(label: str, getter, *, as_percent: bool = True) -> None:
        cells = []
        for name in scorecards:
            try:
                value = getter(scorecards[name])
            except Exception:
                value = None
            if value is None:
                cells.append("-")
            elif as_percent:
                cells.append(_pct(value))
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join([label] + cells) + " |")

    add("参数", lambda c: f"{c['router']['parameters']:,}", as_percent=False)
    add("router_dim", lambda c: c["router"]["arch_config"]["router_dim"], as_percent=False)
    add("每条记录地址字节", lambda c: c["storage"]["address_bytes_per_record"], as_percent=False)
    add("**Top-1 正确率**", lambda c: c["metrics"]["top1"])
    add("Recall@1", lambda c: c["metrics"]["recall1"])
    add("Recall@3", lambda c: c["metrics"]["recall3"])
    add("Recall@5", lambda c: c["metrics"]["recall5"])
    add("MRR", lambda c: c["metrics"]["mrr"])
    add("nDCG@3", lambda c: c["metrics"]["ndcg3"])
    add("多跳证据全中(Top-3)", lambda c: c["metrics"]["all_evidence_in_top3"])
    add("hop 正确率", lambda c: c["metrics"]["hop_accuracy"])
    add("hop 欠预测率", lambda c: c["metrics"]["hop_under_prediction"])
    add("need F1", lambda c: c["metrics"]["thr0.50"]["need_f1"])
    add("need 召回", lambda c: c["metrics"]["thr0.50"]["need_recall"])
    add("**未知拒答率**", lambda c: c["metrics"]["thr0.50"]["specificity_unknown_refusal"])
    add("**已知问题被误拒率**", lambda c: c["metrics"]["thr0.50"]["known_question_refusal_rate"])
    add("未知问题被误读率", lambda c: c["metrics"]["thr0.50"]["unknown_question_read_rate"])
    add("仲裁准确率", lambda c: c["metrics"]["thr0.50"]["abstention_accuracy"])
    add("已知问题 need 概率均值", lambda c: c["metrics"]["known_mean_need_prob"])
    add("未知问题 need 概率均值", lambda c: c["metrics"]["unknown_mean_need_prob"])
    add("平均分数余量", lambda c: c["metrics"]["mean_score_margin"], as_percent=False)
    add("单查询延迟 ms(GPU)", lambda c: c["latency"].get("cuda", {}).get("single_query_latency_ms_mean"), as_percent=False)
    add("路由 QPS(GPU, 单查询)", lambda c: c["latency"].get("cuda", {}).get("queries_per_second"), as_percent=False)
    add("批量 QPS(GPU, batch=64)", lambda c: c["latency"].get("cuda", {}).get("batched_qps_64"), as_percent=False)
    add("批量 QPS(GPU, batch=256)", lambda c: c["latency"].get("cuda", {}).get("batched_qps_256"), as_percent=False)
    add("批量延迟 ms(GPU, batch=64)", lambda c: c["latency"].get("cuda", {}).get("batched_latency_ms_64"), as_percent=False)
    return "\n".join(rows)


def _format_family_tables(scorecards: dict[str, dict[str, Any]]) -> str:
    """Per-family breakdown, because the aggregate hides family-level failures."""

    families: list[str] = []
    for card in scorecards.values():
        for family in card.get("by_family", {}):
            if family not in families:
                families.append(family)
    metrics = [
        ("episodes", lambda block: block["episodes"], False),
        ("Top-1 正确率", lambda block: block["top1"], True),
        ("Recall@3", lambda block: block["recall3"], True),
        ("MRR", lambda block: block["mrr"], True),
        ("hop 正确率", lambda block: block["hop_accuracy"], True),
        ("未知拒答率", lambda block: block["thr0.50"]["specificity_unknown_refusal"], True),
        ("已知问题被误拒率", lambda block: block["thr0.50"]["known_question_refusal_rate"], True),
        ("未知问题被误读率", lambda block: block["thr0.50"]["unknown_question_read_rate"], True),
        ("平均分数余量", lambda block: block["mean_score_margin"], False),
    ]
    sections: list[str] = []
    for family in families:
        lines = [f"\n### family = `{family}`\n", "| metric | " + " | ".join(scorecards) + " |", "|" + "---|" * (len(scorecards) + 1)]
        for label, getter, as_percent in metrics:
            cells = []
            for card in scorecards.values():
                block = card.get("by_family", {}).get(family)
                if block is None:
                    cells.append("-")
                    continue
                try:
                    value = getter(block)
                except Exception:
                    value = None
                if value is None:
                    cells.append("-")
                elif as_percent:
                    cells.append(_pct(value))
                elif isinstance(value, float):
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join([label] + cells) + " |")
        sections.append("\n".join(lines))
    return "\n".join(sections)


def _format_threshold_tables(scorecards: dict[str, dict[str, Any]]) -> str:
    """Operating curve for the read/abstain decision.

    A router cannot be judged at one threshold: refusing more unknown questions
    also risks refusing answerable ones.  This shows both error directions.
    """

    thresholds = sorted(
        {key for card in scorecards.values() for key in card["metrics"] if key.startswith("thr")},
        key=lambda key: float(key[3:]),
    )
    lines = [
        "\n### 读取/拒答门槛曲线\n",
        "| run | threshold | need F1 | need 召回 | 未知拒答率 | 已知问题被误拒率 | 未知被误读率 | tp/tn/fp/fn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, card in scorecards.items():
        for key in thresholds:
            block = card["metrics"][key]
            counts = block["counts"]
            lines.append(
                "| {label} | {thr} | {f1} | {rec} | {spec} | {known} | {unknown} | {tp}/{tn}/{fp}/{fn} |".format(
                    label=label,
                    thr=key[3:],
                    f1=_pct(block["need_f1"]),
                    rec=_pct(block["need_recall"]),
                    spec=_pct(block["specificity_unknown_refusal"]),
                    known=_pct(block["known_question_refusal_rate"]),
                    unknown=_pct(block["unknown_question_read_rate"]),
                    tp=counts["tp"], tn=counts["tn"], fp=counts["fp"], fn=counts["fn"],
                )
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=CHECKPOINT_PATH")
    parser.add_argument("--train-file", default="data/router_training_v3/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v3/eval.jsonl")
    parser.add_argument("--feature-cache-dir", default="checkpoints/router_shared/feature_cache")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--output", default="router_scorecard.json")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    train_path = _resolve_path(args.train_file)
    eval_path = _resolve_path(args.eval_file)
    train = _read_episodes(train_path)
    evaluation = _read_episodes(eval_path)
    namespace = argparse.Namespace(
        feature_cache_dir=args.feature_cache_dir,
        model_path=args.model_path,
        max_key_tokens=256,
        hidden_size=2560,
        rebuild_features=False,
        precompute_features=False,
        gpu_memory_gb=0.0,
        no_4bit=False,
        encode_batch_size=1,
        precompute_log_every=256,
    )
    vectors, lookup, feature_meta = _prepare_feature_cache(namespace, train, evaluation, train_path, eval_path)
    eval_data = _episode_tensors(evaluation, lookup)
    eval_data["categories"] = [str((row.get("metadata") or {}).get("category", "")) for row in evaluation]
    print(json.dumps({"feature_bank": list(vectors.shape), "eval_episodes": len(evaluation)}, ensure_ascii=False), flush=True)

    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7)
    scorecards: dict[str, dict[str, Any]] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects LABEL=CHECKPOINT, got {spec!r}")
        label, path_value = spec.split("=", 1)
        path = Path(path_value)
        if not path.exists():
            print(f"skipping {label}: {path} does not exist", flush=True)
            continue
        router, arch_config, info = load_router_any(path)
        device = torch.device(args.device)
        router.to(device)
        metrics, per_episode = score_router(
            router, eval_data, vectors, device=device, batch_size=args.batch_size, thresholds=thresholds
        )
        # Cross-check the base metrics against the training-time evaluator.
        reference = _evaluate(router, eval_data, vectors, device=device, batch_size=args.batch_size, threshold=0.5)
        consistency = {
            key: round(metrics[key] - reference[key], 6)
            for key in ("top1", "mrr", "hop_accuracy")
            if key in reference
        }
        # Match definitions: the trainer's route_recall_at3 == our hit3.
        if "route_recall_at3" in reference:
            consistency["hit3"] = round(metrics["hit3"] - reference["route_recall_at3"], 6)
        consistency.update(
            {
                key: round(metrics["thr0.50"][key] - reference[key], 6)
                for key in ("need_precision", "need_recall", "need_f1", "specificity_unknown_refusal", "abstention_accuracy")
                if key in reference
            }
        )
        latency = measure_latency(
            router, eval_data, vectors,
            devices=[args.device, "cpu"] if args.device == "cuda" else ["cpu"],
            samples=args.latency_samples, warmup=args.latency_warmup,
        )
        router_dim = int(arch_config["router_dim"])
        scorecards[label] = {
            "checkpoint": str(path),
            "router": info,
            "storage": {
                "address_bytes_per_record": router_dim * 4,
                "address_mb_per_1m_records": router_dim * 4 * 1_000_000 / 1e6,
                "checkpoint_bytes": info["checkpoint_bytes"],
            },
            "latency": latency,
            "metrics": metrics,
            "by_family": _breakdown(per_episode, "family", thresholds=(0.5,)),
            "by_category": _breakdown(per_episode, "category", thresholds=(0.5,)),
            "protocol_consistency_delta": consistency,
        }
        print(json.dumps({
            "label": label,
            "kind": info["kind"],
            "parameters": info["parameters"],
            "top1": metrics["top1"],
            "recall3": metrics["recall3"],
            "mrr": metrics["mrr"],
            "hop_accuracy": metrics["hop_accuracy"],
            "unknown_refusal": metrics["thr0.50"]["specificity_unknown_refusal"],
            "known_refusal_rate": metrics["thr0.50"]["known_question_refusal_rate"],
            "gpu_latency_ms": latency.get("cuda", {}).get("single_query_latency_ms_mean"),
        }, ensure_ascii=False), flush=True)

    Path(args.output).write_text(json.dumps(scorecards, ensure_ascii=False, indent=2), encoding="utf-8")
    table = _format_table(scorecards)
    print(table, flush=True)
    if args.markdown:
        report = (
            table
            + "\n"
            + _format_family_tables(scorecards)
            + "\n"
            + _format_threshold_tables(scorecards)
            + "\n"
        )
        Path(args.markdown).write_text(report, encoding="utf-8")
        print(f"wrote {args.markdown}", flush=True)
        print(_format_family_tables(scorecards), flush=True)
        print(_format_threshold_tables(scorecards), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
