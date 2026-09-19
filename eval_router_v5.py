"""Multi-axis scorecard for the v5 dataset (streaming + memory-mapped bank).

Same metrics, breakdowns, threshold curve and percentage formatting as
``eval_router_scorecard``; the difference is the data path:

* episodes are **streamed** from the 1.15 GB ``train.jsonl`` / 288 MB
  ``eval.jsonl`` instead of being parsed into RAM;
* the 10.86 GB feature bank is **memory-mapped** from the NVMe cache;
* the evaluation set is the full frozen v5 eval (21,920 episodes covering all ten
  mega categories), so the report can break every metric down per category.

Usage::

    python -m V2_dpskw.eval_router_v5 ^
        --run "V2-512(v5)=checkpoints/router_v5_v2_512/router_best.pt" ^
        --run "XL-512(v5)=checkpoints/router_v5_xl512/router_best.pt" ^
        --output router_scorecard_v5.json --markdown router_scorecard_v5.md
"""

from __future__ import annotations

import argparse
import json
import sys
import torch
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_router_scorecard import (
    _breakdown,
    _format_family_tables,
    _format_table,
    _format_threshold_tables,
    load_router_any,
    measure_latency,
    score_router,
)
from V2_dpskw.train_router_v5 import _resolve, load_feature_bank, stream_episode_tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=CHECKPOINT_PATH")
    parser.add_argument("--train-file", default="data/router_training_v5/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v5/eval.jsonl")
    parser.add_argument("--feature-cache", default=r"H:\Memory\nm_cache\nm_router_v5\feature_cache")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--max-key-tokens", type=int, default=256)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latency-samples", type=int, default=150)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument(
        "--latency-overrides",
        default="",
        help=(
            "JSON from bench_router_latency.py; replaces this tool's single-shot latency "
            "numbers with interleaved round-robin medians.  A single-shot sweep measures "
            "models back to back, so a ~4% gap between two identical architectures is drift, "
            "not a regression."
        ),
    )
    parser.add_argument("--output", default="router_scorecard_v5.json")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    train_path = _resolve(args.train_file)
    eval_path = _resolve(args.eval_file)
    cache_dir = _resolve(args.feature_cache)
    bank, lookup, manifest = load_feature_bank(
        cache_dir, train_path=train_path, eval_path=eval_path,
        model_path=args.model_path, max_key_tokens=args.max_key_tokens,
    )
    vectors = torch.from_numpy(bank)
    eval_data = stream_episode_tensors(eval_path, lookup, max_candidates=args.candidate_count)
    print(json.dumps({
        "eval_episodes": len(eval_data["families"]),
        "answerable": sum(1 for row in eval_data["need"].tolist() if row >= 0.5),
        "categories": {name: eval_data["categories"].count(name) for name in sorted(set(eval_data["categories"]))},
    }, ensure_ascii=False), flush=True)

    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    latency_overrides: dict[str, dict] = {}
    if args.latency_overrides:
        latency_overrides = json.loads(Path(args.latency_overrides).read_text(encoding="utf-8"))
        print(json.dumps({"latency_overrides_from": args.latency_overrides,
                          "labels": sorted(latency_overrides)}, ensure_ascii=False), flush=True)
    scorecards: dict[str, dict] = {}
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
        latency = measure_latency(
            router, eval_data, vectors,
            devices=[args.device, "cpu"] if args.device == "cuda" else ["cpu"],
            samples=args.latency_samples, warmup=args.latency_warmup,
        )
        override = latency_overrides.get(label)
        if override:
            median_ms = float(override["single_ms_median"])
            latency["cuda"] = {
                "single_query_latency_ms_mean": median_ms,
                "single_query_latency_ms_p50": median_ms,
                "single_query_latency_ms_p95": float(override["single_ms_max"]),
                "queries_per_second": 1000.0 / median_ms,
                "batched_qps_64": float(override["batch64_qps_median"]),
                "batched_qps_256": float(override["batch256_qps_median"]),
                "batched_latency_ms_64": 1000.0 / float(override["batch64_qps_median"]),
                "samples": int(override.get("rounds", 0)),
                "source": "bench_router_latency.py (interleaved round-robin medians)",
            }
        router_dim = int(arch_config["router_dim"])
        scorecards[label] = {
            "checkpoint": str(path),
            "router": info,
            "storage": {
                "address_bytes_per_record": router_dim * 4,
                "address_mb_per_1m_records": router_dim * 4,
                "checkpoint_bytes": info["checkpoint_bytes"],
            },
            "latency": latency,
            "metrics": metrics,
            "by_family": _breakdown(per_episode, "family", thresholds=(0.5,)),
            "by_category": _breakdown(per_episode, "category", thresholds=(0.5,)),
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
    categories = _format_category_table(scorecards)
    print(categories, flush=True)
    if args.markdown:
        report = "\n".join([
            table,
            _format_family_tables(scorecards),
            _format_threshold_tables(scorecards),
            categories,
        ]) + "\n"
        Path(args.markdown).write_text(report, encoding="utf-8")
        print(f"wrote {args.markdown}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


def _format_category_table(scorecards: dict[str, dict]) -> str:
    """Per mega-category table: the 91% of the v5 eval that the old set lacked."""

    from V2_dpskw.eval_router_scorecard import _pct

    categories: list[str] = []
    for card in scorecards.values():
        for name in card.get("by_category", {}):
            if name and name not in categories:
                categories.append(name)
    rows = ["\n### 按 mega 类别拆解（每类 2,000 条）\n"]
    headers = ["类别", "episodes", "metric"] + list(scorecards)
    for name in ["", *sorted(categories)]:
        block_rows = []
        for label, getter, percent in (
            ("Top-1 正确率", lambda b: b["top1"], True),
            ("Recall@3", lambda b: b["recall3"], True),
            ("MRR", lambda b: b["mrr"], True),
            ("多跳证据全中", lambda b: b["all_evidence_in_top3"], True),
            ("hop 正确率", lambda b: b["hop_accuracy"], True),
            ("未知拒答率", lambda b: b["thr0.50"]["specificity_unknown_refusal"], True),
            ("已知问题被误拒率", lambda b: b["thr0.50"]["known_question_refusal_rate"], True),
            ("未知问题被误读率", lambda b: b["thr0.50"]["unknown_question_read_rate"], True),
        ):
            cells = []
            episodes = "-"
            for card in scorecards.values():
                block = card.get("by_category", {}).get(name)
                if block is None:
                    cells.append("-")
                    continue
                episodes = str(block["episodes"])
                value = getter(block)
                cells.append(_pct(value) if percent else value)
            block_rows.append("| " + " | ".join([name or "(未分类)", episodes, label] + cells) + " |")
        rows.extend(block_rows)
    header = "| 类别 | episodes | metric | " + " | ".join(scorecards) + " |"
    divider = "|" + "---|" * (len(scorecards) + 3)
    return rows[0] + "\n" + header + "\n" + divider + "\n" + "\n".join(rows[1:])


if __name__ == "__main__":
    raise SystemExit(main())
