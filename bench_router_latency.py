"""Repeatable router latency/throughput benchmark with interleaved measurements.

The scorecard measured each router once, back to back.  Two routers with the same
architecture and the same parameter count then differed by ~4% on single-query
latency, which is the same order as run-to-run drift — not evidence of a real
regression.  This tool settles that by:

* loading every router once and measuring them **round-robin**, so slow drift in
  machine state cannot favour one model over another;
* repeating the whole sweep and reporting min/median/spread per model;
* reporting the same axes the verdict uses (single query, batch-64, batch-256).

Usage::

    python -m V2_dpskw.bench_router_latency ^
        --run "deployed=V2-128:path.pt" --run "v2-128-v6=..." --rounds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_router_scorecard import load_router_any
from V2_dpskw.train_router_v5 import _resolve, load_feature_bank, stream_episode_tensors


def _time_call(fn, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=KIND:PATH")
    parser.add_argument("--train-file", default="data/router_training_v6/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v6/eval.jsonl")
    parser.add_argument("--feature-cache", default=r"H:\Memory\nm_cache\nm_router_v6\feature_cache")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--single-samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", default="router_latency_bench.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path, eval_path = _resolve(args.train_file), _resolve(args.eval_file)
    bank, lookup, _ = load_feature_bank(
        _resolve(args.feature_cache), train_path=train_path, eval_path=eval_path,
        model_path=args.model_path, max_key_tokens=256,
    )
    vectors = torch.from_numpy(bank)
    data = stream_episode_tensors(eval_path, lookup, max_candidates=32)

    routers = []
    for spec in args.run:
        label, rest = spec.split("=", 1)
        path = Path(rest.split(":", 1)[1] if ":" in rest else rest)
        router, arch, info = load_router_any(path)
        routers.append((label, router.to(device).eval(), info["parameters"]))
        print(json.dumps({"loaded": label, "parameters": info["parameters"],
                          "router_dim": arch["router_dim"]}), flush=True)

    def single_query(model):
        index = 0

        def call():
            nonlocal index
            index = (index + 1) % 1000
            q = vectors[data["query_indices"][index]].to(device=device, dtype=torch.float32).unsqueeze(0)
            c = vectors[data["candidate_indices"][index]].to(device=device, dtype=torch.float32).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(q, c)

        return call

    def batched(model, batch):
        idx = torch.arange(batch, dtype=torch.long)
        q = vectors[data["query_indices"][idx]].to(device=device, dtype=torch.float32)
        c = vectors[data["candidate_indices"][idx]].to(device=device, dtype=torch.float32)

        def call():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(q, c)

        return call

    samples: dict[str, dict[str, list[float]]] = {label: {"single": [], "b64": [], "b256": []} for label, _, _ in routers}
    for label, model, _params in routers:          # warm every model before timing any
        call = single_query(model)
        for _ in range(args.warmup):
            call()
        for batch, key in ((64, "b64"), (256, "b256")):
            bcall = batched(model, batch)
            for _ in range(3):
                bcall()

    for round_index in range(args.rounds):
        for label, model, _params in routers:      # round-robin: drift hits everyone
            call = single_query(model)
            elapsed = [_time_call(call, device) for _ in range(args.single_samples)]
            samples[label]["single"].append(statistics.median(elapsed) * 1000)
            for batch, key in ((64, "b64"), (256, "b256")):
                bcall = batched(model, batch)
                reps = 5
                started = time.perf_counter()
                for _ in range(reps):
                    bcall()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total = time.perf_counter() - started
                samples[label][key].append(batch * reps / total)
        print(json.dumps({"round": round_index + 1,
                          "single_ms": {label: round(min(samples[label]["single"]), 4) for label, _, _ in routers}}), flush=True)

    print()
    print("| router | params | single_query_ms (min/median/spread) | batch64_QPS | batch256_QPS |")
    print("|---|---:|---|---:|---:|")
    report = {}
    for label, _model, params in routers:
        s = samples[label]
        row = {
            "parameters": params,
            "rounds": args.rounds,
            "single_ms_min": min(s["single"]),
            "single_ms_median": statistics.median(s["single"]),
            "single_ms_max": max(s["single"]),
            "single_ms_spread_pct": (max(s["single"]) - min(s["single"])) / min(s["single"]) * 100,
            "batch64_qps_median": statistics.median(s["b64"]),
            "batch256_qps_median": statistics.median(s["b256"]),
            "batch64_qps_spread_pct": (max(s["b64"]) - min(s["b64"])) / max(1e-9, min(s["b64"])) * 100,
            "batch256_qps_spread_pct": (max(s["b256"]) - min(s["b256"])) / max(1e-9, min(s["b256"])) * 100,
        }
        report[label] = row
        print("| {label} | {params:,} | {a:.4f} / {b:.4f} / {c:.1f}% | {d:,.0f} | {e:,.0f} |".format(
            label=label, params=params, a=row["single_ms_min"], b=row["single_ms_median"],
            c=row["single_ms_spread_pct"], d=row["batch64_qps_median"], e=row["batch256_qps_median"]))
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
