"""Turn a v6 scorecard into an explicit per-axis dominance verdict.

"Complete dominance" is a claim about every axis at once, so it should be
computed, not narrated: for each axis this compares the candidate router against
the *best* value achieved by any baseline router (the deployed 128-dim router and
the two v3-trained 512-dim routers) and reports pass/fail with the delta.

Lower-is-better axes (latency, storage, false-refusal rate) are handled too, so a
faster-but-worse model cannot look like a win.

Usage::

    python -m V2_dpskw.verdict_router_v6 ^
        --scorecard router_scorecard_v6.json ^
        --candidate "XL-512 v6 best" ^
        --baseline-prefix "V2-128 deployed" --baseline-prefix "V2-512 v3" ^
        --markdown router_verdict_v6.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pct(value: Any) -> str:
    return f"{value * 100:.2f}%" if isinstance(value, float) else ("-" if value is None else str(value))


def _axis_specs() -> list[tuple[str, Callable[[dict], Any], str]]:
    return [
        ("Top-1 正确率", lambda c: c["metrics"]["top1"], "max"),
        ("Recall@1", lambda c: c["metrics"]["recall1"], "max"),
        ("Recall@3", lambda c: c["metrics"]["recall3"], "max"),
        ("Recall@5", lambda c: c["metrics"]["recall5"], "max"),
        ("MRR", lambda c: c["metrics"]["mrr"], "max"),
        ("nDCG@3", lambda c: c["metrics"]["ndcg3"], "max"),
        ("多跳证据全中(Top-3)", lambda c: c["metrics"]["all_evidence_in_top3"], "max"),
        ("多跳证据全中(仅多正例)", lambda c: c["metrics"]["all_evidence_in_top3_multi"], "max"),
        ("hop 正确率", lambda c: c["metrics"]["hop_accuracy"], "max"),
        ("hop 欠预测率", lambda c: c["metrics"]["hop_under_prediction"], "min"),
        ("need F1 (门槛 0.50)", lambda c: c["metrics"]["thr0.50"]["need_f1"], "max"),
        ("need 召回 (门槛 0.50)", lambda c: c["metrics"]["thr0.50"]["need_recall"], "max"),
        ("未知拒答率 (门槛 0.50)", lambda c: c["metrics"]["thr0.50"]["specificity_unknown_refusal"], "max"),
        ("已知问题被误拒率", lambda c: c["metrics"]["thr0.50"]["known_question_refusal_rate"], "min"),
        ("未知问题被误读率", lambda c: c["metrics"]["thr0.50"]["unknown_question_read_rate"], "min"),
        ("仲裁准确率", lambda c: c["metrics"]["thr0.50"]["abstention_accuracy"], "max"),
        ("平均分数余量", lambda c: c["metrics"]["mean_score_margin"], "max"),
        ("单查询延迟 ms (GPU)", lambda c: (c.get("latency", {}).get("cuda") or {}).get("single_query_latency_ms_mean"), "min"),
        ("路由 QPS (GPU, 单查询)", lambda c: (c.get("latency", {}).get("cuda") or {}).get("queries_per_second"), "max"),
        ("批量 QPS (GPU, batch=64)", lambda c: (c.get("latency", {}).get("cuda") or {}).get("batched_qps_64"), "max"),
        ("批量 QPS (GPU, batch=256)", lambda c: (c.get("latency", {}).get("cuda") or {}).get("batched_qps_256"), "max"),
        # Address bytes per record is the real deployment cost of a wider router.
        # Parameter count is deliberately NOT an axis: the objective is a router
        # with more dimensions/parameters, so a larger router is not a regression.
        ("地址字节/记录", lambda c: c.get("storage", {}).get("address_bytes_per_record"), "min"),
    ]


#: Speed axes carry a relative tolerance, quality axes do not.
#: bench_router_latency.py measures a 2.8-5.3% run-to-run spread, so a ~1% gap
#: between two routers with identical architecture, size and address geometry is
#: noise; treating it as a defeat would be a measurement artefact, not a finding.
NOISY_AXES = {
    "单查询延迟 ms (GPU)",
    "路由 QPS (GPU, 单查询)",
    "批量 QPS (GPU, batch=64)",
    "批量 QPS (GPU, batch=256)",
}


def _axis_specs_with_tolerance() -> list[tuple[str, Callable[[dict], Any], str, bool]]:
    return [(name, getter, direction, name in NOISY_AXES) for name, getter, direction in _axis_specs()]


def _value(card: dict, getter: Callable[[dict], Any]) -> Any:
    try:
        value = getter(card)
    except Exception:
        return None
    return value if isinstance(value, (int, float)) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", default="router_scorecard_v6.json")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--baseline-prefix", action="append", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9, help="pass threshold slack")
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.03,
        help="relative slack for speed axes, set to the measured run-to-run spread (default 3%%)",
    )
    parser.add_argument("--output", default="router_verdict_v6.json")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    cards = json.loads(Path(args.scorecard).read_text(encoding="utf-8"))
    baselines = {
        label: card for label, card in cards.items()
        if any(label.startswith(prefix) for prefix in args.baseline_prefix)
    }
    if not baselines:
        raise SystemExit(f"no baseline rows matched {args.baseline_prefix} in {args.scorecard}")

    lines = ["| 指标 | 最强旧基线 | 基线值 | 候选值 | 差值 | 判定 |", "|---|---|---:|---:|---:|---|"]
    rows: list[dict[str, Any]] = []
    passed = failed = skipped = 0
    for name, getter, direction, is_noisy in _axis_specs_with_tolerance():
        baseline_values = {label: _value(card, getter) for label, card in baselines.items()}
        baseline_values = {label: value for label, value in baseline_values.items() if value is not None}
        if not baseline_values:
            skipped += 1
            continue
        best_label = (max if direction == "max" else min)(baseline_values, key=baseline_values.get)
        best_value = baseline_values[best_label]
        # Speed axes get a relative band equal to the benchmark's own spread.
        slack = abs(best_value) * args.relative_tolerance if is_noisy else args.tolerance
        for candidate in args.candidate:
            card = cards.get(candidate)
            if card is None:
                continue
            value = _value(card, getter)
            if value is None:
                skipped += 1
                continue
            better = value >= best_value - slack if direction == "max" else value <= best_value + slack
            delta = value - best_value
            rows.append({
                "axis": name, "candidate": candidate, "baseline": best_label,
                "baseline_value": best_value, "candidate_value": value,
                "delta": delta, "direction": direction, "pass": bool(better),
                "relative_tolerance_applied": bool(is_noisy),
                "relative_delta_pct": (delta / best_value * 100) if best_value else None,
            })
            if better:
                passed += 1
            else:
                failed += 1

    for row in rows:
        is_rate = abs(row["baseline_value"]) <= 1.5 and abs(row["candidate_value"]) <= 1.5 and not row["axis"].startswith(("单查询", "路由 QPS", "地址字节", "参数量"))
        shown_base = _pct(row["baseline_value"]) if is_rate else f"{row['baseline_value']:.4f}"
        shown_cand = _pct(row["candidate_value"]) if is_rate else f"{row['candidate_value']:.4f}"
        shown_delta = (f"{row['delta'] * 100:+.2f}pp" if is_rate else f"{row['delta']:+.4f}")
        mark = "通过" if row["pass"] else "**未通过**"
        lines.append(f"| {row['axis']} | {row['baseline']} | {shown_base} | {shown_cand} | {shown_delta} | {mark} |")

    verdict = {
        "candidates": args.candidate,
        "baselines": list(baselines),
        "passed_axes": passed,
        "failed_axes": failed,
        "skipped_axes": skipped,
        # Dominance is a claim about ONE router, so it is reported per candidate:
        # an aggregate over candidates hides that some of them dominate and others
        # do not, which is exactly the situation here.
        "dominates": failed == 0 and passed > 0,
        "per_candidate": {
            label: {
                "passed": sum(1 for row in rows if row["candidate"] == label and row["pass"]),
                "failed": sum(1 for row in rows if row["candidate"] == label and not row["pass"]),
                "dominates": bool(rows) and all(
                    row["pass"] for row in rows if row["candidate"] == label
                ) and any(row["candidate"] == label for row in rows),
                "failed_axes": [
                    row["axis"] for row in rows if row["candidate"] == label and not row["pass"]
                ],
            }
            for label in args.candidate
        },
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_lines = ["\n### 逐候选判定（全方位超越是对单个路由器说的）\n",
                     "| 路由器 | 通过 | 未通过 | 全方位超越 | 未通过的轴 |", "|---|---:|---:|---|---|"]
    for label, block in verdict["per_candidate"].items():
        summary_lines.append("| {label} | {p} | {f} | {d} | {axes} |".format(
            label=label, p=block["passed"], f=block["failed"],
            d="**是**" if block["dominates"] else "否",
            axes=", ".join(block["failed_axes"]) or "-"))
    summary = "\n".join(summary_lines) + (
        f"\n\n**汇总 {passed} 项通过 / {failed} 项未通过 / {skipped} 项无数据**\n")
    table = "\n".join(lines) + "\n" + summary
    print(table, flush=True)
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0 if verdict["dominates"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
