"""Render the zero-overlap phase results into one comparison report.

Reads the scorecards produced for this phase and prints / writes a single table set,
so every number quoted in the write-up comes from a stored scorecard rather than from
scrollback.  All rates are rendered as percentages, never bare decimals.

Usage::

    python -m V2_dpskw.report_zero_overlap_phase --output zero_overlap_phase.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: (json path, human label, unit) -- unit is "pct", "int" or "num".
#: Paths are tuples because the threshold key itself contains a dot ("thr0.50").
#: Policy axes live under metrics["thr0.50"], matching eval_router_scorecard.py.
METRICS = [
    (("metrics", "top1"), "Top-1 正确率", "pct"),
    (("metrics", "recall3"), "Recall@3", "pct"),
    (("metrics", "mrr"), "MRR", "pct"),
    (("metrics", "all_evidence_in_top3"), "多跳证据全中(Top-3)", "pct"),
    (("metrics", "hop_accuracy"), "hop 正确率", "pct"),
    (("metrics", "thr0.50", "specificity_unknown_refusal"), "未知拒答率", "pct"),
    (("metrics", "thr0.50", "known_question_refusal_rate"), "已知问题被误拒率", "pct"),
    (("metrics", "thr0.50", "unknown_question_read_rate"), "未知问题被误读率", "pct"),
]

STORAGE_METRICS = [
    (("storage", "address_bytes_per_record"), "每条记录地址字节", "int"),
    (("router", "parameters"), "参数量", "int"),
]

LATENCY_METRICS = [
    (("latency", "cuda", "single_query_latency_ms_p50"), "单查询延迟中位数 ms (GPU)", "num"),
    (("latency", "cuda", "queries_per_second"), "路由 QPS (GPU, 单查询)", "num"),
    (("latency", "cuda", "batched_qps_64"), "批量 QPS (GPU, batch=64)", "num"),
    (("latency", "cuda", "batched_qps_256"), "批量 QPS (GPU, batch=256)", "num"),
]


def pluck(body: dict, path: tuple[str, ...]):
    """Read a nested path from a run body, returning None when any part is missing."""
    current = body
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def render(value, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "pct":
        return f"{100 * float(value):.2f}%"
    if unit == "int":
        return f"{int(value):,}"
    return f"{float(value):.4f}"


def table(title: str, scorecards: list[tuple[str, Path]], note: str = "") -> list[str]:
    loaded = []
    for caption, path in scorecards:
        if not path.exists():
            continue
        loaded.append((caption, path, json.loads(path.read_text(encoding="utf-8"))))
    if not loaded:
        return [f"## {title}", "", "_no scorecard found_", ""]

    labels: list[str] = []
    for _, _, data in loaded:
        for label in data:
            if label not in labels:
                labels.append(label)

    lines = [f"## {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| 指标 | " + " | ".join(labels) + " |")
    lines.append("|---" * (len(labels) + 1) + "|")

    def row(nice: str, extract) -> str:
        cells = []
        for label in labels:
            value = None
            for _, _, data in loaded:
                body = data.get(label)
                if body is not None:
                    value = extract(body)
                    break
            cells.append(value)
        return f"| {nice} | " + " | ".join(cells) + " |"

    for path, nice, unit in METRICS:
        lines.append(row(nice, lambda body, p=path, u=unit: render(pluck(body, p), u)))
    for path, nice, unit in STORAGE_METRICS:
        lines.append(row(nice, lambda body, p=path, u=unit: render(pluck(body, p), u)))
    for path, nice, unit in LATENCY_METRICS:
        lines.append(row(nice, lambda body, p=path, u=unit: render(pluck(body, p), u)))
    lines.append("")

    for caption, path, data in loaded:
        families = {}
        for label in labels:
            body = data.get(label)
            if body:
                for family in body.get("by_family", {}):
                    families[family] = True
        if len(families) > 1:
            lines.append(f"### {caption} — 分类别 Top-1")
            lines.append("")
            lines.append("| 类别 | " + " | ".join(labels) + " |")
            lines.append("|---" * (len(labels) + 1) + "|")
            for family in sorted(families):
                cells = []
                for label in labels:
                    body = data.get(label) or {}
                    block = (body.get("by_family") or {}).get(family) or {}
                    cells.append(render(block.get("top1"), "pct"))
                lines.append(f"| {family} | " + " | ".join(cells) + " |")
            lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="zero_overlap_phase.md")
    args = parser.parse_args()

    lines = [
        "# 零重叠改写（zero-overlap paraphrase）阶段结果",
        "",
        "所有速率均为百分比。数据来源为各阶段保存的 scorecard JSON，不引用滚动日志。",
        "",
    ]
    lines += table(
        "零重叠评测集（300 条，24 个**未见过的**改写问法，24 个同形候选，随机猜测 4.17%）",
        [("基线", ROOT / "zero_overlap_baseline.json"),
         ("冻结策略重放", ROOT / "replay_check_zov.json")],
        note="`基线` 含部署态路由器与 v6；`冻结策略重放` 为合并数据重放训练后的路由器。",
    )
    lines += table(
        "v6 评测集（21,920 条，全部 22 轴）",
        [("基线", ROOT / "forget_check_zov.json"),
         ("冻结策略重放", ROOT / "replay_check_v6.json")],
        note="`基线` 同时包含朴素微调（仅零重叠数据）以显示灾难性遗忘，以及 v6 原始路由器。",
    )
    text = "\n".join(lines) + "\n"
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
