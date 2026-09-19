"""Verify dominance on the policy axes across the *whole* threshold sweep, not just 0.50.

The scorecard's headline verdict compares the refusal axes at the single default
read threshold (0.50).  A candidate that merely happens to sit on the right side of one
threshold could still be worse at 0.30 or 0.80.  This script checks every threshold in
the recorded sweep, so "not worse than the baseline on 未知拒答率 / 已知问题被误拒率 /
未知问题被误读率" is established across the curve rather than at one point.

Usage::

    python -m V2_dpskw.verify_threshold_sweep --scorecard router_scorecard_final.json ^
        --baseline "V2-128 deployed(v3)" --candidate "REPLAY-128 v7 final"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: (metric key, label, direction) -- "max" means higher is better.
AXES = [
    ("need_f1", "need F1", "max"),
    ("need_recall", "need 召回", "max"),
    ("need_precision", "need 精确率", "max"),
    ("specificity_unknown_refusal", "未知拒答率", "max"),
    ("known_question_refusal_rate", "已知问题被误拒率", "min"),
    ("unknown_question_read_rate", "未知问题被误读率", "min"),
    ("abstention_accuracy", "仲裁准确率", "max"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", default="router_scorecard_final.json")
    parser.add_argument("--baseline", default="V2-128 deployed(v3)")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", default="threshold_sweep_check.json")
    parser.add_argument("--markdown", default="threshold_sweep_check.md")
    args = parser.parse_args()

    data = json.loads(Path(args.scorecard).read_text(encoding="utf-8"))
    if args.baseline not in data:
        raise SystemExit(f"baseline {args.baseline!r} not in scorecard: {list(data)}")
    thresholds = sorted(k for k in data[args.baseline]["metrics"] if k.startswith("thr"))

    lines = ["# 拒答策略轴 · 全门槛曲线验证", "",
             f"基线：`{args.baseline}`。门槛取值：{', '.join(t[3:] for t in thresholds)}。", "",
             "全部为百分比。`通过` 表示该候选在每个门槛上都不劣于基线。", ""]
    report: dict = {"baseline": args.baseline, "thresholds": thresholds, "candidates": {}}

    for candidate in args.candidate:
        if candidate not in data:
            raise SystemExit(f"candidate {candidate!r} not in scorecard")
        rows = []
        all_pass = True
        for key, label, direction in AXES:
            cells = []
            axis_pass = True
            for threshold in thresholds:
                base_value = data[args.baseline]["metrics"][threshold].get(key)
                cand_value = data[candidate]["metrics"][threshold].get(key)
                if base_value is None or cand_value is None:
                    cells.append("-")
                    continue
                if direction == "max":
                    ok = cand_value >= base_value - 1e-12
                    delta = cand_value - base_value
                else:
                    ok = cand_value <= base_value + 1e-12
                    delta = cand_value - base_value
                axis_pass = axis_pass and ok
                cells.append(f"{100 * cand_value:.2f}%" + ("" if ok else " ✗"))
            all_pass = all_pass and axis_pass
            rows.append({"axis": label, "direction": direction,
                         "values_pct": cells, "axis_passed": axis_pass})
        report["candidates"][candidate] = {"rows": rows, "sweep_dominates": all_pass}

    lines.append("| 轴 | " + " | ".join(f"门槛 {t[3:]}" for t in thresholds) + " | 全门槛通过 |")
    lines.append("|---" * (len(thresholds) + 2) + "|")
    for candidate in args.candidate:
        body = report["candidates"][candidate]
        for row in body["rows"]:
            lines.append(
                f"| {candidate} · {row['axis']} | " + " | ".join(row["values_pct"])
                + f" | {'是' if row['axis_passed'] else '**否**'} |"
            )
    lines.append("")
    lines.append("## 基线在各门槛上的值（作对照）")
    lines.append("")
    lines.append("| 轴 | " + " | ".join(f"门槛 {t[3:]}" for t in thresholds) + " |")
    lines.append("|---" * (len(thresholds) + 1) + "|")
    for key, label, _ in AXES:
        cells = []
        for threshold in thresholds:
            value = data[args.baseline]["metrics"][threshold].get(key)
            cells.append("-" if value is None else f"{100 * value:.2f}%")
        lines.append(f"| {args.baseline} · {label} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    Path(args.markdown).write_text(text, encoding="utf-8")
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    for candidate in args.candidate:
        verdict = "通过" if report["candidates"][candidate]["sweep_dominates"] else "未通过"
        print(f"[{candidate}] 全门槛 7 轴判定: {verdict}")
    return 0 if all(report["candidates"][c]["sweep_dominates"] for c in args.candidate) else 2


if __name__ == "__main__":
    raise SystemExit(main())
