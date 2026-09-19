"""Side-by-side comparison of the NM2.1 and original-NM2 memory batteries.

Reads the tagged battery outputs and prints one comparison table, so the NM2.1 numbers
are never quoted from scrollback.  All rates are printed as percentages.

Usage::

    python -m V2_dpskw.compare_nm2_batteries --tag NM2.1=nm2_1 --tag 原版NM2=nm2_orig
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # missing or malformed run
        return {"__error__": str(exc)}


def dig(data: dict, *path, default=None):
    current = data
    for step in path:
        if not isinstance(current, dict) or step not in current:
            return default
        current = current[step]
    return current


def first_key(data: dict) -> str | None:
    if not data or "__error__" in data:
        return None
    return next(iter(data))


def row_for(tag: str) -> dict:
    a = load(Path(f"{tag}_e2e.json"))
    b = load(Path(f"{tag}_critical_e2e.json"))
    c = load(Path(f"{tag}_runtime_e2e.json"))
    d = load(Path(f"{tag}_restart.json"))
    ka, kb = first_key(a), first_key(b)
    # The runtime evaluator labels its runs by the knobs under test, so take whichever
    # run key the file actually holds rather than assuming one.
    c_key = next(iter((c.get("results") or {}).keys()), "0.00") if "__error__" not in c else "0.00"
    return {
        "A_cases": dig(a, ka, "summary", "cases") if ka else None,
        "A_overall": dig(a, ka, "summary", "accuracy_pct") if ka else None,
        "A_answerable": dig(a, ka, "summary", "answerable_accuracy_pct") if ka else None,
        "A_unknown_refusal": dig(a, ka, "summary", "unknown_refusal_pct") if ka else None,
        "A_known_false_refusal": dig(a, ka, "summary", "wrong_abstention_pct") if ka else None,
        "B_accuracy": dig(b, kb, "summary", "accuracy_pct") if kb else None,
        "B_wrong_attribute": dig(b, kb, "summary", "wrong_attribute_pct") if kb else None,
        "B_read": dig(b, kb, "summary", "read_pct") if kb else None,
        "B_records": dig(b, kb, "summary", "mean_records_selected") if kb else None,
        "C_answerable": dig(c, "results", c_key, "accuracy_pct"),
        "C_wrong_attribute": dig(c, "results", c_key, "wrong_attribute_pct"),
        "C_unknown_leak": dig(c, "results", c_key, "unknown_leak_pct"),
        "C_active_min": dig(c, "results", c_key, "records_active_min"),
        "C_active_max": dig(c, "results", c_key, "records_active_max"),
        "D_recalled": dig(d, "router_recalled_after_restart"),
        "D_answer_correct": dig(d, "generated_contains_expected"),
        "D_cleanup": dig(d, "cleanup_applied"),
    }


ROW_LABELS = [
    ("A_cases", "A 用例数", "int"),
    ("A_overall", "A 总体正确率", "pct"),
    ("A_answerable", "A 可回答正确率", "pct"),
    ("A_unknown_refusal", "A 未知拒答率", "pct"),
    ("A_known_false_refusal", "A 已知问题被误拒率", "pct"),
    ("B_accuracy", "B 回答正确率", "pct"),
    ("B_wrong_attribute", "B 答成别的属性", "pct"),
    ("B_read", "B 触发读取", "pct"),
    ("B_records", "B 平均选中记录", "num"),
    ("C_answerable", "C 可回答正确率", "pct"),
    ("C_wrong_attribute", "C 答成别的属性", "pct"),
    ("C_unknown_leak", "C 未知泄漏率", "pct"),
    ("C_active_min", "C 活跃记录下限", "int"),
    ("C_active_max", "C 活跃记录上限", "int"),
    ("D_recalled", "D 重启后召回", "bool"),
    ("D_answer_correct", "D 重启后作答正确", "bool"),
    ("D_cleanup", "D 清理生效", "bool"),
]


def render(value, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "pct":
        return f"{float(value):.2f}%"
    if unit == "int":
        return f"{int(value):,}"
    if unit == "num":
        return f"{float(value):.4f}"
    return "通过" if value else "**未通过**"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", action="append", required=True, help="LABEL=TAG")
    parser.add_argument("--output", default="nm2_battery_comparison.json")
    parser.add_argument("--markdown", default="nm2_battery_comparison.md")
    args = parser.parse_args()

    pairs = []
    for item in args.tag:
        label, _, tag = item.partition("=")
        pairs.append((label, tag, row_for(tag)))

    lines = ["# NM2.1 与原版 NM2 的整体记忆测试对照", "",
             "同一套电池、同一份运行时，只有模型包不同。全部为百分比。", "",
             "| 指标 | " + " | ".join(label for label, _, _ in pairs) + " |",
             "|---" * (len(pairs) + 1) + "|"]
    for key, nice, unit in ROW_LABELS:
        cells = [render(body.get(key), unit) for _, _, body in pairs]
        lines.append(f"| {nice} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    Path(args.markdown).write_text(text, encoding="utf-8")
    Path(args.output).write_text(
        json.dumps({label: body for label, _, body in pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(text)
    print(f"wrote {args.markdown} and {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
