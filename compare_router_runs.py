"""Compare the new MemoryRouterXL runs against the 512-dim MemoryRouterV2 baseline.

All runs share the same frozen ``router_training_v3`` eval set (SHA-256 checked
by the trainers), the same selection score and the same metric implementation,
so the only intended difference is router capacity.

Usage (from the fork root)::

    python -m V2_dpskw.compare_router_runs
    python -m V2_dpskw.compare_router_runs --markdown router_xl_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent

#: Baseline produced by the original (GPT-era) trainer in the upstream project.
DEFAULT_BASELINE = Path(
    r"H:\Memory\dynamic_memory_lab\checkpoints\natural_memory_v2_router_512\memory_router_large_training.json"
)

METRIC_KEYS = (
    ("route_top1", "Top-1"),
    ("route_recall_at3", "Recall@3"),
    ("route_mrr", "MRR"),
    ("need_f1", "need F1"),
    ("need_specificity", "specificity"),
    ("abstention_accuracy", "abstention"),
    ("hop_accuracy", "hop acc"),
)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return None


def _row(label: str, params: int | str, summary: dict[str, Any] | None, *, source: str) -> dict[str, Any]:
    if summary is None:
        return {"label": label, "parameters": params, "source": source, "status": "missing"}
    best = summary.get("best_eval") or summary.get("final_eval") or {}
    final = summary.get("final_eval") or {}
    row: dict[str, Any] = {
        "label": label,
        "parameters": params,
        "source": source,
        "status": "ok",
        "router_dim": summary.get("router_dim"),
        "num_heads": summary.get("num_heads"),
        "steps": summary.get("steps"),
        "eval_episodes": summary.get("eval_episodes"),
        "eval_sha256": summary.get("eval_sha256"),
        "best_step": summary.get("best_step"),
        "best_selection_score": summary.get("best_selection_score"),
        "best": {key: best.get(key) for key, _ in METRIC_KEYS},
        "final": {key: final.get(key) for key, _ in METRIC_KEYS},
    }
    if "arch_config" in summary:
        row["arch"] = summary["arch_config"]
    if isinstance(summary.get("parameters"), dict):
        row["parameters"] = summary["parameters"].get("total", params)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="training summary json; repeatable. Defaults to the XL runs in checkpoints/.",
    )
    parser.add_argument("--output", default="router_xl_comparison.json")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    run_paths = args.run or [
        str(PROJECT_ROOT / "checkpoints/router_xl_1024/router_xl_training.json"),
        str(PROJECT_ROOT / "checkpoints/router_xl_2048/router_xl_training.json"),
    ]

    baseline_path = Path(args.baseline)
    baseline_summary = _load(baseline_path)
    rows: list[dict[str, Any]] = [
        _row(
            "MemoryRouterV2 512 (baseline)",
            4741902,
            baseline_summary,
            source=str(baseline_path),
        )
    ]
    for run in run_paths:
        path = Path(run)
        summary = _load(path)
        label = (summary or {}).get("label") or path.parent.name
        rows.append(_row(label, "?", summary, source=str(path)))

    report = {"baseline": str(baseline_path), "runs": run_paths, "rows": rows}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    header = ["router", "params", "dim", "steps", "best step", "score"] + [name for _, name in METRIC_KEYS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        if row["status"] != "ok":
            lines.append("| " + " | ".join([row["label"], str(row["parameters"]), "-", "-", "-", "-"] + ["-"] * len(METRIC_KEYS)) + " |")
            continue
        best = row["best"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    f"{row['parameters']:,}" if isinstance(row["parameters"], int) else str(row["parameters"]),
                    str(row.get("router_dim", "-")),
                    str(row.get("steps", "-")),
                    str(row.get("best_step", "-")),
                    f"{row['best_selection_score']:.4f}" if isinstance(row.get("best_selection_score"), (int, float)) else "-",
                ]
                + [
                    f"{best[key]:.4f}" if isinstance(best.get(key), (int, float)) else "-"
                    for key, _ in METRIC_KEYS
                ]
            )
            + " |"
        )
    table = "\n".join(lines)
    print(table, flush=True)
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
