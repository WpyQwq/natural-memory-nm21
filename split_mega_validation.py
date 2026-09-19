"""Create a deterministic, category-balanced train/eval split for mega memory data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--eval-per-category", type=int, default=2000)
    args = parser.parse_args()

    source = Path(args.input)
    train_path = Path(args.train_output)
    eval_path = Path(args.eval_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_category: dict[str, list[str]] = defaultdict(list)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            category = str(row.get("category", "unknown"))
            rows_by_category[category].append(line)

    if not rows_by_category:
        raise SystemExit("input contains no rows")
    if any(len(rows) <= args.eval_per_category for rows in rows_by_category.values()):
        raise SystemExit("eval-per-category leaves no training rows in at least one category")

    train_counts: Counter[str] = Counter()
    eval_counts: Counter[str] = Counter()
    with train_path.open("w", encoding="utf-8") as train_handle, eval_path.open("w", encoding="utf-8") as eval_handle:
        for category in sorted(rows_by_category):
            rows = rows_by_category[category]
            split_at = len(rows) - args.eval_per_category
            for line in rows[:split_at]:
                train_handle.write(line + "\n")
                train_counts[category] += 1
            for line in rows[split_at:]:
                eval_handle.write(line + "\n")
                eval_counts[category] += 1

    manifest = {
        "source": str(source),
        "eval_per_category": args.eval_per_category,
        "categories": sorted(rows_by_category),
        "train_counts": dict(train_counts),
        "eval_counts": dict(eval_counts),
        "train_rows": sum(train_counts.values()),
        "eval_rows": sum(eval_counts.values()),
    }
    manifest_path = train_path.parent / "mega_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
