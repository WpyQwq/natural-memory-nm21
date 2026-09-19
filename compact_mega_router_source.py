"""Reduce mega memory records to the evidence needed by a router episode.

The original stress file keeps very large long-context fact lists.  The
router only needs the gold evidence plus a small local context; global hard
negatives are supplied later by prepare_memory_router_dataset.py.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split()).strip()


def compact(row: dict[str, Any], local_limit: int) -> dict[str, Any]:
    facts = [fact for fact in row.get("facts", []) if isinstance(fact, dict) and clean(fact.get("text"))]
    acceptable = [clean(value) for value in row.get("acceptable", []) if clean(value)]
    positive = [fact for fact in facts if any(value.lower() in clean(fact.get("text")).lower() for value in acceptable)]
    category = clean(row.get("category")) or "unknown"

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(fact: dict[str, Any]) -> None:
        key = clean(fact.get("text"))
        if key and key not in seen and len(selected) < local_limit:
            selected.append(fact)
            seen.add(key)

    for fact in positive:
        add(fact)
    if category == "unknown_abstention":
        for fact in facts[:local_limit]:
            add(fact)
    else:
        for fact in facts[:local_limit]:
            add(fact)
        for fact in reversed(facts[-local_limit:]):
            add(fact)

    output = dict(row)
    output["facts"] = selected
    output["metadata"] = {**(row.get("metadata") if isinstance(row.get("metadata"), dict) else {}), "compact_router_source": True}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-limit", type=int, default=8)
    args = parser.parse_args()
    if args.local_limit < 1:
        raise SystemExit("local-limit must be positive")

    counts: Counter[str] = Counter()
    source = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = compact(json.loads(line), args.local_limit)
            counts[clean(row.get("category")) or "unknown"] += 1
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"input": str(source), "output": str(output), "rows": sum(counts.values()), "categories": dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
