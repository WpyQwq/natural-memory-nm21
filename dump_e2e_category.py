"""Dump one category of a stored run next to its corpus case.

Shows the anchor, the positive (answer) fact, the distractor facts actually
written into the bank, and the model's verbatim reply -- the minimum needed to
tell "old value won" apart from "wrong attribute was injected".
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.rescore_e2e import load_run


def load_corpus_full(path: Path):
    """Same ordering as the harness: sorted category, file order inside a category."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            candidates = [str(c.get("text", "")) for c in (row.get("candidates") or []) if isinstance(c, dict)]
            positives = [candidates[i] for i in (row.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
            rows.append({
                "category": str(meta.get("category", "")),
                "query": str(row.get("query", "")),
                "acceptable": [str(v) for v in (meta.get("acceptable") or []) if str(v).strip()],
                "positives": positives,
                "candidates": candidates,
                "meta": meta,
            })
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    ordered: list[dict] = []
    for category in sorted(grouped):
        ordered.extend(grouped[category])
    return ordered


def main(corpus, run, category, limit=8, per_category=25):
    cases = load_corpus_full(Path(corpus))
    rows = load_run(Path(run))
    picked = 0
    idx_used = 0
    seen = defaultdict(int)
    for case, row in zip(cases, rows):
        c = case["category"]
        if c != category:
            continue
        idx_used += 1
        picked += 1
        if idx_used > limit:
            break
        print(f"--- #{idx_used} Q: {case['query']}")
        print(f"    anchor  : {case['acceptable']}")
        print(f"    positive: {case['positives']}")
        print(f"    written : {row.get('written')}  correct={row.get('correct')} matched={row.get('matched')}")
        print(f"    reply   : {row.get('reply')!r}")
        others = [f for f in case["candidates"] if f not in case["positives"]]
        for o in others[:4]:
            print(f"      other : {o}")
        extra = {k: v for k, v in case["meta"].items() if k not in {"category", "acceptable"}}
        if extra:
            print(f"    meta    : {json.dumps(extra, ensure_ascii=False)[:300]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
