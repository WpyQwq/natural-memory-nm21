"""Independent audit of a router dataset: category integrity and train/eval leakage.

The builder reports its own ``group_overlap`` check, but a benchmark that will be
quoted as evidence deserves an independent audit that streams the frozen files and
verifies, without trusting the builder's counters:

* per-family / per-category episode counts and the answerable-vs-unknown split;
* candidate width, positive-count and hop distributions per category (a category
  that silently lost its positives would make every router look identical);
* **text-level leakage**: how many eval queries and eval candidate texts also
  occur in the training split, and how many ``group_id`` values overlap.

Usage::

    python -m V2_dpskw.audit_router_dataset ^
        --train-file data/router_training_v5/train.jsonl ^
        --eval-file data/router_training_v5/eval.jsonl ^
        --output router_dataset_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _digest(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8", "replace")).digest()


def _scan(path: Path) -> dict:
    per_category: dict[str, Counter] = defaultdict(Counter)
    families = Counter()
    categories = Counter()
    group_ids: set[str] = set()
    query_digests: set[bytes] = set()
    candidate_digests: set[bytes] = set()
    positive_by_query: dict[bytes, set[bytes]] = defaultdict(set)
    queries_by_digest: dict[bytes, str] = {}
    episodes = 0
    started = time.perf_counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            episodes += 1
            family = str(row.get("family", "?") or "?")
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            category = str(metadata.get("category", "") or "(none)")
            positives = row.get("positive_indices") or []
            candidates = row.get("candidates") or []
            families[family] += 1
            categories[category] += 1
            group_ids.add(str(row.get("group_id", "")))
            bucket = per_category[category]
            bucket["episodes"] += 1
            bucket["answerable" if positives else "unknown"] += 1
            bucket[f"candidates_{len(candidates)}"] += 1
            bucket[f"positives_{min(len(positives), 4)}"] += 1
            bucket[f"hop_{int(row.get('hop', 0))}"] += 1
            bucket["candidate_slots"] += len(candidates)
            bucket["positive_total"] += len(positives)
            query = str(row.get("query", "")).strip()
            query_digest = _digest(query) if query else None
            if query_digest is not None:
                query_digests.add(query_digest)
                queries_by_digest.setdefault(query_digest, query)
            texts: list[bytes] = []
            for position, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    continue
                text = str(candidate.get("text", "")).strip()
                if not text:
                    continue
                digest = _digest(text)
                candidate_digests.add(digest)
                texts.append(digest)
            # Evidence that a memorising router could exploit: the query text
            # together with the exact positive evidence it should retrieve.
            if query_digest is not None:
                for position in positives:
                    if 0 <= int(position) < len(texts):
                        positive_by_query[query_digest].add(texts[int(position)])
    return {
        "episodes": episodes,
        "families": dict(families),
        "categories": dict(categories),
        "per_category": {name: dict(counter) for name, counter in sorted(per_category.items())},
        "group_ids": group_ids,
        "query_digests": query_digests,
        "candidate_digests": candidate_digests,
        "positive_by_query": dict(positive_by_query),
        "queries_by_digest": queries_by_digest,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/router_training_v5/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v5/eval.jsonl")
    parser.add_argument("--output", default="router_dataset_audit.json")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    eval_path = Path(args.eval_file)
    print(f"scanning {train_path} ...", flush=True)
    train = _scan(train_path)
    print(f"scanning {eval_path} ...", flush=True)
    evaluation = _scan(eval_path)

    report = {
        "train": {key: value for key, value in train.items() if not key.endswith("_digests") and key not in {"group_ids", "positive_by_query", "queries_by_digest"}},
        "eval": {key: value for key, value in evaluation.items() if not key.endswith("_digests") and key not in {"group_ids", "positive_by_query", "queries_by_digest"}},
        "leakage": {},
    }
    overlapping_queries = evaluation["query_digests"] & train["query_digests"]
    shared_evidence = [
        digest for digest in overlapping_queries
        if train["positive_by_query"].get(digest) and evaluation["positive_by_query"].get(digest)
        and (train["positive_by_query"][digest] & evaluation["positive_by_query"][digest])
    ]
    report["leakage"] = {
        "group_id_overlap": len(train["group_ids"] & evaluation["group_ids"]),
        "eval_queries_seen_in_train": len(overlapping_queries),
        "eval_queries_total": len(evaluation["query_digests"]),
        "query_overlap_rate": len(overlapping_queries) / max(1, len(evaluation["query_digests"])),
        # The number that actually matters: a repeated query is only leakage when
        # the same query also carries the same positive evidence in both splits.
        "queries_with_shared_positive_evidence": len(shared_evidence),
        "queries_with_shared_positive_evidence_examples": [
            evaluation["queries_by_digest"].get(digest, "")[:60] for digest in shared_evidence[:5]
        ],
        "eval_candidate_texts_seen_in_train": len(evaluation["candidate_digests"] & train["candidate_digests"]),
        "eval_candidate_texts_total": len(evaluation["candidate_digests"]),
        "candidate_overlap_rate": (
            len(evaluation["candidate_digests"] & train["candidate_digests"])
            / max(1, len(evaluation["candidate_digests"]))
        ),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"families": report["eval"]["families"], "categories": report["eval"]["categories"],
                      "leakage": report["leakage"]}, ensure_ascii=False, indent=2), flush=True)
    print("\n| category | episodes | answerable | unknown | avg candidates | avg positives | hop dist |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---|", flush=True)
    for name, block in report["eval"]["per_category"].items():
        episodes = block.get("episodes", 0)
        print("| {name} | {ep} | {ans} | {unk} | {cand:.2f} | {pos:.2f} | {hop} |".format(
            name=name,
            ep=episodes,
            ans=block.get("answerable", 0),
            unk=block.get("unknown", 0),
            cand=block.get("candidate_slots", 0) / max(1, episodes),
            pos=block.get("positive_total", 0) / max(1, episodes),
            hop={key.split("_")[1]: value for key, value in block.items() if key.startswith("hop_")},
        ), flush=True)
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
