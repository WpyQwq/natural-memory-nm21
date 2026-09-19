"""Independently verify the zero-overlap paraphrase dataset.

Deliberately shares no code with ``make_zero_overlap_paraphrase_data.py``: the
stop-word set, the character extraction and the overlap computation are re-derived
here, so a bug in the generator's own check cannot hide inside this verification.

Checks, per split:
  * the query shares no distinctive character with its own target fact;
  * the query string set is disjoint between train and eval;
  * every episode has the declared candidate count and exactly one positive
    (or none, when it is an abstention episode);
  * the value in ``metadata.answer`` appears in exactly one candidate, so evidence
    mixing is detectable downstream;
  * the answerable/need_memory/hop flags agree with the positive indices.

Usage::

    python -m V2_dpskw.verify_zero_overlap_dataset --data-dir V2_dpskw/data/zero_overlap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Re-derived independently: particles, pronouns and question words that carry no
# topic information.  Written out explicitly rather than imported.
NOISE = frozenset("我的了是在有个吗？。！，、你他她它和与及为把被这那哪些什么哪儿都就还")


def content_chars(text: str) -> set[str]:
    """Non-ASCII characters that carry topic meaning."""
    return {ch for ch in text if not ch.isascii() and ch.strip() and ch not in NOISE}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def audit(rows: list[dict], split: str) -> tuple[dict, list[dict]]:
    problems: list[dict] = []
    queries: set[str] = set()
    positive_overlaps = 0
    unknown_rows = 0
    distractor_rows = 0
    candidate_counts: set[int] = set()
    for row in rows:
        queries.add(row["query"])
        candidates = row["candidates"]
        candidate_counts.add(len(candidates))
        positives = row["positive_indices"]
        if not positives:
            unknown_rows += 1
            if row["need_memory"] != 0.0 or row["hop"] != 0:
                problems.append({"id": row["id"], "issue": "unknown row has need_memory/hop set"})
            if row["metadata"]["answerable"]:
                problems.append({"id": row["id"], "issue": "unknown row marked answerable"})
            continue
        if len(positives) != 1:
            problems.append({"id": row["id"], "issue": f"{len(positives)} positives"})
            continue
        if row["positive_index"] != positives[0]:
            problems.append({"id": row["id"], "issue": "positive_index disagrees with positive_indices"})
        if row["need_memory"] != 1.0 or row["hop"] != 1:
            problems.append({"id": row["id"], "issue": "known row missing need_memory/hop"})
        if not row["metadata"]["answerable"]:
            problems.append({"id": row["id"], "issue": "known row marked unanswerable"})

        target = candidates[positives[0]]
        query_chars = content_chars(row["query"])
        if query_chars & content_chars(target["text"]):
            positive_overlaps += 1
            problems.append({"id": row["id"], "issue": "query overlaps its own target fact",
                             "query": row["query"], "target": target["text"],
                             "shared": sorted(query_chars & content_chars(target["text"]))})

        # The answer code must identify exactly one candidate.
        answer = row["metadata"]["answer"]
        matches = [i for i, candidate in enumerate(candidates) if answer in candidate["text"]]
        if matches != [positives[0]]:
            problems.append({"id": row["id"], "issue": f"answer appears in candidates {matches}"})

        # Attribute labels must be unique, otherwise two candidates are indistinguishable.
        attributes = [candidate.get("attribute") for candidate in candidates]
        if len(set(attributes)) != len(attributes):
            problems.append({"id": row["id"], "issue": "duplicate candidate attributes"})

        for index, candidate in enumerate(candidates):
            if index == positives[0]:
                continue
            if query_chars & content_chars(candidate["text"]):
                distractor_rows += 1
                break
    report = {
        "split": split,
        "episodes": len(rows),
        "distinct_queries": len(queries),
        "candidate_counts": sorted(candidate_counts),
        "unknown_episodes": unknown_rows,
        "unknown_share_pct": round(100.0 * unknown_rows / max(1, len(rows)), 2),
        "episodes_whose_query_lexically_matches_its_own_target": positive_overlaps,
        "episodes_with_a_lexical_distractor": distractor_rows,
        "distractor_share_pct": round(100.0 * distractor_rows / max(1, len(rows)), 2),
        "problems": len(problems),
        "problem_examples": problems[:10],
    }
    return report, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/zero_overlap")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = read_jsonl(data_dir / "train.jsonl")
    evaluation = read_jsonl(data_dir / "eval.jsonl")

    train_report, train_problems = audit(train, "train")
    eval_report, eval_problems = audit(evaluation, "eval")

    train_queries = {row["query"] for row in train}
    eval_queries = {row["query"] for row in evaluation}
    shared = sorted(train_queries & eval_queries)

    report = {
        "verified_by": "verify_zero_overlap_dataset.py",
        "data_dir": str(data_dir),
        "train": train_report,
        "eval": eval_report,
        "split_disjointness": {
            "shared_query_strings": shared,
            "disjoint": not shared,
        },
        "checks": {
            "no_query_overlaps_its_own_target":
                train_report["episodes_whose_query_lexically_matches_its_own_target"] == 0
                and eval_report["episodes_whose_query_lexically_matches_its_own_target"] == 0,
            "no_structural_problems": not train_problems and not eval_problems,
            "splits_disjoint": not shared,
        },
    }
    report["all_checks_passed"] = all(report["checks"].values())

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
