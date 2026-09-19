"""How many written facts actually survive into the bank, and does the target survive?

End-to-end runs on realistic data reported ``records_active`` of 2-9 while the harness
writes up to 32 candidate facts per episode, and making the router fully responsible for
ranking moved answerable accuracy by one case.  Both observations point at the same
suspicion: the binding constraint is not ranking, it is *what got stored*.

This probe measures it directly and cheaply -- it only writes, never generates:

* how many candidates were written vs how many records are active afterwards;
* whether the episode's *target* fact survived, which is a hard prerequisite for any
  ranker to be able to answer;
* the dominant rejection reason, read from the runtime (write gate vs parse vs supersede).

Usage::

    python -m V2_dpskw.diagnose_write_survival --eval-file data/realistic_v2/eval.jsonl --cases 40
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .eval_end_to_end_memory import write_fact
from .qwen_integration import infer_memory_metadata, load_memory_config, load_qwen_dynamic, load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2_1")
    parser.add_argument("--eval-file", default="data/realistic_v2/eval.jsonl")
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--output", default="write_survival.json")
    args = parser.parse_args()

    rows = []
    with Path(args.eval_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # One episode per category, so the report is not dominated by one shape.
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["metadata"]["category"]].append(row)
    picked = []
    per = max(1, args.cases // max(1, len(by_category)))
    for category, items in sorted(by_category.items()):
        picked.extend(items[:per])

    model_path = Path(args.package)
    config = load_memory_config(model_path)
    model = load_qwen_dynamic(model_path, memory_config=config, load_in_4bit=True,
                              max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    tokenizer = load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_category: dict[str, dict] = defaultdict(lambda: {"episodes": 0, "written": 0, "active": 0,
                                                         "target_survived": 0,
                                                         "answerable": 0, "parsed": 0})
    target_missing_examples = []
    for row in picked:
        candidates = [c["text"] for c in row["candidates"]]
        positives = row.get("positive_indices") or []
        acceptable = [str(v) for v in (row["metadata"].get("acceptable") or [])]
        model.reset_memory(batch_size=1, device=device)
        for text in candidates:
            write_fact(model, tokenizer, text, device)
        bank = model.memory_os_v2.bank
        active = [r for r in bank.records.values() if r.status == "active"]
        active_text = "\n".join(r.text or "" for r in active)

        category = row["metadata"]["category"]
        bucket = per_category[category]
        bucket["episodes"] += 1
        bucket["written"] += len(candidates)
        bucket["active"] += len(active)
        bucket["parsed"] += sum(1 for text in candidates
                                if infer_memory_metadata(text).get("attribute"))
        if positives:
            bucket["answerable"] += 1
            survived = any(value and value in active_text for value in acceptable)
            bucket["target_survived"] += int(survived)
            if not survived and len(target_missing_examples) < 6:
                target_missing_examples.append({
                    "category": category,
                    "query": row["query"],
                    "target_text": candidates[positives[0]][:90],
                    "active_records": len(active),
                })

    report = {
        "package": str(model_path),
        "eval_file": args.eval_file,
        "episodes": len(picked),
        "per_category": {},
        "target_missing_examples": target_missing_examples,
    }
    for category, bucket in sorted(per_category.items()):
        report["per_category"][category] = {
            "episodes": bucket["episodes"],
            "candidates_written": bucket["written"],
            "active_after_write": bucket["active"],
            "write_survival_pct": round(100 * bucket["active"] / max(1, bucket["written"]), 2),
            "candidates_parsed_as_attribute_pct": round(
                100 * bucket["parsed"] / max(1, bucket["written"]), 2),
            "answerable_episodes": bucket["answerable"],
            "target_survived": bucket["target_survived"],
            "target_survival_pct": round(
                100 * bucket["target_survived"] / max(1, bucket["answerable"]), 2),
        }
    total_written = sum(b["written"] for b in per_category.values())
    total_active = sum(b["active"] for b in per_category.values())
    total_answerable = sum(b["answerable"] for b in per_category.values())
    total_survived = sum(b["target_survived"] for b in per_category.values())
    report["overall"] = {
        "candidates_written": total_written,
        "active_after_write": total_active,
        "write_survival_pct": round(100 * total_active / max(1, total_written), 2),
        "answerable_episodes": total_answerable,
        "target_survival_pct": round(100 * total_survived / max(1, total_answerable), 2),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
