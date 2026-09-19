"""Re-score stored end-to-end runs offline with the current scorer.

Motivation: the first scorer used raw substring containment, so a reply that said
``值班人 -5259`` (one space) counted as wrong while ``值班人-5259`` counted as
right.  A stored run therefore cannot be compared to a later run unless both are
re-scored with the same scorer -- and re-running a 4B model to learn that costs
minutes per run.  This tool reads the stored replies, pairs them back to the
corpus case by (category, query, in-category index), and recomputes every metric.

Usage::

    python -m V2_dpskw.rescore_e2e --corpus data/realistic_v2/eval.jsonl ^
        --run rv2_new_e2e.json --run rv2_supersede_e2e.json

Prints both the legacy (raw containment) and corrected (whitespace-squashed)
verdicts side by side, so the size of the formatting artefact is explicit.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from V2_dpskw.eval_scoring import is_refusal, squash


def load_run(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = next(iter(data))
    body = data[key]
    rows = body if isinstance(body, list) else body.get("rows", [])
    if not rows:
        raise SystemExit(f"{path}: no rows found")
    return rows


def load_corpus(path: Path, per_category: int) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            category = str(metadata.get("category", "") or "(policy)")
            if len(by_category[category]) >= per_category:
                continue
            acceptable = [str(v) for v in (metadata.get("acceptable") or []) if str(v).strip()]
            candidates = [str(c.get("text", "")) for c in (row.get("candidates") or []) if isinstance(c, dict)]
            positives = [candidates[i] for i in (row.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
            by_category[category].append({
                "category": category,
                "query": str(row.get("query", "")),
                "acceptable": acceptable,
                "answerable": bool(positives),
            })
    cases: list[dict] = []
    for category in sorted(by_category):
        cases.extend(by_category[category])
    return cases


def score(reply: str, case: dict, *, squashed: bool) -> dict:
    if squashed:
        haystack = squash(reply)
        accepted = [v for v in case["acceptable"] if v and squash(v) in haystack]
    else:
        haystack = reply.lower()
        accepted = [v for v in case["acceptable"] if v and v.lower() in haystack]
    refused = is_refusal(reply)
    correct = bool(accepted) or (refused and not case["answerable"])
    return {
        "correct": bool(correct),
        "wrongly_abstained": bool(case["answerable"] and refused and not accepted),
    }


def evaluate(run_rows: list[dict], cases: list[dict], *, squashed: bool) -> dict:
    buckets: dict[str, dict] = defaultdict(lambda: {"n": 0, "ok": 0, "ans": 0, "ans_ok": 0,
                                                    "unk": 0, "unk_ok": 0, "wrong_abst": 0})
    for row, case in zip(run_rows, cases):
        b = buckets[case["category"]]
        s = score(row.get("reply", ""), case, squashed=squashed)
        b["n"] += 1
        b["ok"] += int(s["correct"])
        if case["answerable"]:
            b["ans"] += 1
            b["ans_ok"] += int(s["correct"])
            b["wrong_abst"] += int(s["wrongly_abstained"])
        else:
            b["unk"] += 1
            b["unk_ok"] += int(s["correct"])
    return buckets


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def totals(buckets: dict) -> dict:
    agg = {"n": 0, "ok": 0, "ans": 0, "ans_ok": 0, "unk": 0, "unk_ok": 0, "wrong_abst": 0}
    for b in buckets.values():
        for k in agg:
            agg[k] += b[k]
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_corpus(args.corpus, args.per_category)
    print(f"corpus: {args.corpus}  cases: {len(cases)}")
    payload = {}

    for path in args.run:
        rows = load_run(path)
        if len(rows) != len(cases):
            print(f"!! {path.name}: {len(rows)} rows vs {len(cases)} cases -- pairing may be wrong")
        mismatch = sum(1 for r, c in zip(rows, cases)
                       if r.get("category") != c["category"] or r.get("query") != c["query"])
        if mismatch:
            raise SystemExit(f"{path.name}: {mismatch} rows do not pair with the corpus; aborting")

        legacy = evaluate(rows, cases, squashed=False)
        fixed = evaluate(rows, cases, squashed=True)
        lt, ft = totals(legacy), totals(fixed)
        print(f"\n=== {path.name} ===")
        print(f"{'category':<20}{'legacy':>16}{'corrected':>16}")
        for cat in sorted(fixed):
            print(f"{cat:<20}{pct(legacy[cat]['ok'], legacy[cat]['n']):>14.2f}%{pct(fixed[cat]['ok'], fixed[cat]['n']):>15.2f}%")
        print(f"{'TOTAL':<20}{pct(lt['ok'], lt['n']):>14.2f}%{pct(ft['ok'], ft['n']):>15.2f}%")
        print(f"{'  answerable':<20}{pct(lt['ans_ok'], lt['ans']):>14.2f}%{pct(ft['ans_ok'], ft['ans']):>15.2f}%"
              f"   ({ft['ans_ok']}/{ft['ans']})")
        print(f"{'  unknown refusal':<20}{pct(lt['unk_ok'], lt['unk']):>14.2f}%{pct(ft['unk_ok'], ft['unk']):>15.2f}%"
              f"   ({ft['unk_ok']}/{ft['unk']})")
        print(f"{'  wrong abstention':<20}{pct(lt['wrong_abst'], lt['ans']):>14.2f}%{pct(ft['wrong_abst'], ft['ans']):>15.2f}%")
        payload[path.stem] = {
            "cases": ft["n"],
            "legacy_accuracy_pct": round(pct(lt["ok"], lt["n"]), 2),
            "accuracy_pct": round(pct(ft["ok"], ft["n"]), 2),
            "answerable_accuracy_pct": round(pct(ft["ans_ok"], ft["ans"]), 2),
            "unknown_refusal_pct": round(pct(ft["unk_ok"], ft["unk"]), 2),
            "wrong_abstention_pct": round(pct(ft["wrong_abst"], ft["ans"]), 2),
            "per_category": {
                c: {
                    "cases": fixed[c]["n"],
                    "legacy_accuracy_pct": round(pct(legacy[c]["ok"], legacy[c]["n"]), 2),
                    "accuracy_pct": round(pct(fixed[c]["ok"], fixed[c]["n"]), 2),
                }
                for c in sorted(fixed)
            },
        }

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
