"""Side-by-side LoCoMo comparison: memory system vs strong-RAG baselines.

Reads stored replies only -- no GPU -- and reports several metrics per system so
no conclusion rests on one brittle string match:

* **exact**        -- benchmark answer string found in the reply (the project's
  primary metric; whitespace-insensitive now).
* **token recall** -- fraction of the answer's content tokens present.  A reply
  that says "she went to an LGBTQ support group on 7 May" against the benchmark's
  "7 May 2023" is partly right; exact matching alone would hide that.
* **token all**    -- every content token present (a strict paraphrase-tolerant
  variant).
* **refusal**      -- on adversarial questions (no answer exists), did the system
  decline?  And, worse, did it *assert the trap answer* -- the plausible wrong
  answer the benchmark supplies?

Pairing between a run's rows and the corpus is verified by query equality before
anything is scored; a silent misalignment once produced a fake 0.94 AUC in this
project, so it is checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_scoring import is_refusal, squash


def load_corpus(path: Path, per_category: int = 40):
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                grouped[str((row.get("metadata") or {}).get("category", ""))].append(row)
    cases = []
    for category in sorted(grouped):
        cases.extend(grouped[category][:per_category])
    return cases


def load_runs(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for label, body in data.items():
        out[label] = body if isinstance(body, list) else body.get("rows", [])
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in squash(text).split() if t]


def score_row(row: dict, case: dict) -> dict:
    meta = case.get("metadata") or {}
    reply = row.get("reply", "")
    squashed_reply = squash(reply)
    acceptable = [str(a) for a in (meta.get("acceptable") or []) if str(a).strip()]
    exact = any(squash(a) in squashed_reply for a in acceptable)
    tokens = [str(t) for t in (meta.get("answer_tokens") or [])]
    present = [t for t in tokens if t in squashed_reply]
    token_recall = (len(present) / len(tokens)) if tokens else 0.0
    token_all = bool(tokens) and len(present) == len(tokens)
    refused = is_refusal(reply)
    trap = str(meta.get("adversarial_answer") or "")
    asserted_trap = bool(trap) and squash(trap) in squashed_reply
    return {"exact": exact, "token_recall": token_recall, "token_all": token_all,
            "refused": refused, "asserted_trap": asserted_trap,
            "answer_tokens": len(tokens)}


def evaluate(rows: list[dict], cases: list[dict]) -> dict:
    if len(rows) != len(cases):
        raise SystemExit(f"row/case count mismatch: {len(rows)} vs {len(cases)}")
    for row, case in zip(rows, cases):
        if str(row.get("query", "")).strip() != str(case.get("query", "")).strip():
            raise SystemExit(f"pairing mismatch: {row.get('query')!r} vs {case.get('query')!r}")
    buckets = defaultdict(lambda: defaultdict(list))
    for row, case in zip(rows, cases):
        s = score_row(row, case)
        cat = str((case.get("metadata") or {}).get("category", "?"))
        buckets["all"][cat].append(s)
        buckets["overall"][cat].append(s)
    return buckets


def summarize(buckets) -> dict:
    out = {}
    for scope, per_cat in buckets.items():
        flat = [s for rows in per_cat.values() for s in rows]
        if not flat:
            continue
        ans = [s for s in flat if s["answer_tokens"] or not s["asserted_trap"]]
        out[scope] = {
            "n": len(flat),
            "exact_pct": 100 * sum(s["exact"] for s in flat) / len(flat),
            "token_recall_pct": 100 * sum(s["token_recall"] for s in flat) / len(flat),
            "token_all_pct": 100 * sum(s["token_all"] for s in flat) / len(flat),
            "refused_pct": 100 * sum(s["refused"] for s in flat) / len(flat),
            "asserted_trap_pct": 100 * sum(s["asserted_trap"] for s in flat) / len(flat),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/net_locomo/eval.jsonl"))
    parser.add_argument("--run", action="append", required=True, help="LABEL=path.json")
    parser.add_argument("--per-category", type=int, default=40)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_corpus(args.corpus, args.per_category)
    print(f"corpus cases: {len(cases)}")
    answerable = sum(1 for c in cases if (c.get("metadata") or {}).get("answerable"))
    print(f"  answerable {answerable}   adversarial(no answer) {len(cases) - answerable}\n")

    payload = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs = load_runs(Path(path))
        for run_label, rows in runs.items():
            name = f"{label}:{run_label}" if len(runs) > 1 else label
            result = summarize(evaluate(rows, cases))
            payload[name] = result

    header = f"{'system':<26}{'exact':>9}{'tok-recall':>12}{'tok-all':>9}{'refused':>9}{'asserted-trap':>15}"
    print(header)
    print("-" * len(header))
    for name, result in payload.items():
        o = result.get("overall", {})
        print(f"{name:<26}{o.get('exact_pct', 0):>8.2f}%{o.get('token_recall_pct', 0):>11.2f}%"
              f"{o.get('token_all_pct', 0):>8.2f}%{o.get('refused_pct', 0):>8.2f}%"
              f"{o.get('asserted_trap_pct', 0):>14.2f}%")

    print("\nper-category exact match")
    categories = sorted({c for result in payload.values() for c in result.get("overall", {})})
    print(f"{'category':<16}" + "".join(f"{name[:18]:>20}" for name in payload))
    for cat in categories:
        line = f"{cat:<16}"
        for name in payload:
            v = payload[name].get("overall", {}).get(cat)
            line += f"{v['exact_pct']:>19.2f}%" if v else f"{'-':>20}"
        print(line)

    print("\nper-category refusal rate (adversarial is the axis that matters)")
    print(f"{'category':<16}" + "".join(f"{name[:18]:>20}" for name in payload))
    for cat in categories:
        line = f"{cat:<16}"
        for name in payload:
            v = payload[name].get("overall", {}).get(cat)
            line += f"{v['refused_pct']:>19.2f}%" if v else f"{'-':>20}"
        print(line)

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
