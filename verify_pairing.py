"""Prove that stored run rows pair correctly with corpus cases.

Pairing is by position (``build_cases`` emits categories in sorted order, file
order inside a category), so any drift between the corpus as it is *now* and the
corpus as it was *when the run happened* would silently pair the wrong anchor
with the wrong reply -- and repeated queries inside a category would hide it.

The stored rows carry ``matched``, the list the legacy scorer produced at run
time.  Recomputing it from the paired case must reproduce that list exactly;
that is an independent witness of correct pairing.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.rescore_e2e import load_corpus, load_run


def legacy_accept(reply: str, case: dict) -> list[str]:
    lowered = reply.lower()
    return [v for v in case["acceptable"] if v and v.lower() in lowered][:3]


def check(corpus_path: Path, run_path: Path, per_category: int = 25) -> bool:
    cases = load_corpus(corpus_path, per_category)
    rows = load_run(run_path)
    print(f"{run_path.name}: {len(rows)} rows / {len(cases)} cases")
    if len(rows) != len(cases):
        print("  !! length mismatch")
        return False
    bad = []
    for i, (row, case) in enumerate(zip(rows, cases)):
        expect = legacy_accept(row.get("reply", ""), case)
        got = row.get("matched") or []
        if expect != got:
            bad.append((i, case, row, expect, got))
    if not bad:
        print("  pairing VERIFIED: every stored 'matched' field reproduces exactly")
        return True
    print(f"  !! {len(bad)} rows disagree with the corpus")
    per_cat = defaultdict(int)
    for _, case, _, _, _ in bad:
        per_cat[case["category"]] += 1
    print(f"  disagreement by category: {dict(per_cat)}")
    for i, case, row, expect, got in bad[:6]:
        print(f"   row#{i} cat={case['category']}")
        print(f"     corpus query={case['query']!r} acceptable={case['acceptable']!r}")
        print(f"     row    query={row.get('query')!r}")
        print(f"     expect matched={expect!r} stored={got!r}")
    return False


if __name__ == "__main__":
    ok = True
    for run in sys.argv[2:]:
        ok &= check(Path(sys.argv[1]), Path(run))
    print("\nALL PAIRINGS OK" if ok else "\nPAIRING PROBLEM")
