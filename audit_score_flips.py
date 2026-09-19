"""Audit every case whose verdict flips when whitespace is ignored.

A flip is only legitimate if the anchor really is present modulo whitespace --
this prints the raw anchor and the raw reply so the difference can be eyeballed.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_scoring import squash
from V2_dpskw.rescore_e2e import load_corpus, load_run


def main(corpus, run, per_category=25):
    cases = load_corpus(Path(corpus), per_category)
    rows = load_run(Path(run))
    assert len(rows) == len(cases)
    flips = []
    for row, case in zip(rows, cases):
        reply = row.get("reply", "")
        legacy = [v for v in case["acceptable"] if v and v.lower() in reply.lower()]
        fixed = [v for v in case["acceptable"] if v and squash(v) in squash(reply)]
        if bool(legacy) != bool(fixed):
            flips.append((case, reply, legacy, fixed))
    print(f"flips: {len(flips)}")
    by_cat = defaultdict(int)
    for case, reply, legacy, fixed in flips:
        by_cat[case["category"]] += 1
    print(dict(by_cat))
    bad = 0
    for case, reply, legacy, fixed in flips:
        if not fixed:
            print(f"!! now WRONG (was right): {case['category']} {case['query']}")
            print(f"   anchors={case['acceptable']!r}\n   reply={reply!r}")
            bad += 1
            continue
        anchor = fixed[0]
        # show the neighbourhood of the anchor inside the reply
        s_reply, s_anchor = squash(reply), squash(anchor)
        pos = s_reply.find(s_anchor)
        print(f"{case['category']:<18} anchor={anchor!r}")
        print(f"   reply = {reply!r}")
        print(f"   match@{pos} (raw anchor absent: {anchor not in reply})")
    print(f"\nunexplained (now-wrong) flips: {bad}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
