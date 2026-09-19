"""Validate the refusal detector in both directions.

Over-firing would inflate the known-question false-refusal rate; under-firing
would inflate the unknown-refusal rate.  Both directions are printed with the
verbatim reply so the judgement can be checked by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_scoring import is_refusal, squash
from V2_dpskw.rescore_e2e import load_corpus, load_run


def main(corpus, run):
    cases = load_corpus(Path(corpus), 25)
    rows = load_run(Path(run))
    unknown_refused, unknown_asserted = [], []
    ans_refused, ans_refused_but_answered = [], []
    for case, row in zip(cases, rows):
        reply = row.get("reply", "")
        refused = is_refusal(reply)
        hit = [v for v in case["acceptable"] if v and squash(v) in squash(reply)]
        if not case["answerable"]:
            (unknown_refused if refused else unknown_asserted).append((case, row))
        elif refused:
            (ans_refused_but_answered if hit else ans_refused).append((case, row))

    print(f"UNANSWERABLE cases where the detector says REFUSED ({len(unknown_refused)})")
    for case, row in unknown_refused:
        print(f"  Q {case['query']}\n    R {row.get('reply','')[:110]}")

    print(f"\nUNANSWERABLE cases where the detector says ASSERTED ({len(unknown_asserted)})")
    for case, row in unknown_asserted:
        print(f"  Q {case['query']}\n    R {row.get('reply','')[:110]}")

    print(f"\nANSWERABLE cases flagged as refusals ({len(ans_refused) + len(ans_refused_but_answered)})")
    for tag, group in (("ALSO-MATCHED-ANCHOR (harmless)", ans_refused_but_answered), ("COUNTED AS FALSE REFUSAL", ans_refused)):
        print(f"  -- {tag}: {len(group)}")
        for case, row in group:
            print(f"     Q {case['query']}\n       R {row.get('reply','')[:110]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
