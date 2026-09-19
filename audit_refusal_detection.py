"""Audit the unknown-refusal axis: are the 0/25 refusals real, or misdetected?

`ABSTENTION_MARKERS` is a fixed keyword list ("不知道", "没有记录", "无法确认",
"未找到", "不清楚", "没有相关信息").  A reply that says
"当前长期记忆中未包含相关信息，无法回答。" is a refusal in plain Chinese but
contains none of those exact strings, so it is scored as a confident wrong answer.

This prints every unknown_category reply so the true refusal rate can be counted
by hand, and groups them by which abstention signal (if any) they carry.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_scoring import ABSTENTION_MARKERS
from V2_dpskw.rescore_e2e import load_corpus, load_run

# Refusal shapes actually observed in this project's replies, expressed as
# patterns rather than exact strings.
REFUSAL_PATTERNS = [
    r"不知道",
    r"没有(相关)?(记录|信息)",
    r"未(包含|找到|记录|提及|提供)",
    r"无法(回答|确认|确定|提供)",
    r"不清楚",
    r"缺乏(具体)?(上下文|信息)",
    r"未(曾)?(在)?(长期)?记忆中(包含|出现|找到|记录)",
    r"证据不足",
    r"无法(从|根据)(记忆|证据)",
    r"没有(足够)?(的)?证据",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS))


def looks_like_refusal(reply: str) -> bool:
    return bool(REFUSAL_RE.search(reply))


def main(corpus, run, category="unknown_attribute"):
    cases = load_corpus(Path(corpus), 25)
    rows = load_run(Path(run))
    n = 0
    markers_hit = 0
    pattern_hit = 0
    for case, row in zip(cases, rows):
        if case["category"] != category:
            continue
        n += 1
        reply = row.get("reply", "")
        m = [x for x in ABSTENTION_MARKERS if x in reply]
        p = bool(looks_like_refusal(reply))
        markers_hit += int(bool(m))
        pattern_hit += int(p)
        print(f"[{n:02d}] markers={m or '-'} refusal_pattern={p}")
        print(f"     Q: {case['query']}")
        print(f"     R: {reply}")
    print(f"\n{category}: {n} cases")
    print(f"  detected by keyword list : {markers_hit}/{n}")
    print(f"  detected by pattern set  : {pattern_hit}/{n}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
