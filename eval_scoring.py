"""Answer scoring for the end-to-end memory harness (single source of truth).

Kept dependency-free on purpose so stored runs can be *re-scored offline* without
loading torch or a 4B checkpoint.

Two measurement defects in the original scorer are fixed here; both had produced
numbers that were wrong enough to misdirect the work:

1. **Whitespace-sensitivity.**  The containment check was a plain
   ``value in reply``, so ``值班人-5259`` scored correct while ``值班人 -5259``
   (one space, same identifier) scored wrong.  All 19 such cases were multi-hop,
   which is why multi-hop appeared to be a 24.00% bottleneck when it is in fact
   100.00%.  Anchors here are identifiers, versions and names, so the containment
   test now runs on a whitespace-squashed form.

2. **Keyword-only refusal detection.**  A refusal was recognised only if the
   reply contained one of a fixed list of strings.  On the `unknown_attribute`
   category the model refused 12 times out of 25 -- in its own words
   ("当前长期记忆中未包含相关信息，无法回答。") -- and *none* of those matched, so
   the unknown-refusal axis read 0.00% instead of 48.00%.  Refusal is now
   recognised by pattern.
"""

from __future__ import annotations

import re
import unicodedata

#: Legacy exact-string list, kept only so older callers keep importing something.
#: Do not use for new measurements -- it misses most real refusals.
ABSTENTION_MARKERS = ("不知道", "没有记录", "无法确认", "未找到", "不清楚", "没有相关信息", "不知道。")

REFUSAL_PATTERNS = (
    r"不知道",
    r"不清楚",
    r"没有(相关)?(的)?(记录|信息|资料|内容)",
    r"未(包含|找到|记录|提及|提供|出现|收录)(相关)?(的)?(信息|记录|内容|资料)?",
    r"未在(长期)?记忆(中)?(包含|出现|找到|记录)",
    r"无法(回答|确认|确定|提供|判断)",
    r"缺乏(具体)?(上下文|信息|线索)",
    r"证据不足",
    r"没有(足够)?(的)?证据",
    r"没有提及",
    r"记忆(中)?(没有|未)(相关)?(记录|信息)",
    # "您的协议编号未知。" -- a refusal that names no fabrication, so it counts.
    r"未知\s*[。.!！]?\s*$",
    # English refusals.  Required the moment a non-Chinese corpus is scored: without
    # these, every English refusal would count as an asserted answer -- precisely the
    # defect that made the Chinese refusal axis read 0.00% instead of 52.00%.
    r"\b(i|we)\s+(do\s*n[o']?t|don't|do\s+not)\s+(know|have|see|recall)",
    r"\b(i'?m|i\s+am)\s+not\s+(sure|certain|able)",
    r"\bno\s+(information|record|records|mention|evidence|data)\b",
    r"\bnot\s+(mentioned|stated|specified|provided|recorded|available|found|clear)\b",
    r"\b(wasn'?t|weren'?t|isn'?t|was\s+not|were\s+not|is\s+not)\s+(mentioned|stated|specified|provided|recorded)\b",
    r"\bcannot\s+be\s+(determined|answered|found|inferred)",
    r"\b(can'?t|cannot|could\s+not|couldn'?t)\s+(answer|determine|tell|find|say|recall)",
    r"\bthere\s+(is|are)\s+no\s+(mention|record|records|information)",
    r"\bunable\s+to\s+(answer|determine|find|provide|recall)",
    r"\bdoes\s+not\s+(mention|state|specify|provide|contain|include)",
    r"\bnothing\s+(in|about|regarding)\b",
    r"\bnot\s+enough\s+(information|evidence)",
    r"\bunknown\s*[.!]?\s*$",
)
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE | re.MULTILINE)
_WS = re.compile(r"\s+", re.UNICODE)


def squash(text: str) -> str:
    """Casefold + strip every whitespace run, including full-width space (U+3000)."""
    if not text:
        return ""
    normalised = unicodedata.normalize("NFKC", text)
    return _WS.sub("", normalised).casefold()


def is_refusal(reply: str) -> bool:
    """True when the reply declines to answer rather than asserting a value."""
    if not reply:
        return False
    return bool(_REFUSAL_RE.search(reply.strip()))


def score_case(case: dict, reply: str) -> dict:
    """Score one emitted reply against a case's ``acceptable`` anchors.

    ``answerable`` cases are correct when any acceptable anchor is present;
    abstention cases are also correct when the model refuses in its own words.
    """
    squashed_reply = squash(reply)
    accepted = [value for value in case["acceptable"] if value and squash(value) in squashed_reply]
    refused = is_refusal(reply)
    if case["answerable"]:
        correct = bool(accepted)
    else:
        correct = bool(accepted) or refused
    return {
        "correct": bool(correct),
        "matched": accepted[:3],
        "refused": refused,
        "abstained": (not case["answerable"]) and refused,
        "wrongly_abstained": case["answerable"] and refused and not accepted,
    }
