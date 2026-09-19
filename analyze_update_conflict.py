"""Decide whether `update_conflict` is ill-posed or the system ignores recency.

Attribute membership is decided with the generator's own frame templates, so a
fact phrased as ``我的代码仓库是 X。`` and one phrased as ``我源码放在 X。`` are
both recognised as belonging to the same attribute -- surface-shape matching
alone misses this, which is exactly the trap this category is built around.

The harness writes ``positives first, then the remaining candidates in corpus
order`` (``eval_end_to_end_memory.build``/``run_router``), so the *last* write
wins on recency.  For each case this reports:

  * which written fact is the newest one for the asked attribute
  * whether that newest fact is the expected answer
  * what the model actually answered, mapped back to a written fact
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_scoring import squash
from V2_dpskw.make_realistic_memory_data import ATTRIBUTE_POOL
from V2_dpskw.rescore_e2e import load_run

FRAME_RE: dict[str, list[re.Pattern]] = {}
for _name, _frames, _questions in ATTRIBUTE_POOL:
    pats = []
    for _f in _frames:
        # Split the *raw* template on the placeholders and escape each literal part
        # separately.  Escaping first would turn ``{s}`` into ``\{s\}`` and the
        # split pattern would no longer match it.
        parts = re.split(r"\{[sv]\}", _f)
        parts = [re.escape(p) for p in parts]
        pats.append(re.compile("".join(p if i == 0 else r"(.+?)" + p for i, p in enumerate(parts)) + r"\Z"))
    FRAME_RE[_name] = pats


def belongs_to(text: str, attribute: str) -> bool:
    t = text.strip()
    return any(p.search(t) for p in FRAME_RE.get(attribute, []))


def value_in(text: str, attribute: str):
    """The ``{v}`` slot of the first frame that matches.

    Frames carry two placeholders (``{s}`` subject and ``{v}`` value), so the
    value is the *last* captured group, not the first.
    """
    for p in FRAME_RE.get(attribute, []):
        m = p.search(text.strip())
        if m:
            groups = [g for g in m.groups() if g]
            return groups[-1].strip() if groups else None
    return None


def load_cases(path: Path, per_category: int = 25):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    grouped = defaultdict(list)
    for row in rows:
        grouped[str((row.get("metadata") or {}).get("category", ""))].append(row)
    out = []
    for category in sorted(grouped):
        for row in grouped[category][:per_category]:
            meta = row.get("metadata") or {}
            candidates = [str(c.get("text", "")) for c in (row.get("candidates") or []) if isinstance(c, dict)]
            positives = [candidates[i] for i in (row.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
            out.append({
                "category": category,
                "query": str(row.get("query", "")),
                "acceptable": [str(v) for v in (meta.get("acceptable") or []) if str(v).strip()],
                "positives": positives,
                "candidates": candidates,
                "meta": meta,
            })
    return out


def main(corpus, run, category="update_conflict", max_facts=6, verbose=10):
    cases = load_cases(Path(corpus))
    rows = load_run(Path(run))
    stats = Counter()
    examples = []
    for case, row in zip(cases, rows):
        if case["category"] != category:
            continue
        attribute = case["meta"].get("attribute", "")
        order = (case["positives"] + [f for f in case["candidates"] if f not in case["positives"]])[:max_facts]
        same_attr = [f for f in order if belongs_to(f, attribute)]
        newest = same_attr[-1] if same_attr else None
        newest_value = value_in(newest, attribute) if newest else None
        answer_value = case["meta"].get("answer", "")
        reply = row.get("reply", "")
        s_reply = squash(reply)

        stats["cases"] += 1
        stats[f"same_attr_written_{len(same_attr)}"] += 1
        newest_is_answer = newest is not None and newest in case["positives"]
        if newest_is_answer:
            stats["newest_is_expected_answer"] += 1
        else:
            stats["newest_is_NOT_answer"] += 1
        answered_expected = bool(answer_value) and squash(answer_value) in s_reply
        answered_newest = bool(newest_value) and squash(newest_value) in s_reply
        if answered_expected:
            stats["answered_expected"] += 1
        if newest_is_answer:
            if answered_expected:
                stats["posed_correct"] += 1
            else:
                stats["posed_wrong"] += 1
        else:
            # the expected answer is not the freshest evidence for the attribute
            stats["illposed_cases"] += 1
            if answered_newest:
                stats["illposed_answered_newest"] += 1
            elif answered_expected:
                stats["illposed_answered_expected"] += 1
            else:
                stats["illposed_answered_neither"] += 1
        if len(examples) < verbose:
            examples.append((case, row, order, same_attr, newest, newest_value, answered_expected, answered_newest))

    print(f"category={category}")
    for k in sorted(stats):
        print(f"  {k:<34}{stats[k]}")
    print("\n--- examples ---")
    for case, row, order, same_attr, newest, newest_value, ae, an in examples:
        print(f"Q {case['query']}   attribute={case['meta'].get('attribute')}  expect={case['acceptable']}")
        for i, f in enumerate(order):
            tags = []
            if f in case["positives"]:
                tags.append("POSITIVE")
            if f in same_attr:
                tags.append("SAME-ATTR")
            if f == newest:
                tags.append("NEWEST")
            print(f"    {i} {f}   [{' '.join(tags)}]")
        print(f"    reply={row.get('reply','')[:64]!r}")
        print(f"    answered_expected={ae}  answered_newest={an}  newest_value={newest_value!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
