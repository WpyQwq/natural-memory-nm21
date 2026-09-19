"""Is there an open-vocabulary signal for "the bank holds the asked attribute"?

Before training any head, measure the cheapest candidate: the maximum cosine
similarity between the query key and the keys of the records that would be in the
bank.  If plain cosine already separates answerable from unanswerable episodes, a
learned gate is unnecessary; if it does not, the AUC sets the bar a learned gate
has to beat.

Reported per category (so the answer for `unknown_attribute` is visible on its
own) and as an overall AUC, plus the operating point of the best threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.make_answerability_data import FeatureBank, write_set


def load_episodes(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC with ties handled by average rank."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank = 0.0
    i = 0
    rank_sum_pos = 0.0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum_pos += avg_rank
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path(r"H:\Memory\nm_cache\nm_realistic_v2\feature_cache"))
    parser.add_argument("--corpus", type=Path, default=Path("data/realistic_v2/eval.jsonl"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    bank = FeatureBank(args.bank)
    scores, labels, cats = [], [], []
    features: dict[str, list[float]] = {"max": [], "margin": [], "zscore": []}
    per_cat = defaultdict(list)
    skipped = 0

    for episode in load_episodes(args.corpus):
        metadata = episode.get("metadata") or {}
        category = str(metadata.get("category", ""))
        candidates = [str(c.get("text", "")) for c in (episode.get("candidates") or []) if isinstance(c, dict)]
        positives = [candidates[i] for i in (episode.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
        records = write_set(episode, include_answer=True)
        qrow = bank.row(str(episode.get("query", "")))
        rows = [bank.row(t) for t in records]
        if qrow is None or not rows or any(r is None for r in rows):
            skipped += 1
            continue
        q = bank.key(qrow)
        q = q / (q.norm() + 1e-6)
        R = torch.stack([bank.key(r) for r in rows])
        R = R / (R.norm(dim=-1, keepdim=True) + 1e-6)
        sims = (R @ q)
        ranked = torch.sort(sims, descending=True).values
        best = float(ranked[0])
        margin = float(ranked[0] - ranked[1]) if ranked.numel() > 1 else float(ranked[0])
        # A record that *is* about the asked attribute should also be unusually
        # similar relative to the whole record set, not just its best neighbour.
        zscore = float((ranked[0] - sims.mean()) / (sims.std() + 1e-6)) if sims.numel() > 1 else 0.0
        label = 1 if positives else 0
        scores.append(best)
        labels.append(label)
        features["max"].append(best)
        features["margin"].append(margin)
        features["zscore"].append(zscore)
        cats.append(category)
        per_cat[category].append((best, label))

    overall = auc(scores, labels)
    print(f"episodes scored: {len(labels)}  (skipped {skipped})")
    print(f"OVERALL AUC of max-cosine            : {overall:.4f}")
    for name, values in features.items():
        print(f"OVERALL AUC of {name:<21}: {auc(values, labels):.4f}")
    print()
    print(f"{'category':<20}{'n':>5}{'pos':>5}{'mean_pos':>10}{'mean_neg':>10}{'auc':>8}")
    payload = {"overall_auc": round(overall, 4), "per_category": {}}
    for name in sorted(per_cat):
        rows = per_cat[name]
        pos = [s for s, l in rows if l]
        neg = [s for s, l in rows if not l]
        a = auc([s for s, _ in rows], [l for _, l in rows])
        mp = float(np.mean(pos)) if pos else float("nan")
        mn = float(np.mean(neg)) if neg else float("nan")
        print(f"{name:<20}{len(rows):>5}{len(pos):>5}{mp:>10.4f}{mn:>10.4f}{a:>8.3f}")
        payload["per_category"][name] = {"n": len(rows), "auc": None if a != a else round(a, 4),
                                         "mean_pos": None if mp != mp else round(mp, 4),
                                         "mean_neg": None if mn != mn else round(mn, 4)}

    # best operating point for refusing the unanswerable while keeping answerable.
    # The margin is the only feature with usable separation (AUC 0.94 vs 0.67 for
    # raw max cosine), so the sweep runs on it.
    print(f"\n{'thr(margin)':>12}{'refused_unknown':>18}{'false_refusal':>16}{'kept_answerable':>18}")
    best_row = None
    for t in [x / 1000 for x in range(0, 200, 10)]:
        unk = [(m, l) for m, l in zip(features["margin"], labels) if l == 0]
        ans = [(m, l) for m, l in zip(features["margin"], labels) if l == 1]
        refused_unknown = sum(1 for m, _ in unk if m < t) / max(1, len(unk))
        false_refusal = sum(1 for m, _ in ans if m < t) / max(1, len(ans))
        print(f"{t:>12.3f}{refused_unknown * 100:>17.2f}%{false_refusal * 100:>15.2f}%"
              f"{(1 - false_refusal) * 100:>17.2f}%")
        # Prefer the point that refuses the most unanswerable while holding false
        # refusals at or below 4% (the 'known question' promise).
        if false_refusal <= 0.04 and (best_row is None or refused_unknown > best_row[2]):
            best_row = (refused_unknown - 2 * false_refusal, t, refused_unknown, false_refusal)
    if best_row is None:
        print("\nno threshold holds false refusal <= 4.00%")
        best_row = (0.0, float("nan"), 0.0, 1.0)
    else:
        print(f"\nbest threshold holding false_refusal <= 4.00%: margin >= {best_row[1]:.3f}"
              f"  -> refused_unknown={best_row[2] * 100:.2f}%  false_refusal={best_row[3] * 100:.2f}%")
    payload["best_threshold"] = {"margin": best_row[1], "refused_unknown_pct": round(best_row[2] * 100, 2),
                                 "false_refusal_pct": round(best_row[3] * 100, 2)}
    payload["auc"] = {name: round(auc(values, labels), 4) for name, values in features.items()}
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
