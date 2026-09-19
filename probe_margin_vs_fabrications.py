"""Would a margin gate fix the fabrications the model commits, or only echo it?

The model already refuses 13 of the 25 unknown-attribute questions in its own
words.  A gate is only worth wiring into the runtime if it catches the *other* 12
-- the ones where the model fabricates a value -- without paying many extra false
refusals on answerable questions.

So this reports, for the stored run's unknown-attribute cases:

* whether the reply was a refusal (model handled it) or a fabrication (gate needed)
* the offline margin for that episode
* the same for answerable episodes, as the false-refusal denominator

and then sweeps the threshold to show refusals_added vs false_refusals_added.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_scoring import is_refusal
from V2_dpskw.fit_answerability_threshold import load_episodes, margin_of
from V2_dpskw.make_answerability_data import FeatureBank, write_set
from V2_dpskw.rescore_e2e import load_run


def load_episodes_sorted(path: Path, per_category: int = 25):
    """Corpus rows in the order the harness ran them: sorted category, then file order.

    Pairing a stored run against a raw-file-order corpus silently misaligns every
    row -- repeated queries inside a category hide it -- so the ordering used by
    ``eval_end_to_end_memory.build_cases`` is reproduced here explicitly.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for episode in load_episodes(path):
        grouped[str((episode.get("metadata") or {}).get("category", ""))].append(episode)
    out = []
    for category in sorted(grouped):
        out.extend(grouped[category][:per_category])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path(r"H:\Memory\nm_cache\nm_realistic_v2\feature_cache"))
    parser.add_argument("--corpus", type=Path, default=Path("data/realistic_v2/eval.jsonl"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    bank = FeatureBank(args.bank)
    rows = load_run(args.run)

    unknown = []   # (margin, model_refused, query, reply)
    answerable = []  # (margin, model_answered_correctly)
    for episode, row in zip(load_episodes_sorted(args.corpus), rows):
        if str(episode.get("query", "")) != str(row.get("query", "")):
            raise SystemExit(f"row/corpus mismatch: {episode.get('query')!r} vs {row.get('query')!r}")
        metadata = episode.get("metadata") or {}
        candidates = [str(c.get("text", "")) for c in (episode.get("candidates") or []) if isinstance(c, dict)]
        positives = [candidates[i] for i in (episode.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
        records = write_set(episode, include_answer=True)
        m = margin_of(bank, str(episode.get("query", "")), records)
        if m is None:
            continue
        reply = row.get("reply", "")
        if positives:
            answerable.append((m, bool(row.get("correct"))))
        else:
            unknown.append((m, is_refusal(reply), str(episode.get("query", "")), reply))

    refused_by_model = [u for u in unknown if u[1]]
    fabricated = [u for u in unknown if not u[1]]
    print(f"unknown-attribute cases with a usable margin: {len(unknown)}")
    print(f"  model already refused : {len(refused_by_model)}   margins "
          f"{sorted(round(u[0], 3) for u in refused_by_model)}")
    print(f"  model FABRICATED      : {len(fabricated)}   margins "
          f"{sorted(round(u[0], 3) for u in fabricated)}")
    print("\nfabricated cases:")
    for m, _r, q, reply in fabricated:
        print(f"  margin={m:+.3f}  Q={q}  R={reply[:60]}")

    print(f"\n{'thr':>7}{'caught_fabrications':>21}{'reflagged_refusals':>21}{'new_false_refusals':>21}")
    ans_margins = [m for m, _ok in answerable]
    for step in range(0, 80, 4):
        t = step / 1000.0
        caught = sum(1 for m, _r, _q, _x in fabricated if m < t)
        reflagged = sum(1 for m, _r, _q, _x in refused_by_model if m < t)
        new_false = sum(1 for m in ans_margins if m < t)
        print(f"{t:>7.3f}{caught:>16}/{len(fabricated):<4}{reflagged:>16}/{len(refused_by_model):<4}"
              f"{new_false:>16}/{len(ans_margins):<4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
