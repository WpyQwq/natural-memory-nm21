"""Fit the answerability margin threshold on train families, evaluate on eval.

`probe_answerability_cosine` swept thresholds on the eval set itself, which is
in-sample and therefore optimistic.  This script does it properly:

1. build the same POSITIVE / NEGATIVE episode variants the head training uses
   (positives = answering facts present; negatives = only records that provably
   belong to *other* attribute families), from `train.jsonl`;
2. pick the margin threshold that refuses the most unanswerable episodes while
   holding false refusals at or below a budget;
3. apply that frozen threshold to `eval.jsonl`, whose attribute families never
   appear in train.

The margin is ``cos(query, best record) - cos(query, second best record)`` -- a
training-free, open-vocabulary signal, which matters because the shipped gate is
a closed 24-class classifier that disables itself outside its vocabulary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.make_answerability_data import FeatureBank, attribute_of_question, write_set


def load_episodes(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def variants(episode: dict):
    """(records, label, kind) triples for one episode.

    Three populations, deliberately kept separate because they are different
    questions:

    * ``answerable``      -- the answering fact is in the bank (label 1)
    * ``unknown``         -- an unknown-attribute episode as constructed: the bank
                             holds only other-family facts (label 0).  This is the
                             deployment case the gate exists for.
    * ``answer_removed``  -- an answerable episode with its answer stripped out,
                             leaving same-shaped records (label 0).  A stress case:
                             the records are drawn from the same episodes, so the
                             margin is only informative if the signal is real.
    """
    candidates = [str(c.get("text", "")) for c in (episode.get("candidates") or []) if isinstance(c, dict)]
    positives = [candidates[i] for i in (episode.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
    out = []
    if positives:
        out.append((write_set(episode, include_answer=True), 1, "answerable"))
        neg = write_set(episode, include_answer=False)
        if neg:
            out.append((neg, 0, "answer_removed"))
    else:
        records = write_set(episode, include_answer=True)
        if records:
            out.append((records, 0, "unknown"))
    return out


def margin_of(bank: FeatureBank, query: str, records: list[str]) -> float | None:
    qrow = bank.row(query)
    rows = [bank.row(t) for t in records]
    if qrow is None or len(rows) < 2 or any(r is None for r in rows):
        return None
    q = bank.key(qrow)
    q = q / (q.norm() + 1e-6)
    R = torch.stack([bank.key(r) for r in rows])
    R = R / (R.norm(dim=-1, keepdim=True) + 1e-6)
    sims = torch.sort(R @ q, descending=True).values
    return float(sims[0] - sims[1])


def collect(bank: FeatureBank, corpus: Path):
    margins, labels, kinds, cats = [], [], [], []
    for episode in load_episodes(corpus):
        category = str((episode.get("metadata") or {}).get("category", ""))
        for records, label, kind in variants(episode):
            m = margin_of(bank, str(episode.get("query", "")), records)
            if m is None:
                continue
            margins.append(m)
            labels.append(label)
            kinds.append(kind)
            cats.append(category)
    return margins, labels, kinds, cats


def auc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        rank_sum_pos += avg_rank * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path(r"H:\Memory\nm_cache\nm_realistic_v2\feature_cache"))
    parser.add_argument("--train", type=Path, default=Path("data/realistic_v2/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/realistic_v2/eval.jsonl"))
    parser.add_argument("--false-refusal-budget", type=float, default=0.04)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    bank = FeatureBank(args.bank)
    tr_m, tr_l, tr_k, tr_c = collect(bank, args.train)
    ev_m, ev_l, ev_k, ev_c = collect(bank, args.eval)
    for name, m, l, k in (("train", tr_m, tr_l, tr_k), ("eval", ev_m, ev_l, ev_k)):
        counts = defaultdict(int)
        for kind in k:
            counts[kind] += 1
        print(f"{name}: {len(l)} variants {dict(counts)}")

    # AUC against the deployment population only: can the margin tell an
    # answerable episode from an unknown-attribute one?
    def sub(m, l, k, kinds):
        return ([x for x, kk in zip(m, k) if kk in kinds],
                [x for x, kk in zip(l, k) if kk in kinds])

    tr_md, tr_ld = sub(tr_m, tr_l, tr_k, {"answerable", "unknown"})
    ev_md, ev_ld = sub(ev_m, ev_l, ev_k, {"answerable", "unknown"})
    print(f"\nAUC (answerable vs unknown)  train: {auc(tr_md, tr_ld):.4f}   eval: {auc(ev_md, ev_ld):.4f}")
    tr_ms, tr_ls = sub(tr_m, tr_l, tr_k, {"answerable", "answer_removed"})
    ev_ms, ev_ls = sub(ev_m, ev_l, ev_k, {"answerable", "answer_removed"})
    print(f"AUC (answerable vs answer-removed)  train: {auc(tr_ms, tr_ls):.4f}   eval: {auc(ev_ms, ev_ld and ev_ls):.4f}")

    budget = args.false_refusal_budget
    tr_ans = [m for m, l in zip(tr_m, tr_l) if l == 1]
    tr_unk = [m for m, l, k in zip(tr_m, tr_l, tr_k) if k == "unknown"]
    best = None
    for step in range(0, 400):
        t = step / 1000.0
        fr = sum(1 for m in tr_ans if m < t) / max(1, len(tr_ans))
        ru = sum(1 for m in tr_unk if m < t) / max(1, len(tr_unk))
        if fr <= budget and (best is None or ru > best[1]):
            best = (t, ru, fr)
    if best is None:
        print("no threshold satisfies the false-refusal budget on train")
        return 1
    threshold, train_ru, train_fr = best
    print(f"\nfitted on TRAIN: margin >= {threshold:.3f}"
          f"  -> refused_unanswerable={train_ru * 100:.2f}%  false_refusal={train_fr * 100:.2f}%")

    ev_ans = [(m, c) for m, l, c in zip(ev_m, ev_l, ev_c) if l == 1]
    ev_unk = [(m, c) for m, l, k, c in zip(ev_m, ev_l, ev_k, ev_c) if k == "unknown"]
    ev_rem = [(m, c) for m, l, k, c in zip(ev_m, ev_l, ev_k, ev_c) if k == "answer_removed"]
    ru = sum(1 for m, _ in ev_unk if m < threshold) / max(1, len(ev_unk))
    fr = sum(1 for m, _ in ev_ans if m < threshold) / max(1, len(ev_ans))
    rr = sum(1 for m, _ in ev_rem if m < threshold) / max(1, len(ev_rem))
    print(f"applied to  EVAL: unknown refused={ru * 100:.2f}% ({len(ev_unk)})  "
          f"false_refusal={fr * 100:.2f}% ({len(ev_ans)})  answer_removed refused={rr * 100:.2f}% ({len(ev_rem)})")

    per_unknown = defaultdict(int)
    for m, c in ev_unk:
        if m < threshold:
            per_unknown[c] += 1
    print(f"\nrefused-by-category {dict(per_unknown)}")
    per_ans_refused = defaultdict(int)
    per_ans_total = defaultdict(int)
    for m, c in ev_ans:
        per_ans_total[c] += 1
        if m < threshold:
            per_ans_refused[c] += 1
    print(f"{'answerable category':<22}{'refused':>9}{'total':>7}")
    for c in sorted(per_ans_total):
        print(f"{c:<22}{per_ans_refused[c]:>9}{per_ans_total[c]:>7}")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "threshold": threshold,
            "train_auc": round(auc(tr_m, tr_l), 4),
            "eval_auc": round(auc(ev_m, ev_l), 4),
            "train_refused_unanswerable_pct": round(train_ru * 100, 2),
            "train_false_refusal_pct": round(train_fr * 100, 2),
            "eval_refused_unanswerable_pct": round(ru * 100, 2),
            "eval_false_refusal_pct": round(fr * 100, 2),
            "eval_refused_by_category": dict(per_unknown),
            "eval_answerable_refused": {c: [per_ans_refused[c], per_ans_total[c]] for c in sorted(per_ans_total)},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
