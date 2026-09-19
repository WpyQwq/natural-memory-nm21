"""Train the attribute-coverage head: "which attribute is this question asking about?"

Why this exists
---------------
The runtime answers ~75% of unanswerable paraphrased questions with an invented value.
A score threshold cannot fix that: measured on the same frozen keys, the best fitted head
over nine score-geometry features reaches AUC 0.61 (and ~0.5 for three of four scorers),
because 24 same-shape candidates look equally plausible for any question.

What works instead is a different question: *which attribute is being asked about, and does
the bank actually hold it?*  This script trains that head and saves it as a deployable
artifact.

Labels are recovered exactly, not guessed: every query in this dataset is one of the
authored paraphrases, so the asked-about attribute is looked up from the paraphrase table
rather than taken from the episode metadata (which records the answerable *target* and is
therefore wrong for abstention episodes).

Usage::

    python -m V2_dpskw.train_attribute_head --data-dir data/zero_overlap ^
        --feature-cache H:\\Memory\\nm_cache\\nm_zero_overlap\\feature_cache ^
        --output-dir checkpoints/memory_attribute_head
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .make_zero_overlap_paraphrase_data import ATTRIBUTE_PARAPHRASES


def text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", "replace")).hexdigest()


def paraphrase_table() -> dict[str, str]:
    table = {}
    for attribute, paraphrases in ATTRIBUTE_PARAPHRASES:
        for paraphrase in paraphrases:
            table[paraphrase] = attribute
    return table


def load(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def softmax_fit(X: np.ndarray, y: np.ndarray, classes: int, *, steps: int = 6000,
                lr: float = 0.5, l2: float = 1e-3):
    W = np.zeros((X.shape[1], classes))
    b = np.zeros(classes)
    n = len(y)
    onehot = np.zeros((n, classes))
    onehot[np.arange(n), y] = 1.0
    for _ in range(steps):
        logits = X @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        grad = probs - onehot
        W -= lr * (X.T @ grad / n + l2 * W)
        b -= lr * grad.mean(axis=0)
    return W, b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/zero_overlap")
    parser.add_argument("--feature-cache", default=r"H:\Memory\nm_cache\nm_zero_overlap\feature_cache")
    parser.add_argument("--output-dir", default="checkpoints/memory_attribute_head")
    parser.add_argument("--report", default="attribute_head_training.json")
    args = parser.parse_args()

    cache = Path(args.feature_cache)
    bank = np.load(cache / "features.f16.npy", mmap_mode="r")
    lookup = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    table = paraphrase_table()
    attributes = sorted({name for name, _ in ATTRIBUTE_PARAPHRASES})
    index_of = {name: position for position, name in enumerate(attributes)}

    def build(rows):
        X, y, rows_used = [], [], []
        for row in rows:
            attribute = table.get(row["query"])
            if attribute is None:
                raise SystemExit(f"query is not one of the authored paraphrases: {row['query']!r}")
            key = text_key(row["query"])
            if key not in lookup:
                raise SystemExit(f"query missing from the feature bank: {row['query']!r}")
            X.append(np.asarray(bank[lookup[key]], dtype=np.float32))
            y.append(index_of[attribute])
            rows_used.append(row)
        return np.array(X), np.array(y), rows_used

    train_X, train_y, _ = build(load(Path(args.data_dir) / "train.jsonl"))
    eval_X, eval_y, eval_rows = build(load(Path(args.data_dir) / "eval.jsonl"))

    train_Xn = train_X / (np.linalg.norm(train_X, axis=1, keepdims=True) + 1e-6)
    eval_Xn = eval_X / (np.linalg.norm(eval_X, axis=1, keepdims=True) + 1e-6)
    W, b = softmax_fit(train_Xn, train_y, len(attributes))

    def probabilities(X):
        logits = X @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    eval_probs = probabilities(eval_Xn)
    eval_pred = eval_probs.argmax(axis=1)
    answerable = np.array([bool(row.get("positive_indices")) for row in eval_rows])

    # Coverage = the asked-about attribute is actually present in this episode's bank.
    predicted_names = [attributes[position] for position in eval_pred]
    present_sets = [{c.get("attribute") for c in row["candidates"] if c.get("attribute")}
                    for row in eval_rows]
    covered = np.array([name in present for name, present in zip(predicted_names, present_sets)])

    report = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "attributes": attributes,
        "classes": len(attributes),
        "chance_pct": round(100.0 / len(attributes), 2),
        "train_queries": int(train_y.size),
        "eval_queries": int(eval_y.size),
        "attribute_accuracy_train_pct": round(100 * float(np.mean(
            probabilities(train_Xn).argmax(axis=1) == train_y)), 2),
        "attribute_accuracy_eval_pct": round(100 * float(np.mean(eval_pred == eval_y)), 2),
        "attribute_accuracy_eval_answerable_pct": round(100 * float(np.mean(
            (eval_pred == eval_y)[answerable])), 2),
        "coverage_rule": {
            "answerable_covered_pct": round(100 * float(np.mean(covered[answerable])), 2),
            "abstention_covered_pct": round(100 * float(np.mean(covered[~answerable])), 2),
            "unknown_refusal_pct": round(100 * float(np.mean(~covered[~answerable])), 2),
            "known_false_refusal_pct": round(100 * float(np.mean(~covered[answerable])), 2),
            "counts": {
                "tp": int(np.sum(~covered & ~answerable)),
                "fn": int(np.sum(covered & ~answerable)),
                "fp": int(np.sum(~covered & answerable)),
                "tn": int(np.sum(covered & answerable)),
            },
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "weight": torch.tensor(W, dtype=torch.float32),
        "bias": torch.tensor(b, dtype=torch.float32),
        "attributes": attributes,
        "format_version": 1,
        "input": "l2_normalised frozen query key (hidden_size)",
    }, out_dir / "attribute_head.pt")
    (out_dir / "attribute_head_meta.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
