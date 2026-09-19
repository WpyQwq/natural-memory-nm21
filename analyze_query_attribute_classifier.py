"""Do the frozen features encode *which attribute* a paraphrased question asks about?

The previous measurement killed the score-geometry route to abstention (best fitted-head AUC
~0.61, and near chance for three of four scorers).  The remaining structural route is:

    map the question to the attribute it asks about, then check whether the bank holds it

That only works if attribute identity is recoverable from the query's frozen key *for
phrasings never seen during training*.  This script tests exactly that, and nothing else:

* query vectors are the same 2560-dim frozen keys the runtime scores;
* a linear classifier is fitted on the **train** split's queries and scored on the **eval**
  split's queries, which are *different paraphrases of the same attributes* -- so the
  reported number is generalisation to unseen wording, not memorisation;
* chance level is 1 / (number of attributes).

It also reports the "none" case that abstention needs: accuracy when the query's attribute
is absent from the candidate set (the unknown episodes), i.e. can we even tell that the
question refers to something the bank does not hold.

Usage::

    python -m V2_dpskw.analyze_query_attribute_classifier
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", "replace")).hexdigest()


def load(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def softmax_linear_fit(X: np.ndarray, y: np.ndarray, classes: int, *, steps: int = 4000,
                       lr: float = 0.5, l2: float = 1e-3):
    """Plain multinomial logistic regression (no sklearn dependency)."""
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
    parser.add_argument("--output", default="query_attribute_classifier.json")
    parser.add_argument("--markdown", default="query_attribute_classifier.md")
    parser.add_argument("--holdout-attributes", type=int, default=0,
                        help=("train on all but N attributes and evaluate only on those N; "
                              "tests whether the head generalises to attributes it has never seen"))
    parser.add_argument("--none-class", action="store_true",
                        help=("train an explicit NONE class using queries of attributes held out "
                              "of the vocabulary, then test on a *different* held-out set"))
    args = parser.parse_args()

    cache = Path(args.feature_cache)
    bank = np.load(cache / "features.f16.npy", mmap_mode="r")
    lookup = json.loads((cache / "index.json").read_text(encoding="utf-8"))

    attributes = sorted({row["metadata"]["attribute"] for row in load(Path(args.data_dir) / "train.jsonl")})
    index_of = {name: position for position, name in enumerate(attributes)}
    classes = len(attributes)

    def vectors(rows):
        X, y, unknown, present = [], [], [], []
        for row in rows:
            key = text_key(row["query"])
            if key not in lookup:
                raise SystemExit(f"query missing from the bank: {row['query']!r}")
            X.append(np.asarray(bank[lookup[key]], dtype=np.float32))
            y.append(index_of[row["metadata"]["attribute"]])
            unknown.append(0 if row.get("positive_indices") else 1)
            # Which attributes does this episode's candidate set actually hold?  The
            # abstention question is whether the *asked-about* attribute is among them.
            present.append({c.get("attribute") for c in row["candidates"] if c.get("attribute")})
        return np.array(X), np.array(y), np.array(unknown), present

    train_X, train_y, _, _ = vectors(load(Path(args.data_dir) / "train.jsonl"))
    eval_X, eval_y, eval_unknown, eval_present = vectors(load(Path(args.data_dir) / "eval.jsonl"))

    # L2-normalise and scale: the frozen keys are pooled hidden states.
    train_Xn = train_X / (np.linalg.norm(train_X, axis=1, keepdims=True) + 1e-6)
    eval_Xn = eval_X / (np.linalg.norm(eval_X, axis=1, keepdims=True) + 1e-6)

    holdout_report = None
    if args.none_class:
        # The production shape of the problem: the bank holds a *vocabulary* of attributes,
        # and a user may ask about one outside it.  Train K attribute classes PLUS an
        # explicit NONE class, where NONE examples are real queries about attributes that
        # are deliberately excluded from the vocabulary.  A separate, further held-out set
        # of attributes is then used for testing, so NONE recall is measured out of sample.
        order = np.argsort([attributes[position] for position in range(classes)])
        vocabulary = set(order[:12].tolist())
        none_train = set(order[12:18].tolist())
        none_eval = set(order[18:].tolist())
        compact = {original: position for position, original in enumerate(sorted(vocabulary))}
        none_label = len(compact)

        rows_train = np.concatenate([np.where(np.isin(train_y, list(vocabulary)))[0],
                                     np.where(np.isin(train_y, list(none_train)))[0]])
        labels_train = np.array([
            compact.get(int(train_y[index]), none_label) for index in rows_train])

        W_n, b_n = softmax_linear_fit(train_Xn[rows_train], labels_train, len(compact) + 1)

        def predict(X: np.ndarray) -> np.ndarray:
            return (X @ W_n + b_n).argmax(axis=1)

        eval_answerable = eval_unknown == 0
        vocab_rows = eval_answerable & np.isin(eval_y, list(vocabulary))
        none_rows = eval_answerable & np.isin(eval_y, list(none_eval))
        vocab_pred = predict(eval_Xn[vocab_rows])
        none_pred = predict(eval_Xn[none_rows])
        vocab_correct = float(np.mean(vocab_pred == np.array(
            [compact[int(label)] for label in eval_y[vocab_rows]])))
        none_recall = float(np.mean(none_pred == none_label))
        none_precision = float(np.sum(none_pred == none_label) / max(1, none_pred.size))
        none_report = {
            "vocabulary_attributes": [attributes[position] for position in sorted(vocabulary)],
            "none_train_attributes": [attributes[position] for position in sorted(none_train)],
            "none_eval_attributes": [attributes[position] for position in sorted(none_eval)],
            "train_rows": int(rows_train.size),
            "eval_vocabulary_queries": int(vocab_rows.sum()),
            "eval_never_seen_queries": int(none_rows.sum()),
            "chance_pct": round(100.0 / (len(compact) + 1), 2),
            "vocabulary_accuracy_pct": round(100 * vocab_correct, 2),
            "never_seen_rejected_pct": round(100 * none_recall, 2),
            "known_false_rejection_pct": round(
                100 * float(np.mean(vocab_pred == none_label)), 2),
            "note": ("an explicit NONE class trained on held-out attributes; the never-seen "
                     "attributes used for testing are disjoint from both the vocabulary and "
                     "the NONE training set"),
        }
        Path(args.output).write_text(json.dumps(none_report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(json.dumps(none_report, ensure_ascii=False, indent=2))
        return 0

    if args.holdout_attributes > 0:
        # Open-set test: N attributes are removed from the *training* label set entirely,
        # so the head has never seen them.  A production bank cannot know in advance which
        # attributes a user will ask about, so the question that matters is: when a
        # never-trained attribute is asked about, does the head (a) land on some *present*
        # attribute -- which would make the coverage rule answer wrongly -- or (b) betray
        # its uncertainty, giving a usable reject option?
        rng = np.random.default_rng(20260912)
        held = set(rng.choice(classes, size=min(args.holdout_attributes, classes - 2),
                              replace=False).tolist())
        kept = sorted(set(range(classes)) - held)
        compact = {original: position for position, original in enumerate(kept)}
        keep_train = np.array([label in compact for label in train_y])
        W_h, b_h = softmax_linear_fit(
            train_Xn[keep_train],
            np.array([compact[label] for label in train_y[keep_train]]),
            len(kept))

        def max_probability(X: np.ndarray) -> np.ndarray:
            logits = X @ W_h + b_h
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            return probs.max(axis=1)

        held_mask = np.array([(label in held) and (unknown == 0)
                              for label, unknown in zip(eval_y, eval_unknown)])
        kept_mask = np.array([(label in compact) and (unknown == 0)
                              for label, unknown in zip(eval_y, eval_unknown)])
        held_prob = max_probability(eval_Xn[held_mask])
        kept_prob = max_probability(eval_Xn[kept_mask])
        held_pred = (eval_Xn[held_mask] @ W_h + b_h).argmax(axis=1)
        # Would the closed-set head claim a *present* attribute for a never-trained one?
        held_pred_names = [attributes[kept[position]] for position in held_pred]
        held_present = np.array([
            name in present for name, present in zip(held_pred_names, np.array(eval_present, dtype=object)[held_mask])
        ])
        threshold = float(np.percentile(kept_prob, 5)) if kept_prob.size else 0.0
        holdout_report = {
            "held_out_attributes": sorted(attributes[position] for position in held),
            "trained_on_attributes": len(kept),
            "eval_held_out_queries": int(held_mask.sum()),
            "eval_kept_queries": int(kept_mask.sum()),
            "max_prob_held_out": {
                "p10": round(float(np.percentile(held_prob, 10)), 4),
                "p50": round(float(np.percentile(held_prob, 50)), 4),
                "p90": round(float(np.percentile(held_prob, 90)), 4),
            },
            "max_prob_kept": {
                "p10": round(float(np.percentile(kept_prob, 10)), 4),
                "p50": round(float(np.percentile(kept_prob, 50)), 4),
                "p90": round(float(np.percentile(kept_prob, 90)), 4),
            },
            "openset_auc_held_out_vs_kept": round(
                float(np.mean(held_prob[:, None] < kept_prob[None, :])), 4),
            "closed_set_false_claim_pct": round(100 * float(np.mean(held_present)), 2),
            "reject_threshold_from_kept_p05": round(threshold, 4),
            "held_out_rejected_pct_at_threshold": round(
                100 * float(np.mean(held_prob < threshold)), 2),
            "kept_retained_pct_at_threshold": round(
                100 * float(np.mean(kept_prob >= threshold)), 2),
            "note": ("held-out attributes never appear as training labels, so this is the "
                     "open-set case a real user bank faces"),
        }
        Path(args.output).write_text(json.dumps(holdout_report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(json.dumps(holdout_report, ensure_ascii=False, indent=2))
        return 0
    W, b = softmax_linear_fit(train_Xn, train_y, classes)
    eval_logits = eval_Xn @ W + b
    eval_pred = eval_logits.argmax(axis=1)
    train_pred = (train_Xn @ W + b).argmax(axis=1)

    answerable = eval_unknown == 0
    accuracy_all = float(np.mean(eval_pred == eval_y))
    accuracy_answerable = float(np.mean((eval_pred == eval_y)[answerable]))

    report = {
        "attributes": classes,
        "chance_pct": round(100.0 / classes, 2),
        "train_queries": len(train_y),
        "eval_queries": len(eval_y),
        "distinct_train_queries": len({text_key(t) for t in []}) or None,
        "train_accuracy_pct": round(100 * float(np.mean(train_pred == train_y)), 2),
        "eval_accuracy_pct": round(100 * accuracy_all, 2),
        "eval_accuracy_answerable_pct": round(100 * accuracy_answerable, 2),
        "note": ("eval queries are different paraphrases of the same attributes, so this is "
                 "generalisation to unseen wording"),
    }

    # Per-attribute breakdown for the answerable eval queries.
    per_attribute = {}
    for position, name in enumerate(attributes):
        mask = (eval_y == position) & answerable
        if mask.sum():
            per_attribute[name] = {
                "eval_queries": int(mask.sum()),
                "correct_pct": round(100 * float(np.mean(eval_pred[mask] == position)), 2),
            }
    report["per_attribute"] = per_attribute

    # --- the structural abstention signal -------------------------------------------
    # Predicted asked-about attribute vs the attributes the candidate set actually holds.
    # Answerable episodes should land inside the set; abstention episodes should land
    # outside it.  A categorical split here means abstention needs no fragile threshold.
    predicted_names = [attributes[position] for position in eval_pred]
    inside = np.array([name in present for name, present in zip(predicted_names, eval_present)])
    coverage = {
        "rule": "predict the asked-about attribute; refuse when the candidate set lacks it",
        "answerable_inside_pct": round(100 * float(np.mean(inside[answerable])), 2),
        "unknown_inside_pct": round(100 * float(np.mean(inside[~answerable])), 2),
        "answerable_cases": int(answerable.sum()),
        "unknown_cases": int((~answerable).sum()),
    }
    # What a naive "refuse when outside" rule would yield on this split:
    refused = ~inside
    tp = int(np.sum(refused & ~answerable))      # unknown correctly refused
    fn = int(np.sum(~refused & ~answerable))     # unknown wrongly answered
    fp = int(np.sum(refused & answerable))       # answerable wrongly refused
    tn = int(np.sum(~refused & answerable))      # answerable correctly answered
    coverage.update({
        "unknown_refusal_pct": round(100 * tp / max(1, tp + fn), 2),
        "known_false_refusal_pct": round(100 * fp / max(1, fp + tn), 2),
        "counts": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    })
    report["coverage_rule"] = coverage
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 冻结特征是否编码「提问问的是哪个属性」？", "",
        f"属性数 {classes}（随机基线 {100.0/classes:.2f}%）；训练查询 {len(train_y)} 条，"
        f"评测查询 {len(eval_y)} 条（**同为这些属性的不同改写，训练时从未见过**）。", "",
        "| 指标 | 值 |", "|---|---:|",
        f"| 训练集准确率 | {report['train_accuracy_pct']:.2f}% |",
        f"| **评测集准确率（未见改写）** | **{report['eval_accuracy_pct']:.2f}%** |",
        f"| 评测集·仅可回答子集 | {report['eval_accuracy_answerable_pct']:.2f}% |",
        f"| 随机基线 | {report['chance_pct']:.2f}% |", "",
        "## 结构化拒答规则（判属性 → 查是否在库里）", "",
        f"规则：{coverage['rule']}。", "",
        "| 指标 | 值 |", "|---|---:|",
        f"| 可回答 episode 中「预测属性在候选集内」 | **{coverage['answerable_inside_pct']:.2f}%**（{coverage['answerable_cases']} 条） |",
        f"| 未知 episode 中「预测属性在候选集内」 | **{coverage['unknown_inside_pct']:.2f}%**（{coverage['unknown_cases']} 条） |",
        f"| 未知拒答率（规则命中） | **{coverage['unknown_refusal_pct']:.2f}%** |",
        f"| 已知问题被误拒率 | **{coverage['known_false_refusal_pct']:.2f}%** |",
        f"| 混淆计数 | tp={coverage['counts']['tp']} fn={coverage['counts']['fn']} fp={coverage['counts']['fp']} tn={coverage['counts']['tn']} |",
        "",
        "## 按属性（评测集可回答）", "",
        "| 属性 | 评测问法数 | 正确率 |", "|---|---:|---:|",
    ]
    for name, body in sorted(per_attribute.items(), key=lambda kv: -kv[1]["correct_pct"]):
        lines.append(f"| {name} | {body['eval_queries']} | {body['correct_pct']:.2f}% |")
    text = "\n".join(lines) + "\n"
    Path(args.markdown).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
