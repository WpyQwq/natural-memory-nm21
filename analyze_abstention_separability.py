"""Can "the bank does not contain this attribute" be detected at all?

The runtime answers 75% of unanswerable paraphrased questions with an invented code.  A
score threshold was already measured to be useless: the retrieval top-score distributions
of answerable and unanswerable episodes overlap almost entirely.  Before building any
classifier head, this script asks the prior question properly and cheaply -- **without
generating text** -- by scoring every candidate of every episode with the same scorers the
runtime uses and testing how well answerability can be predicted from the resulting score
geometry (top-1, margin, top-k mean, spread...).

Method, so the number is trustworthy:

* features come from the frozen feature bank, i.e. exactly the 2560-dim keys the runtime
  scores;
* a logistic head is **fitted on the train split and evaluated on the eval split**, so the
  reported AUC is not an in-sample artefact;
* the reported ceiling is the AUC of the single best feature and of the fitted head, plus
  the achievable operating points (unknown-flagging rate vs answerable rejection rate).

If the AUC is near 0.5 the conclusion is firm: abstention cannot be recovered from score
geometry on this feature set, and the fix has to be structural.

Usage::

    python -m V2_dpskw.analyze_abstention_separability --with-text-retriever
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .eval_router_scorecard import load_router_any


def text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", "replace")).hexdigest()


def load_episodes(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_vectors(row: dict, lookup: dict, bank: np.ndarray):
    query = row["query"]
    candidates = [c["text"] for c in row["candidates"]]
    missing = [t for t in [query, *candidates] if text_key(t) not in lookup]
    if missing:
        raise SystemExit(f"{row['id']}: {len(missing)} texts missing from the bank")
    return bank[lookup[text_key(query)]], bank[[lookup[text_key(t)] for t in candidates]]


def features_from_scores(scores: np.ndarray) -> dict:
    order = np.sort(scores)[::-1]
    top1 = float(order[0])
    top2 = float(order[1]) if order.size > 1 else 0.0
    top3 = float(order[2]) if order.size > 2 else top2
    spread = float(np.std(scores))
    median = float(np.median(scores))
    return {
        "top1": top1,
        "top2": top2,
        "margin": top1 - top2,
        "top3_mean": float(np.mean(order[:3])),
        "top1_minus_median": top1 - median,
        "std": spread,
        "z_top1": (top1 - float(np.mean(scores))) / (spread + 1e-6),
        # how many candidates sit close to the best one: a truly held fact should be
        # clearly ahead of 23 unrelated same-shape facts
        "n_within_10pct": float(np.sum(scores >= top1 - 0.10 * max(abs(top1), 1e-6))),
        "entropy": float(-np.sum(np.exp(scores - top1) / np.sum(np.exp(scores - top1))
                                 * (scores - top1))),
    }


FEATURE_NAMES = ["top1", "top2", "margin", "top3_mean", "top1_minus_median",
                 "std", "z_top1", "n_within_10pct", "entropy"]


def auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties averaged."""
    positives = values[labels == 1]
    negatives = values[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positives, negatives]))
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    # average ties
    combined = np.concatenate([positives, negatives])
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            mask = inverse == index
            ranks[mask] = ranks[mask].mean()
    rank_sum = ranks[: positives.size].sum()
    return float((rank_sum - positives.size * (positives.size + 1) / 2)
                 / (positives.size * negatives.size))


def collect(rows, lookup, bank, scorers) -> tuple[dict[str, list[dict]], np.ndarray]:
    per_scorer: dict[str, list[dict]] = {name: [] for name in scorers}
    labels = []
    for row in rows:
        query_vector, candidate_vectors = episode_vectors(row, lookup, bank)
        query = torch.from_numpy(np.ascontiguousarray(query_vector)).float().reshape(1, -1)
        candidates = torch.from_numpy(
            np.ascontiguousarray(candidate_vectors)).float().reshape(1, -1, bank.shape[1])
        labels.append(1 if row.get("positive_indices") else 0)
        with torch.no_grad():
            for name, fn in scorers.items():
                scores = fn(query, candidates).reshape(-1).numpy()
                per_scorer[name].append(features_from_scores(scores))
    return per_scorer, np.array(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/zero_overlap")
    parser.add_argument("--feature-cache", default=r"H:\Memory\nm_cache\nm_zero_overlap\feature_cache")
    parser.add_argument("--router", default="checkpoints/router_replay_v7_v2_128/memory_router_v2.pt")
    parser.add_argument("--with-text-retriever", action="store_true")
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="abstention_separability.json")
    parser.add_argument("--markdown", default="abstention_separability.md")
    args = parser.parse_args()

    cache = Path(args.feature_cache)
    bank = np.load(cache / "features.f16.npy", mmap_mode="r")
    lookup = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)

    router, _, _ = load_router_any(Path(args.router))
    router = router.to(device).eval()

    def cosine(query, candidates):
        return F.cosine_similarity(query, candidates.reshape(-1, candidates.shape[-1]), dim=-1)

    def router_score(query, candidates):
        projected = router.encode_key(candidates.reshape(-1, candidates.shape[-1]).to(device))
        scores, _ = router.projected_scores(
            query.to(device), projected.reshape(1, -1, projected.shape[-1]))
        return torch.sigmoid(scores.reshape(-1)).cpu()

    scorers = {"cosine": cosine, "router": router_score}

    retriever = None
    if args.with_text_retriever:
        from .qwen_integration import load_qwen_dynamic

        model = load_qwen_dynamic(args.package)
        model.eval()
        if model.text_retriever is None or not getattr(model, "_text_retriever_ready", False):
            raise SystemExit("packaged text_retriever is not ready")
        retriever = model.text_retriever.to(device).eval()

        def retriever_score(query, candidates):
            with torch.no_grad():
                out = retriever(query.to(device),
                                candidates.reshape(-1, candidates.shape[-1]).to(device))
            return torch.sigmoid(out.reshape(-1)).cpu()

        scorers["text_retriever"] = retriever_score
        scorers["blend_50_50"] = lambda q, c: 0.5 * retriever_score(q, c) + 0.5 * router_score(q, c)

    train_rows = load_episodes(Path(args.data_dir) / "train.jsonl")
    eval_rows = load_episodes(Path(args.data_dir) / "eval.jsonl")
    print(json.dumps({"train_episodes": len(train_rows), "eval_episodes": len(eval_rows),
                      "train_unknown": sum(1 for r in train_rows if not r.get("positive_indices")),
                      "eval_unknown": sum(1 for r in eval_rows if not r.get("positive_indices")),
                      "scorers": list(scorers)}), flush=True)

    train_feats, train_labels = collect(train_rows, lookup, bank, scorers)
    eval_feats, eval_labels = collect(eval_rows, lookup, bank, scorers)

    report: dict = {"scorers": {}, "train_episodes": len(train_rows), "eval_episodes": len(eval_rows)}

    for name in scorers:
        entry: dict = {"single_feature_auc_on_eval": {}}
        for feature in FEATURE_NAMES:
            values = np.array([row[feature] for row in eval_feats[name]], dtype=float)
            entry["single_feature_auc_on_eval"][feature] = round(auc(eval_labels, values), 4)

        # Logistic head, fitted on train only.
        X_train = np.array([[row[f] for f in FEATURE_NAMES] for row in train_feats[name]], dtype=np.float64)
        X_eval = np.array([[row[f] for f in FEATURE_NAMES] for row in eval_feats[name]], dtype=np.float64)
        mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-9
        Xtr = (X_train - mean) / std
        Xev = (X_eval - mean) / std
        weights = np.zeros(Xtr.shape[1])
        bias = 0.0
        # plain gradient descent with L2; no sklearn dependency
        for _ in range(4000):
            logits = Xtr @ weights + bias
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad_w = Xtr.T @ (probs - train_labels) / len(train_labels) + 1e-3 * weights
            grad_b = float(np.mean(probs - train_labels))
            weights -= 0.5 * grad_w
            bias -= 0.5 * grad_b
        eval_probs = 1.0 / (1.0 + np.exp(-(Xev @ weights + bias)))
        entry["head_auc_on_eval"] = round(auc(eval_labels, eval_probs), 4)
        entry["head_weights"] = {f: round(float(w), 4) for f, w in zip(FEATURE_NAMES, weights)}

        # Operating points: keep answerable episodes (maximise) while flagging unknowns.
        curve = []
        for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            flagged_unknown = float(np.mean(eval_probs[eval_labels == 0] < threshold))
            rejected_answerable = float(np.mean(eval_probs[eval_labels == 1] < threshold))
            curve.append({
                "threshold": threshold,
                "unknown_flagged_pct": round(100 * flagged_unknown, 2),
                "answerable_rejected_pct": round(100 * rejected_answerable, 2),
            })
        entry["operating_curve"] = curve
        report["scorers"][name] = entry

    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 未知问题可分性分析（能否从分数几何判断\"库里没有这个属性\"）", "",
             f"训练集 {len(train_rows)} 条（未知 {report['train_episodes'] and sum(1 for r in train_rows if not r.get('positive_indices'))}），"
             f"评测集 {len(eval_rows)} 条（未知 {sum(1 for r in eval_rows if not r.get('positive_indices'))}）。",
             "逻辑回归头**只在训练集上拟合**，AUC 在评测集上计算。AUC 0.5 = 完全不可分。", ""]
    for name, entry in report["scorers"].items():
        lines.append(f"## 打分器 `{name}`")
        lines.append("")
        lines.append(f"**单特征最佳 AUC**：" + ", ".join(
            f"{k} {v:.4f}" for k, v in sorted(entry["single_feature_auc_on_eval"].items(),
                                              key=lambda kv: -kv[1])[:4]))
        lines.append("")
        lines.append(f"**拟合头 AUC（评测集）**：{entry['head_auc_on_eval']:.4f}")
        lines.append("")
        lines.append("| 判定阈值 | 标出未知的比例 | 误拒可回答的比例 |")
        lines.append("|---:|---:|---:|")
        for point in entry["operating_curve"]:
            lines.append("| {threshold:.2f} | {unknown_flagged_pct:.2f}% | {answerable_rejected_pct:.2f}% |".format(**point))
        lines.append("")
    text = "\n".join(lines) + "\n"
    Path(args.markdown).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
