"""Open-vocabulary attribute matching: can a question be matched to an attribute *name*?

The closed 24-way head is domain-locked: on a bank whose attributes it never saw, it can
only answer with one of its 24 labels and therefore refuses everything (measured: A-segment
answerable accuracy 0.00%).  The production shape needs no fixed vocabulary at all --
the bank's own attribute names are the candidate set, and they change as the user writes.

This script measures whether the *frozen* keys already support that, with no training:

    score(question, name) = cosine(query_key, encode(attribute_name))

and reports (a) whether the asked-about attribute is ranked first among the names actually
present in the bank, and (b) whether the best score separates "the bank holds it" from
"the bank does not", which is what an abstention rule needs.

Usage::

    python -m V2_dpskw.analyze_open_vocabulary_attribute_match
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .make_zero_overlap_paraphrase_data import ATTRIBUTE_PARAPHRASES


def text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", "replace")).hexdigest()


def auc(labels: np.ndarray, values: np.ndarray) -> float:
    positive = values[labels == 1]
    negative = values[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    return float(np.mean(positive[:, None] > negative[None, :])
                 + 0.5 * np.mean(positive[:, None] == negative[None, :]))


def load(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/zero_overlap")
    parser.add_argument("--feature-cache", default=r"H:\Memory\nm_cache\nm_zero_overlap\feature_cache")
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--output", default="open_vocabulary_match.json")
    parser.add_argument("--markdown", default="open_vocabulary_match.md")
    args = parser.parse_args()

    from .qwen_integration import load_qwen_dynamic, load_tokenizer

    cache = Path(args.feature_cache)
    bank = np.load(cache / "features.f16.npy", mmap_mode="r")
    lookup = json.loads((cache / "index.json").read_text(encoding="utf-8"))

    model = load_qwen_dynamic(args.package)
    model.eval()
    tokenizer = load_tokenizer(args.package)
    device = next(model.parameters()).device

    names = [name for name, _ in ATTRIBUTE_PARAPHRASES]
    with torch.no_grad():
        encoded = tokenizer(names, return_tensors="pt", padding=True, truncation=True,
                            max_length=64)
        name_keys = model._encode_model_key(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device)).float()
        name_keys = F.normalize(name_keys, dim=-1).cpu()

    eval_rows = load(Path(args.data_dir) / "eval.jsonl")
    answerable_flags, top1_hits, best_scores, present_counts = [], [], [], []
    for row in eval_rows:
        query = torch.from_numpy(
            np.asarray(bank[lookup[text_key(row["query"])]], dtype=np.float32))
        query = F.normalize(query.reshape(1, -1), dim=-1)
        present = [c["attribute"] for c in row["candidates"] if c.get("attribute")]
        indices = [names.index(name) for name in dict.fromkeys(present) if name in names]
        if not indices:
            continue
        scores = (query @ name_keys[indices].t()).reshape(-1)
        best = int(scores.argmax())
        best_scores.append(float(scores[best]))
        present_counts.append(len(indices))
        answerable_flags.append(1 if row.get("positive_indices") else 0)
        # The asked-about attribute for abstention episodes is deliberately absent from the
        # candidate set, so a "hit" only makes sense for answerable episodes.
        if row.get("positive_indices"):
            target = row["metadata"]["attribute"]
            top1_hits.append(names[indices[best]] == target)

    labels = np.array(answerable_flags)
    best = np.array(best_scores)
    report = {
        "attribute_names": len(names),
        "episodes": len(labels),
        "answerable": int(labels.sum()),
        "abstention": int((labels == 0).sum()),
        "mean_present_attributes": round(float(np.mean(present_counts)), 2),
        "top1_attribute_accuracy_answerable_pct": round(100 * float(np.mean(top1_hits)), 2),
        "chance_pct": round(100.0 / float(np.mean(present_counts)), 2),
        "best_score_answerable_p50": round(float(np.percentile(best[labels == 1], 50)), 4),
        "best_score_abstention_p50": round(float(np.percentile(best[labels == 0], 50)), 4),
        "best_score_auc_covered_vs_uncovered": round(auc(labels, best), 4),
        "note": ("training-free: cosine between the frozen query key and the frozen encoding "
                 "of each attribute NAME actually present in the bank"),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 开放词表属性匹配（零训练，用库里的属性名直接匹配）", "",
             f"库内属性名候选 {report['attribute_names']} 个；评测 {report['episodes']} 条"
             f"（可回答 {report['answerable']} / 不可回答 {report['abstention']}）；"
             f"平均候选数 {report['mean_present_attributes']}。", "",
             "| 指标 | 值 |", "|---|---:|",
             f"| 可回答 episode 的 Top-1 属性识别准确率 | **{report['top1_attribute_accuracy_answerable_pct']:.2f}%** |",
             f"| 随机基线 | {report['chance_pct']:.2f}% |",
             f"| 最佳匹配分中位数（可回答） | {report['best_score_answerable_p50']:.4f} |",
             f"| 最佳匹配分中位数（不可回答） | {report['best_score_abstention_p50']:.4f} |",
             f"| **覆盖 vs 不覆盖 AUC** | **{report['best_score_auc_covered_vs_uncovered']:.4f}** |"]
    text = "\n".join(lines) + "\n"
    Path(args.markdown).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
