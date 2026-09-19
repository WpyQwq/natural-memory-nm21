"""Measure why the write path retires unrelated facts.

Observed symptom: writing 20 distinct-attribute facts into a fresh bank leaves only 12
active records, and in 8 of 16 critical e2e cases the *target* record is already
``retracted`` before the query even runs.  No ranker can recover a record that no longer
exists, which caps end-to-end accuracy at 50% regardless of router quality.

The write path (``qwen_integration._write_text_memory``) retires an existing record when
it decides the new text is a *confirmed update* of it:

    confirmed_update = (text_retriever score >= text_memory_semantic_update_threshold)   # 0.95
    ... or, with no learned score, lexical overlap >= text_memory_update_overlap_threshold # 0.30

and then retracts records that share the slot or satisfy ``score >= 0.95 and shared >= 2``.

The guard ``shared >= 2`` cannot discriminate anything for this fact template: every fact
is "我的<属性>是 <代号>。" so the template tokens (我的 / 是 / 。) are shared by *all* pairs.

This script measures that claim directly and cheaply, without touching the write path and
without mutating any bank: it encodes the fact texts with the frozen backbone, scores every
unordered pair with the packaged ``text_retriever``, counts the token overlap exactly the
way ``_rank_v2_text_matches`` does, and reports how many pairs the write path would treat
as a confirmed update.

Usage::

    python -m V2_dpskw.measure_write_path_updates --package qwen3_5_4b_natural_memory_v2
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attributes", type=int, default=20)
    parser.add_argument("--output", default="write_path_update_measurement.json")
    args = parser.parse_args()

    from .make_zero_overlap_paraphrase_data import ATTRIBUTE_PARAPHRASES
    from .qwen_integration import load_qwen_dynamic, load_tokenizer

    attributes = [name for name, _ in ATTRIBUTE_PARAPHRASES][: args.attributes]
    facts = ["我的%s是 VAL-%s。" % (name, "A%07d" % index)
             for index, name in enumerate(attributes)]

    model = load_qwen_dynamic(args.package)
    model.eval()
    device = next(model.parameters()).device
    args.device = str(device)
    tokenizer = load_tokenizer(args.package)

    ready = {
        "text_retriever_ready": bool(getattr(model, "_text_retriever_ready", False)),
        "router_ready": bool(getattr(model, "_memory_router_v2_ready", False)),
        "text_retriever_params": (
            sum(p.numel() for p in model.text_retriever.parameters())
            if model.text_retriever is not None else 0
        ),
    }
    print(json.dumps({"phase": "ready", **ready}), flush=True)
    if not ready["text_retriever_ready"]:
        raise SystemExit("packaged text_retriever is not ready; this measurement needs it")

    encoded = tokenizer(facts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.no_grad():
        keys = model._encode_model_key(encoded["input_ids"], encoded["attention_mask"])
    keys = keys.reshape(len(facts), -1)
    token_ids = [torch.unique(encoded["input_ids"][index][encoded["attention_mask"][index].bool()])
                 for index in range(len(facts))]

    pairs = []
    with torch.no_grad():
        for left, right in itertools.combinations(range(len(facts)), 2):
            if model.text_retriever is not None:
                score = float(torch.sigmoid(model.text_retriever(
                    keys[left].reshape(1, -1), keys[right].reshape(1, 1, -1)
                )).reshape(-1)[0].item())
            else:
                score = float(torch.nn.functional.cosine_similarity(
                    keys[left].reshape(1, -1), keys[right].reshape(1, -1)).item())
            shared = int(torch.isin(token_ids[left], token_ids[right]).sum().item())
            pairs.append({
                "left": attributes[left],
                "right": attributes[right],
                "retriever_score": score,
                "shared_tokens": shared,
                "template_shared": sorted(
                    tokenizer.decode(token_ids[left][torch.isin(token_ids[left], token_ids[right])]).split()
                ),
                "would_be_confirmed_update": score >= 0.95,
                "would_be_retracted_by_second_condition": score >= 0.95 and shared >= 2,
            })

    scores = sorted(item["retriever_score"] for item in pairs)
    total = len(pairs)
    confirmed = sum(1 for item in pairs if item["would_be_confirmed_update"])
    guard_pass = sum(1 for item in pairs if item["would_be_retracted_by_second_condition"])

    def percentile(fraction: float) -> float:
        if not scores:
            return 0.0
        return scores[min(len(scores) - 1, int(fraction * len(scores)))]

    report = {
        "package": args.package,
        "facts": len(facts),
        "distinct_attribute_pairs": total,
        "retriever": ready,
        "retriever_score_distribution": {
            "min_pct": round(100 * scores[0], 2) if scores else None,
            "p50_pct": round(100 * percentile(0.50), 2),
            "p90_pct": round(100 * percentile(0.90), 2),
            "p99_pct": round(100 * percentile(0.99), 2),
            "max_pct": round(100 * scores[-1], 2) if scores else None,
        },
        "pairs_scoring_at_or_above_update_threshold_0.95": confirmed,
        "pairs_at_or_above_0.95_pct": round(100 * confirmed / max(1, total), 2),
        "pairs_where_the_shared_ge_2_guard_does_not_block": guard_pass,
        "shared_token_counts": sorted({item["shared_tokens"] for item in pairs}),
        "template_token_share_note": (
            "every fact shares the 我的/是 template tokens, so 'shared >= 2' is satisfied "
            "by unrelated attributes and cannot act as an update guard"
        ),
        "worst_pairs": sorted(pairs, key=lambda item: item["retriever_score"], reverse=True)[:8],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
