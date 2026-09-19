"""Compare the rankers that can order a record set, on the zero-overlap eval set.

Why this exists
---------------
``memory_os_v2.py`` gives a record's *learned* score to ``record_scorer`` and discards
whatever ``MemoryRouterV2.projected_scores`` produced for it.  The deployed adapter
ships no ``text_retriever.pt``, so ``_text_retriever_ready`` is False and that callback
falls back to plain cosine similarity between the query key and the record key -- both
produced by the *frozen* backbone.  Measured consequence: swapping or even randomising
the router changes end-to-end answers not at all.

This script measures, on identical frozen features, which of those candidate rankers can
actually put the right fact first:

* ``cosine``  -- ``F.cosine_similarity(query_key, candidate_key)``, i.e. what the runtime
  uses today for fact records;
* each router checkpoint, scored through ``projected_scores`` exactly as the runtime and
  the training protocol do.

Usage::

    python -m V2_dpskw.compare_record_rankers ^
        --eval-file data/zero_overlap/eval.jsonl ^
        --feature-cache H:\\Memory\\nm_cache\\nm_zero_overlap\\feature_cache ^
        --run "REPLAY-128=checkpoints/router_replay_v7_v2_128/memory_router_v2.pt" ^
        --run "V2-128-v6=checkpoints/router_v6_v2_128/memory_router_v2.pt"
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


def stream_episodes(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def episode_tensors(row: dict, lookup: dict[str, int], bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = row["query"]
    candidates = [candidate["text"] for candidate in row["candidates"]]
    missing = [text for text in [query, *candidates] if text_key(text) not in lookup]
    if missing:
        raise SystemExit(f"{row['id']}: {len(missing)} text(s) absent from the bank")
    query_vector = bank[lookup[text_key(query)]]
    candidate_vectors = bank[[lookup[text_key(text)] for text in candidates]]
    return query_vector, candidate_vectors


def rank_metrics(scores: torch.Tensor, positive: int) -> tuple[int, bool, bool]:
    order = torch.argsort(scores, descending=True).tolist()
    rank = order.index(positive) + 1
    return rank, rank == 1, rank <= 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", default="data/zero_overlap/eval.jsonl")
    parser.add_argument("--feature-cache", default=r"H:\Memory\nm_cache\nm_zero_overlap\feature_cache")
    parser.add_argument("--run", action="append", required=True, help="LABEL=CHECKPOINT_PATH")
    parser.add_argument("--with-text-retriever", action="store_true",
                        help="also score with the packaged text_retriever (costs one model load)")
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="zero_overlap_ranker_comparison.json")
    args = parser.parse_args()

    cache = Path(args.feature_cache)
    bank = np.load(cache / "features.f16.npy", mmap_mode="r")
    lookup = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)

    routers = []
    for spec in args.run:
        label, _, path_value = spec.partition("=")
        router, arch_config, info = load_router_any(Path(path_value))
        router = router.to(device).eval()
        routers.append((label, router))
        print(json.dumps({"loaded": label, "path": path_value, "arch": str(arch_config),
                          "parameters": sum(p.numel() for p in router.parameters())}), flush=True)

    stats = {label: {"top1": 0, "recall3": 0, "mrr": 0.0} for label, _ in routers}
    stats["cosine"] = {"top1": 0, "recall3": 0, "mrr": 0.0}

    # The packaged text_retriever is the module that actually orders fact records at
    # runtime (memory_os_v2._record_scores overwrites the router score for records that
    # carry a semantic_key).  The feature bank already stores exactly the 2560-dim model
    # keys it consumes, so it can be scored on identical inputs with no re-encoding.
    retriever = None
    if args.with_text_retriever:
        from .qwen_integration import load_qwen_dynamic

        model = load_qwen_dynamic(args.package)
        model.eval()
        if model.text_retriever is None or not getattr(model, "_text_retriever_ready", False):
            raise SystemExit("packaged text_retriever is not ready")
        retriever = model.text_retriever.to(device).eval()
        stats["text_retriever"] = {"top1": 0, "recall3": 0, "mrr": 0.0}
        print(json.dumps({"loaded": "text_retriever",
                          "parameters": sum(p.numel() for p in retriever.parameters())}), flush=True)

    episodes = answerable = 0

    for row in stream_episodes(Path(args.eval_file)):
        positives = row.get("positive_indices") or []
        if not positives:
            continue
        positive = int(positives[0])
        query_vector, candidate_vectors = episode_tensors(row, lookup, bank)
        query = torch.from_numpy(np.ascontiguousarray(query_vector)).to(device).float().reshape(1, -1)
        candidates = torch.from_numpy(
            np.ascontiguousarray(candidate_vectors)).to(device).float().reshape(1, -1, bank.shape[1])
        episodes += 1
        answerable += 1

        # 1) frozen-key cosine, the runtime's current ranker for fact records
        cosine_scores = F.cosine_similarity(query, candidates.reshape(-1, candidates.shape[-1]), dim=-1)
        rank, top1, in3 = rank_metrics(cosine_scores, positive)
        stats["cosine"]["top1"] += int(top1)
        stats["cosine"]["recall3"] += int(in3)
        stats["cosine"]["mrr"] += 1.0 / rank

        if retriever is not None:
            with torch.no_grad():
                retriever_scores = torch.sigmoid(
                    retriever(query, candidates.reshape(-1, candidates.shape[-1]))
                ).reshape(-1)
            rank, top1, in3 = rank_metrics(retriever_scores, positive)
            stats["text_retriever"]["top1"] += int(top1)
            stats["text_retriever"]["recall3"] += int(in3)
            stats["text_retriever"]["mrr"] += 1.0 / rank

        for label, router in routers:
            with torch.no_grad():
                # Same call shape the runtime's _score_candidates uses: candidates are
                # projected into the router's compact address space, then pair-scored.
                projected = router.encode_key(candidates.reshape(-1, candidates.shape[-1]))
                scores, _ = router.projected_scores(query, projected.reshape(1, -1, projected.shape[-1]))
            scores = scores.reshape(-1)
            rank, top1, in3 = rank_metrics(scores, positive)
            stats[label]["top1"] += int(top1)
            stats[label]["recall3"] += int(in3)
            stats[label]["mrr"] += 1.0 / rank

    report = {
        "eval_file": args.eval_file,
        "episodes": episodes,
        "answerable_episodes": answerable,
        "chance_top1_pct": round(100.0 / 24, 2),
        "rankers": {
            label: {
                "top1_pct": round(100.0 * body["top1"] / max(1, episodes), 2),
                "recall3_pct": round(100.0 * body["recall3"] / max(1, episodes), 2),
                "mrr_pct": round(100.0 * body["mrr"] / max(1, episodes), 2),
            }
            for label, body in stats.items()
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
