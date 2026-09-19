"""Fail-fast check that router training can reuse a frozen feature cache.

The XL router reuses the exact same frozen Qwen features as the 512-dim
baseline, so training must never re-encode the corpus.  ``_prepare_feature_cache``
silently loads the 4B model when the cache looks stale, which would cost ~20
minutes and 9 GiB of VRAM.  This tool runs the real cache-validation path with
the model loader replaced by a hard failure, so a stale cache is reported in
seconds instead of being discovered halfway through a training launch.

Usage (from the fork root, e.g. H:\\Memory\\V2_dpskw)::

    python -m V2_dpskw.check_router_cache ^
        --train-file data/router_training_v3/train.jsonl ^
        --eval-file  data/router_training_v3/eval.jsonl ^
        --feature-cache-dir checkpoints/router_shared/feature_cache ^
        --model-path qwen3_5_4b_natural_memory_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import V2_dpskw.train_memory_router_large as trainer


def _explode(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError(
        "feature cache is stale: the trainer would now load the 4B Qwen model to "
        "re-encode the corpus. Fix the cache or pass --rebuild-features deliberately."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/router_training_v3/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v3/eval.jsonl")
    parser.add_argument("--feature-cache-dir", default="checkpoints/router_shared/feature_cache")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--max-key-tokens", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2560)
    args = parser.parse_args()

    train_path = trainer._resolve_path(args.train_file)
    eval_path = trainer._resolve_path(args.eval_file)
    cache_dir = trainer._resolve_path(args.feature_cache_dir)
    model_path = trainer._resolve_path(args.model_path)
    print(json.dumps({
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "feature_cache_dir": str(cache_dir),
        "model_path": str(model_path),
    }, ensure_ascii=False, indent=2), flush=True)

    train = trainer._read_episodes(train_path)
    evaluation = trainer._read_episodes(eval_path)
    texts, _lookup = trainer._collect_texts(train + evaluation)
    expected = {
        "format_version": 1,
        "train_sha256": trainer._sha256(train_path),
        "eval_sha256": trainer._sha256(eval_path),
        "model_path": str(model_path),
        "max_key_tokens": int(args.max_key_tokens),
        "hidden_size": int(args.hidden_size),
        "text_count": len(texts),
        "dtype": "float16_cpu",
    }
    saved_path = trainer._cache_meta_path(cache_dir)
    saved = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.exists() else {}
    compatible = trainer._cache_is_compatible(cache_dir, expected)
    for key in sorted(expected):
        mark = "==" if saved.get(key) == expected[key] else "!="
        print(f"{mark} {key}: expected={expected[key]!r} saved={saved.get(key)!r}", flush=True)
    if not compatible:
        print("CACHE MISS", flush=True)
        return 1

    # Run the real code path with the model loader disabled.
    trainer.load_qwen_dynamic = _explode  # type: ignore[assignment]
    trainer.load_tokenizer = _explode  # type: ignore[assignment]
    vectors, lookup, meta = trainer._prepare_feature_cache(
        argparse.Namespace(
            feature_cache_dir=str(cache_dir),
            model_path=str(model_path),
            max_key_tokens=int(args.max_key_tokens),
            hidden_size=int(args.hidden_size),
            rebuild_features=False,
            precompute_features=True,
            gpu_memory_gb=0.0,
            no_4bit=False,
            encode_batch_size=1,
            precompute_log_every=256,
        ),
        train,
        evaluation,
        train_path,
        eval_path,
    )
    print(json.dumps({
        "verdict": "CACHE HIT",
        "feature_shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "unique_texts": len(lookup),
        "train_episodes": len(train),
        "eval_episodes": len(evaluation),
        "meta": meta,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
