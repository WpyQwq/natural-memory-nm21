"""Probe whether a faster encoder precision reproduces the frozen 4-bit features.

The v6 feature encode needs ~3 hours because the 4B model runs in 4-bit NF4,
which dequantises every weight on every forward.  A bf16 forward is usually much
faster, but the deployed routers were trained on 4-bit features, so switching
precision is only acceptable if it does not move the *downstream* numbers.

This tool answers that empirically instead of by assumption:

1. rebuild the exact text list of an existing cache deterministically from the
   dataset (and verify the ``sha1`` set matches the reference ``index.json``);
2. encode it with a chosen precision into a second cache;
3. report feature-level agreement (cosine) against the reference bank.

Feeding both caches to ``eval_router_scorecard`` then shows whether the router
metrics - and therefore any conclusion drawn from them - survive the change.

Usage::

    python -m V2_dpskw.probe_encoder_precision ^
        --train-file data/router_training_v3/train.jsonl ^
        --eval-file data/router_training_v3/eval.jsonl ^
        --reference-cache checkpoints/router_shared/feature_cache ^
        --output-dir H:\\Memory\\nm_cache\\nm_probe_bf16 --precision bf16
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.qwen_integration import load_qwen_dynamic, load_tokenizer


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent / path


def _text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="replace")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_texts(paths: list[Path]) -> tuple[list[str], dict[str, int]]:
    """Reproduce the trainer's text order: first appearance across train+eval."""

    texts: list[str] = []
    lookup: dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                values = [str(row.get("query", ""))]
                for candidate in row.get("candidates") or []:
                    if isinstance(candidate, dict):
                        values.append(str(candidate.get("text", "")))
                for text in values:
                    text = text.strip()
                    if not text:
                        continue
                    key = _text_key(text)
                    if key not in lookup:
                        lookup[key] = len(texts)
                        texts.append(text)
    return texts, lookup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/router_training_v3/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v3/eval.jsonl")
    parser.add_argument("--reference-cache", default="checkpoints/router_shared/feature_cache")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("bf16", "4bit"), default="bf16")
    parser.add_argument("--max-key-tokens", type=int, default=256)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--gpu-memory-gb", type=float, default=10.5)
    args = parser.parse_args()

    train_path = _resolve(args.train_file)
    eval_path = _resolve(args.eval_file)
    reference_dir = _resolve(args.reference_cache)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_index = json.loads((reference_dir / "index.json").read_text(encoding="utf-8"))
    texts, lookup = collect_texts([train_path, eval_path])
    print(json.dumps({
        "phase": "texts",
        "collected": len(texts),
        "reference": len(reference_index),
        "sha_set_identical": set(lookup) == set(reference_index),
        "row_order_identical": lookup == reference_index,
    }, ensure_ascii=False), flush=True)

    if args.gpu_memory_gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.95, args.gpu_memory_gb * 1024**3 / total), device=0)

    tokenizer = load_tokenizer(_resolve(args.model_path))
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=args.max_key_tokens, padding=False)["input_ids"]
    lengths = np.array([max(1, min(len(ids), args.max_key_tokens)) for ids in encoded], dtype=np.int64)
    order = np.argsort(lengths, kind="stable")
    print(json.dumps({"phase": "tokenized", "texts": len(texts), "mean_len": float(lengths.mean())}, ensure_ascii=False), flush=True)

    four_bit = args.precision == "4bit"
    model = load_qwen_dynamic(
        _resolve(args.model_path), load_in_4bit=four_bit,
        max_memory={0: f"{args.gpu_memory_gb:.1f}GiB", "cpu": "48GiB"} if torch.cuda.is_available() else None,
    )
    model.eval()
    hidden = int(model.memory.hidden_size)
    device = next(model.base_model.parameters()).device

    bank = np.lib.format.open_memmap(
        output_dir / "features.f16.npy", mode="w+", dtype=np.float16, shape=(len(texts), hidden)
    )
    boundaries = np.flatnonzero(np.diff(lengths[order])) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    started = time.perf_counter()
    done = 0
    for start, end in zip(starts.tolist(), ends.tolist()):
        rows = order[start:end]
        length = int(lengths[start])
        capacity = max(1, min(args.max_batch, args.token_budget // max(1, length)))
        for offset in range(0, len(rows), capacity):
            block = rows[offset : offset + capacity]
            ids = torch.tensor([encoded[i][:length] for i in block], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                vectors = model._encode_model_key(ids, mask).detach().to(torch.float16).cpu().numpy()
            bank[block] = vectors
            done += len(block)
        if done % 5000 < capacity:
            rate = done / max(1e-9, time.perf_counter() - started)
            print(json.dumps({"phase": "encode", "precision": args.precision, "done": done,
                              "total": len(texts), "texts_per_second": round(rate, 1)}), flush=True)
    bank.flush()
    seconds = time.perf_counter() - started
    tokens = int(lengths.sum())
    print(json.dumps({
        "phase": "encode_done", "precision": args.precision, "texts": len(texts),
        "seconds": round(seconds, 1), "texts_per_second": round(len(texts) / max(1e-9, seconds), 1),
        "tokens_per_second": round(tokens / max(1e-9, seconds), 1),
    }, ensure_ascii=False), flush=True)

    (output_dir / "index.json").write_text(json.dumps(lookup, ensure_ascii=False), encoding="utf-8")
    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"dtype": f"float16_{args.precision}", "bank": "features.f16.npy",
                     "text_count": len(texts), "encoded_by": "probe_encoder_precision.py",
                     "precision": args.precision})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Feature-level agreement against the reference bank.
    reference_path = reference_dir / ("features.pt" if (reference_dir / "features.pt").exists() else "features.f16.npy")
    if reference_path.suffix == ".pt":
        reference = torch.load(reference_path, map_location="cpu", weights_only=True).float()
    else:
        reference = torch.from_numpy(np.load(reference_path, mmap_mode="r")).float()
    mine = torch.from_numpy(np.asarray(bank, dtype=np.float32))
    common = [key for key in lookup if key in reference_index]
    rows_mine = torch.tensor([lookup[key] for key in common])
    rows_ref = torch.tensor([reference_index[key] for key in common])
    a = reference[rows_ref]
    b = mine[rows_mine]
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(json.dumps({
        "phase": "compare", "precision": args.precision, "shared_texts": len(common),
        "cosine_min": float(cosine.min()), "cosine_mean": float(cosine.mean()),
        "cosine_p01": float(cosine.kthvalue(max(1, int(len(cosine) * 0.01))).values),
        "max_abs_diff": float((a - b).abs().max()),
    }, ensure_ascii=False), flush=True)
    del model, reference, mine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
