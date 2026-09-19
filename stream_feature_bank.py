"""Streaming, multi-threaded frozen-feature builder for router training data.

The original builder loaded every episode into RAM and encoded one text at a time.
That does not scale to the 87k-episode v5 dataset:

* parsing ``train.jsonl`` (1.15 GB) into Python objects costs several GB of RAM,
* the resulting feature bank is 2.12M x 2560 fp16 = **10.86 GB**,
* single-text encoding would need many hours.

This tool instead:

1. **scans** the JSONL files line by line, keeps only a ``sha1 -> row`` lookup and
   spills the unique texts to disk (never holding the parsed episodes);
2. **tokenizes** with a thread pool (Hugging Face fast tokenizers release the GIL);
3. **encodes** into a memory-mapped ``.npy`` bank, in exact-token-length groups so
   the frozen Qwen representation is identical to the original cache, with a
   token budget per batch so a long-text batch cannot blow up VRAM;
4. writes a ``manifest.json`` compatible with the trainer's cache validation.

Peak RAM is the lookup dictionary plus the token/length arrays; the 10.86 GB bank
stays on disk and is memory-mapped by the trainer.

Usage::

    python -m V2_dpskw.stream_feature_bank ^
        --train-file data/router_training_v5/train.jsonl ^
        --eval-file data/router_training_v5/eval.jsonl ^
        --model-path qwen3_5_4b_natural_memory_v2 ^
        --output-dir checkpoints/router_v5/feature_cache ^
        --tokenizer-threads 8 --max-batch 128 --token-budget 8192 --gpu-memory-gb 10
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.qwen_integration import load_qwen_dynamic, load_tokenizer

BANK_NAME = "features.f16.npy"
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.json"
TEXTS_NAME = "texts.jsonl"
TOKENS_NAME = "tokens.i32"
LENGTHS_NAME = "lengths.i32"
PROGRESS_NAME = "progress.json"
FORMAT_VERSION = 2


def _encode_method_fingerprint() -> str:
    """Hash only the method that decides the feature values.

    The fork is shared with other work, so ``qwen_integration.py`` can change
    between two halves of one encode.  Hashing the whole file would be too
    coarse (unrelated edits would block a legitimate resume); hashing exactly
    ``_encode_model_key`` guarantees that a resumed run keeps writing features
    that are comparable with the rows written before the interruption.

    The compiled code object is hashed rather than ``inspect.getsource`` text:
    ``getsource`` slices the *current* file by the code object's stored line
    numbers, so any edit above the method silently shifts the slice and hashes a
    neighbouring function instead (measured: it returned the ``compact_context``
    body for ``_encode_model_key``).  That turned a safety check into a random
    value and would have refused a legitimate resume.  Bytecode is stable across
    line shifts and still changes when the method's logic changes.
    """

    try:
        import marshal

        from V2_dpskw.qwen_integration import QwenDynamicMemoryModel

        code = QwenDynamicMemoryModel._encode_model_key.__code__
        digest = hashlib.sha256()
        digest.update(marshal.dumps(code.co_code))
        digest.update(repr(code.co_consts).encode("utf-8"))
        digest.update(repr(code.co_names).encode("utf-8"))
        digest.update(repr(code.co_varnames).encode("utf-8"))
        digest.update(repr(sorted(code.co_freevars)).encode("utf-8"))
        return digest.hexdigest()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable:{type(exc).__name__}"


def _write_progress(output_dir: Path, completed: int, total: int) -> None:
    """Record how many length-sorted rows are durably written."""

    payload = {
        "completed_rows": int(completed),
        "total_rows": int(total),
        "encode_method_sha256": _encode_method_fingerprint(),
    }
    target = output_dir / PROGRESS_NAME
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload), encoding="utf-8")
    temp.replace(target)


def _read_progress(output_dir: Path, *, text_count: int) -> int:
    target = output_dir / PROGRESS_NAME
    if not target.exists():
        return 0
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return 0
    completed = int(payload.get("completed_rows", 0))
    if int(payload.get("total_rows", -1)) != int(text_count):
        _report({"phase": "resume_rejected", "reason": "text count changed", "progress": payload})
        return 0
    recorded = str(payload.get("encode_method_sha256", ""))
    current = _encode_method_fingerprint()
    if recorded and recorded != current:
        _report({"phase": "resume_rejected", "reason": "encode method changed since the partial run",
                 "recorded": recorded, "current": current})
        return 0
    bank = output_dir / BANK_NAME
    if not bank.exists():
        return 0
    # Validate by array shape, not by byte size: a .npy file carries a header.
    try:
        probe = np.load(bank, mmap_mode="r")
        shape = tuple(probe.shape)
    except Exception:
        return 0
    finally:
        probe = None
    if len(shape) != 2 or int(shape[0]) != int(text_count):
        _report({"phase": "resume_rejected", "reason": "bank shape mismatch",
                 "bank_shape": list(shape), "expected_rows": int(text_count)})
        return 0
    if completed >= text_count:
        return 0
    _report({"phase": "resume_accepted", "completed_rows": completed,
             "encode_method_sha256": recorded[:16]})
    return completed


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _report(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _iter_episode_texts(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(kind, text)`` for the query and every candidate of each episode.

    Episodes are streamed: nothing but the current line is ever materialised, so
    a 1.15 GB file costs no lasting RAM.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = str(row.get("query", "")).strip()
            if query:
                yield "query", query
            candidates = row.get("candidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        text = str(candidate.get("text", "")).strip()
                        if text:
                            yield "candidate", text


def _encode_keys(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, skip_lm_head: bool, backbone: str = "outer") -> torch.Tensor:
    """Masked-mean key encoding.

    ``backbone="outer"`` is the canonical path (identical to
    ``QwenDynamicMemoryModel._encode_model_key``) and is the default so that a
    feature bank never mixes two code paths.

    ``backbone="inner"`` calls the inner transformer directly, which skips the
    LM head entirely.  Measured on 192 *distinct* length-32 texts: 977 ms vs
    1099 ms per batch (1.13x faster) and 4.80 GiB vs 6.98 GiB peak VRAM, with
    feature agreement cosine 0.9999911.  Note that ``skip_lm_head`` is the wrong
    lever for the same goal: asking for one token of logits is 1.55x *slower*
    (see the flag help), because the sliced view is not contiguous.
    """

    if backbone == "inner" and hasattr(model.base_model, "model"):
        target = model.base_model.model
    else:
        if not skip_lm_head:
            return model._encode_model_key(input_ids, attention_mask)
        target = model.base_model
    previous_read = model.runtime.read_enabled
    previous_update = model.runtime.update_enabled
    model.runtime.read_enabled = False
    model.runtime.update_enabled = False
    try:
        embedding_layer = model.base_model.get_input_embeddings()
        input_ids = input_ids.to(embedding_layer.weight.device)
        attention_mask = attention_mask.to(input_ids.device)
        try:
            output = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                logits_to_keep=1,
            )
        except TypeError:
            # Architectures without a logits_to_keep argument.
            output = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        hidden_states = getattr(output, "hidden_states", None)
        hidden = (hidden_states[-1] if hidden_states is not None else output.last_hidden_state).float()
        weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        key = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.nn.functional.normalize(key, dim=-1)
    finally:
        model.runtime.read_enabled = previous_read
        model.runtime.update_enabled = previous_update


def phase_scan(paths: list[Path], output_dir: Path) -> tuple[int, dict[str, int]]:
    """Collect unique texts, spilling them to disk in row order."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lookup: dict[str, int] = {}
    kinds: dict[str, int] = {"query": 0, "candidate": 0}
    started = time.perf_counter()
    with (output_dir / TEXTS_NAME).open("w", encoding="utf-8") as sink:
        for path in paths:
            for kind, text in _iter_episode_texts(path):
                digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
                if digest in lookup:
                    continue
                lookup[digest] = len(lookup)
                kinds[kind] += 1
                sink.write(json.dumps(text, ensure_ascii=False) + "\n")
                if len(lookup) % 250000 == 0:
                    _report({"phase": "scan", "unique_texts": len(lookup),
                             "seconds": round(time.perf_counter() - started, 1)})
    with (output_dir / INDEX_NAME).open("w", encoding="utf-8") as handle:
        json.dump(lookup, handle, ensure_ascii=False)
    _report({"phase": "scan_done", "unique_texts": len(lookup), "kinds": kinds,
             "seconds": round(time.perf_counter() - started, 1)})
    return len(lookup), kinds


def phase_tokenize(
    output_dir: Path,
    tokenizer: Any,
    *,
    text_count: int,
    max_key_tokens: int,
    threads: int,
    chunk: int,
    progress_every: int,
) -> tuple[np.memmap, np.memmap]:
    """Tokenize every unique text with a thread pool."""

    tokens = np.memmap(output_dir / TOKENS_NAME, dtype=np.int32, mode="w+", shape=(text_count, max_key_tokens))
    lengths = np.memmap(output_dir / LENGTHS_NAME, dtype=np.int32, mode="w+", shape=(text_count,))
    started = time.perf_counter()
    done = 0

    def tokenize_chunk(items: list[tuple[int, str]]) -> list[tuple[int, list[int]]]:
        rows = [row for row, _ in items]
        encoded = tokenizer(
            [text for _, text in items],
            add_special_tokens=False,
            truncation=True,
            max_length=max_key_tokens,
            padding=False,
        )["input_ids"]
        return list(zip(rows, encoded))

    with (output_dir / TEXTS_NAME).open("r", encoding="utf-8") as source, ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        batch: list[tuple[int, str]] = []
        pending = []
        for row, line in enumerate(source):
            batch.append((row, json.loads(line)))
            if len(batch) >= chunk:
                pending.append(pool.submit(tokenize_chunk, batch))
                batch = []
            if len(pending) >= max(2, threads * 2):
                for future in pending:
                    for row, ids in future.result():
                        length = min(len(ids), max_key_tokens)
                        lengths[row] = length
                        if length:
                            tokens[row, :length] = np.asarray(ids[:length], dtype=np.int32)
                        else:
                            lengths[row] = 1
                            tokens[row, 0] = 0
                        done += 1
                pending = []
                if done % progress_every < chunk:
                    rate = done / max(1e-9, time.perf_counter() - started)
                    _report({"phase": "tokenize", "done": done, "total": text_count,
                             "texts_per_second": round(rate, 1)})
        if batch:
            pending.append(pool.submit(tokenize_chunk, batch))
        for future in pending:
            for row, ids in future.result():
                length = min(len(ids), max_key_tokens)
                if length == 0:
                    length = 1
                    ids = [0]
                lengths[row] = length
                tokens[row, :length] = np.asarray(ids[:length], dtype=np.int32)
                done += 1
    tokens.flush()
    lengths.flush()
    _report({"phase": "tokenize_done", "texts": int(done),
             "seconds": round(time.perf_counter() - started, 1)})
    return tokens, lengths


def phase_encode(
    output_dir: Path,
    model: Any,
    tokens: np.memmap,
    lengths: np.memmap,
    *,
    text_count: int,
    hidden_size: int,
    max_batch: int,
    token_budget: int,
    progress_every: int,
    skip_lm_head: bool = True,
    backbone: str = "outer",
    resume_from: int = 0,
    fill_from: int = 0,
    fill_to: int = 0,
) -> int:
    """Encode in exact-token-length groups, writing into a memory-mapped bank.

    Returns the number of entries of the length-sorted order that are complete, so
    an interrupted run can resume instead of re-encoding hours of GPU work.
    """

    # "w+" recreates the whole bank; it is only correct for a genuinely fresh run.
    # A resume and a gap-fill must both open the existing file read/write, or they
    # would wipe the rows that were already paid for.
    fresh_run = int(resume_from) == 0 and int(fill_to) == 0
    bank = np.lib.format.open_memmap(
        output_dir / BANK_NAME, mode="w+" if fresh_run else "r+", dtype=np.float16, shape=(text_count, hidden_size)
    )
    length_values = np.asarray(lengths, dtype=np.int64)
    order = np.argsort(length_values, kind="stable")
    sorted_lengths = length_values[order]
    boundaries = np.flatnonzero(np.diff(sorted_lengths)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    device = next(model.base_model.parameters()).device
    started = time.perf_counter()
    done = int(resume_from)
    calls = 0
    processed = 0
    if resume_from:
        _report({"phase": "resume", "completed_rows": resume_from, "total": text_count})
    for start, end in zip(starts.tolist(), ends.tolist()):
        # Resume must trim *within* the partially finished group.  Skipping any
        # group whose ``start`` is below the checkpoint silently dropped every row
        # of that group above the checkpoint: the length-32 group spans ~680k rows,
        # so an interrupted resume lost 294,306 rows while still reporting success.
        low = max(start, done) if resume_from else start
        if fill_to:
            low = max(low, fill_from)
            end = min(end, fill_to)
        if end <= low:
            continue
        rows = order[low:end]
        length = int(sorted_lengths[start])
        capacity = max(1, min(max_batch, token_budget // max(1, length)))
        for offset in range(0, len(rows), capacity):
            block = rows[offset : offset + capacity]
            ids = torch.from_numpy(np.asarray(tokens[block, :length], dtype=np.int64)).to(device, non_blocking=True)
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                vectors = _encode_keys(model, ids, mask, skip_lm_head=skip_lm_head, backbone=backbone).detach().to(torch.float16).cpu().numpy()
            bank[block] = vectors
            done += len(block)
            processed += len(block)
            calls += 1
            if done % progress_every < len(block):
                bank.flush()
                # A gap-fill run does not produce a contiguous prefix, so it must
                # not overwrite the resume checkpoint of the prefix run.
                if not fill_to:
                    _write_progress(output_dir, done, text_count)
                elapsed = max(1e-9, time.perf_counter() - started)
                remaining = (text_count - done) / max(1e-9, processed / elapsed)
                _report({"phase": "encode", "done": done, "total": text_count,
                         "texts_per_second": round(processed / elapsed, 1),
                         "eta_seconds": round(remaining, 0), "length": length, "batch": len(block)})
    bank.flush()
    if not fill_to:
        _write_progress(output_dir, done, text_count)
    _report({"phase": "encode_done", "texts": done, "model_calls": calls,
             "seconds": round(time.perf_counter() - started, 1), "resumed_from": resume_from,
             "fill_from": fill_from, "fill_to": fill_to,
             "complete": bool(not fill_to and done >= text_count)})
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/router_training_v5/train.jsonl")
    parser.add_argument("--eval-file", default="data/router_training_v5/eval.jsonl")
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--output-dir", default="checkpoints/router_v5/feature_cache")
    parser.add_argument("--max-key-tokens", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--tokenizer-threads", type=int, default=8)
    parser.add_argument("--tokenize-chunk", type=int, default=1024)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--gpu-memory-gb", type=float, default=10.0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--skip-lm-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "request one token of logits instead of all tokens.  Features are identical "
            "(measured cosine 0.9999996) but on this bitsandbytes 4-bit stack it is "
            "1.55x SLOWER: the quantised matmul falls into a slow path for the sliced "
            "non-contiguous view (1050 ms vs 679 ms per 192x32 batch), even though it "
            "saves 1.1 GiB of VRAM.  Off by default; kept for the record."
        ),
    )
    parser.add_argument("--backbone", choices=("outer", "inner"), default="outer", help="inner calls the inner transformer directly (1.13x faster, 2.2 GiB less VRAM, features agree at cosine 0.999991); default keeps one consistent code path for a whole bank")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted encode from progress.json")
    parser.add_argument("--fill-rows", default="", help="encode only this half-open row range in length-sorted order, e.g. 750021:1044327")
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument("--limit-texts", type=int, default=0, help="smoke test cap")
    args = parser.parse_args()

    train_path = _resolve(args.train_file)
    eval_path = _resolve(args.eval_file)
    output_dir = _resolve(args.output_dir)
    model_path = _resolve(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _report({"phase": "start", "train_file": str(train_path), "eval_file": str(eval_path),
             "output_dir": str(output_dir), "model_path": str(model_path)})

    text_count, kinds = phase_scan([train_path, eval_path], output_dir)
    if args.limit_texts:
        text_count = min(text_count, args.limit_texts)

    if args.gpu_memory_gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(0.92, max(0.05, args.gpu_memory_gb * 1024**3 / total)), device=0
        )
    tokenizer = load_tokenizer(model_path)
    tokens, lengths = phase_tokenize(
        output_dir, tokenizer, text_count=text_count, max_key_tokens=args.max_key_tokens,
        threads=args.tokenizer_threads, chunk=args.tokenize_chunk, progress_every=args.progress_every,
    )
    del tokenizer
    gc.collect()

    max_memory = None
    if args.gpu_memory_gb > 0 and torch.cuda.is_available():
        max_memory = {0: f"{args.gpu_memory_gb:.1f}GiB", "cpu": "48GiB"}
    model = load_qwen_dynamic(model_path, load_in_4bit=not args.no_4bit, max_memory=max_memory)
    model.eval()
    hidden_size = int(model.memory.hidden_size)
    if hidden_size != args.hidden_size:
        _report({"phase": "hidden_size_override", "expected": args.hidden_size, "actual": hidden_size})
    fill_from = fill_to = 0
    if args.fill_rows:
        try:
            fill_from, fill_to = (int(part) for part in args.fill_rows.split(":"))
        except Exception:
            raise SystemExit("--fill-rows expects START:END")
        if not (0 <= fill_from < fill_to <= text_count):
            raise SystemExit(f"--fill-rows out of range for {text_count} texts")
    encoded = phase_encode(
        output_dir, model, tokens, lengths, text_count=text_count, hidden_size=hidden_size,
        max_batch=args.max_batch, token_budget=args.token_budget, progress_every=args.progress_every,
        skip_lm_head=args.skip_lm_head,
        backbone=args.backbone,
        resume_from=_read_progress(output_dir, text_count=text_count) if (args.resume and not fill_to) else 0,
        fill_from=fill_from,
        fill_to=fill_to,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    manifest = {
        "format_version": FORMAT_VERSION,
        "train_sha256": _sha256(train_path),
        "eval_sha256": _sha256(eval_path),
        "model_path": str(args.model_path),
        "max_key_tokens": int(args.max_key_tokens),
        "hidden_size": int(hidden_size),
        "text_count": int(text_count),
        "dtype": "float16_memmap",
        "bank": BANK_NAME,
        "index": INDEX_NAME,
        "texts": TEXTS_NAME,
        "kinds": kinds,
        # A manifest is only written with complete=true when this process encoded
        # every row it was asked to; the trainer refuses an incomplete bank.  This
        # flag exists because an interrupted encode used to still emit a manifest
        # that looked complete, and a training run was started on zero rows.
        "complete": bool(not fill_to and encoded >= text_count),
        "encoded_rows": int(encoded),
        "encoding": {
            "tokenizer_threads": args.tokenizer_threads,
            "max_batch": args.max_batch,
            "token_budget": args.token_budget,
            "grouping": "exact_token_length",
            "skip_lm_head": bool(args.skip_lm_head),
            "backbone": args.backbone,
            "fill_rows": args.fill_rows or None,
        },
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _report({"phase": "done", **manifest})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
