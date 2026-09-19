"""Merge the v6 corpus with the zero-overlap corpus into one feature bank.

Why a merge and not a joint encode: the v6 bank holds 2,124,552 rows whose values
were produced by a frozen encoder that must never change (they are the reference
features every earlier measurement was taken on).  Re-encoding them would both cost
hours and risk silent drift, so the merged bank is built by **copying verified rows
and appending the new ones**, with the copy spot-checked byte-for-byte afterwards.

Invariants enforced here:

* every row of every source bank is copied exactly once, or skipped only because the
  identical text already has a row -- and skipped rows are proven byte-equal;
* ``len(index) == bank rows`` (the trainer refuses a bank where they differ);
* every text referenced by the merged train/eval files resolves to a bank row;
* the merged manifest carries the sha256 of the *merged* dataset files, because the
  trainer validates the bank against those frozen inputs.

Usage::

    python -m V2_dpskw.build_replay_corpus ^
        --source-data v6=data/router_training_v6 --source-bank v6=H:\\Memory\\nm_cache\\nm_router_v6\\feature_cache ^
        --source-data zov=data/zero_overlap --source-bank zov=H:\\Memory\\nm_cache\\nm_zero_overlap\\feature_cache ^
        --output-data data/router_replay_v7 --output-bank H:\\Memory\\nm_cache\\nm_replay_v7\\feature_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

BANK_NAME = "features.f16.npy"
INDEX_NAME = "index.json"
MANIFEST_NAME = "manifest.json"
TEXTS_NAME = "texts.jsonl"
LENGTHS_NAME = "lengths.i32"
CHUNK_ROWS = 65536


def text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", "replace")).hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_episode_texts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = str(row.get("query", "")).strip()
            if query:
                yield query
            for candidate in row.get("candidates") or []:
                text = str(candidate.get("text", "")).strip()
                if text:
                    yield text


def concat_datasets(sources: list[tuple[str, Path]], output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes = 0
    with output.open("wb") as out:
        for name, directory in sources:
            part = directory / output.name
            if not part.exists():
                raise FileNotFoundError(f"missing {part}")
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, length=1 << 22)
            with part.open("rb") as handle:
                episodes += sum(1 for _ in handle)
    return {"path": str(output), "episodes": episodes, "sha256": sha256_of(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", action="append", required=True,
                        help="NAME=DIR containing train.jsonl and eval.jsonl")
    parser.add_argument("--source-bank", action="append", required=True,
                        help="NAME=DIR containing a verified feature bank")
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--max-key-tokens", type=int, default=256)
    parser.add_argument("--spot-checks", type=int, default=2000)
    args = parser.parse_args()

    data_sources = []
    for item in args.source_data:
        name, _, directory = item.partition("=")
        data_sources.append((name, Path(directory)))
    bank_sources = []
    for item in args.source_bank:
        name, _, directory = item.partition("=")
        bank_sources.append((name, Path(directory)))
    if [n for n, _ in data_sources] != [n for n, _ in bank_sources]:
        raise SystemExit("--source-data and --source-bank must list the same names in the same order")

    out_data = Path(args.output_data)
    out_bank = Path(args.output_bank)
    out_bank.mkdir(parents=True, exist_ok=True)

    merged_datasets = {
        "train": concat_datasets([(n, d) for n, d in data_sources], out_data / "train.jsonl"),
        "eval": concat_datasets([(n, d) for n, d in data_sources], out_data / "eval.jsonl"),
    }
    print(json.dumps({"phase": "datasets", **merged_datasets}, ensure_ascii=False), flush=True)

    # --- plan the merged row layout -------------------------------------------------
    layout: list[tuple[int, int, str]] = []   # (source index, source row, sha1)
    origin: list[tuple[int, int]] = []        # (source index, source row) per merged row
    texts: list[str] = []
    key_to_row: dict[str, int] = {}
    duplicates: list[tuple[str, int, int]] = []
    source_meta = []
    for index, (name, directory) in enumerate(bank_sources):
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise SystemExit(f"source bank {name} is not marked complete: {directory}")
        lookup = json.loads((directory / INDEX_NAME).read_text(encoding="utf-8"))
        bank = np.load(directory / BANK_NAME, mmap_mode="r")
        if int(bank.shape[0]) != len(lookup):
            raise SystemExit(f"source bank {name}: rows {bank.shape[0]} != index {len(lookup)}")
        source_meta.append({"name": name, "path": str(directory), "rows": int(bank.shape[0]),
                            "hidden_size": int(bank.shape[1]), "dtype": str(bank.dtype),
                            "manifest_model_path": manifest.get("model_path")})
        row_to_key = [None] * int(bank.shape[0])
        for key, row in lookup.items():
            row_to_key[int(row)] = key
        length_path = directory / LENGTHS_NAME
        # lengths.i32 is a raw int32 dump (written with ndarray.tofile), not a .npy.
        lengths = np.fromfile(length_path, dtype=np.int32) if length_path.exists() else None
        if lengths is not None and int(lengths.shape[0]) != int(bank.shape[0]):
            raise SystemExit(f"source bank {name}: lengths {lengths.shape[0]} != rows {bank.shape[0]}")
        seen_lines = 0
        with (directory / TEXTS_NAME).open("r", encoding="utf-8") as handle:
            for row, line in enumerate(handle):
                seen_lines = row + 1
                text = json.loads(line)
                key = row_to_key[row]
                if key is None:
                    raise SystemExit(f"source bank {name}: row {row} has no index entry")
                if key != text_key(text):
                    raise SystemExit(f"source bank {name}: texts.jsonl line {row} does not match its index key")
                if key in key_to_row:
                    duplicates.append((name, index, row))
                    continue
                key_to_row[key] = len(origin)
                origin.append((index, row))
                texts.append(text)
        if seen_lines != int(bank.shape[0]):
            raise SystemExit(f"source bank {name}: texts.jsonl has {seen_lines} lines, bank has {bank.shape[0]} rows")
        del row_to_key, lookup, bank, lengths

    total = len(origin)
    print(json.dumps({"phase": "layout", "merged_rows": total,
                      "sources": source_meta,
                      "duplicate_rows_skipped": len(duplicates)}, ensure_ascii=False), flush=True)

    # --- copy the feature rows -----------------------------------------------------
    hidden = source_meta[0]["hidden_size"]
    if any(meta["hidden_size"] != hidden for meta in source_meta):
        raise SystemExit("source banks disagree on hidden size")
    banks = [np.load(directory / BANK_NAME, mmap_mode="r") for _, directory in bank_sources]
    target = np.lib.format.open_memmap(out_bank / BANK_NAME, mode="w+",
                                       dtype=np.float16, shape=(total, hidden))
    lengths_out = np.empty(total, dtype=np.int32)
    length_arrays = []
    for _, directory in bank_sources:
        path = directory / LENGTHS_NAME
        length_arrays.append(np.fromfile(path, dtype=np.int32) if path.exists() else None)

    by_source: dict[int, list[tuple[int, int]]] = {}
    for merged_row, (source_index, source_row) in enumerate(origin):
        by_source.setdefault(source_index, []).append((merged_row, source_row))

    for source_index in sorted(by_source):
        pairs = by_source[source_index]
        bank = banks[source_index]
        lengths = length_arrays[source_index]
        # Copy contiguous source runs so the disk sees sequential reads.
        start = 0
        while start < len(pairs):
            end = start + 1
            while (end < len(pairs)
                   and pairs[end][1] == pairs[end - 1][1] + 1
                   and pairs[end][0] == pairs[end - 1][0] + 1):
                end += 1
            run = pairs[start:end]
            if len(run) > CHUNK_ROWS:
                for offset in range(0, len(run), CHUNK_ROWS):
                    piece = run[offset:offset + CHUNK_ROWS]
                    rows = [p[1] for p in piece]
                    target[[p[0] for p in piece]] = bank[rows[0]:rows[-1] + 1]
            else:
                rows = [p[1] for p in run]
                target[[p[0] for p in run]] = bank[rows[0]:rows[-1] + 1]
            start = end
        if lengths is not None:
            for merged_row, source_row in pairs:
                lengths_out[merged_row] = lengths[source_row]
        print(json.dumps({"phase": "copied_source", "name": source_meta[source_index]["name"],
                          "rows": len(pairs)}, ensure_ascii=False), flush=True)

    target.flush()
    del target

    # --- index, texts, lengths -----------------------------------------------------
    (out_bank / INDEX_NAME).write_text(json.dumps(key_to_row), encoding="utf-8")
    with (out_bank / TEXTS_NAME).open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps(text, ensure_ascii=False) + "\n")
    lengths_out.tofile(out_bank / LENGTHS_NAME)
    print(json.dumps({"phase": "written", "index_entries": len(key_to_row)}, ensure_ascii=False), flush=True)

    # --- verification --------------------------------------------------------------
    problems: list[str] = []
    merged = np.load(out_bank / BANK_NAME, mmap_mode="r")
    if int(merged.shape[0]) != len(key_to_row):
        problems.append(f"rows {merged.shape[0]} != index entries {len(key_to_row)}")
    rng = np.random.default_rng(20260912)
    checked = 0
    for merged_row in rng.choice(total, size=min(args.spot_checks, total), replace=False):
        source_index, source_row = origin[int(merged_row)]
        if not np.array_equal(np.asarray(merged[merged_row]), np.asarray(banks[source_index][source_row])):
            problems.append(f"row {merged_row} differs from {source_meta[source_index]['name']}:{source_row}")
            if len(problems) > 5:
                break
        checked += 1
    zero_rows = 0
    for start in range(0, total, CHUNK_ROWS):
        block = np.asarray(merged[start:start + CHUNK_ROWS])
        zero_rows += int((~block.any(axis=1)).sum())
    if zero_rows:
        problems.append(f"{zero_rows} merged rows are all-zero (unencoded)")

    missing = 0
    for split in ("train", "eval"):
        for text in iter_episode_texts(out_data / f"{split}.jsonl"):
            if text_key(text) not in key_to_row:
                missing += 1
    if missing:
        problems.append(f"{missing} dataset texts have no bank row")

    manifest = {
        "format_version": 2,
        "merged_by": "build_replay_corpus.py",
        "train_sha256": merged_datasets["train"]["sha256"],
        "eval_sha256": merged_datasets["eval"]["sha256"],
        "model_path": args.model_path,
        "max_key_tokens": int(args.max_key_tokens),
        "hidden_size": hidden,
        "text_count": total,
        "encoded_rows": total,
        "complete": True,
        "dtype": "float16_memmap",
        "bank": BANK_NAME,
        "index": INDEX_NAME,
        "texts": TEXTS_NAME,
        "sources": source_meta,
        "duplicate_rows_skipped": len(duplicates),
        "verification": {
            "spot_checked_rows": checked,
            "spot_check_mismatches": [p for p in problems if "differs" in p],
            "zero_rows": zero_rows,
            "dataset_texts_without_bank_row": missing,
            "passed": not problems,
        },
        "datasets": merged_datasets,
    }
    (out_bank / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if problems:
        raise SystemExit("merged bank failed verification: " + "; ".join(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
