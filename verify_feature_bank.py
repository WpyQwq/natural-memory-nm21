"""Verify a streamed feature bank is actually complete, and record the evidence.

The encoder writes into a preallocated memory-mapped ``.npy`` file, so an
interrupted or mis-resumed run leaves *zero* rows that look like valid data to
everything downstream.  That failure already happened once: a resume guard
skipped whole length groups, 494k rows stayed zero, and a training run started
on them before anyone noticed.

This tool makes the check explicit and auditable:

1. scan every row of the bank and count all-zero rows;
2. map any zero rows back to length-sorted order and print the missing ranges,
   so a targeted ``--fill-rows`` run can repair them;
3. when nothing is missing, set ``complete=true`` in the manifest and record that
   the claim is backed by a full scan (the trainer refuses an incomplete bank).

Usage::

    python -m V2_dpskw.verify_feature_bank --cache-dir H:\\Memory\\nm_cache\\nm_router_v6\\feature_cache
    python -m V2_dpskw.verify_feature_bank --cache-dir ... --no-update-manifest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BANK_NAME = "features.f16.npy"
MANIFEST_NAME = "manifest.json"
LENGTHS_NAME = "lengths.i32"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--chunk", type=int, default=100000)
    parser.add_argument("--no-update-manifest", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    bank = np.load(cache_dir / BANK_NAME, mmap_mode="r")
    rows = int(bank.shape[0])
    started = time.perf_counter()
    zero = np.zeros(rows, dtype=bool)
    for start in range(0, rows, max(1, args.chunk)):
        block = np.asarray(bank[start : start + args.chunk])
        zero[start : start + len(block)] = ~block.any(axis=1)
        if start % (max(1, args.chunk) * 5) == 0:
            print(json.dumps({"phase": "scan", "scanned": min(start + args.chunk, rows),
                              "rows": rows, "zero_so_far": int(zero.sum())}), flush=True)

    report: dict = {
        "bank": str(cache_dir / BANK_NAME),
        "rows": rows,
        "zero_rows": int(zero.sum()),
        "scan_seconds": round(time.perf_counter() - started, 1),
        "missing_ranges_length_sorted": [],
        "complete": bool(zero.sum() == 0),
    }
    lengths_path = cache_dir / LENGTHS_NAME
    if lengths_path.exists():
        lengths = np.fromfile(lengths_path, dtype=np.int32)
        if len(lengths) == rows:
            order = np.argsort(lengths, kind="stable")
            rank = np.empty(rows, dtype=np.int64)
            rank[order] = np.arange(rows)
            missing_ranks = np.sort(rank[zero]) if zero.any() else np.array([], dtype=np.int64)
            if len(missing_ranks):
                runs: list[list[int]] = []
                start = prev = int(missing_ranks[0])
                for value in missing_ranks[1:]:
                    value = int(value)
                    if value != prev + 1:
                        runs.append([start, prev + 1])
                        start = value
                    prev = value
                runs.append([start, prev + 1])
                report["missing_ranges_length_sorted"] = [
                    {"from": a, "to": b, "rows": b - a, "token_length": int(lengths[order[a]])}
                    for a, b in runs
                ]
                report["fill_command_hint"] = " ".join(
                    f"--fill-rows {a}:{b}" for a, b in runs
                )

    if report["complete"] and not args.no_update_manifest:
        manifest_path = cache_dir / MANIFEST_NAME
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["complete"] = True
            manifest["encoded_rows"] = rows
            manifest["completeness_verified_by"] = "full_bank_zero_row_scan"
            manifest["completeness_scan_seconds"] = report["scan_seconds"]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            report["manifest_updated"] = True

    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
