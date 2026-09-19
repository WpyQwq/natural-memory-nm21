"""One-off migration: repoint feature-bank paths from D:\\ to H:\\Memory\\nm_cache.

The feature banks were moved off D: because that volume ran short of space.  Only
*live* references are rewritten -- defaults, run scripts and documentation.  Historical
run records (``metrics.jsonl``, ``training_stdout.log``, ``router_v5_training.json``,
``feature_bank_verification.json``) are deliberately left untouched: they are evidence of
what actually ran and where the data lived at that moment, and rewriting them would
falsify the record.

Run with ``--apply`` to write; without it, the script only reports what would change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NEW_ROOT = r"H:\Memory\nm_cache"

#: old path fragment -> new path fragment (with the escaped form handled separately)
MOVES = {
    r"D:\nm_router_v6\feature_cache": NEW_ROOT + r"\nm_router_v6\feature_cache",
    r"D:\nm_router_v5\feature_cache": NEW_ROOT + r"\nm_router_v5\feature_cache",
    r"D:\nm_zero_overlap\feature_cache": NEW_ROOT + r"\nm_zero_overlap\feature_cache",
    r"D:\nm_replay_v7\feature_cache": NEW_ROOT + r"\nm_replay_v7\feature_cache",
    r"D:\nm_probe_bf16": NEW_ROOT + r"\nm_probe_bf16",
}

#: Live files only.  Historical logs and per-run training records are excluded on purpose.
TARGETS = [
    "bench_router_latency.py",
    "build_replay_corpus.py",
    "compare_record_rankers.py",
    "chain_v5.ps1",
    "chain_v6.ps1",
    "eval_router_v5.py",
    "probe_encoder_precision.py",
    "train_router_v5.py",
    "train_and_score_v6_128.ps1",
    "train_and_score_v6.ps1",
    "run_router_v6.ps1",
    "run_router_v5.ps1",
    "verify_feature_bank.py",
    "README_FORK.md",
    "V6_FINAL_REPORT.md",
    "ZERO_OVERLAP_FINDINGS.md",
    "ZERO_OVERLAP_PHASE.md",
    "zero_overlap_phase.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    changes = []
    for relative in TARGETS:
        path = ROOT / relative
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        text = original
        for old, new in MOVES.items():
            text = text.replace(old, new)
            # Escaped form, e.g. inside Python docstrings and PowerShell here-strings.
            escaped_old = old.replace("\\", "\\\\")
            escaped_new = new.replace("\\", "\\\\")
            text = text.replace(escaped_old, escaped_new)
        if text != original:
            count = sum(original.count(old) + original.count(old.replace("\\", "\\\\"))
                        for old in MOVES)
            changes.append({"file": relative, "replacements": count})
            if args.apply:
                path.write_text(text, encoding="utf-8")

    report = {
        "applied": args.apply,
        "new_root": NEW_ROOT,
        "files_changed": changes,
        "total_replacements": sum(item["replacements"] for item in changes),
        "note": ("historical run records were intentionally left unmodified: "
                 "checkpoints/*/metrics.jsonl, checkpoints/*/training_stdout.log, "
                 "checkpoints/*/router_v5_training.json, feature_bank_verification.json"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Verify no live reference remains.
    stale = []
    for relative in TARGETS:
        path = ROOT / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hits = [old for old in MOVES if old in text or old.replace("\\", "\\\\") in text]
        if hits:
            stale.append({"file": relative, "remaining": hits})
    if stale:
        print(json.dumps({"remaining_d_references": stale}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"remaining_d_references": []}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
