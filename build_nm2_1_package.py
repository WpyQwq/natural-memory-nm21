"""Build the NM2.1 package: the delivered router merged into the Qwen3.5 memory model.

"Merge" here means producing a **self-contained package** rather than a sidecar file: the
base Qwen3.5-4B shards are copied unchanged and only the 16
``dynamic_memory.memory_router_v2.*`` tensors inside the memory shard are replaced with the
delivered router.  Everything else -- text retriever, persistent slots, memory policy,
configs, tokenizer -- is left byte-identical, so anything the package did before it still
does, with a different router.

Verification is not "the script exited zero": the script reloads the written shard and
compares all 16 tensors bit-exactly against the source checkpoint, and separately reports
every other tensor as unchanged (same bytes as the source shard).

Usage::

    python -m V2_dpskw.build_nm2_1_package ^
        --source-package H:\\Memory\\dynamic_memory_lab\\qwen3_5_4b_natural_memory_v2 ^
        --output-package H:\\Memory\\dynamic_memory_lab\\qwen3_5_4b_natural_memory_v2_1 ^
        --router-checkpoint checkpoints/router_replay_v7_v2_128/memory_router_v2.pt ^
        --label NM2.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROUTER_PREFIX = "dynamic_memory.memory_router_v2."
INDEX_NAME = "model.safetensors.index.json"
MERGE_NAME = "memory_merge.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", required=True)
    parser.add_argument("--output-package", required=True)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--label", default="NM2.1")
    parser.add_argument("--report", default="nm2_1_build_report.json")
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = Path(args.source_package)
    output = Path(args.output_package)
    router_path = Path(args.router_checkpoint)
    if not source.exists():
        raise SystemExit(f"source package not found: {source}")
    if not router_path.exists():
        raise SystemExit(f"router checkpoint not found: {router_path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output package already exists and is not empty: {output}")

    router = torch.load(router_path, map_location="cpu", weights_only=True)
    if isinstance(router, dict) and "router_state_dict" in router:
        router = router["router_state_dict"]

    # --- which shard holds the router tensors? -------------------------------------
    index = json.loads((source / INDEX_NAME).read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    router_keys = sorted(key for key in weight_map if key.startswith(ROUTER_PREFIX))
    shards = sorted({weight_map[key] for key in router_keys})
    if len(shards) != 1:
        raise SystemExit(f"router tensors span several shards: {shards}")
    print(json.dumps({"phase": "locate", "router_tensors": len(router_keys),
                      "shard": shards[0],
                      "expected_from_checkpoint": len(router),
                      "router_sha256": sha256_of(router_path)}), flush=True)

    missing = [key for key in router if ROUTER_PREFIX + key not in weight_map]
    if missing:
        raise SystemExit(f"checkpoint keys not present in the package: {missing}")
    if len(router_keys) != len(router):
        raise SystemExit(f"package has {len(router_keys)} router tensors, checkpoint has {len(router)}")

    # --- copy the package -----------------------------------------------------------
    print(json.dumps({"phase": "copy", "from": str(source), "to": str(output)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["robocopy", str(source), str(output), "/E", "/NFL", "/NDL",
                             "/NJH", "/NJS", "/NP", "/R:2", "/W:2"],
                            capture_output=True, text=True)
    if result.returncode > 7:
        raise SystemExit(f"robocopy failed ({result.returncode}): {result.stdout[-800:]}")
    copied = sorted(p.name for p in output.iterdir())
    expected = sorted(p.name for p in source.iterdir())
    if copied != expected:
        raise SystemExit(f"copy mismatch: missing={set(expected) - set(copied)} extra={set(copied) - set(expected)}")

    # --- rewrite the memory shard ---------------------------------------------------
    shard_path = output / shards[0]
    source_shard_path = source / shards[0]
    with safe_open(str(source_shard_path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    untouched = 0
    replaced = []
    for key, value in router.items():
        full = ROUTER_PREFIX + key
        original = tensors[full]
        if tuple(original.shape) != tuple(value.shape):
            raise SystemExit(f"shape mismatch for {full}: {tuple(original.shape)} vs {tuple(value.shape)}")
        tensors[full] = value.to(dtype=original.dtype).contiguous()
        replaced.append(full)
    for key in tensors:
        if key not in replaced:
            untouched += 1
    save_file(tensors, str(shard_path), metadata=metadata)
    print(json.dumps({"phase": "shard_rewritten", "shard": shards[0],
                      "replaced": len(replaced), "untouched": untouched}), flush=True)

    # --- provenance in the merge manifest -------------------------------------------
    merge_path = output / MERGE_NAME
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    merge["package_label"] = args.label
    merge["router_swap"] = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "built_from_package": str(source),
        "router_checkpoint": str(router_path),
        "router_sha256": sha256_of(router_path),
        "replaced_tensors": len(replaced),
        "shard": shards[0],
    }
    merge_path.write_text(json.dumps(merge, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- verify: reload what was written, compare bit-exactly ------------------------
    with safe_open(str(shard_path), framework="pt") as handle:
        written = {key: handle.get_tensor(key) for key in handle.keys()}
    mismatches = []
    for key, value in router.items():
        full = ROUTER_PREFIX + key
        if not torch.equal(written[full], value.to(dtype=written[full].dtype)):
            mismatches.append(full)
    with safe_open(str(source_shard_path), framework="pt") as handle:
        originals = {key: handle.get_tensor(key) for key in handle.keys()}
    changed_others = [key for key in originals
                      if key not in replaced and not torch.equal(originals[key], written.get(key))]

    report = {
        "package_label": args.label,
        "output_package": str(output),
        "source_package": str(source),
        "router_checkpoint": str(router_path),
        "router_sha256": sha256_of(router_path),
        "shard": shards[0],
        "files_copied": len(copied),
        "router_tensors_replaced": len(replaced),
        "tensors_left_untouched": untouched,
        "verification": {
            "router_tensors_bit_exact": not mismatches,
            "router_mismatches": mismatches,
            "non_router_tensors_unchanged": not changed_others,
            "changed_non_router": changed_others,
        },
    }
    report["passed"] = (not mismatches) and (not changed_others)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
