"""Check that a new router checkpoint is a drop-in replacement for the deployed one.

The shipped router lives inside the model package's memory safetensors shard under
``dynamic_memory.memory_router_v2.*``.  Before claiming a new router "replaces" it,
verify mechanically that:

1. the key sets and tensor shapes are identical (modulo the shard prefix), so the
   swap is a rename rather than a runtime change;
2. the checkpoint loads into the same ``MemoryRouterV2`` construction the runtime
   uses, and produces finite scores on real candidates;
3. the runtime path (``PagedMemoryBankV2`` driven by that router) still routes.

Usage::

    python -m V2_dpskw.check_router_swap ^
        --package qwen3_5_4b_natural_memory_v2 ^
        --candidate checkpoints/router_v6_v2_128/router_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import MemoryRouterV2, PagedMemoryBankV2

PREFIX = "dynamic_memory.memory_router_v2."


def _shard_for(package: Path, needle: str) -> tuple[Path, dict]:
    index = json.loads((package / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    keys = [key for key in weight_map if needle in key]
    if not keys:
        raise SystemExit(f"package has no tensors matching {needle!r}")
    shard = package / weight_map[keys[0]]
    with shard.open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len).decode("utf-8"))
    return shard, header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--candidate", default="checkpoints/router_v6_v2_128/router_best.pt")
    parser.add_argument("--output", default="router_swap_check.json")
    args = parser.parse_args()

    package = Path(args.package)
    candidate_path = Path(args.candidate)
    shard, header = _shard_for(package, "memory_router_v2.")
    deployed = {
        key[len(PREFIX):]: tuple(value["shape"])
        for key, value in header.items()
        if key.startswith(PREFIX)
    }
    payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
    state = payload.get("router_state_dict", payload)
    candidate = {key: tuple(value.shape) for key, value in state.items()}

    report: dict = {
        "package": str(package),
        "shard": str(shard),
        "deployed_keys": len(deployed),
        "candidate_keys": len(candidate),
        "missing_in_candidate": sorted(set(deployed) - set(candidate)),
        "extra_in_candidate": sorted(set(candidate) - set(deployed)),
        "shape_mismatches": {
            key: {"deployed": deployed[key], "candidate": candidate[key]}
            for key in set(deployed) & set(candidate)
            if deployed[key] != candidate[key]
        },
    }
    report["drop_in_compatible"] = not (
        report["missing_in_candidate"] or report["extra_in_candidate"] or report["shape_mismatches"]
    )

    # 2. Load into the runtime construction and score real candidates.
    infer = {
        "hidden_size": int(candidate["query_projection.weight"][1]),
        "router_dim": int(candidate["query_projection.weight"][0]),
        "num_heads": int(candidate["head_gate.weight"][0]),
        "max_hops": int(candidate["hop_controller.2.weight"][0]) - 1,
    }
    router = MemoryRouterV2(**infer)
    router.load_state_dict(state, strict=True)
    router.eval()
    query = torch.randn(2, infer["hidden_size"])
    candidates = torch.randn(2, 32, infer["hidden_size"])
    with torch.inference_mode():
        out = router(query, candidates)
    report["runtime_construction"] = infer
    report["scores_finite"] = bool(torch.isfinite(out["scores"]).all())
    report["score_shape"] = list(out["scores"].shape)

    # 3. Drive the paged bank with it, as the memory OS does.
    bank = PagedMemoryBankV2(
        infer["hidden_size"], router=router, page_capacity=2, max_pages=64, hot_pages=2,
        top_k_pages=2, top_k_records=3, max_hops=infer["max_hops"], coarse_index_bits=8,
    )
    for index in range(3):
        bank.write(text=f"fact {index}", key=torch.randn(infer["hidden_size"]),
                   entity=f"entity-{index}", attribute="value", value=str(index), confidence=0.9)
    records, decision = bank.query(query_key=torch.randn(infer["hidden_size"]), query_text="fact 1")
    report["bank_routed_records"] = len(records)
    report["bank_key_dim"] = int(bank.key_dim)
    report["bank_routing_ok"] = bool(records) and int(bank.key_dim) == infer["router_dim"]

    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    ok = report["drop_in_compatible"] and report["scores_finite"] and report["bank_routing_ok"]
    print("\nVERDICT:", "DROP-IN REPLACEMENT OK" if ok else "NOT A DROP-IN REPLACEMENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
