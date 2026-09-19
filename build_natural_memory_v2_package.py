"""Build an embedded Natural Memory v2 package from the current v1 package.

Unchanged Qwen shards are hard-linked when the filesystem permits it.  The
custom memory shard is rewritten once to include the trained V2 router and a
compact V2 page payload, while the official model shards remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.memory_os_v2 import MemoryOSV2, MemoryRouterV2


def _link_or_copy(source: Path, target: Path, *, allow_copy: bool) -> None:
    try:
        os.link(source, target)
    except OSError:
        try:
            # Some Windows volumes reject hardlinks but allow symlinks.  A
            # symlink keeps the multi-gigabyte Qwen shards shared by both
            # packages and avoids an avoidable disk-space spike.
            os.symlink(source, target)
            return
        except OSError:
            pass
        if not allow_copy:
            raise RuntimeError(
                "the destination filesystem does not support hard links; "
                "refusing to duplicate multi-gigabyte Qwen shards. "
                "Re-run with --allow-copy-base only when enough disk space "
                "has been explicitly reserved"
            )
        shutil.copy2(source, target)


def _pack_v2_payload(payload: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    metadata = dict(payload)

    def pack_record(item: dict[str, Any], prefix: str) -> dict[str, Any]:
        item = dict(item)
        for field in ("key", "summary", "semantic_key", "token_ids", "token_mask"):
            value = item.pop(field, None)
            if isinstance(value, torch.Tensor):
                name = f"dynamic_memory.v2.{prefix}.{field}"
                tensors[name] = value.detach().cpu().contiguous()
                item[f"{field}_ref"] = name
        return item

    metadata["records"] = [
        pack_record(item, f"records.{index}")
        for index, item in enumerate(payload.get("records", []))
    ]
    metadata["quarantine"] = [
        pack_record(item, f"quarantine.{index}")
        for index, item in enumerate(payload.get("quarantine", []))
    ]
    metadata["pages"] = []
    for index, item in enumerate(payload.get("pages", [])):
        item = dict(item)
        for field in ("key", "summary"):
            value = item.pop(field, None)
            if isinstance(value, torch.Tensor):
                name = f"dynamic_memory.v2.pages.{index}.{field}"
                tensors[name] = value.detach().cpu().contiguous()
                item[f"{field}_ref"] = name
        metadata["pages"].append(item)
    return tensors, metadata


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = Path(args.base_package)
    output_dir = base_dir if args.in_place else Path(args.output_dir)
    if not args.in_place and output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if not args.in_place:
        output_dir.mkdir(parents=True)

    base_manifest = json.loads((base_dir / "memory_merge.json").read_text(encoding="utf-8"))
    base_config = json.loads((base_dir / "memory_config.json").read_text(encoding="utf-8"))
    memory_name = str(base_manifest.get("memory_weights", "model.safetensors-00003-of-00003.safetensors"))
    source_memory = base_dir / memory_name
    if not source_memory.exists():
        raise FileNotFoundError(source_memory)
    output_memory_name = (
        str(args.in_place_memory_name)
        if args.in_place
        else memory_name
    )

    sources = [] if args.in_place else [
        source
        for source in base_dir.iterdir()
        if source.is_file()
        and source.name not in {
            memory_name,
            "memory_config.json",
            "memory_merge.json",
            "model.safetensors.index.json",
        }
    ]
    if not args.allow_copy_base:
        probe_source = next(
            (source for source in sources if source.name.endswith(".safetensors")),
            None,
        )
        if probe_source is not None:
            probe_target = output_dir / ".hardlink-probe"
            try:
                _link_or_copy(probe_source, probe_target, allow_copy=False)
            finally:
                if probe_target.exists() or probe_target.is_symlink():
                    probe_target.unlink()

    for source in sources:
        _link_or_copy(source, output_dir / source.name, allow_copy=args.allow_copy_base)

    source_metadata: dict[str, str] = {}
    with safe_open(str(source_memory), framework="pt", device="cpu") as handle:
        base_tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        source_metadata = dict(handle.metadata() or {})

    controller_config: dict[str, Any] = {}
    if args.controller_adapter:
        controller_dir = Path(args.controller_adapter)
        controller_config = json.loads(
            (controller_dir / "memory_config.json").read_text(encoding="utf-8")
        )
        memory_path = controller_dir / "memory.pt"
        if memory_path.exists():
            memory_state = torch.load(memory_path, map_location="cpu", weights_only=True)
            base_tensors.update(
                {
                    f"dynamic_memory.memory.{key}": value.detach().cpu().contiguous()
                    for key, value in memory_state.items()
                }
            )
        for filename, prefix in (
            ("text_retriever.pt", "dynamic_memory.text_retriever."),
            ("memory_policy.pt", "dynamic_memory.memory_policy."),
        ):
            state_path = controller_dir / filename
            if not state_path.exists():
                continue
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            base_tensors.update(
                {
                    f"{prefix}{key}": value.detach().cpu().contiguous()
                    for key, value in state.items()
                }
            )
        surgery_path = controller_dir / "surgery.pt"
        if surgery_path.exists():
            surgery = torch.load(surgery_path, map_location="cpu", weights_only=True)
            for layer, value in surgery.get("blend_logits", {}).items():
                base_tensors[f"dynamic_memory.blend_logits.{layer}"] = (
                    value.detach().cpu().contiguous()
                )

    router = MemoryRouterV2(
        int(base_config["hidden_size"]),
        router_dim=args.router_dim,
        num_heads=args.num_heads,
        max_hops=args.max_hops,
    )
    router_state = torch.load(args.router_checkpoint, map_location="cpu", weights_only=True)
    router.load_state_dict(router_state, strict=True)
    for key, value in router.state_dict().items():
        base_tensors[f"dynamic_memory.memory_router_v2.{key}"] = value.detach().cpu().contiguous()

    memory_os = MemoryOSV2(
        int(base_config["hidden_size"]),
        router=router,
    )
    # Migrate the existing model-owned hot facts into V2 address pages.  The
    # old bank remains intact; this is only a compatibility seed for the new
    # hierarchical route.
    legacy_prefix = "dynamic_memory.persistent."
    token_ids = base_tensors.get(f"{legacy_prefix}text_token_ids")
    token_mask = base_tensors.get(f"{legacy_prefix}text_token_mask")
    slot_valid = base_tensors.get(f"{legacy_prefix}text_slot_valid")
    slot_keys = base_tensors.get(f"{legacy_prefix}text_slot_keys")
    if all(isinstance(value, torch.Tensor) for value in (token_ids, token_mask, slot_valid, slot_keys)):
        for batch_index in range(slot_valid.shape[0]):
            for slot in range(slot_valid.shape[1]):
                if not bool(slot_valid[batch_index, slot].item()):
                    continue
                ids = token_ids[batch_index, slot][token_mask[batch_index, slot]]
                memory_os.write(
                    text=f"legacy_hot_slot:{batch_index}:{slot}",
                    key=slot_keys[batch_index, slot],
                    summary=slot_keys[batch_index, slot],
                    memory_type="legacy_hot_text",
                    importance=0.95,
                    confidence=0.95,
                    source="v1_migration",
                    slot_index=slot,
                    token_ids=ids,
                    token_mask=torch.ones_like(ids, dtype=torch.bool),
                    trusted=True,
                )

    if args.in_place:
        # An in-place controller upgrade must not erase an existing user's V2
        # records. The current memory shard and its metadata are already the
        # authoritative snapshot; only the controller/router tensors change.
        v2_metadata_text = source_metadata.get("memory_os_v2_payload")
    else:
        v2_tensors, v2_metadata = _pack_v2_payload(memory_os.export_payload())
        base_tensors.update(v2_tensors)
        v2_metadata_text = json.dumps(v2_metadata, ensure_ascii=False, separators=(",", ":"))
    save_file(
        base_tensors,
        str(output_dir / output_memory_name),
        metadata=(
            {
                "format": "qwen_dynamic_memory_embedded_v2",
                **({"memory_os_v2_payload": v2_metadata_text} if v2_metadata_text else {}),
            }
        ),
    )

    memory_config = dict(base_config)
    saved = dict(memory_config.get("memory_config", {}))
    saved.update(dict(controller_config.get("memory_config", {})))
    saved.update(
        {
            "memory_version": 2,
            "hierarchical_memory": True,
            "memory_router_dim": args.router_dim,
            "memory_router_heads": args.num_heads,
            "memory_page_capacity": args.page_capacity,
            "memory_max_pages": args.max_pages,
            "memory_hot_pages": args.hot_pages,
            "memory_top_k_pages": args.top_k_pages,
            "memory_top_k_records": args.top_k_records,
            "memory_max_hops": args.max_hops,
            "memory_coarse_index_bits": args.coarse_index_bits,
            "memory_v2_read_threshold": args.read_threshold,
            "memory_v2_write_threshold": args.write_threshold,
            "memory_storage_mode": args.memory_storage_mode,
            "memory_storage_path": args.memory_storage_path,
            "memory_resident_pages": args.memory_resident_pages,
            "memory_gpu_cache_records": args.memory_gpu_cache_records,
            "memory_gpu_cache_tokens": args.memory_gpu_cache_tokens,
            "memory_gpu_cache_reserve_mb": args.memory_gpu_cache_reserve_mb,
            "memory_gpu_cache_adaptive": args.memory_gpu_cache_adaptive,
            "kv_budget_tokens": args.kv_budget,
            "kv_hard_max_tokens": args.kv_hard_max,
            "kv_compaction_trigger": args.kv_trigger,
            "kv_keep_recent_tokens": args.kv_keep_recent,
            "persistent_memory": True,
        }
    )
    saved.setdefault("text_memory_semantic_update_threshold", 0.95)
    saved.setdefault("memory_min_read_margin", 0.0)
    saved.setdefault("memory_require_evidence", False)
    memory_config["memory_config"] = saved
    memory_config["router_v2_ready"] = True
    memory_config["checkpoint_contains_user_memory"] = True
    (output_dir / "memory_config.json").write_text(
        json.dumps(memory_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    index = json.loads((base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.setdefault("weight_map", {})
    for key in base_tensors:
            weight_map[key] = output_memory_name
    index.setdefault("metadata", {})["total_size"] = int(
        sum(value.numel() * value.element_size() for value in base_tensors.values())
    )
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = dict(base_manifest)
    manifest.update(
        {
            "format_version": 2,
            "format": "qwen_dynamic_memory_v2_embedded",
            "base_model": str(base_dir),
            "source_router_checkpoint": str(args.router_checkpoint),
            "source_controller_adapter": str(args.controller_adapter) if args.controller_adapter else None,
            "checkpoint_contains_user_memory": True,
            "memory_weights": output_memory_name,
            "memory_config": saved,
        }
    )
    (output_dir / "memory_merge.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "memory_shard": str(output_dir / output_memory_name),
        "memory_tensor_count": len(base_tensors),
        "v2_records": memory_os.stats()["records"],
        "v2_pages": memory_os.stats()["pages"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", default="W:/Flash/model/V2_dpskw/qwen3_5_4b_memory_merged_v13")
    parser.add_argument("--output-dir", default="W:/Flash/model/V2_dpskw/qwen3_5_4b_memory_merged_v2")
    parser.add_argument("--router-checkpoint", default="W:/Flash/model/V2_dpskw/checkpoints/natural_memory_v2_router/memory_router_v2.pt")
    parser.add_argument(
        "--controller-adapter",
        default=None,
        help="optional trained policy/retriever adapter to embed into the output package",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="upgrade the selected package without duplicating its large Qwen shards",
    )
    parser.add_argument(
        "--in-place-memory-name",
        default="model.safetensors-00003-of-00003.safetensors",
        help="target memory shard name for an in-place upgrade",
    )
    parser.add_argument("--router-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--page-capacity", type=int, default=32)
    parser.add_argument("--max-pages", type=int, default=32768)
    parser.add_argument("--hot-pages", type=int, default=8)
    parser.add_argument("--top-k-pages", type=int, default=4)
    parser.add_argument("--top-k-records", type=int, default=8)
    parser.add_argument("--coarse-index-bits", type=int, default=20)
    parser.add_argument("--read-threshold", type=float, default=0.65)
    parser.add_argument("--write-threshold", type=float, default=0.50)
    parser.add_argument("--memory-storage-mode", choices=("embedded", "tiered"), default="embedded")
    parser.add_argument("--memory-storage-path", default=None)
    parser.add_argument("--memory-resident-pages", type=int, default=256)
    parser.add_argument("--memory-gpu-cache-records", type=int, default=256)
    parser.add_argument("--memory-gpu-cache-tokens", type=int, default=131072)
    parser.add_argument("--memory-gpu-cache-reserve-mb", type=int, default=2048)
    parser.add_argument(
        "--memory-gpu-cache-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep the VRAM cache below the reserve line and fall back to system RAM when needed",
    )
    parser.add_argument("--kv-budget", type=int, default=32768)
    parser.add_argument("--kv-hard-max", type=int, default=131072)
    parser.add_argument("--kv-trigger", type=float, default=0.90)
    parser.add_argument("--kv-keep-recent", type=int, default=8192)
    parser.add_argument(
        "--allow-copy-base",
        action="store_true",
        help="allow copying the frozen Qwen shards when hard links are unavailable; requires substantial free disk space",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))
