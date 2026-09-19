"""Merge dynamic-memory tensors into a copy of a local safetensors shard.

The base Qwen shards are never modified.  The output directory contains
hardlinks for unchanged files and a newly written second shard with the
memory tensors appended to its safetensors payload.  The normal HF loader
continues to see the original Qwen keys; ``qwen_integration`` loads the
embedded ``dynamic_memory.*`` keys when it sees ``memory_merge.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _read_header(path: Path) -> tuple[dict[str, Any], int, int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"not a safetensors file: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    return json.loads(header_bytes.decode("utf-8")), int(header_length), 8 + int(header_length)


def _load_adapter_tensors(adapter_dir: Path, memory_state: Path | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    memory = torch.load(adapter_dir / "memory.pt", map_location="cpu", weights_only=True)
    tensors.update({f"dynamic_memory.memory.{key}": value.detach().cpu().contiguous() for key, value in memory.items()})

    retriever_path = adapter_dir / "text_retriever.pt"
    if retriever_path.exists():
        retriever = torch.load(retriever_path, map_location="cpu", weights_only=True)
        tensors.update(
            {f"dynamic_memory.text_retriever.{key}": value.detach().cpu().contiguous() for key, value in retriever.items()}
        )

    policy_path = adapter_dir / "memory_policy.pt"
    if policy_path.exists():
        policy = torch.load(policy_path, map_location="cpu", weights_only=True)
        tensors.update(
            {f"dynamic_memory.memory_policy.{key}": value.detach().cpu().contiguous() for key, value in policy.items()}
        )

    surgery_path = adapter_dir / "surgery.pt"
    if surgery_path.exists():
        surgery = torch.load(surgery_path, map_location="cpu", weights_only=True)
        for layer, value in surgery.get("blend_logits", {}).items():
            tensors[f"dynamic_memory.blend_logits.{layer}"] = value.detach().cpu().contiguous()

    metadata = json.loads((adapter_dir / "memory_config.json").read_text(encoding="utf-8"))
    memory_config = dict(metadata.get("memory_config", {}))

    persistent_source = memory_state
    if persistent_source is None and (adapter_dir / "persistent_memory.pt").exists():
        persistent_source = adapter_dir / "persistent_memory.pt"
    if persistent_source is not None:
        payload = torch.load(persistent_source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("memory_state"), torch.Tensor):
            raise ValueError(f"memory state file has no tensor memory_state: {persistent_source}")
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                tensors[f"dynamic_memory.persistent.{key}"] = value.detach().cpu().contiguous()
        memory_config["persistent_memory"] = True

    if not tensors:
        raise ValueError("no tensors were found to merge")
    metadata["memory_config"] = memory_config
    metadata["checkpoint_contains_user_memory"] = persistent_source is not None
    return tensors, metadata


def _write_merged_shard(base_shard: Path, extra_shard: Path, output_shard: Path) -> None:
    base_header, base_header_length, base_data_start = _read_header(base_shard)
    extra_header, _, extra_data_start = _read_header(extra_shard)
    if "__metadata__" in extra_header:
        extra_metadata = extra_header.pop("__metadata__")
    else:
        extra_metadata = {}
    if "__metadata__" in base_header:
        base_metadata = base_header.pop("__metadata__")
    else:
        base_metadata = {}
    collisions = set(base_header).intersection(extra_header)
    if collisions:
        raise ValueError(f"safetensors key collision while merging: {sorted(collisions)[:4]}")

    base_payload_size = base_shard.stat().st_size - base_data_start
    merged_header: dict[str, Any] = {}
    new_metadata = dict(base_metadata)
    new_metadata.update({str(key): str(value) for key, value in extra_metadata.items()})
    merged_header["__metadata__"] = new_metadata

    for key, entry in base_header.items():
        item = dict(entry)
        start, end = item["data_offsets"]
        item["data_offsets"] = [int(start), int(end)]
        merged_header[key] = item
    # Copy the entries: the offset-fixup loop mutates merged_header and must
    # not mutate extra_header, otherwise the second loop iteration applies the
    # base offset repeatedly.
    merged_header.update({key: dict(entry) for key, entry in extra_header.items()})

    # Offsets are relative to the beginning of the data section, so changing
    # the header length does not move the base tensors in that coordinate
    # system.  Recompute the header until its padded size is stable.
    while True:
        encoded = json.dumps(merged_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        padded_length = (len(encoded) + 7) // 8 * 8
        for key, entry in base_header.items():
            start, end = entry["data_offsets"]
            merged_header[key]["data_offsets"] = [int(start), int(end)]
        # ``data_offsets`` are relative to the beginning of the data section,
        # not absolute file offsets.  The extra payload follows the base
        # payload inside that merged data section.
        extra_start = base_payload_size
        for key, entry in extra_header.items():
            start, end = entry["data_offsets"]
            merged_header[key]["data_offsets"] = [int(extra_start + start), int(extra_start + end)]
        encoded_next = json.dumps(merged_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        next_length = (len(encoded_next) + 7) // 8 * 8
        if next_length == padded_length:
            encoded = encoded_next + b" " * (next_length - len(encoded_next))
            break

    output_shard.parent.mkdir(parents=True, exist_ok=True)
    with output_shard.open("wb") as output, base_shard.open("rb") as base, extra_shard.open("rb") as extra:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        base.seek(base_data_start)
        shutil.copyfileobj(base, output, length=16 * 1024 * 1024)
        extra.seek(extra_data_start)
        shutil.copyfileobj(extra, output, length=16 * 1024 * 1024)


def _hardlink_base_files(base_dir: Path, output_dir: Path, merged_shard_name: str) -> None:
    for source in base_dir.iterdir():
        if not source.is_file() or source.name in {"memory_config.json", "memory_merge.json", merged_shard_name}:
            continue
        destination = output_dir / source.name
        try:
            os.link(source, destination)
        except OSError:
            try:
                # Some Windows volumes reject hardlinks but allow symlinks;
                # this keeps the unchanged 5GB shard from being duplicated.
                os.symlink(source, destination)
            except OSError:
                # Last-resort fallback for filesystems that allow neither.
                # The large first shard is still copied only when necessary.
                shutil.copy2(source, destination)


def merge_package(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    memory_state: Path | None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    base_index = base_dir / "model.safetensors.index.json"
    index = json.loads(base_index.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    if len(shard_names) != 2 or "model.safetensors-00002-of-00002.safetensors" not in shard_names:
        raise ValueError(f"expected the local two-shard Qwen layout, got {shard_names}")

    output_dir.mkdir(parents=True)
    merged_shard_name = "model.safetensors-00002-of-00002.safetensors"
    _hardlink_base_files(base_dir, output_dir, merged_shard_name)

    tensors, adapter_metadata = _load_adapter_tensors(adapter_dir, memory_state)
    extra_shard = output_dir / "memory_extra.safetensors"
    save_file(tensors, str(extra_shard), metadata={"format": "qwen_dynamic_memory_embedded_v1"})
    _write_merged_shard(
        base_dir / merged_shard_name,
        extra_shard,
        output_dir / merged_shard_name,
    )
    # This is a self-owned staging file; the merged shard is the only copy
    # that belongs in the output package.
    extra_shard.unlink()

    (output_dir / "memory_config.json").write_text(
        json.dumps(adapter_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "format": "qwen_dynamic_memory_merged_shard",
        "shard_mode": "combined",
        "memory_weights": merged_shard_name,
        "base_model": str(base_dir),
        "source_adapter": str(adapter_dir),
        "source_memory_state": str(memory_state) if memory_state is not None else None,
        "tensor_prefix": "dynamic_memory.",
        "checkpoint_contains_user_memory": bool(adapter_metadata.get("checkpoint_contains_user_memory")),
        "memory_config": adapter_metadata.get("memory_config", {}),
    }
    (output_dir / "memory_merge.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def add_memory_shard(
    base_dir: Path,
    adapter_dir: Path,
    shard_path: Path,
    memory_state: Path | None,
) -> dict[str, Any]:
    """Add a small extra-only memory shard beside an existing Qwen model."""

    if shard_path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard: {shard_path}")
    if not (base_dir / "config.json").exists() or not (base_dir / "model.safetensors.index.json").exists():
        raise ValueError(f"not a complete local Qwen model directory: {base_dir}")

    tensors, adapter_metadata = _load_adapter_tensors(adapter_dir, memory_state)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(shard_path),
        metadata={"format": "qwen_dynamic_memory_extra_shard_v1"},
    )
    index_path = base_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.setdefault("weight_map", {})
    collisions = sorted(set(weight_map).intersection(tensors))
    if collisions:
        raise ValueError(f"weight-map key collision while adding memory shard: {collisions[:4]}")
    weight_map.update({key: shard_path.name for key in tensors})
    tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    index.setdefault("metadata", {})["total_size"] = int(
        index.get("metadata", {}).get("total_size", 0) + tensor_bytes
    )
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    memory_config_path = base_dir / "memory_config.json"
    memory_config_path.write_text(
        json.dumps(adapter_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "format": "qwen_dynamic_memory_extra_shard",
        "shard_mode": "extra_only",
        "memory_weights": shard_path.name,
        "base_model": str(base_dir),
        "source_adapter": str(adapter_dir),
        "source_memory_state": str(memory_state) if memory_state is not None else None,
        "tensor_prefix": "dynamic_memory.",
        "checkpoint_contains_user_memory": bool(adapter_metadata.get("checkpoint_contains_user_memory")),
        "memory_config": adapter_metadata.get("memory_config", {}),
    }
    (base_dir / "memory_merge.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--add-shard", default=None, help="add a small extra-only shard beside the base model")
    parser.add_argument("--memory-state", default=None)
    args = parser.parse_args()
    if args.add_shard:
        manifest = add_memory_shard(
            Path(args.base_model),
            Path(args.adapter),
            Path(args.add_shard),
            Path(args.memory_state) if args.memory_state else None,
        )
        output_shard = Path(args.add_shard)
        print(f"merged_package={Path(args.base_model).resolve()}")
    else:
        if not args.output:
            parser.error("--output is required unless --add-shard is used")
        manifest = merge_package(
            Path(args.base_model),
            Path(args.adapter),
            Path(args.output),
            Path(args.memory_state) if args.memory_state else None,
        )
        output_shard = Path(args.output) / str(manifest["memory_weights"])
        print(f"merged_package={Path(args.output).resolve()}")
    print(f"merged_shard={output_shard.resolve()}")
    print(f"merged_size_gb={output_shard.stat().st_size / (1024**3):.3f}")
    print(f"contains_user_memory={manifest['checkpoint_contains_user_memory']}")


if __name__ == "__main__":
    main()
