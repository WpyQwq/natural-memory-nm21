"""Run large-scale natural-language and real-repository memory benchmarks.

The benchmark keeps the memory bank in process RAM and caps the model process
at a declared HBM budget.  It measures two separate workloads:

* naturalistic multi-turn memory episodes from the local native-memory corpus;
* real source and documentation from this repository, expanded to a large
  page library and queried through actual file/symbol questions.

The primary outputs are peak VRAM and correctness.  Retrieval is evaluated on
every query, while generation is evaluated on a substantial holdout subset so
that a high retrieval score cannot be mistaken for end-to-end chat quality.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .qwen_integration import format_memory_evidence, load_qwen_dynamic, load_tokenizer
from .stream_chat_qwen_memory import _chat_tensor, _memory_system_prefix


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MEMORY_MODEL = PROJECT_ROOT / "qwen3_5_4b_natural_memory_v2"
REFUSAL_MARKERS = ("不知道", "没有记录", "无相关", "未找到", "不清楚", "无法确认")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() or path.exists() else PROJECT_ROOT / path


def _max_memory(gpu_memory_gb: float) -> dict[Any, str] | None:
    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return None
    return {0: f"{gpu_memory_gb:.1f}GiB", "cpu": "64GiB"}


def _set_cuda_process_cap(gpu_memory_gb: float) -> None:
    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(0.95, max(0.05, gpu_memory_gb * 1024**3 / total))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _encode_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    *,
    batch_size: int = 16,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start : start + max(1, batch_size)]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        outputs.append(model._encode_model_key(input_ids, mask).detach().cpu())
    if not outputs:
        return torch.empty((0, model.memory.hidden_size), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def _query_token_ids(tokenizer: Any, text: str) -> torch.Tensor:
    return torch.tensor(
        tokenizer(text, add_special_tokens=False)["input_ids"],
        dtype=torch.long,
    )


def _normal(text: Any) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def _contains_answer(response: str, expected: str) -> bool:
    expected_normal = _normal(expected)
    return bool(expected_normal) and expected_normal in _normal(response)


def _is_refusal(response: str) -> bool:
    """Recognize concise and natural-language abstentions."""

    normalized = _normal(response)
    if any(marker in normalized for marker in REFUSAL_MARKERS):
        return True
    return bool(
        re.search(
            r"(没有|无|未|不包含|无法).{0,80}(记录|资料|信息|数据|找到|知道|访问|交互)",
            normalized,
        )
    )


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    items = sorted(float(value) for value in values)
    if not items:
        return {"mean": None, "median": None, "p95": None, "max": None}
    index = min(len(items) - 1, max(0, int(round(0.95 * (len(items) - 1)))))
    return {
        "mean": mean(items),
        "median": median(items),
        "p95": items[index],
        "max": items[-1],
    }


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _native_cases(data_root: Path, max_cases: int) -> list[dict[str, Any]]:
    """Load train/eval native-memory episodes without loading large files."""

    rows: list[dict[str, Any]] = []
    sources = (
        ("train", data_root / "native_memory" / "train.jsonl"),
        ("eval", data_root / "native_memory" / "eval.jsonl"),
    )
    remaining = max(1, max_cases)
    for split, path in sources:
        if not path.exists() or remaining <= 0:
            continue
        loaded = _load_jsonl(path, remaining)
        for row in loaded:
            rows.append({"split": split, **row})
        remaining -= len(loaded)
    if not rows:
        raise FileNotFoundError(f"native memory corpus not found under {data_root}")
    return rows


def _fact_text(chunk: dict[str, Any]) -> str:
    messages = chunk.get("messages") or []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return str(chunk.get("text", ""))


def _build_general_records(
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert local conversation episodes into versioned memory records."""

    records: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for case in cases:
        subject = str(case.get("subject", ""))
        attribute = str(case.get("attribute", ""))
        chunks = case.get("memory_chunks") or []
        for chunk in chunks:
            if not isinstance(chunk, dict) or float(chunk.get("write_label", 0.0)) < 0.5:
                continue
            text = _fact_text(chunk).strip()
            value = str(chunk.get("value", ""))
            if not text or not value:
                continue
            records.append(
                {
                    "text": text,
                    "entity": subject,
                    "attribute": attribute,
                    "value": value,
                    "memory_type": "personal_fact",
                    "importance": 0.9,
                    "confidence": 0.95,
                    "source": "native_memory_corpus",
                    "trusted": True,
                    "force": True,
                }
            )
        query_messages = case.get("query") or []
        query = ""
        for message in query_messages:
            if isinstance(message, dict) and message.get("role") == "user":
                query = str(message.get("content", ""))
                break
        if not query:
            query = f"请查询 {subject} 的 {attribute}。"
        answer = str(case.get("answer", ""))
        answerable = bool(case.get("answerable", False)) and answer not in REFUSAL_MARKERS
        query_rows.append(
            {
                "id": str(case.get("id", "")),
                "split": str(case.get("split", "unknown")),
                "query": query,
                "subject": subject,
                "attribute": attribute,
                "expected": answer,
                "answerable": answerable,
            }
        )
    return records, query_rows


def _source_files() -> list[Path]:
    allowed = {".py", ".md", ".json"}
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if "__pycache__" in path.parts or "checkpoints" in path.parts:
            continue
        if path.name.endswith(".safetensors"):
            continue
        files.append(path)
    return sorted(files)


def _source_chunks(files: list[Path], *, chars_per_chunk: int = 1800) -> list[str]:
    chunks: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text:
            continue
        for start in range(0, len(text), chars_per_chunk):
            piece = text[start : start + chars_per_chunk].strip()
            if piece:
                chunks.append(piece)
    if not chunks:
        raise RuntimeError("repository source corpus is empty")
    return chunks


def _project_targets(files: list[Path], limit: int) -> list[dict[str, Any]]:
    pattern = re.compile(r"^\s*(class|async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)")
    candidates: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() != ".py":
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = pattern.match(line)
            if match is None:
                continue
            kind = match.group(1).replace("async ", "")
            name = match.group(2)
            text = (
                f"真实代码库事实：文件 {relative} 的第 {line_number} 行定义了 "
                f"{kind} {name}。原始代码行：{line.strip()}"
            )
            candidates.append(
                {
                    "file": relative,
                    "line": line_number,
                    "kind": kind,
                    "name": name,
                    "text": text,
                    "expected": relative,
                }
            )
    if not candidates:
        raise RuntimeError("no Python symbols found in repository corpus")
    if len(candidates) <= limit:
        return candidates
    # Evenly sample the repository instead of measuring only the first file.
    indices = [int(index * len(candidates) / limit) for index in range(limit)]
    return [candidates[index] for index in indices]


def _build_project_records(
    files: list[Path],
    *,
    record_count: int,
    target_count: int,
    chunk_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    chunks = _source_chunks(files)
    targets = _project_targets(files, target_count)
    rng = random.Random(20260905)
    positions = rng.sample(range(record_count), min(len(targets), record_count))
    position_to_target = dict(zip(positions, targets))
    records: list[dict[str, Any]] = []
    for index in range(record_count):
        target = position_to_target.get(index)
        if target is None:
            text = (
                f"项目源码快照分片 {index}：\n"
                f"{chunks[index % len(chunks)]}"
            )
            records.append(
                {
                    "text": text,
                    "key_kind": "random_filler",
                    "memory_type": "repository_chunk",
                    "importance": 0.4,
                    "confidence": 0.8,
                    "source": "real_repository_snapshot",
                    "trusted": True,
                    "force": True,
                }
            )
            continue
        records.append(
            {
                "text": target["text"],
                "entity": target["file"],
                "attribute": f"symbol:{target['name']}",
                "value": target["file"],
                "memory_type": "repository_symbol",
                "importance": 0.9,
                "confidence": 0.95,
                "source": "real_repository_snapshot",
                "target": target,
                "trusted": True,
                "force": True,
            }
        )
    queries: list[dict[str, Any]] = []
    for target in targets:
        queries.extend(
            [
                {
                    "query": f"在真实代码库中，文件 {target['file']} 里的 {target['name']} 定义在哪个文件？",
                    "expected": target["expected"],
                    "target": target,
                },
                {
                    "query": f"请从项目记忆查找：{target['file']} 的 {target['kind']} {target['name']} 位于哪里？",
                    "expected": target["expected"],
                    "target": target,
                },
            ]
        )
    return records, queries, {
        "source_file_count": len(files),
        "source_characters": sum(path.stat().st_size for path in files),
        "source_chunk_count": len(chunks),
        "capacity_tokens": int(record_count * chunk_tokens),
        "chunk_tokens": int(chunk_tokens),
        "target_count": len(targets),
    }


def _prepare_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    semantic_indices = [
        index for index, record in enumerate(records)
        if record.get("key_kind") != "random_filler"
    ]
    semantic_keys = _encode_texts(
        model,
        tokenizer,
        [str(records[index]["text"]) for index in semantic_indices],
        device,
        batch_size=batch_size,
    )
    key_by_index = {
        index: semantic_keys[position]
        for position, index in enumerate(semantic_indices)
    }
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        item = {key: value for key, value in record.items() if key not in {"target", "key_kind"}}
        if record.get("key_kind") == "random_filler":
            generator = torch.Generator(device="cpu").manual_seed(900000 + index)
            key = F.normalize(torch.randn(model.memory.hidden_size, generator=generator), dim=0)
            semantic_key = None
        else:
            key = F.normalize(key_by_index[index].float(), dim=0)
            semantic_key = key
        item["key"] = key
        item["summary"] = key
        # Background corpus chunks are intentionally not semantic candidates
        # in this bounded stress run.  Leaving their semantic key unset lets
        # the bank use the learned reranker for real evidence while retaining
        # the chunks as corpus noise rather than pretending their random
        # addresses were learned representations.
        item["semantic_key"] = semantic_key
        if record.get("key_kind") != "random_filler":
            # Match the production writer: retrieved memories enter Qwen as
            # an internal system-message prefix, while record.text remains
            # the raw fact used by the address/routing layer.
            evidence_text = format_memory_evidence(
                str(record["text"]),
                entity=str(record.get("entity", "")),
                attribute=str(record.get("attribute", "")),
                value=str(record.get("value", "")),
            )
            storage = _memory_system_prefix(
                tokenizer,
                "以下是与当前用户相关的已保存长期记忆。仅在问题相关时使用，"
                "只能依据明确证据；先核对实体、属性和已确认值；冲突优先最新可靠来源，"
                "不要拼接不确定候选，证据不足就明确说不知道；涉及名称、路径、token、"
                "参数或结论时，原样复述证据中的关键短语：\n"
                + evidence_text,
            )
            token_ids = storage["input_ids"].reshape(-1).to(dtype=torch.long)
            token_mask = storage["attention_mask"].reshape(-1).to(dtype=torch.bool)
            item["token_ids"] = token_ids
            item["token_mask"] = token_mask
        prepared.append(item)
    return prepared


def _clear_memory_bank(model: Any, device: torch.device) -> None:
    model.memory_os_v2 = model._new_memory_os_v2(model.memory.hidden_size)
    model.reset_memory(batch_size=1, device=device)


def _direct_read(
    model: Any,
    tokenizer: Any,
    query_key: torch.Tensor,
    row: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    query = str(row["query"])
    token_ids = _query_token_ids(tokenizer, query)
    _sync(device)
    started = time.perf_counter()
    records, decision = model.memory_os_v2.read(
        query_key=query_key,
        query_text=query,
        query_token_ids=token_ids,
        top_k_pages=4,
        top_k_records=2,
        max_hops=3,
    )
    _sync(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    returned_values = [record.value for record in records]
    if row.get("answerable", True):
        hit = any(_contains_answer(value, str(row.get("expected", ""))) for value in returned_values)
    else:
        hit = len(records) == 0
    return {
        "id": row.get("id", ""),
        "query": query,
        "expected": row.get("expected", ""),
        "answerable": bool(row.get("answerable", True)),
        "hit": bool(hit),
        "returned_values": returned_values,
        "returned_ids": [record.record_id for record in records],
        "stop_reason": decision.stop_reason,
        "coarse_candidates": model.memory_os_v2.bank._last_coarse_candidates,
        "elapsed_ms": elapsed_ms,
    }


def _record_audit_view(record: Any) -> dict[str, Any]:
    """Keep generation diagnostics readable without serializing embeddings."""

    return {
        "record_id": str(getattr(record, "record_id", "")),
        "entity": str(getattr(record, "entity", "")),
        "attribute": str(getattr(record, "attribute", "")),
        "value": str(getattr(record, "value", "")),
        "status": str(getattr(record, "status", "")),
        "version": int(getattr(record, "version", 0)),
        "source": str(getattr(record, "source", "")),
        "text_preview": str(getattr(record, "text", ""))[:300],
    }


def _chat_generate(
    model: Any,
    tokenizer: Any,
    query: str,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    encoded = _chat_tensor(
        tokenizer,
        query,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    raw = tokenizer(query, add_special_tokens=False, return_tensors="pt")
    query_ids = raw["input_ids"].to(device)
    query_mask = raw.get("attention_mask")
    if query_mask is None:
        query_mask = torch.ones_like(query_ids)
    _sync(device)
    generation_started = time.perf_counter()
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        update_memory=False,
        memory_query_input_ids=query_ids,
        memory_query_attention_mask=query_mask.to(device),
        memory_query_text=query,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    _sync(device)
    generation_seconds = time.perf_counter() - generation_started
    response_ids = output[0, encoded["input_ids"].shape[1] :]
    response = tokenizer.decode(response_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()

    runtime = getattr(model, "runtime", None)
    raw_decisions = list(getattr(runtime, "v2_last_decisions", []) or [])
    decisions: list[dict[str, Any]] = []
    selected_record_ids: list[str] = []
    for raw_decision in raw_decisions:
        decision = dict(raw_decision)
        record_ids = [str(item) for item in decision.get("record_ids", [])]
        selected_record_ids.extend(record_ids)
        decisions.append(
            {
                "need_memory": bool(decision.get("need_memory", False)),
                "record_ids": record_ids,
                "page_ids": [str(item) for item in decision.get("page_ids", [])],
                "hop_count": int(decision.get("hop_count", 0)),
                "confidence": float(decision.get("confidence", 0.0)),
                "top_score": float(decision.get("top_score", 0.0)),
                "score_margin": float(decision.get("score_margin", 0.0)),
                "evidence_score": float(decision.get("evidence_score", 0.0)),
                "stop_reason": str(decision.get("stop_reason", "")),
            }
        )
    selected_record_ids = list(dict.fromkeys(selected_record_ids))
    records_by_id = getattr(getattr(model, "memory_os_v2", None), "bank", None)
    records_by_id = getattr(records_by_id, "records", {})
    selected_records = [
        _record_audit_view(records_by_id[record_id])
        for record_id in selected_record_ids
        if record_id in records_by_id
    ]
    return {
        "response": response,
        "public_prompt_tokens": int(encoded["input_ids"].shape[1]),
        "generated_tokens": int(response_ids.numel()),
        "generation_seconds": float(generation_seconds),
        "prefix_used": bool(getattr(runtime, "text_prefix_used", False)),
        "guard_used": bool(getattr(runtime, "text_guard_used", False)),
        "prefix_tokens": int(getattr(runtime, "text_prefix_tokens", 0)),
        "memory_read_seconds": float(getattr(runtime, "text_read_seconds", 0.0)),
        "selected_record_ids": selected_record_ids,
        "selected_records": selected_records,
        "selected_values": [record["value"] for record in selected_records],
        "decisions": decisions,
        "stop_reasons": [decision["stop_reason"] for decision in decisions],
    }


def _generation_eval(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    max_new_tokens: int,
    reference_retrieval_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        generated = _chat_generate(model, tokenizer, str(row["query"]), device, max_new_tokens)
        response = str(generated["response"])
        answerable = bool(row.get("answerable", True))
        expected = str(row.get("expected", ""))
        retrieved_target = answerable and any(
            _contains_answer(str(record.get("value", "")), expected)
            or _contains_answer(str(record.get("text_preview", "")), expected)
            for record in generated["selected_records"]
        )
        reference = (reference_retrieval_rows or {}).get(str(row.get("id", "")))
        reference_target = bool(reference and reference.get("hit", False))
        passed = (
            _contains_answer(response, expected)
            if answerable
            else _is_refusal(response)
        )
        if answerable and passed:
            error_class = "correct"
        elif not answerable and passed:
            error_class = "correct_refusal"
        elif not answerable:
            error_class = "refusal_failure"
        elif reference_target and not retrieved_target:
            error_class = "generation_retrieval_divergence"
        elif not generated["selected_record_ids"]:
            error_class = "no_memory_prefix"
        elif not retrieved_target:
            error_class = "retrieval_miss_or_wrong_prefix"
        else:
            error_class = "correct_evidence_ignored_or_overridden"
        output_rows.append(
            {
                "id": row.get("id", ""),
                "query": row["query"],
                "expected": expected,
                "response": response,
                "answerable": answerable,
                "correct": bool(passed),
                "retrieval_target_found": bool(retrieved_target),
                "reference_retrieval_hit": reference_target,
                "reference_retrieved_values": list(reference.get("returned_values", [])) if reference else [],
                "retrieved_values": list(generated["selected_values"]),
                "retrieved_ids": list(generated["selected_record_ids"]),
                "selected_records": list(generated["selected_records"]),
                "prefix_used": bool(generated["prefix_used"]),
                "prefix_tokens": int(generated["prefix_tokens"]),
                "memory_read_seconds": float(generated["memory_read_seconds"]),
                "decisions": list(generated["decisions"]),
                "stop_reasons": list(generated["stop_reasons"]),
                "error_class": error_class,
            }
        )
    answerable_rows = [row for row in output_rows if row["answerable"]]
    unknown_rows = [row for row in output_rows if not row["answerable"]]
    error_counts = Counter(row["error_class"] for row in output_rows)
    failed_answerable = [row for row in answerable_rows if not row["correct"]]
    return {
        "cases": len(output_rows),
        "answerable_cases": len(answerable_rows),
        "answerable_correct": sum(int(row["correct"]) for row in answerable_rows),
        "answerable_accuracy": sum(int(row["correct"]) for row in answerable_rows) / max(1, len(answerable_rows)),
        "unknown_cases": len(unknown_rows),
        "unknown_correct": sum(int(row["correct"]) for row in unknown_rows),
        "unknown_refusal_accuracy": sum(int(row["correct"]) for row in unknown_rows) / max(1, len(unknown_rows)),
        "error_class_counts": dict(sorted(error_counts.items())),
        "failed_answerable_cases": len(failed_answerable),
        "failed_answerable_with_correct_evidence": sum(
            int(row["retrieval_target_found"]) for row in failed_answerable
        ),
        "failed_answerable_with_reference_retrieval_hit": sum(
            int(row["reference_retrieval_hit"]) for row in failed_answerable
        ),
        "failed_answerable_with_retrieval_miss": sum(
            int(not row["retrieval_target_found"]) for row in failed_answerable
        ),
        "rows": output_rows,
    }


def _retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    unknown = [row for row in rows if not row["answerable"]]
    return {
        "queries": len(rows),
        "answerable_queries": len(answerable),
        "answerable_hits": sum(int(row["hit"]) for row in answerable),
        "answerable_recall": sum(int(row["hit"]) for row in answerable) / max(1, len(answerable)),
        "unknown_queries": len(unknown),
        "unknown_correct_abstentions": sum(int(row["hit"]) for row in unknown),
        "unknown_abstention_accuracy": sum(int(row["hit"]) for row in unknown) / max(1, len(unknown)),
        "read_latency_ms": _quantiles(row["elapsed_ms"] for row in rows),
        "mean_coarse_candidates": mean(row["coarse_candidates"] for row in rows) if rows else 0.0,
        "max_coarse_candidates": max((row["coarse_candidates"] for row in rows), default=0),
    }


def _vram_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"allocated_gb": None, "reserved_gb": None, "peak_allocated_gb": None, "peak_reserved_gb": None}
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=r"H:\Memory")
    parser.add_argument("--memory-model", default=str(DEFAULT_MEMORY_MODEL))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "real_scale_memory_4b.json"))
    parser.add_argument("--general-cases", type=int, default=640)
    parser.add_argument("--project-records", type=int, default=8192)
    parser.add_argument("--project-targets", type=int, default=256)
    parser.add_argument("--project-chunk-tokens", type=int, default=512)
    parser.add_argument("--generation-general", type=int, default=128)
    parser.add_argument("--generation-project", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-gb", type=float, default=10.0)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    _set_cuda_process_cap(args.gpu_memory_gb)

    tokenizer = load_tokenizer(_path(args.base_model))
    use_4bit = not args.no_4bit
    print("loading Natural Memory 4B")
    model = load_qwen_dynamic(
        _path(args.memory_model),
        load_in_4bit=use_4bit,
        max_memory=_max_memory(args.gpu_memory_gb),
    )
    model.configure_memory_grounding_guard(tokenizer)
    model.eval()
    model.memory_config.memory_top_k_records = 2
    model.memory_os_v2.bank.top_k_records = 2
    device = model._find_layer_device()
    model.reset_memory(batch_size=1, device=device)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline_vram = _vram_snapshot(device)

    data_root = _path(args.data_root)
    general_cases = _native_cases(data_root, max(1, int(args.general_cases)))
    general_raw_records, general_queries = _build_general_records(general_cases)
    general_records = _prepare_records(
        model,
        tokenizer,
        general_raw_records,
        device,
        batch_size=max(1, int(args.encode_batch_size)),
    )
    _sync(device)
    general_write_start = time.perf_counter()
    model.memory_os_v2.write_batch(general_records)
    _sync(device)
    general_write_ms = (time.perf_counter() - general_write_start) * 1000.0
    general_query_keys = _encode_texts(
        model,
        tokenizer,
        [str(row["query"]) for row in general_queries],
        device,
        batch_size=max(1, int(args.encode_batch_size)),
    )
    general_retrieval_rows = [
        _direct_read(model, tokenizer, general_query_keys[index], row, device)
        for index, row in enumerate(general_queries)
    ]
    general_retrieval = _retrieval_summary(general_retrieval_rows)
    general_generation_rows = [
        row
        for row in general_queries
        if row["split"] == "eval"
    ][: max(1, int(args.generation_general))]
    general_generation = _generation_eval(
        model,
        tokenizer,
        general_generation_rows,
        device,
        max_new_tokens=max(1, int(args.max_new_tokens)),
        reference_retrieval_rows={str(row["id"]): row for row in general_retrieval_rows},
    )
    general_vram = _vram_snapshot(device)

    _clear_memory_bank(model, device)
    project_files = _source_files()
    project_raw_records, project_queries, project_meta = _build_project_records(
        project_files,
        record_count=max(128, int(args.project_records)),
        target_count=max(1, int(args.project_targets)),
        chunk_tokens=max(1, int(args.project_chunk_tokens)),
    )
    project_records = _prepare_records(
        model,
        tokenizer,
        project_raw_records,
        device,
        batch_size=max(1, int(args.encode_batch_size)),
    )
    _sync(device)
    project_write_start = time.perf_counter()
    model.memory_os_v2.write_batch(project_records)
    _sync(device)
    project_write_ms = (time.perf_counter() - project_write_start) * 1000.0
    project_query_keys = _encode_texts(
        model,
        tokenizer,
        [str(row["query"]) for row in project_queries],
        device,
        batch_size=max(1, int(args.encode_batch_size)),
    )
    project_retrieval_rows = []
    for index, row in enumerate(project_queries):
        direct_row = dict(row)
        direct_row["answerable"] = True
        project_retrieval_rows.append(
            _direct_read(model, tokenizer, project_query_keys[index], direct_row, device)
        )
    project_retrieval = _retrieval_summary(project_retrieval_rows)
    project_generation_rows = [
        {"id": f"project-{index}", "query": row["query"], "expected": row["expected"], "answerable": True}
        for index, row in enumerate(project_queries[: max(1, int(args.generation_project))])
    ]
    project_generation = _generation_eval(
        model,
        tokenizer,
        project_generation_rows,
        device,
        max_new_tokens=max(1, int(args.max_new_tokens)),
        reference_retrieval_rows={str(row["id"]): row for row in project_retrieval_rows},
    )
    project_vram = _vram_snapshot(device)

    report = {
        "benchmark": "real_scale_memory_4b",
        "model": str(_path(args.memory_model)),
        "base_model": str(_path(args.base_model)),
        "quantization": "4bit_nf4" if use_4bit else "none",
        "gpu_memory_cap_gb": float(args.gpu_memory_gb),
        "device": str(device),
        "priority_metrics": ["peak_vram", "correctness"],
        "baseline_vram_after_load": baseline_vram,
        "general_chat_memory": {
            "source": "local_native_memory_train_plus_eval",
            "cases": len(general_cases),
            "records_written": len(general_records),
            "bank_records": len(model.memory_os_v2.bank.records),
            "write_ms": general_write_ms,
            "retrieval": general_retrieval,
            "generation": general_generation,
            "vram_after_general": general_vram,
        },
        "project_repository_memory": {
            **project_meta,
            "records_written": len(project_records),
            "bank_records": len(model.memory_os_v2.bank.records),
            "page_count": len(model.memory_os_v2.bank.pages),
            "write_ms": project_write_ms,
            "retrieval": project_retrieval,
            "generation": project_generation,
            "vram_after_project": project_vram,
        },
        "limitations": [
            "本地 native_memory 语料是工程内置的自然语言基准，不等同于真实用户导出数据。",
            "项目库目标来自当前仓库的真实源码和文档；大规模背景页用于压力测试，主要考察有界路由和显存。",
            "生成质量与检索召回分别报告，不能用检索正确率替代端到端聊天正确率。",
            "没有启用 SQLite 或磁盘分页；记忆主体保持在进程 RAM，只有命中的记录进入有界显存缓存。",
        ],
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "general_retrieval": general_retrieval,
        "general_generation": {key: value for key, value in general_generation.items() if key != "rows"},
        "project_retrieval": project_retrieval,
        "project_generation": {key: value for key, value in project_generation.items() if key != "rows"},
        "vram": {"baseline": baseline_vram, "general": general_vram, "project": project_vram},
    }, ensure_ascii=False, indent=2))
    print(f"saved={output}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
