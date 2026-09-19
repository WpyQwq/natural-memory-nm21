"""Local embedding + bounded reranking baseline for Natural Memory comparisons.

This is intentionally independent of Natural Memory.  It uses a locally
cached BERT encoder on CPU for dense retrieval, followed by a transparent
feature reranker over only the top candidate set.  It is a stronger baseline
than the lexical Chunk RAG path, but it is not a public cross-encoder model.
The report records that distinction explicitly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .benchmark_compare_baselines_4b import (
    _base_generate,
    _contains_answer,
    _is_refusal,
    _quality_summary,
    _rag_prompt,
    _record_entry,
    _vram_snapshot,
)
from .benchmark_real_scale_memory_4b import (
    _build_general_records,
    _build_project_records,
    _max_memory,
    _native_cases,
    _path,
    _project_targets,
    _set_cuda_process_cap,
    _source_files,
)
from .qwen_integration import load_qwen_base, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
TERM_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_\-]+")


def _terms(text: str) -> set[str]:
    return set(TERM_PATTERN.findall(str(text).lower()))


def _resolve_local_encoder(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"embedding model config not found: {path}")
        return path
    cache = Path(r"C:\Users\Administrator\.cache\huggingface\hub\models--bert-base-chinese\snapshots")
    candidates = sorted(
        (path for path in cache.glob("*") if (path / "config.json").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "no local bert-base-chinese snapshot found; pass --embedding-model"
        )
    return candidates[0]


class LocalEmbeddingReranker:
    """CPU dense index followed by a bounded transparent reranker."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 16,
        candidate_k: int = 32,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("the local embedding baseline is CPU-only by default")
        self.max_length = max(8, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self.candidate_k = max(1, int(candidate_k))
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.encoder = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.encoder.eval()
        self.records: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.term_to_indices: dict[str, set[int]] = {}
        self.entity_to_indices: dict[str, set[int]] = {}
        self.attribute_to_indices: dict[str, set[int]] = {}
        self.index_build_seconds = 0.0

    def new_index(self, *, candidate_k: int | None = None) -> "LocalEmbeddingReranker":
        """Create an empty index view sharing the loaded CPU encoder."""

        view = object.__new__(type(self))
        view.model_path = self.model_path
        view.device = self.device
        view.max_length = self.max_length
        view.batch_size = self.batch_size
        view.candidate_k = max(1, int(candidate_k or self.candidate_k))
        view.tokenizer = self.tokenizer
        view.encoder = self.encoder
        view.records = []
        view.embeddings = None
        view.term_to_indices = {}
        view.entity_to_indices = {}
        view.attribute_to_indices = {}
        view.index_build_seconds = 0.0
        return view

    @staticmethod
    def _pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=last_hidden.dtype).unsqueeze(-1)
        pooled = (last_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.nn.functional.normalize(pooled, dim=-1)

    @torch.inference_mode()
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        items = [str(text) for text in texts]
        output: list[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            batch = self.tokenizer(
                items[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            hidden = self.encoder(**batch).last_hidden_state
            pooled = self._pool(hidden, batch["attention_mask"])
            output.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        if not output:
            return np.zeros((0, 768), dtype=np.float32)
        return np.concatenate(output, axis=0)

    def add(self, records: list[dict[str, Any]]) -> None:
        started = time.perf_counter()
        raw_records = [
            _record_entry(record)
            for record in records
            if str(record.get("text", "")).strip()
        ]
        # A production RAG baseline should not return superseded versions of
        # the same entity/attribute as competing evidence. The source corpus
        # is ordered, so the last version is treated as current.
        latest_by_conflict: dict[tuple[str, str], int] = {}
        for index, record in enumerate(raw_records):
            entity = record["entity"].strip().lower()
            attribute = record["attribute"].strip().lower()
            if entity and attribute:
                latest_by_conflict[(entity, attribute)] = index
        self.records = [
            record
            for index, record in enumerate(raw_records)
            if not (
                record["entity"].strip().lower()
                and record["attribute"].strip().lower()
            )
            or latest_by_conflict[
                (
                    record["entity"].strip().lower(),
                    record["attribute"].strip().lower(),
                )
            ] == index
        ]
        self.embeddings = self.encode([record["text"] for record in self.records])
        self.term_to_indices = {}
        self.entity_to_indices = {}
        self.attribute_to_indices = {}
        for index, record in enumerate(self.records):
            for term in record["terms"]:
                self.term_to_indices.setdefault(term, set()).add(index)
            entity = record["entity"].strip().lower()
            attribute = record["attribute"].strip().lower()
            if entity:
                self.entity_to_indices.setdefault(entity, set()).add(index)
            if attribute:
                self.attribute_to_indices.setdefault(attribute, set()).add(index)
        self.index_build_seconds = time.perf_counter() - started

    @staticmethod
    def _lexical_score(query_terms: set[str], record_terms: set[str]) -> float:
        shared = len(query_terms.intersection(record_terms))
        return shared / math.sqrt(max(1, len(query_terms) * len(record_terms)))

    @staticmethod
    def _rerank_score(
        dense_score: float,
        lexical_score: float,
        entity_match: bool,
        attribute_match: bool,
    ) -> float:
        # A fixed, auditable reranker keeps this baseline independent of the
        # Natural Memory router.  Dense similarity supplies recall; exact
        # entity/attribute signals resolve near-duplicate records.
        return (
            0.35 * dense_score
            + 0.15 * lexical_score
            + 0.30 * float(entity_match)
            + 0.20 * float(attribute_match)
        )

    def retrieve(self, query: str, *, top_k: int) -> tuple[list[dict[str, Any]], float]:
        if self.embeddings is None or not self.records:
            return [], 0.0
        started = time.perf_counter()
        query_embedding = self.encode([query])[0]
        dense = np.matmul(self.embeddings, query_embedding)
        query_lower = str(query).strip().lower()
        query_terms = _terms(query)
        candidate_count = min(len(self.records), max(int(top_k), self.candidate_k))
        if candidate_count == len(self.records):
            candidate_indices = set(range(len(self.records)))
        else:
            candidate_indices = set(
                np.argpartition(-dense, candidate_count - 1)[:candidate_count].tolist()
            )
        # Dense retrieval supplies semantic recall. Exact rare terms are
        # unioned into the candidate set so random IDs, filenames, symbols,
        # and personal attributes cannot be lost before reranking.
        for term in query_terms:
            hits = self.term_to_indices.get(term)
            if hits is not None and len(hits) <= 256:
                candidate_indices.update(hits)
        for address_map in (self.entity_to_indices, self.attribute_to_indices):
            for address, hits in address_map.items():
                if len(address) >= 4 and address in query_lower and len(hits) <= 512:
                    candidate_indices.update(hits)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index in candidate_indices:
            record = self.records[int(index)]
            entity = record["entity"].strip().lower()
            attribute = record["attribute"].strip().lower()
            lexical = self._lexical_score(query_terms, record["terms"])
            score = self._rerank_score(
                (float(dense[int(index)]) + 1.0) * 0.5,
                lexical,
                bool(entity and entity in query_lower),
                bool(attribute and attribute in query_lower),
            )
            scored.append((score, -int(index), record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [item[2] for item in scored[: max(1, int(top_k))]], elapsed_ms


def _run_system(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    index: LocalEmbeddingReranker,
    device: torch.device,
    *,
    top_k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        query = str(row["query"])
        retrieved, retrieve_ms = index.retrieve(query, top_k=top_k)
        prompt = _rag_prompt(query, retrieved)
        generated = _base_generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=max_new_tokens,
        )
        expected = str(row.get("expected", ""))
        answerable = bool(row.get("answerable", True))
        correct = (
            _contains_answer(generated["response"], expected)
            if answerable
            else _is_refusal(generated["response"])
        )
        retrieved_target = answerable and any(
            _contains_answer(item["value"], expected)
            or _contains_answer(item["text"], expected)
            for item in retrieved
        )
        output_rows.append(
            {
                "id": row.get("id", ""),
                "query": query,
                "expected": expected,
                "answerable": answerable,
                "correct": bool(correct),
                "retrieved_target": bool(retrieved_target),
                "retrieved_count": len(retrieved),
                "retriever_ms": retrieve_ms,
                "retrieved_values": [item["value"] for item in retrieved],
                **generated,
            }
        )
    summary = _quality_summary(output_rows)
    summary["mean_retriever_ms"] = (
        mean(row["retriever_ms"] for row in output_rows) if output_rows else 0.0
    )
    summary["answerable_retrieval_recall"] = (
        sum(int(row["retrieved_target"]) for row in output_rows if row["answerable"])
        / max(1, sum(int(row["answerable"]) for row in output_rows))
    )
    summary["rows"] = output_rows
    return summary


def _release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=r"H:\Memory")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "strong_rag_compare_4b.json"))
    parser.add_argument("--general-cases", type=int, default=640)
    parser.add_argument("--project-records", type=int, default=8192)
    parser.add_argument("--project-targets", type=int, default=256)
    parser.add_argument("--generation-general", type=int, default=128)
    parser.add_argument("--generation-project", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--gpu-memory-gb", type=float, default=10.0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--reuse-original-report",
        default=None,
        help="reuse original Qwen and lexical Chunk RAG results from an earlier report",
    )
    args = parser.parse_args()
    _set_cuda_process_cap(args.gpu_memory_gb)

    tokenizer = load_tokenizer(_path(args.base_model))
    general_cases = _native_cases(_path(args.data_root), max(1, int(args.general_cases)))
    general_raw_records, general_queries = _build_general_records(general_cases)
    general_rows = [
        row for row in general_queries if row["split"] == "eval"
    ][: max(1, int(args.generation_general))]
    project_files = _source_files()
    project_raw_records, project_queries, project_meta = _build_project_records(
        project_files,
        record_count=max(128, int(args.project_records)),
        target_count=max(1, int(args.project_targets)),
        chunk_tokens=512,
    )
    project_rows = [
        {
            "id": f"project-{index}",
            "query": row["query"],
            "expected": row["expected"],
            "answerable": True,
        }
        for index, row in enumerate(project_queries[: max(1, int(args.generation_project))])
    ]

    encoder_path = _resolve_local_encoder(args.embedding_model)
    embedder = LocalEmbeddingReranker(
        encoder_path,
        max_length=args.embedding_max_length,
        batch_size=args.embedding_batch_size,
        candidate_k=args.candidate_k,
    )
    print(f"loading local embedding model on CPU: {encoder_path}")
    print(f"building general dense index: records={len(general_raw_records)}")
    embedder.add(general_raw_records)
    general_index_seconds = embedder.index_build_seconds
    print(f"general_index_seconds={general_index_seconds:.3f}")
    general_index = embedder

    # Build the project index separately so the two corpora cannot influence
    # one another's candidates.
    project_embedder = embedder.new_index(candidate_k=args.candidate_k)
    print(f"building project dense index: records={len(project_raw_records)}")
    project_embedder.add(project_raw_records)
    project_index_seconds = project_embedder.index_build_seconds
    print(f"project_index_seconds={project_index_seconds:.3f}")

    reused = None
    if args.reuse_original_report:
        reused = json.loads(_path(args.reuse_original_report).read_text(encoding="utf-8"))
        original = reused.get("systems", {}).get("original_qwen")
        if not isinstance(original, dict):
            raise ValueError("reuse report does not contain systems.original_qwen")
        base_reference = original
    else:
        base_reference = None

    print("loading original Qwen 4B for strong-RAG generation")
    base = load_qwen_base(
        _path(args.base_model),
        load_in_4bit=not args.no_4bit,
        max_memory=_max_memory(args.gpu_memory_gb),
    )
    base.eval()
    device = base.get_input_embeddings().weight.device
    load_vram = _vram_snapshot(device)
    strong_general = _run_system(
        base,
        tokenizer,
        general_rows,
        general_index,
        device,
        top_k=args.top_k,
        max_new_tokens=max(1, args.max_new_tokens),
    )
    strong_project = _run_system(
        base,
        tokenizer,
        project_rows,
        project_embedder,
        device,
        top_k=args.top_k,
        max_new_tokens=max(1, args.max_new_tokens),
    )
    _release(base)

    report = {
        "benchmark": "strong_rag_compare_4b",
        "base_model": str(_path(args.base_model)),
        "embedding_model": str(encoder_path),
        "quantization": "4bit_nf4" if not args.no_4bit else "none",
        "gpu_memory_cap_gb": float(args.gpu_memory_gb),
        "project": project_meta,
        "protocol": {
            "same_tokenizer": True,
            "same_sampling": "greedy",
            "top_k": int(args.top_k),
            "candidate_k": int(args.candidate_k),
            "embedding_device": "cpu",
            "embedding_max_length": int(args.embedding_max_length),
            "embedding_batch_size": int(args.embedding_batch_size),
            "generation_general": len(general_rows),
            "generation_project": len(project_rows),
            "baseline_reference": str(_path(args.reuse_original_report)) if args.reuse_original_report else None,
            "baseline_note": "BERT embedding plus fixed feature reranker; not a hosted public cross-encoder reranker",
        },
        "index_build": {
            "general_records": len(general_raw_records),
            "general_indexed_records": len(general_index.records),
            "general_seconds": general_index_seconds,
            "project_records": len(project_raw_records),
            "project_indexed_records": len(project_embedder.records),
            "project_seconds": project_index_seconds,
        },
        "systems": {
            "original_qwen_reference": base_reference,
            "strong_rag": {
                "load_vram": load_vram,
                "general": strong_general,
                "project": strong_project,
            },
        },
        "limitations": [
            "The embedding encoder is local bert-base-chinese, not a dedicated multilingual embedding model.",
            "The reranker is a fixed transparent feature reranker, not a trained public cross-encoder.",
            "Embedding/index build time is reported separately from per-query retrieval and Qwen generation.",
            "The reference baseline is copied from the same-protocol report when --reuse-original-report is used.",
        ],
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "strong_rag": {
            "general": {k: v for k, v in strong_general.items() if k != "rows"},
            "project": {k: v for k, v in strong_project.items() if k != "rows"},
        },
        "index_build": report["index_build"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
