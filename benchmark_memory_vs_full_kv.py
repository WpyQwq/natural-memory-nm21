"""Paired teacher/student benchmark for Natural Memory versus full KV context."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .qwen_integration import load_memory_config, load_qwen_base, load_qwen_dynamic, load_tokenizer
from .stream_chat_qwen_memory import _chat_tensor, _write_turn


PROJECT_ROOT = Path(__file__).resolve().parent


def _path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _max_memory(gpu_memory_gb: float) -> dict[Any, str] | None:
    """Cap CUDA placement so a benchmark cannot consume the whole HBM."""

    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return None
    return {0: f"{gpu_memory_gb:.1f}GiB", "cpu": "64GiB"}


def _set_cuda_process_cap(gpu_memory_gb: float) -> None:
    """Make the benchmark fail safely instead of growing past its HBM budget."""

    if gpu_memory_gb <= 0.0 or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(0.95, max(0.05, gpu_memory_gb * 1024**3 / total))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def _normalize(text: str) -> str:
    return re.sub(r"[\s`*_#，。！？、；：,.!?;:'\"（）()\[\]{}]", "", str(text)).lower()


def _contains(text: str, choices: Iterable[str]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(choice) and _normalize(choice) in normalized for choice in choices)


_ABSTENTION_MARKERS = (
    "不知道",
    "没有记录",
    "无法确定",
    "未找到相关信息",
    "无法访问",
    "没有访问权限",
    "无法查询",
    "无法获取",
    "无法得知",
    "不能确定",
    "没有能力",
)


def _is_abstention(response: str) -> bool:
    """Recognize a truthful no-evidence response in answer-unavailable cases."""

    if _contains(response, _ABSTENTION_MARKERS):
        return True
    if _contains(response, ("没有访问或存储", "没有读取或存储", "不具备访问或存储")):
        return True
    # Qwen often expresses the same abstention as a longer capability
    # disclaimer, e.g. "没有访问或存储...的能力".  This is still a no-evidence
    # answer and must not be scored as a hallucinated personal fact.
    return bool(
        re.search(r"没有[^。！？\n]{0,24}(能力|权限)", response)
        or re.search(r"无法[^。！？\n]{0,24}(访问|查询|获取|确定|得知|读取|存储)", response)
    )


def _passed(response: str, case: dict[str, Any]) -> bool:
    if _contains(response, case.get("forbidden", [])):
        return False
    acceptable = _contains(response, case.get("acceptable", []))
    answerable = bool(case.get("metadata", {}).get("answerable", True))
    return acceptable or (not answerable and _is_abstention(response))


def _read_cases(
    path: Path,
    *,
    limit: int | None,
    offset: int,
    category: str | None,
    per_category_limit: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    category_counts: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            case = json.loads(raw)
            if category and case.get("category") != category:
                continue
            case_category = str(case.get("category", "unknown"))
            if per_category_limit is not None:
                if per_category_limit < 1:
                    raise ValueError("per_category_limit must be positive")
                if category_counts[case_category] >= per_category_limit:
                    continue
                category_counts[case_category] += 1
            if offset > 0:
                offset -= 1
                continue
            rows.append(case)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("no validation cases selected")
    return rows


def _teacher_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for fact in case["facts"]:
        messages.append({"role": "user", "content": str(fact["text"])})
        messages.append({"role": "assistant", "content": str(fact.get("assistant", "好的。"))})
    messages.append({"role": "user", "content": str(case["query"])})
    return messages


@torch.inference_mode()
def _generate_teacher(model: Any, tokenizer: Any, case: dict[str, Any], max_new_tokens: int) -> str:
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer.apply_chat_template(
        _teacher_messages(case),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        # This is a quality/parity benchmark.  Disable generation KV so the
        # evaluator cannot exceed the declared HBM placement cap while it
        # repeatedly loads teacher and student models on a 12GB card.
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    del output, encoded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response


@torch.inference_mode()
def _generate_student(
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    *,
    max_new_tokens: int,
    force_write: bool,
) -> tuple[str, dict[str, Any]]:
    device = model._find_layer_device()
    model.reset_memory(batch_size=1, device=device)
    write_rows: list[dict[str, Any]] = []
    for fact in case["facts"]:
        changed = _write_turn(
            model,
            tokenizer,
            str(fact["text"]),
            device,
            force_write=force_write,
        )
        last_slot = model.runtime.text_last_written_slot
        write_rows.append(
            {
                "kind": fact.get("kind", "fact"),
                "should_write": bool(fact.get("should_write", True)),
                "changed": bool(changed),
                "slot": int(last_slot[0].item()) if isinstance(last_slot, torch.Tensor) else -1,
            }
        )
    encoded = {key: value.to(device) for key, value in _chat_tensor(
        tokenizer,
        str(case["query"]),
    ).items()}
    query = tokenizer(str(case["query"]), add_special_tokens=False, return_tensors="pt")
    query_ids = query["input_ids"].to(device)
    query_mask = query.get("attention_mask")
    if query_mask is None:
        query_mask = torch.ones_like(query_ids)
    query_mask = query_mask.to(device)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        update_memory=False,
        memory_query_input_ids=query_ids,
        memory_query_attention_mask=query_mask,
        memory_query_text=str(case["query"]),
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    diagnostics = {
        "writes": write_rows,
        "valid_slots": int(model.runtime.text_slot_valid.sum().item())
        if isinstance(model.runtime.text_slot_valid, torch.Tensor)
        else 0,
        "prefix_used": bool(model.runtime.text_prefix_used),
        "read_slots": (
            model.runtime.text_read_slots.detach().cpu().tolist()
            if isinstance(model.runtime.text_read_slots, torch.Tensor)
            else []
        ),
        "read_relevance": (
            model.runtime.text_read_relevance.detach().cpu().tolist()
            if isinstance(model.runtime.text_read_relevance, torch.Tensor)
            else []
        ),
        "v2": {
            **model.memory_v2_stats(),
            "last_decisions": list(model.runtime.v2_last_decisions),
        },
    }
    del output, encoded, query, query_ids, query_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response, diagnostics


def _release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, list[float]] = defaultdict(list)
    answerable: dict[str, list[float]] = defaultdict(list)
    stop_reasons: dict[str, int] = defaultdict(int)
    prefix_used = 0
    route_records = 0
    abstained = 0
    elapsed = []
    for row in rows:
        categories[str(row["category"])].append(float(row["passed"]))
        answerable[str(row["category"])].append(float(row.get("answerable", True)))
        elapsed.append(float(row.get("elapsed_seconds", 0.0)))
        diagnostics = row.get("diagnostics", {})
        if diagnostics.get("prefix_used"):
            prefix_used += 1
        if row.get("abstained"):
            abstained += 1
        v2 = diagnostics.get("v2", {}) if isinstance(diagnostics, dict) else {}
        for decision in v2.get("last_decisions", []) if isinstance(v2, dict) else []:
            stop_reasons[str(decision.get("stop_reason", "unknown"))] += 1
        route_records += sum(
            len(decision.get("record_ids", []))
            for decision in v2.get("last_decisions", [])
            if isinstance(decision, dict)
        ) if isinstance(v2, dict) else 0
    return {
        "cases": len(rows),
        "accuracy": sum(float(row["passed"]) for row in rows) / max(1, len(rows)),
        "answerable_cases": sum(
            int(bool(row.get("answerable", True))) for row in rows
        ),
        "prefix_used_cases": prefix_used,
        "abstention_cases": abstained,
        "average_latency_seconds": sum(elapsed) / max(1, len(elapsed)),
        "p95_latency_seconds": sorted(elapsed)[min(len(elapsed) - 1, int(len(elapsed) * 0.95))]
        if elapsed else 0.0,
        "retrieved_record_count": route_records,
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "categories": {
            category: {
                "cases": len(values),
                "accuracy": sum(values) / len(values),
                "answerable_cases": int(sum(answerable[category])),
            }
            for category, values in sorted(categories.items())
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--adapter", default=None, help="optional candidate adapter; omitted uses embedded policy")
    parser.add_argument("--data", default="data/mega_validation/memory_validation_100k.jsonl")
    parser.add_argument("--output", default="mega_memory_vs_full_kv_report.json")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=None,
        help="select this many cases from every category before applying generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=8.0,
        help="hard CUDA placement cap; 0 disables the cap",
    )
    parser.add_argument("--force-write", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    _set_cuda_process_cap(args.gpu_memory_gb)
    model_path = _path(args.model_path)
    data_path = _path(args.data)
    output_path = _path(args.output)
    cases = _read_cases(
        data_path,
        limit=args.limit,
        offset=args.offset,
        category=args.category,
        per_category_limit=args.per_category_limit,
    )
    tokenizer = load_tokenizer(model_path)
    use_4bit = not args.no_4bit
    report: dict[str, Any] = {
        "format_version": 1,
        "model_path": str(model_path),
        "data": str(data_path),
        "selected_cases": len(cases),
        "force_write": bool(args.force_write),
        "quantization": "4bit_nf4" if use_4bit else "none",
        "gpu_memory_cap_gb": args.gpu_memory_gb if args.gpu_memory_gb > 0 else None,
    }

    started = time.perf_counter()
    max_memory = _max_memory(args.gpu_memory_gb)
    teacher = load_qwen_base(model_path, load_in_4bit=use_4bit, max_memory=max_memory)
    teacher.eval()
    teacher_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        case_started = time.perf_counter()
        response = _generate_teacher(teacher, tokenizer, case, args.max_new_tokens)
        teacher_rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "response": response,
                "passed": _passed(response, case),
                "abstained": _is_abstention(response),
                "answerable": bool(case.get("metadata", {}).get("answerable", True)),
                "elapsed_seconds": time.perf_counter() - case_started,
            }
        )
        if index % 16 == 0:
            print(f"teacher {index}/{len(cases)}")
    report["teacher"] = _summarize(teacher_rows)
    _release(teacher)
    teacher = None

    adapter_path = _path(args.adapter) if args.adapter else None
    config_source = adapter_path or model_path
    config = load_memory_config(config_source)
    config.persistent_memory = True
    config.natural_language_memory = True
    config.automatic_memory = True
    student = load_qwen_dynamic(
        model_path,
        memory_config=config,
        load_in_4bit=use_4bit,
        max_memory=max_memory,
    )
    if adapter_path is not None:
        student.load_memory_adapter(adapter_path, strict=True)
    student.eval()
    student_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        case_started = time.perf_counter()
        response, diagnostics = _generate_student(
            student,
            tokenizer,
            case,
            max_new_tokens=args.max_new_tokens,
            force_write=args.force_write,
        )
        student_rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "response": response,
                "passed": _passed(response, case),
                "abstained": _is_abstention(response),
                "answerable": bool(case.get("metadata", {}).get("answerable", True)),
                "elapsed_seconds": time.perf_counter() - case_started,
                "diagnostics": diagnostics,
            }
        )
        if index % 16 == 0:
            print(f"student {index}/{len(cases)}")
    report["student"] = _summarize(student_rows)
    _release(student)
    student = None

    teacher_accuracy = float(report["teacher"]["accuracy"])
    student_accuracy = float(report["student"]["accuracy"])
    teacher_by_id = {row["id"]: row for row in teacher_rows}
    paired_teacher_pass = sum(bool(teacher_by_id[row["id"]]["passed"]) for row in student_rows)
    paired_student_pass = sum(bool(row["passed"]) and teacher_by_id[row["id"]]["passed"] for row in student_rows)
    category_gate: dict[str, Any] = {}
    for category in sorted({str(case["category"]) for case in cases}):
        teacher_cat = [row for row in teacher_rows if row["category"] == category]
        student_cat = [row for row in student_rows if row["category"] == category]
        t = sum(float(row["passed"]) for row in teacher_cat) / max(1, len(teacher_cat))
        s = sum(float(row["passed"]) for row in student_cat) / max(1, len(student_cat))
        category_gate[category] = {"teacher_accuracy": t, "student_accuracy": s, "ratio": s / max(t, 1e-9), "pass": s >= 0.95 * t}
    report["parity"] = {
        "teacher_accuracy": teacher_accuracy,
        "student_accuracy": student_accuracy,
        "student_to_teacher_ratio": student_accuracy / max(teacher_accuracy, 1e-9),
        "paired_teacher_pass": paired_teacher_pass,
        "paired_student_pass": paired_student_pass,
        "paired_ratio": paired_student_pass / max(1, paired_teacher_pass),
        "category_gate": category_gate,
        "required_ratio": 0.95,
        "pass": student_accuracy >= 0.95 * teacher_accuracy and all(item["pass"] for item in category_gate.values()),
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    report["failures"] = [row for row in student_rows if not row["passed"]][:100]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"teacher": report["teacher"], "student": report["student"], "parity": report["parity"], "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
