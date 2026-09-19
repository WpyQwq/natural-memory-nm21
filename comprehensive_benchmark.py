"""Compare original Qwen and Native Memory on general regression tasks."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .qwen_integration import load_memory_config, load_qwen_base, load_qwen_dynamic, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--adapter", default="V2_dpskw/qwen_memory_adapter_native_v3")
    parser.add_argument("--data", default="V2_dpskw/data/comprehensive_general.jsonl")
    parser.add_argument("--output", default="V2_dpskw/comprehensive_benchmark_native_v3.json")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--perf-repeats", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--category-drop-limit", type=float, default=0.10)
    parser.add_argument("--overall-drop-limit", type=float, default=0.05)
    return parser.parse_args()


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    if not cases:
        raise ValueError(f"no benchmark cases found in {path}")
    return cases


def normalize(text: str) -> str:
    return re.sub(r"[\s`*_#，。！？、；：,.!?;:'\"（）()\[\]{}]", "", text).lower()


def contains_answer(text: str, acceptable: list[str]) -> bool:
    normalized = normalize(text)
    for answer in acceptable:
        expected = normalize(str(answer))
        if not expected:
            continue
        if expected.isdigit() and len(expected) == 1:
            if re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", normalized):
                return True
        elif expected in normalized:
            return True
    return False


def prompt_inputs(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) or hasattr(encoded, "__getitem__") else encoded
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
    else:
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _decode_generation(tokenizer: Any, output: torch.Tensor, prompt: dict[str, torch.Tensor]) -> str:
    start = prompt["input_ids"].shape[1]
    return tokenizer.decode(output[0, start:].detach().cpu().tolist(), skip_special_tokens=True).strip()


def evaluate_model(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    *,
    adapted: bool,
    max_new_tokens: int,
    perf_repeats: int,
) -> dict[str, Any]:
    device = model._find_layer_device() if adapted else model.get_input_embeddings().weight.device
    rows: list[dict[str, Any]] = []
    category_values: dict[str, list[float]] = defaultdict(list)
    category_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()
    for case in cases:
        if adapted:
            model.reset_memory()
        prompt = prompt_inputs(tokenizer, str(case["prompt"]), device)
        with torch.inference_mode():
            if adapted:
                output = model.generate(
                    **prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    update_memory=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                output = model.generate(
                    **prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
        generated = _decode_generation(tokenizer, output, prompt)
        passed = contains_answer(generated, list(case["acceptable"]))
        category = str(case["category"])
        category_values[category].append(float(passed))
        if len(category_examples[category]) < 3:
            category_examples[category].append(
                {
                    "id": case["id"],
                    "prompt": case["prompt"],
                    "acceptable": case["acceptable"],
                    "generated": generated,
                    "passed": passed,
                }
            )
        rows.append(
            {
                "id": case["id"],
                "category": category,
                "acceptable": case["acceptable"],
                "generated": generated,
                "passed": passed,
            }
        )
    elapsed = time.perf_counter() - started

    perf_case = cases[0]
    if adapted:
        model.reset_memory()
    perf_prompt = prompt_inputs(tokenizer, str(perf_case["prompt"]), device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    perf_start = time.perf_counter()
    generated_tokens = 0
    for _ in range(max(1, perf_repeats)):
        if adapted:
            model.reset_memory()
        with torch.inference_mode():
            if adapted:
                perf_output = model.generate(
                    **perf_prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    update_memory=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                perf_output = model.generate(
                    **perf_prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
        generated_tokens += int(perf_output.shape[1] - perf_prompt["input_ids"].shape[1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    perf_elapsed = time.perf_counter() - perf_start
    score = sum(sum(values) for values in category_values.values()) / len(cases)
    return {
        "device": str(device),
        "cases": len(cases),
        "elapsed_seconds": elapsed,
        "overall_score": score,
        "categories": {
            category: {
                "count": len(values),
                "score": sum(values) / len(values),
                "examples": category_examples[category],
            }
            for category, values in sorted(category_values.items())
        },
        "performance": {
            "repeats": max(1, perf_repeats),
            "max_new_tokens": max_new_tokens,
            "tokens_per_second": generated_tokens / max(perf_elapsed, 1e-9),
            "seconds_per_run": perf_elapsed / max(1, perf_repeats),
        },
        "rows": rows,
    }


def release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cases = load_cases(args.data)
    tokenizer = load_tokenizer(args.model_path)
    use_4bit = not args.no_4bit
    report: dict[str, Any] = {
        "model_path": str(Path(args.model_path).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "data": str(Path(args.data).resolve()),
        "quantization": "4bit_nf4" if use_4bit else "none",
        "cases": len(cases),
    }

    print("loading baseline")
    baseline = load_qwen_base(args.model_path, load_in_4bit=use_4bit)
    baseline.eval()
    baseline_device = baseline.get_input_embeddings().weight.device
    if baseline_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(baseline_device)
    report["baseline"] = evaluate_model(
        baseline,
        tokenizer,
        cases,
        adapted=False,
        max_new_tokens=args.max_new_tokens,
        perf_repeats=args.perf_repeats,
    )
    if baseline_device.type == "cuda":
        report["baseline"]["peak_memory_gb"] = torch.cuda.max_memory_allocated(baseline_device) / 1024**3
    release(baseline)

    config = load_memory_config(args.adapter)
    config.persistent_memory = False
    print(f"loading native adapter mode={config.mode} layers={config.layer_indices}")
    adapted = load_qwen_dynamic(
        args.model_path,
        memory_config=config,
        load_in_4bit=use_4bit,
    )
    adapted.load_memory_adapter(args.adapter)
    adapted.eval()
    adapted_device = adapted._find_layer_device()
    if adapted_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(adapted_device)
    report["native_memory"] = evaluate_model(
        adapted,
        tokenizer,
        cases,
        adapted=True,
        max_new_tokens=args.max_new_tokens,
        perf_repeats=args.perf_repeats,
    )
    if adapted_device.type == "cuda":
        report["native_memory"]["peak_memory_gb"] = torch.cuda.max_memory_allocated(adapted_device) / 1024**3

    baseline_score = report["baseline"]["overall_score"]
    native_score = report["native_memory"]["overall_score"]
    baseline_categories = report["baseline"]["categories"]
    native_categories = report["native_memory"]["categories"]
    category_deltas = {
        category: native_categories[category]["score"] - baseline_categories[category]["score"]
        for category in baseline_categories.keys() & native_categories.keys()
    }
    report["regression"] = {
        "overall_delta": native_score - baseline_score,
        "category_deltas": category_deltas,
        "overall_drop_limit": args.overall_drop_limit,
        "category_drop_limit": args.category_drop_limit,
        "overall_regression_alert": native_score < baseline_score - args.overall_drop_limit,
        "category_regression_alerts": {
            category: delta < -args.category_drop_limit for category, delta in category_deltas.items()
        },
        "pass": native_score >= baseline_score - args.overall_drop_limit
        and all(delta >= -args.category_drop_limit for delta in category_deltas.values()),
    }
    legacy_path = Path(args.adapter).parent / "benchmark_qwen_native_v3_eval.json"
    if legacy_path.exists():
        report["same_memory_benchmark"] = json.loads(legacy_path.read_text(encoding="utf-8"))
    native_eval_path = Path(args.adapter) / "native_eval_report.json"
    if native_eval_path.exists():
        report["native_memory_holdout"] = json.loads(native_eval_path.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("baseline", "native_memory", "regression")}, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
