"""Compare the unmodified local Qwen checkpoint with a memory-surgery adapter."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch

from .qwen_integration import QwenMemoryConfig, load_memory_config, load_qwen_base, load_qwen_dynamic, load_tokenizer
from .train_qwen_memory import encode_messages, load_records, pad_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--data", default="V2_dpskw/data/demo_stream.jsonl")
    parser.add_argument(
        "--adapter",
        default="V2_dpskw/qwen_memory_adapter_surgery_smoke",
        help="dynamic-memory adapter directory; its memory_config.json selects the surgery mode",
    )
    parser.add_argument("--output", default="V2_dpskw/benchmark_qwen.json")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def _adapter_config(adapter_dir: str | Path) -> QwenMemoryConfig:
    return load_memory_config(adapter_dir)


def _score_output(output: Any, labels: torch.Tensor) -> tuple[float, int, int, int, bool]:
    shifted_labels = labels[..., 1:]
    predictions = output.logits[..., :-1, :].argmax(dim=-1)
    target_positions = shifted_labels != -100
    token_count = int(target_positions.sum().item())
    if output.loss is None or token_count == 0:
        raise RuntimeError("benchmark example has no supervised target tokens")
    correct_tokens = int((predictions[target_positions] == shifted_labels[target_positions]).sum().item())
    first_target = target_positions.nonzero(as_tuple=False)[0]
    first_token_correct = int(
        predictions[first_target[0], first_target[1]] == shifted_labels[first_target[0], first_target[1]]
    )
    sequence_ok = correct_tokens == token_count
    return float(output.loss.detach().item()), token_count, correct_tokens, first_token_correct, sequence_ok


def _evaluate_base(model: Any, tokenizer: Any, records: list[dict[str, Any]], max_length: int) -> dict[str, float]:
    device = model.get_input_embeddings().weight.device
    total_nll = 0.0
    total_tokens = 0
    correct_tokens = 0
    correct_first_tokens = 0
    correct_sequences = 0
    for record in records:
        query = encode_messages(tokenizer, record["query"], max_length)
        input_ids, attention_mask, labels = pad_batch([query], int(tokenizer.pad_token_id))
        with torch.inference_mode():
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
                use_cache=False,
            )
        loss, tokens, tokens_correct, first_token_correct, sequence_ok = _score_output(output, labels.to(device))
        total_nll += loss * tokens
        total_tokens += tokens
        correct_tokens += tokens_correct
        correct_first_tokens += first_token_correct
        correct_sequences += int(sequence_ok)
    mean_loss = total_nll / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "token_accuracy": correct_tokens / total_tokens,
        "first_target_token_accuracy": correct_first_tokens / len(records),
        "exact_sequence_accuracy": correct_sequences / len(records),
        "supervised_tokens": total_tokens,
    }


def _evaluate_dynamic(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
) -> dict[str, float]:
    device = model._find_layer_device()
    pad_id = int(tokenizer.pad_token_id)
    total_nll = 0.0
    total_tokens = 0
    correct_tokens = 0
    correct_first_tokens = 0
    correct_sequences = 0
    for record in records:
        model.reset_memory()
        memory = encode_messages(tokenizer, record["memory"], max_length)
        query = encode_messages(tokenizer, record["query"], max_length)
        memory_input, memory_mask, _ = pad_batch([memory], pad_id)
        query_input, query_mask, query_labels = pad_batch([query], pad_id)
        with torch.inference_mode():
            memory_output = model(
                input_ids=memory_input.to(device),
                attention_mask=memory_mask.to(device),
                read_memory=False,
                update_memory=True,
                return_memory=True,
                use_cache=False,
            )
            output = model(
                input_ids=query_input.to(device),
                attention_mask=query_mask.to(device),
                labels=query_labels.to(device),
                memory_state=memory_output.memory,
                read_memory=True,
                update_memory=False,
                return_memory=True,
                use_cache=False,
            )
        loss, tokens, tokens_correct, first_token_correct, sequence_ok = _score_output(
            output, query_labels.to(device)
        )
        total_nll += loss * tokens
        total_tokens += tokens
        correct_tokens += tokens_correct
        correct_first_tokens += first_token_correct
        correct_sequences += int(sequence_ok)
    model.reset_memory()
    mean_loss = total_nll / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "token_accuracy": correct_tokens / total_tokens,
        "first_target_token_accuracy": correct_first_tokens / len(records),
        "exact_sequence_accuracy": correct_sequences / len(records),
        "supervised_tokens": total_tokens,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(name: str, fn: Callable[[], int], repeats: int, warmup: int, device: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    started = time.perf_counter()
    processed = 0
    for _ in range(repeats):
        processed += fn()
    _sync(device)
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed / repeats,
        "tokens_per_second": processed / elapsed,
    }


def _generation_prompt(tokenizer: Any, messages: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") and "input_ids" in encoded else encoded
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _measure_base_generation(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    repeats: int,
    warmup: int,
    max_new_tokens: int,
) -> dict[str, float]:
    device = model.get_input_embeddings().weight.device
    prompt = _generation_prompt(tokenizer, record["query"][:-1], device)

    def run() -> int:
        output = model.generate(
            **prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        return int(output.shape[1] - prompt["input_ids"].shape[1])

    return _measure("baseline_generation", run, repeats, warmup, device)


def _measure_dynamic_generation(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int,
    repeats: int,
    warmup: int,
    max_new_tokens: int,
) -> dict[str, float]:
    device = model._find_layer_device()
    pad_id = int(tokenizer.pad_token_id)
    memory = encode_messages(tokenizer, record["memory"], max_length)
    memory_input, memory_mask, _ = pad_batch([memory], pad_id)
    prompt = _generation_prompt(tokenizer, record["query"][:-1], device)

    def run() -> int:
        model.reset_memory()
        with torch.inference_mode():
            model(
                input_ids=memory_input.to(device),
                attention_mask=memory_mask.to(device),
                read_memory=False,
                update_memory=True,
                return_memory=True,
                use_cache=False,
            )
            output = model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                update_memory=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        return int(output.shape[1] - prompt["input_ids"].shape[1])

    result = _measure("dynamic_generation", run, repeats, warmup, device)
    model.reset_memory()
    return result


def _clean_generated(text: str) -> str:
    return text.replace(" ", "").replace("\r", "").replace("\n", "").strip()


def _generation_quality_base(model: Any, tokenizer: Any, records: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    device = model.get_input_embeddings().weight.device
    contains = 0
    prefixes = 0
    examples = []
    for record in records:
        prompt = _generation_prompt(tokenizer, record["query"][:-1], device)
        with torch.inference_mode():
            output = model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = _clean_generated(tokenizer.decode(output[0, prompt["input_ids"].shape[1] :], skip_special_tokens=True))
        expected = str(record["answer"])
        contains += int(expected in generated)
        prefixes += int(generated.startswith(expected))
        if len(examples) < 3:
            examples.append({"expected": expected, "generated": generated})
    return {
        "answer_contains_accuracy": contains / len(records),
        "answer_prefix_accuracy": prefixes / len(records),
        "examples": examples,
    }


def _generation_quality_dynamic(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = model._find_layer_device()
    pad_id = int(tokenizer.pad_token_id)
    contains = 0
    prefixes = 0
    examples = []
    for record in records:
        model.reset_memory()
        memory = encode_messages(tokenizer, record["memory"], max_length)
        memory_input, memory_mask, _ = pad_batch([memory], pad_id)
        prompt = _generation_prompt(tokenizer, record["query"][:-1], device)
        with torch.inference_mode():
            model(
                input_ids=memory_input.to(device),
                attention_mask=memory_mask.to(device),
                read_memory=False,
                update_memory=True,
                return_memory=True,
                use_cache=False,
            )
            output = model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                update_memory=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = _clean_generated(tokenizer.decode(output[0, prompt["input_ids"].shape[1] :], skip_special_tokens=True))
        expected = str(record["answer"])
        contains += int(expected in generated)
        prefixes += int(generated.startswith(expected))
        if len(examples) < 3:
            examples.append({"expected": expected, "generated": generated})
    model.reset_memory()
    return {
        "answer_contains_accuracy": contains / len(records),
        "answer_prefix_accuracy": prefixes / len(records),
        "examples": examples,
    }


def _release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup must be >= 0")
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    records = load_records(args.data)
    use_4bit = not args.no_4bit
    results: dict[str, Any] = {
        "model_path": str(Path(args.model_path).resolve()),
        "data": str(Path(args.data).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "records": len(records),
        "max_length": args.max_length,
        "quantization": "4bit_nf4" if use_4bit else "none",
    }

    print("loading baseline")
    baseline = load_qwen_base(args.model_path, load_in_4bit=use_4bit)
    baseline.eval()
    baseline_device = baseline.get_input_embeddings().weight.device
    if baseline_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(baseline_device)
    results["baseline"] = {
        "device": str(baseline_device),
        "scores": _evaluate_base(baseline, tokenizer, records, args.max_length),
        "generation_quality": _generation_quality_base(baseline, tokenizer, records, args.max_new_tokens),
        "generation": _measure_base_generation(
            baseline, tokenizer, records[0], args.repeats, args.warmup, args.max_new_tokens
        ),
    }
    if baseline_device.type == "cuda":
        results["baseline"]["peak_memory_gb"] = torch.cuda.max_memory_allocated(baseline_device) / 1024**3
    _release(baseline)
    baseline = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_config = _adapter_config(args.adapter)
    print(f"loading dynamic mode={memory_config.mode} layers={memory_config.layer_indices}")
    dynamic = load_qwen_dynamic(
        args.model_path,
        memory_config=memory_config,
        load_in_4bit=use_4bit,
    )
    dynamic.load_memory_adapter(args.adapter)
    dynamic.eval()
    dynamic_device = dynamic._find_layer_device()
    if dynamic_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dynamic_device)
    results["dynamic_memory"] = {
        "device": str(dynamic_device),
        "mode": memory_config.mode,
        "layers": list(dynamic.layer_indices),
        "scores": _evaluate_dynamic(dynamic, tokenizer, records, args.max_length),
        "generation_quality": _generation_quality_dynamic(
            dynamic, tokenizer, records, args.max_length, args.max_new_tokens
        ),
        "generation": _measure_dynamic_generation(
            dynamic,
            tokenizer,
            records[0],
            args.max_length,
            args.repeats,
            args.warmup,
            args.max_new_tokens,
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in dynamic.trainable_parameters),
    }
    if dynamic_device.type == "cuda":
        results["dynamic_memory"]["peak_memory_gb"] = torch.cuda.max_memory_allocated(dynamic_device) / 1024**3
    _release(dynamic)
    dynamic = None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved={output_path.resolve()}")


if __name__ == "__main__":
    main()
