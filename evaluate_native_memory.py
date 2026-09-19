"""Evaluate the learned native memory controller on held-out streaming records."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer
from .train_qwen_memory import encode_messages, pad_batch
from .train_native_memory import load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--adapter-dir", default="V2_dpskw/qwen_memory_adapter_native")
    parser.add_argument("--data", default="V2_dpskw/data/native_memory/eval.jsonl")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--restart-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--direct-logit-scale-override",
        type=float,
        default=None,
        help="temporarily override the adapter scale for generation diagnostics",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _is_refusal(text: str) -> bool:
    # The base model may phrase an abstention as "没有相关记录" or "没有访问
    # 权限" rather than the exact training answer "不知道。".  Count these
    # as safe abstentions; the report also keeps the raw generation.
    return any(marker in text for marker in ("不知道", "没有", "无相关", "未找到", "不清楚", "不确定", "无法", "不能"))


def _encode_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if isinstance(encoded, dict):
        return encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return torch.tensor([encoded], dtype=torch.long)


def _query_forward(model: Any, tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, float]:
    query_item = encode_messages(tokenizer, record["query"], max_length)
    query_input, query_mask, query_labels = pad_batch([query_item], int(tokenizer.pad_token_id))
    device = model._find_layer_device()
    output = model(
        input_ids=query_input.to(device),
        attention_mask=query_mask.to(device),
        labels=query_labels.to(device),
        read_memory=True,
        update_memory=False,
        return_memory=True,
        use_cache=False,
    )
    logits = output.logits[..., :-1, :]
    labels = query_labels.to(logits.device)[..., 1:]
    valid = labels.ne(-100)
    predictions = logits.argmax(dim=-1)
    token_accuracy = float((predictions[valid] == labels[valid]).float().mean()) if bool(valid.any()) else float("nan")
    return {"loss": float(output.loss.detach()), "token_accuracy": token_accuracy}


@torch.no_grad()
def _generate_query(model: Any, tokenizer: Any, record: dict[str, Any], max_new_tokens: int) -> str:
    prompt = _encode_prompt(tokenizer, record["query"][:-1])
    device = model._find_layer_device()
    generated = model.generate(
        input_ids=prompt.to(device),
        attention_mask=torch.ones_like(prompt, device=device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        update_memory=False,
    )
    new_tokens = generated[:, prompt.shape[1] :]
    return tokenizer.decode(new_tokens[0].detach().cpu().tolist(), skip_special_tokens=True).strip()


def _controller_step(model: Any, tokenizer: Any, chunk: dict[str, Any], max_length: int) -> tuple[float, float]:
    item = encode_messages(tokenizer, chunk["messages"], max_length)
    inputs, mask, _ = pad_batch([item], int(tokenizer.pad_token_id))
    device = model._find_layer_device()
    model(
        input_ids=inputs.to(device),
        attention_mask=mask.to(device),
        read_memory=False,
        update_memory=True,
        return_memory=True,
        use_cache=False,
    )
    write = model.memory.last_write_probability
    forget = model.memory.last_forget_probability
    if write is None or forget is None:
        raise RuntimeError("adapter does not expose native controller probabilities")
    return float(write.detach().mean()), float(forget.detach().mean())


def evaluate_records(model: Any, tokenizer: Any, records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    controller_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        model.reset_memory()
        for chunk_index, chunk in enumerate(record["memory_chunks"]):
            write, forget = _controller_step(model, tokenizer, chunk, args.max_length)
            write_target = float(chunk.get("write_label", 1.0))
            forget_target = float(chunk.get("forget_label", 0.0))
            kind = str(chunk.get("kind", "unknown"))
            write_correct = float((write >= 0.5) == (write_target >= 0.5))
            forget_correct = float((forget >= 0.5) == (forget_target >= 0.5))
            row = {
                "record_id": record.get("id"),
                "chunk_index": chunk_index,
                "kind": kind,
                "write_probability": write,
                "write_target": write_target,
                "write_correct": write_correct,
                "forget_probability": forget,
                "forget_target": forget_target,
                "forget_correct": forget_correct,
            }
            controller_rows.append(row)
            by_kind[kind]["write_correct"].append(write_correct)
            by_kind[kind]["forget_correct"].append(forget_correct)
            by_kind[kind]["write_probability"].append(write)
            by_kind[kind]["forget_probability"].append(forget)

        query = _query_forward(model, tokenizer, record, args.max_length)
        generated = _generate_query(model, tokenizer, record, args.max_new_tokens)
        answer = str(record.get("answer", ""))
        answerable = bool(record.get("answerable", False))
        contains_answer = bool(answer) and answer in generated if answerable else False
        says_unknown = _is_refusal(generated)
        generation_rows.append(
            {
                "record_id": record.get("id"),
                "answerable": answerable,
                "answer": answer,
                "generated": generated,
                "contains_answer": contains_answer,
                "says_unknown": says_unknown,
            }
        )
        query_rows.append({"record_id": record.get("id"), **query})

    controller_metrics = {
        "write_accuracy": _mean([row["write_correct"] for row in controller_rows]),
        "forget_accuracy": _mean([row["forget_correct"] for row in controller_rows]),
        "write_bce_proxy": _mean([
            -(row["write_target"] * math.log(max(row["write_probability"], 1e-7))
            + (1.0 - row["write_target"]) * math.log(max(1.0 - row["write_probability"], 1e-7))
        )
            for row in controller_rows
        ]),
        "forget_bce_proxy": _mean([
            -(row["forget_target"] * math.log(max(row["forget_probability"], 1e-7))
            + (1.0 - row["forget_target"]) * math.log(max(1.0 - row["forget_probability"], 1e-7))
        )
            for row in controller_rows
        ]),
        "by_kind": {
            kind: {
                "count": len(values["write_correct"]),
                "write_accuracy": _mean(values["write_correct"]),
                "forget_accuracy": _mean(values["forget_correct"]),
                "write_probability": _mean(values["write_probability"]),
                "forget_probability": _mean(values["forget_probability"]),
            }
            for kind, values in by_kind.items()
        },
    }
    answerable_rows = [row for row in generation_rows if row["answerable"]]
    unknown_rows = [row for row in generation_rows if not row["answerable"]]
    query_metrics = {
        "mean_loss": _mean([row["loss"] for row in query_rows]),
        "token_accuracy": _mean([row["token_accuracy"] for row in query_rows]),
    }
    generation_metrics = {
        "answerable_count": len(answerable_rows),
        "answer_containment": _mean([float(row["contains_answer"]) for row in answerable_rows]),
        "unknown_count": len(unknown_rows),
        "unknown_refusal": _mean([float(row["says_unknown"]) for row in unknown_rows]),
    }
    return {
        "controller": controller_metrics,
        "query": query_metrics,
        "generation": generation_metrics,
        "controller_rows": controller_rows,
        "query_rows": query_rows,
        "generation_rows": generation_rows,
    }


def restart_probe(model_path: str, adapter_dir: str, tokenizer: Any, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Save state, recreate the model, and query without passing history."""
    config = load_memory_config(adapter_dir)
    config.persistent_memory = True
    if args.direct_logit_scale_override is not None:
        config.direct_logit_scale = args.direct_logit_scale_override
    first_model = load_qwen_dynamic(model_path, memory_config=config, load_in_4bit=not args.no_4bit)
    first_model.load_memory_adapter(adapter_dir)
    first_model.reset_memory()
    for chunk in record["memory_chunks"]:
        _controller_step(first_model, tokenizer, chunk, args.max_length)
    state_path = Path(adapter_dir) / "native_restart_probe_memory.pt"
    first_model.save_runtime_memory(state_path)
    del first_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    second_model = load_qwen_dynamic(model_path, memory_config=config, load_in_4bit=not args.no_4bit)
    second_model.load_memory_adapter(adapter_dir)
    second_model.load_runtime_memory(state_path)
    generated = _generate_query(second_model, tokenizer, record, args.max_new_tokens)
    answer = str(record.get("answer", ""))
    return {
        "record_id": record.get("id"),
        "generated_after_restart": generated,
        "answer": answer,
        "contains_answer": answer in generated if record.get("answerable") else False,
        "state_path": str(state_path),
    }


def main() -> None:
    args = parse_args()
    records = load_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]
    tokenizer = load_tokenizer(args.model_path)
    config = load_memory_config(args.adapter_dir)
    config.persistent_memory = True
    if args.direct_logit_scale_override is not None:
        config.direct_logit_scale = args.direct_logit_scale_override
    model = load_qwen_dynamic(args.model_path, memory_config=config, load_in_4bit=not args.no_4bit)
    model.load_memory_adapter(args.adapter_dir)
    model.eval()
    report = evaluate_records(model, tokenizer, records, args)
    if args.restart_test:
        probe_record = next(record for record in records if record.get("answerable"))
        report["restart_probe"] = restart_probe(args.model_path, args.adapter_dir, tokenizer, probe_record, args)
    report["model_path"] = str(args.model_path)
    report["adapter_dir"] = str(args.adapter_dir)
    report["data"] = str(args.data)
    report_path = Path(args.report) if args.report else Path(args.adapter_dir) / "native_eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("controller", "query", "generation", "restart_probe") if key in report}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
