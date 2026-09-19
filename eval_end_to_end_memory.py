"""End-to-end answer evaluation for Natural Memory routers.

The router scorecard measures a *proxy*: whether the right evidence is ranked top.
The product promise is an answer, so this harness closes the loop on real hardware:

    write the evidence -> model reads it with the router under test
    -> Qwen generates the answer -> score the emitted text

It exists because a router can improve ranking while the emitted answer stays
wrong (the project's own notes say "recall != answer").  Cases are taken from the
frozen v6 eval, stratified per category, and every router is run over the *same*
cases with the same writes so the comparison is paired.

Scoring uses the source ``acceptable`` strings: for answerable episodes those are
the expected values, and for abstention categories they are the refusal phrases,
so a single containment check covers both.

Usage::

    python -m V2_dpskw.eval_end_to_end_memory ^
        --package qwen3_5_4b_natural_memory_v2 ^
        --eval-file data/router_training_v6/eval.jsonl ^
        --per-category 20 ^
        --router "deployed=" ^
        --router "V2-128-v6=checkpoints/router_v6_v2_128/router_best.pt" ^
        --output router_end_to_end_v6.json --markdown router_end_to_end_v6.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.qwen_integration import (
    QwenMemoryConfig,
    format_memory_evidence,
    infer_memory_metadata,
    load_memory_config,
    load_qwen_dynamic,
    load_tokenizer,
)

from V2_dpskw.eval_scoring import ABSTENTION_MARKERS, score_case  # noqa: F401  (re-exported)


def _chat_tensor(tokenizer, user_text: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, enable_thinking=False,
    )
    return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}


def _system_prefix(tokenizer, content: str) -> dict[str, torch.Tensor]:
    """Encode a system-only prefix by cutting the template at the boundary.

    The chat template refuses a system-only message list ("no user query found"),
    so this mirrors ``stream_chat_qwen_memory._memory_system_prefix``: render a
    system + placeholder user turn, then truncate at the second ``<|im_start|>``.
    """

    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": content},
         {"role": "user", "content": "__memory_query_boundary__"}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, enable_thinking=False,
    )
    input_ids = full["input_ids"]
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    positions = (input_ids[0] == int(im_start)).nonzero(as_tuple=False).flatten()
    if positions.numel() < 2:
        raise RuntimeError("could not locate the system/user memory boundary")
    end = int(positions[1].item())
    return {"input_ids": input_ids[:, :end], "attention_mask": torch.ones((1, end), dtype=torch.long)}


@torch.inference_mode()
def write_fact(model, tokenizer, text: str, device: torch.device) -> bool:
    """Store one fact through the model's own write controller."""

    encoded = {key: value.to(device) for key, value in _chat_tensor(tokenizer, text).items()}
    metadata = infer_memory_metadata(text)
    evidence = format_memory_evidence(
        text, entity=str(metadata.get("entity", "")),
        attribute=str(metadata.get("attribute", "")), value=str(metadata.get("value", "")),
    )
    prefix = _system_prefix(
        tokenizer,
        "以下是与当前用户相关的已保存长期记忆。仅在问题相关时使用，只能依据明确证据；"
        "先核对实体、属性和已确认值；冲突优先最新可靠来源，不要拼接不确定候选，"
        "证据不足就明确说不知道；涉及名称、路径、token、参数或结论时，原样复述证据中的关键短语：\n"
        + evidence,
    )
    key = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    storage = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    model(
        **encoded,
        read_memory=False,
        update_memory=True,
        return_memory=True,
        use_cache=False,
        memory_text_input_ids=prefix["input_ids"].to(device),
        memory_text_attention_mask=prefix["attention_mask"].to(device),
        memory_key_input_ids=key["input_ids"].to(device),
        memory_key_attention_mask=key.get("attention_mask", torch.ones_like(key["input_ids"])).to(device),
        memory_storage_input_ids=storage["input_ids"].to(device),
        memory_storage_attention_mask=storage.get("attention_mask", torch.ones_like(storage["input_ids"])).to(device),
        force_memory_write=True,
        memory_text=text,
    )
    last = model.runtime.text_last_written_slot
    return bool((last >= 0).any()) if isinstance(last, torch.Tensor) else False


@torch.inference_mode()
def answer(model, tokenizer, query: str, device: torch.device, max_new_tokens: int) -> str:
    """One turn: let the router decide what to read, then generate."""

    encoded = {key: value.to(device) for key, value in _chat_tensor(tokenizer, query).items()}
    query_tokens = tokenizer(query, add_special_tokens=False, return_tensors="pt")
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        update_memory=False,
        memory_query_input_ids=query_tokens["input_ids"].to(device),
        memory_query_attention_mask=query_tokens.get(
            "attention_mask", torch.ones_like(query_tokens["input_ids"])
        ).to(device),
        memory_query_text=query,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output[0][encoded["input_ids"].shape[1]:] if isinstance(output, torch.Tensor) else output
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_cases(path: Path, per_category: int) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            category = str(metadata.get("category", "") or "(policy)")
            if len(by_category[category]) >= per_category:
                continue
            acceptable = [str(value) for value in (metadata.get("acceptable") or []) if str(value).strip()]
            candidates = [str(c.get("text", "")) for c in (row.get("candidates") or []) if isinstance(c, dict)]
            positives = [candidates[i] for i in (row.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
            by_category[category].append({
                "category": category,
                "query": str(row.get("query", "")),
                "acceptable": acceptable,
                "positives": positives,
                "facts": candidates[:32],
                "answerable": bool(positives),
            })
    cases: list[dict] = []
    for category in sorted(by_category):
        cases.extend(by_category[category])
    return cases


def evidence_write_order(case: dict, facts: int, *, answer_last: bool = True) -> list[str]:
    """Order in which a case's evidence is written into the bank.

    ``answer_last`` (default) writes distractors first and the answering facts
    last, so the expected answer is the freshest evidence -- which is how a real
    session arrives, oldest fact first.

    Writing the answering facts *first* (the original behaviour) made every later
    same-attribute fact newer than the expected answer.  On ``update_conflict``
    that inverted the question: the expected answer was the oldest of five writes
    for the asked attribute in 25/25 cases, and the newest was a random
    distractor, so the category was unanswerable by construction and pinned at
    44.00%.

    The answering facts get *reserved* slots rather than being appended to a
    truncated list.  ``noise_context`` carries 8 distractors against 1 answering
    fact, so a naive ``(distractors + positives)[:facts]`` dropped the answer
    entirely and scored 0.00% where the old order scored 72.00%.
    """
    positives = case["positives"]
    room = max(0, facts - len(positives))
    distractors = [f for f in case["facts"] if f not in positives][:room]
    return (distractors + positives) if answer_last else (positives + distractors)


def run_router(model, tokenizer, device, cases, *, max_new_tokens: int, facts: int, label: str,
               answer_last: bool = True) -> dict:
    per_category: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "answerable": 0, "answerable_correct": 0,
                                                         "unknown": 0, "unknown_correct": 0, "wrongly_abstained": 0})
    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        # ``reset_memory`` (not ``reset_runtime_memory``) is required here: the V2
        # paged bank only accepts writes while ``runtime.use_persistent_state`` is
        # True, and ``reset_runtime_memory`` sets it to False for "temporary"
        # sessions.  Using it silently produced an empty bank, a router decision of
        # ``below_read_threshold`` and 0% accuracy for every router.
        # This harness never persists, so the package on disk is untouched.
        model.reset_memory(batch_size=1, device=device)
        # Chronology matters; see evidence_write_order for why the answering facts
        # are reserved rather than appended.
        to_write = evidence_write_order(case, facts, answer_last=answer_last)
        written = 0
        for text in to_write[:facts]:
            if write_fact(model, tokenizer, text, device):
                written += 1
        reply = answer(model, tokenizer, case["query"], device, max_new_tokens)
        scored = score_case(case, reply)
        bucket = per_category[case["category"]]
        bucket["n"] += 1
        bucket["correct"] += int(scored["correct"])
        if case["answerable"]:
            bucket["answerable"] += 1
            bucket["answerable_correct"] += int(scored["correct"])
        else:
            bucket["unknown"] += 1
            bucket["unknown_correct"] += int(scored["correct"])
        bucket["wrongly_abstained"] += int(scored["wrongly_abstained"])
        rows.append({"category": case["category"], "query": case["query"], "reply": reply[:200],
                     "written": written, **scored})
        if index % 25 == 0:
            print(json.dumps({"router": label, "case": index, "total": len(cases),
                              "elapsed_s": round(time.perf_counter() - started, 1),
                              "running_accuracy_pct": round(100 * sum(r["correct"] for r in rows) / len(rows), 2)}),
                  flush=True)
    total = len(rows)
    answerable = [r for r, c in zip(rows, cases) if c["answerable"]]
    unknown = [r for r, c in zip(rows, cases) if not c["answerable"]]
    summary = {
        "router": label,
        "cases": total,
        "write_order": "answer-last" if answer_last else "answer-first",
        "scorer": "whitespace-insensitive containment (eval_scoring.score_case)",
        "accuracy_pct": 100 * sum(r["correct"] for r in rows) / max(1, total),
        "answerable_cases": len(answerable),
        "answerable_accuracy_pct": 100 * sum(r["correct"] for r in answerable) / max(1, len(answerable)),
        "unknown_cases": len(unknown),
        "unknown_refusal_pct": 100 * sum(r["correct"] for r in unknown) / max(1, len(unknown)),
        "wrong_abstention_pct": 100 * sum(r["wrongly_abstained"] for r in answerable) / max(1, len(answerable)),
        "seconds": round(time.perf_counter() - started, 1),
        "per_category": {
            name: {
                "cases": block["n"],
                "accuracy_pct": 100 * block["correct"] / max(1, block["n"]),
                "answerable": block["answerable"],
                "answerable_accuracy_pct": 100 * block["answerable_correct"] / max(1, block["answerable"]),
                "unknown": block["unknown"],
                "unknown_refusal_pct": 100 * block["unknown_correct"] / max(1, block["unknown"]),
            }
            for name, block in sorted(per_category.items())
        },
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--eval-file", default="data/router_training_v6/eval.jsonl")
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--facts", type=int, default=6, help="how many evidence texts to write per case")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--top-k-records", type=int, default=0,
                        help="override memory_top_k_records: fewer injected records gives the "
                             "generator less room to cite a different attribute")
    parser.add_argument("--write-order", choices=("answer-last", "answer-first"), default="answer-last",
                        help="answer-last (default) writes distractors first so the expected answer is "
                             "the freshest evidence; answer-first reproduces pre-fix runs, where the "
                             "expected answer was older than the contradicting facts")
    parser.add_argument("--router", action="append", required=True, help="LABEL=PATH ('LABEL=' keeps the deployed router)")
    parser.add_argument("--output", default="router_end_to_end_v6.json")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    cases = build_cases(Path(args.eval_file), args.per_category)
    print(json.dumps({"cases": len(cases), "categories": len({c['category'] for c in cases}),
                      "answerable": sum(1 for c in cases if c["answerable"]),
                      "unknown": sum(1 for c in cases if not c["answerable"])}, ensure_ascii=False), flush=True)

    model_path = Path(args.package)
    memory_config: QwenMemoryConfig = load_memory_config(model_path)
    if args.top_k_records:
        memory_config.memory_top_k_records = int(args.top_k_records)
        print(json.dumps({"top_k_records_override": args.top_k_records}), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_qwen_dynamic(model_path, memory_config=memory_config, load_in_4bit=True,
                             max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    tokenizer = load_tokenizer(model_path)
    print(json.dumps({"model_loaded": True, "router_ready": bool(getattr(model, "_memory_router_v2_ready", False))}), flush=True)

    results = {}
    for spec in args.router:
        label, _, path_value = spec.partition("=")
        if path_value.strip():
            state = torch.load(path_value.strip(), map_location="cpu", weights_only=True)
            state = state.get("router_state_dict", state)
            model.memory_router_v2.load_state_dict(state, strict=True)
            model.memory_router_v2.to(device).eval()
            print(json.dumps({"router_swapped": label, "from": path_value.strip()}), flush=True)
        else:
            print(json.dumps({"router_kept": label, "note": "deployed weights from the package shard"}), flush=True)
        results[label] = run_router(model, tokenizer, device, cases, max_new_tokens=args.max_new_tokens,
                                    facts=args.facts, label=label,
                                    answer_last=args.write_order == "answer-last")
        print(json.dumps(results[label]["summary"], ensure_ascii=False), flush=True)

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| 路由器 | 用例 | 总体正确率 | 可回答正确率 | 未知拒答率 | 已知被误拒率 | 耗时 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, block in results.items():
        s = block["summary"]
        lines.append("| {l} | {c} | {a:.2f}% | {b:.2f}% | {u:.2f}% | {w:.2f}% | {t:.0f}s |".format(
            l=label, c=s["cases"], a=s["accuracy_pct"], b=s["answerable_accuracy_pct"],
            u=s["unknown_refusal_pct"], w=s["wrong_abstention_pct"], t=s["seconds"]))
    table = "\n".join(lines)
    print("\n" + table, flush=True)
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
