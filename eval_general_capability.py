"""General-capability regression with the memory read path ACTIVE.

``comprehensive_benchmark.py`` runs its adapted model without
``memory_query_*``, so the router never executes: it verifies that the memory
*architecture* does not disturb generation, not that the router behaves on
unrelated questions.  In production the router runs on every turn, so the real
risk is a populated memory bank polluting a general question.

This harness therefore measures the harder condition:

1. a plain Qwen baseline with no memory at all;
2. the shipped package with a **deliberately populated** bank of irrelevant
   personal facts, the read path active, for each router under test;
3. per-category and overall scores plus how often the router chose to read.

A correct router abstains on arithmetic/logic/code prompts, so the memory-enabled
scores must match the no-memory baseline.

Usage::

    python -m V2_dpskw.eval_general_capability ^
        --data <comprehensive_general.jsonl> ^
        --router "deployed=" ^
        --router "V2-128-v6=checkpoints/router_v6_v2_128/router_best.pt" ^
        --output router_general_capability.json --markdown router_general_capability.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_end_to_end_memory import _chat_tensor, answer, write_fact
from V2_dpskw.qwen_integration import (
    load_memory_config,
    load_qwen_base,
    load_qwen_dynamic,
    load_tokenizer,
)

#: Unrelated personal facts used to populate the bank before general questions.
DISTRACTOR_FACTS = [
    "我的工作地点是 V-8QPA0VMT。",
    "我的档案代号是 V-2RPEJ33G。",
    "我住在杭州，通勤方式是地铁。",
    "我的项目编号是 PROJECT-M3GWABOJ。",
    "我的默认语言是中文。",
    "我常用的编辑器是 VS Code。",
    "我的工位在 3 号楼 4 层。",
    "我的邮箱后缀是 example.com。",
    "我的常用编程语言是 Python。",
    "我的生日是 3 月 12 日。",
    "我的主管是李工。",
    "我的团队名称是记忆系统组。",
]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"\s+", "", value)


def contains_answer(text: str, acceptable: list[str]) -> bool:
    """Same scoring rule as comprehensive_benchmark.py."""

    normalized = normalize(text)
    for expected in acceptable:
        expected = normalize(str(expected))
        if not expected:
            continue
        if expected.isdigit() and len(expected) == 1:
            if re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", normalized):
                return True
        elif expected in normalized:
            return True
    return False


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.inference_mode()
def run_cases(model, tokenizer, cases, device, *, max_new_tokens: int, use_memory: bool) -> dict:
    per_category: dict[str, list[float]] = defaultdict(list)
    read_decisions = 0
    rows = []
    started = time.perf_counter()
    for case in cases:
        encoded = {key: value.to(device) for key, value in _chat_tensor(tokenizer, str(case["prompt"])).items()}
        kwargs: dict = {}
        if use_memory:
            query_tokens = tokenizer(str(case["prompt"]), add_special_tokens=False, return_tensors="pt")
            kwargs = {
                "update_memory": False,
                "memory_query_input_ids": query_tokens["input_ids"].to(device),
                "memory_query_attention_mask": query_tokens.get(
                    "attention_mask", torch.ones_like(query_tokens["input_ids"])
                ).to(device),
                "memory_query_text": str(case["prompt"]),
            }
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )
        generated = output[0][encoded["input_ids"].shape[1]:] if isinstance(output, torch.Tensor) else output
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        passed = contains_answer(text, list(case["acceptable"]))
        per_category[str(case["category"])].append(float(passed))
        decided = False
        if use_memory and model.runtime.v2_last_decisions:
            decided = bool(model.runtime.v2_last_decisions[-1].get("need_memory"))
            read_decisions += int(decided)
        rows.append({"id": case["id"], "category": case["category"], "passed": passed,
                     "read_memory": decided, "reply": text[:160]})
    summary = {
        "cases": len(rows),
        "overall_pct": 100 * sum(r["passed"] for r in rows) / max(1, len(rows)),
        "per_category_pct": {name: 100 * sum(values) / max(1, len(values)) for name, values in sorted(per_category.items())},
        "read_decisions": read_decisions,
        "seconds": round(time.perf_counter() - started, 1),
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--data", required=True, help="comprehensive_general.jsonl")
    parser.add_argument("--router", action="append", required=True, help="LABEL=PATH ('LABEL=' keeps the deployed router)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--facts", type=int, default=len(DISTRACTOR_FACTS))
    parser.add_argument("--output", default="router_general_capability.json")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    cases = load_cases(Path(args.data))
    print(json.dumps({"cases": len(cases), "categories": len({c['category'] for c in cases}),
                      "distractor_facts": args.facts}, ensure_ascii=False), flush=True)

    model_path = Path(args.package)
    tokenizer = load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    baseline = load_qwen_base(model_path, load_in_4bit=True, max_memory={0: "10.5GiB", "cpu": "48GiB"})
    baseline.eval()
    results["Qwen3.5-4B baseline (no memory)"] = run_cases(
        baseline, tokenizer, cases, device, max_new_tokens=args.max_new_tokens, use_memory=False)
    print(json.dumps({"baseline": results["Qwen3.5-4B baseline (no memory)"]["summary"]}, ensure_ascii=False), flush=True)
    del baseline
    torch.cuda.empty_cache()

    memory_config = load_memory_config(model_path)
    model = load_qwen_dynamic(model_path, memory_config=memory_config, load_in_4bit=True,
                             max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    for spec in args.router:
        label, _, path_value = spec.partition("=")
        if path_value.strip():
            state = torch.load(path_value.strip(), map_location="cpu", weights_only=True)
            model.memory_router_v2.load_state_dict(state.get("router_state_dict", state), strict=True)
            model.memory_router_v2.to(device).eval()
        model.reset_memory(batch_size=1, device=device)
        written = sum(int(write_fact(model, tokenizer, text, device)) for text in DISTRACTOR_FACTS[: args.facts])
        entry = run_cases(model, tokenizer, cases, device, max_new_tokens=args.max_new_tokens, use_memory=True)
        entry["summary"]["facts_written"] = written
        entry["summary"]["router"] = label
        results[label] = entry
        print(json.dumps(entry["summary"], ensure_ascii=False), flush=True)

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    base = results["Qwen3.5-4B baseline (no memory)"]["summary"]
    lines = ["| 条件 | 总体 | " + " | ".join(base["per_category_pct"]) + " | 触发读取次数 |",
             "|---" * (len(base["per_category_pct"]) + 3) + "|"]
    for label, block in results.items():
        s = block["summary"]
        cells = " | ".join(f"{s['per_category_pct'].get(name, float('nan')):.2f}%" for name in base["per_category_pct"])
        delta = s["overall_pct"] - base["overall_pct"]
        lines.append(f"| {label} | {s['overall_pct']:.2f}% ({delta:+.2f}pp) | {cells} | {s.get('read_decisions', 0)}/{s['cases']} |")
    table = "\n".join(lines)
    print("\n" + table, flush=True)
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
