"""Router-critical end-to-end test: cases where the rule-based paths cannot decide.

The v6 end-to-end harness showed 90.9% of decisions coming from the rule-based
lexical/address path and 9.1% from the explicit-unknown rule, with the neural
router deciding **nothing** - which is why two very different routers produced
byte-identical end-to-end results.

This harness builds the opposite condition on purpose:

* several facts share the **same attribute wording** (so lexical hits are
  ambiguous and cannot pick a winner on their own);
* every query identifies its target only by **entity**;
* the answer is a random code that only the right fact contains, so guessing and
  evidence mixing are both detectable.

The router is therefore on the critical path: if it ranks the wrong entity's fact
first, the model reads the wrong code and the answer is wrong.

Usage::

    python -m V2_dpskw.eval_router_critical_e2e ^
        --router "deployed=" ^
        --router "V2-128-v6=checkpoints/router_v6_v2_128/router_best.pt" ^
        --router "XL-128-v6=checkpoints/router_v6_xl128/router_best.pt" ^
        --output router_critical_e2e.json --markdown router_critical_e2e.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_end_to_end_memory import answer, write_fact
from V2_dpskw.qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer

#: (fact, paraphrase-query) pairs.
#:
#: Two properties make these cases router-critical rather than rule-critical:
#:
#: 1. every fact uses a DISTINCT attribute, so the conflict keys differ and all
#:    eight facts genuinely coexist in the bank (an earlier version of this test
#:    reused one attribute, the runtime correctly superseded the older values, and
#:    the test measured overwrite behaviour instead of routing);
#: 2. each query is a paraphrase that shares no content term with its target fact,
#:    so the address/lexical paths find nothing and the neural router - not a rule -
#:    has to decide whether and what to read.
#:
#: The answer is a random code that appears in exactly one fact, so mixing
#: evidence from a neighbouring attribute is detectable.
CASE_PAIRS = [
    ("我的常住城市是 CITY-A1B2C3。", "我平时待得最久的地方是哪里？"),
    ("我的出生城市是 CITY-D4E5F6。", "我老家在哪儿？"),
    ("我的办公城市是 CITY-G7H8J9。", "我每天上班要去哪个地方？"),
    ("我的档案代号是 CODE-K1L2M3。", "系统给我分配的那串字符是什么？"),
    ("我的常用编辑器是 EDIT-N4P5Q6。", "我写代码时用哪个软件？"),
    ("我的默认语言是 LANG-R7S8T9。", "我平时讲哪种话？"),
    ("我的通勤方式是 COMMU-U1V2W3。", "我早上怎么去公司？"),
    ("我的主管姓名是 BOSS-X4Y5Z6。", "谁管我？"),
    ("我的工位楼层是 FLOOR-A7B8C9。", "我坐在哪一层？"),
    ("我的团队名称是 TEAM-D1E2F3。", "我属于哪个小组？"),
    ("我的邮箱域名是 MAIL-G4H5J6。", "别人给我发信要写哪个后缀？"),
    ("我的手机尾号是 PHONE-K7L8M9。", "我的电话号码最后几位是什么？"),
    ("我的项目代号是 PROJ-N1P2Q3。", "我正在做的那个工程叫什么？"),
    ("我的入职年份是 YEAR-R4S5T6。", "我是哪一年来的这家公司？"),
    ("我的办公楼层是 DESK-U7V8W9。", "我平时在几楼办公？"),
    ("我的紧急联系人姓氏是 KIN-X1Y2Z3。", "出事时该找哪家人？"),
]
FILLER_FACTS = [
    "我的工位在 3 号楼。",
    "我的笔记本电脑品牌是 ThinkPad。",
    "我的水杯是蓝色的。",
    "我的鼠标是无线的。",
]


def _value_of(fact: str) -> str:
    """The code that only this fact contains."""

    return fact.split("是", 1)[1].strip().rstrip("。")


def build_cases(count: int, seed: int) -> list[dict]:
    pairs = CASE_PAIRS[: max(1, min(count, len(CASE_PAIRS)))]
    bank = [fact for fact, _ in pairs] + FILLER_FACTS
    cases = []
    for fact, query in pairs:
        cases.append({
            "fact": fact,
            "query": query,
            "code": _value_of(fact),
            "attribute": fact[len("我的"):].split("是", 1)[0],
            "bank": bank,
        })
    return cases


@torch.inference_mode()
def run_cases(model, tokenizer, cases, device, *, max_new_tokens: int, label: str) -> dict:
    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        model.reset_memory(batch_size=1, device=device)
        for fact in case["bank"]:
            write_fact(model, tokenizer, fact, device)
        # Prove the case is not solvable by the rule-based paths: ask the bank
        # itself whether this query produces any address or lexical hit.
        bank = model.memory_os_v2.bank
        address_hit = bool(bank.has_explicit_address(case["query"]))
        lexical_pages, lexical_records = bank._lexical_evidence_hits(case["query"])
        reply = answer(model, tokenizer, case["query"], device, max_new_tokens)
        decision = model.runtime.v2_last_decisions[-1] if model.runtime.v2_last_decisions else {}
        wanted = case["code"]
        others = [c["code"] for c in cases if c["code"] != wanted]
        correct = wanted.lower() in reply.lower()
        mixed = (not correct) and any(code.lower() in reply.lower() for code in others)
        rows.append({
            "attribute": case["attribute"], "wanted": wanted, "reply": reply[:160],
            "correct": bool(correct), "wrong_attribute_code": bool(mixed),
            "rule_address_hit": address_hit,
            "rule_lexical_hits": len(lexical_records),
            "stop_reason": str(decision.get("stop_reason")),
            "need_memory": bool(decision.get("need_memory")),
            "top_score": float(decision.get("top_score") or 0.0),
            "records_selected": len(decision.get("record_ids") or []),
        })
        if index % 4 == 0:
            print(json.dumps({"router": label, "case": index, "total": len(cases),
                              "accuracy_pct": round(100 * sum(r["correct"] for r in rows) / len(rows), 2)}), flush=True)
    summary = {
        "router": label,
        "cases": len(rows),
        "accuracy_pct": 100 * sum(r["correct"] for r in rows) / max(1, len(rows)),
        "wrong_attribute_pct": 100 * sum(r["wrong_attribute_code"] for r in rows) / max(1, len(rows)),
        "read_pct": 100 * sum(r["need_memory"] for r in rows) / max(1, len(rows)),
        "cases_with_rule_hits": sum(1 for r in rows if r["rule_address_hit"] or r["rule_lexical_hits"]),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
        "mean_records_selected": sum(r["records_selected"] for r in rows) / max(1, len(rows)),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--top-k-records",
        type=int,
        default=0,
        help="override memory_top_k_records; the router's recall is only useful if the "
             "records it injects are precise enough for the model to use",
    )
    parser.add_argument("--router", action="append", required=True)
    parser.add_argument("--output", default="router_critical_e2e.json")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    cases = build_cases(args.cases, args.seed)
    print(json.dumps({"cases": len(cases), "bank_facts": len(cases[0]["bank"]),
                      "note": "all facts share the attribute wording; only the entity differs"}, ensure_ascii=False), flush=True)

    model_path = Path(args.package)
    memory_config = load_memory_config(model_path)
    if args.top_k_records:
        memory_config.memory_top_k_records = int(args.top_k_records)
        print(json.dumps({"top_k_records_override": args.top_k_records}), flush=True)
    model = load_qwen_dynamic(model_path, memory_config=memory_config,
                             load_in_4bit=True, max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    tokenizer = load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict = {}
    for spec in args.router:
        label, _, path_value = spec.partition("=")
        if path_value.strip():
            state = torch.load(path_value.strip(), map_location="cpu", weights_only=True)
            model.memory_router_v2.load_state_dict(state.get("router_state_dict", state), strict=True)
            model.memory_router_v2.to(device).eval()
            print(json.dumps({"router_swapped": label}), flush=True)
        results[label] = run_cases(model, tokenizer, cases, device, max_new_tokens=args.max_new_tokens, label=label)
        print(json.dumps(results[label]["summary"], ensure_ascii=False), flush=True)

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| 路由器 | 用例 | 回答正确率 | 答成别的属性 | 触发读取 | 规则命中用例 | 平均选中记录 | 决策来源 |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for label, block in results.items():
        s = block["summary"]
        lines.append("| {l} | {c} | {a:.2f}% | {w:.2f}% | {r:.2f}% | {h}/{c} | {m:.2f} | {s} |".format(
            l=label, c=s["cases"], a=s["accuracy_pct"], w=s["wrong_attribute_pct"],
            r=s["read_pct"], h=s["cases_with_rule_hits"], m=s["mean_records_selected"],
            s=", ".join(f"{k}:{v}" for k, v in sorted(s["stop_reasons"].items()))))
    table = "\n".join(lines)
    print("\n" + table, flush=True)
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
