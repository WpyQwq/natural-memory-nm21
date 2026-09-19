"""Diagnose why the coverage gate stood down (self-gating bypass) on a given bank.

Writes the same facts the runtime evaluator writes, then prints the bank's active conflict
keys, the head vocabulary, and the gate's own applicable/bypassed counters.  This exists
because the gate silently degrading to "no opinion" looks exactly like "the gate does not
work" in end-to-end numbers.

Usage::

    python -m V2_dpskw.diagnose_coverage_gate --package qwen3_5_4b_natural_memory_v2_1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .eval_end_to_end_memory import write_fact
from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer

FACTS = [
    ("常住城市", "CITY-A1B2C3"), ("出生城市", "CITY-D4E5F6"), ("办公城市", "CITY-G7H8J9"),
    ("档案标识", "CODE-K1L2M3"), ("常用编辑器", "EDIT-N4P5Q6"), ("默认语言", "LANG-R7S8T9"),
    ("通勤方式", "COMMU-U1V2W3"), ("主管姓名", "BOSS-X4Y5Z6"), ("工位楼层", "FLOOR-A7B8C9"),
    ("团队名称", "TEAM-D1E2F3"),
]
QUERY = "我平时待得最久的地方是哪里？"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2_1")
    parser.add_argument("--output", default="coverage_gate_diagnosis.json")
    args = parser.parse_args()

    model_path = Path(args.package)
    config = load_memory_config(model_path)
    model = load_qwen_dynamic(model_path, memory_config=config, load_in_4bit=True,
                              max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    tokenizer = load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.reset_memory(batch_size=1, device=device)
    for attribute, value in FACTS:
        write_fact(model, tokenizer, "我的%s是 %s。" % (attribute, value), device)

    active = dict(model.memory_os_v2.bank.active_by_conflict)
    present = sorted({key.split("::", 1)[1] for key in active if "::" in key})
    vocabulary = sorted((model._attribute_head_meta or {}).get("vocabulary", []))
    stats = (model._attribute_head_meta or {}).get("coverage_stats", {})

    from .eval_router_critical_e2e import answer

    reply = answer(model, tokenizer, QUERY, device, 24)
    decision = model.runtime.v2_last_decisions[-1] if model.runtime.v2_last_decisions else {}

    report = {
        "package": str(model_path),
        "gate_enabled": bool(config.memory_coverage_gate),
        "blend": float(config.memory_record_router_blend),
        "facts_written": len(FACTS),
        "bank_records": len(model.memory_os_v2.bank.records),
        "active_records": sum(1 for r in model.memory_os_v2.bank.records.values()
                              if r.status == "active"),
        "active_conflict_keys": len(active),
        "bank_attributes": present,
        "head_vocabulary_size": len(vocabulary),
        "attributes_outside_head_vocabulary": sorted(set(present) - set(vocabulary)),
        "attributes_missing_from_bank": sorted(set(vocabulary) - set(present)),
        "gate_counters": stats,
        "query": QUERY,
        "stop_reason": str(decision.get("stop_reason")),
        "reply": reply[:120],
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
