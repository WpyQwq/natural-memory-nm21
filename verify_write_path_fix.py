"""Verify the write-path fix: unrelated facts survive, real updates still version.

The fix under test gates "this write is an update of an existing record" on a
*structural* same-fact check (the ``entity::attribute`` conflict key the bank already
tracks) instead of trusting the learned pair score alone.  Removing a retraction path is
only safe if it still retires what it is supposed to, so this probe runs three phases on
one model load:

A. **regression target** -- 20 distinct-attribute facts must leave 20 active records
   (before the fix: 12 active, 8 retracted);
B. **positive control** -- re-writing the *same* attribute with a new value must still
   version the fact: the previous record becomes ``superseded`` (not destroyed), the new
   value is active, and the other attributes are untouched.  Without this, "no
   retractions" could simply mean "updates are broken";
C. **retriever-absent control** -- with ``_text_retriever_ready = False`` the same 20
   distinct facts must still all stay active.  This control matters because the lexical
   fallback branch (>= 0.30) is more permissive than the learned one and is only
   unreachable while the retriever is ready.

Usage::

    python -m V2_dpskw.verify_write_path_fix --package qwen3_5_4b_natural_memory_v2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .eval_end_to_end_memory import write_fact
from .memory_os_v2 import STATUS_ACTIVE, STATUS_SUPERSEDED
from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer

ATTRIBUTES = [
    "常住城市", "出生城市", "办公城市", "档案标识", "常用编辑器", "默认语言", "通勤方式",
    "主管姓名", "工位楼层", "团队名称", "邮箱域名", "手机尾号", "项目代号", "入职年份",
    "紧急联系人姓氏", "午餐偏好", "运动习惯", "阅读工具", "起床时间", "咖啡口味",
]


def inventory(model) -> dict:
    records = model.memory_os_v2.bank.records
    statuses = Counter(record.status for record in records.values())
    by_attribute: dict[str, str] = {}
    for record in records.values():
        if record.attribute:
            by_attribute.setdefault(record.attribute, []).append(record.status)
    return {
        "records": len(records),
        "status_counts": dict(statuses),
        "active": statuses.get(STATUS_ACTIVE, 0),
        "superseded": statuses.get(STATUS_SUPERSEDED, 0),
        "retracted": statuses.get("retracted", 0),
        "attribute_status": {k: sorted(v) for k, v in sorted(by_attribute.items())},
    }


def values_for(model, attribute: str) -> list[tuple[str, str]]:
    return [
        (record.value, record.status)
        for record in model.memory_os_v2.bank.records.values()
        if record.attribute == attribute
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--output", default="write_path_fix_verification.json")
    args = parser.parse_args()

    model_path = Path(args.package)
    memory_config = load_memory_config(model_path)
    model = load_qwen_dynamic(model_path, memory_config=memory_config, load_in_4bit=True,
                              max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    tokenizer = load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    facts = ["我的%s是 VAL-%07d。" % (attribute, index)
             for index, attribute in enumerate(ATTRIBUTES)]
    report: dict = {"package": str(model_path), "facts": len(facts), "phases": {}}
    checks: dict[str, bool] = {}

    # --- Phase A: distinct facts must not retire each other -------------------------
    model.reset_memory()
    for fact in facts:
        write_fact(model, tokenizer, fact, device)
    phase_a = inventory(model)
    report["phases"]["A_distinct_facts"] = phase_a
    checks["A_all_distinct_facts_active"] = phase_a["active"] == len(facts)
    checks["A_no_retractions"] = phase_a["retracted"] == 0
    print(json.dumps({"phase": "A", **phase_a}, ensure_ascii=False), flush=True)

    # --- Phase B: a genuine same-attribute update must still version -----------------
    model.reset_memory()
    for fact in facts:
        write_fact(model, tokenizer, fact, device)
    before = inventory(model)
    write_fact(model, tokenizer, "我的常住城市是 VAL-9999999。", device)
    after = inventory(model)
    city_values = values_for(model, "常住城市")
    report["phases"]["B_positive_control"] = {
        "before": before, "after": after, "常住城市_values": city_values,
    }
    active_city = [value for value, status in city_values if status == STATUS_ACTIVE]
    checks["B_new_value_is_active"] = active_city == ["VAL-9999999"]
    checks["B_old_version_superseded_not_destroyed"] = any(
        status == STATUS_SUPERSEDED for _, status in city_values
    )
    checks["B_other_attributes_untouched"] = after["active"] >= len(facts)
    checks["B_no_retractions"] = after["retracted"] == 0
    print(json.dumps({"phase": "B", "常住城市_values": city_values,
                      "active_before": before["active"], "active_after": after["active"],
                      "status_counts_after": after["status_counts"]}, ensure_ascii=False), flush=True)

    # --- Phase C: same behaviour with the learned retriever absent -------------------
    model.reset_memory()
    model._text_retriever_ready = False
    for fact in facts:
        write_fact(model, tokenizer, fact, device)
    phase_c = inventory(model)
    model._text_retriever_ready = True
    report["phases"]["C_retriever_absent"] = phase_c
    checks["C_all_distinct_facts_active_without_retriever"] = phase_c["active"] == len(facts)
    checks["C_no_retractions_without_retriever"] = phase_c["retracted"] == 0
    print(json.dumps({"phase": "C", **phase_c}, ensure_ascii=False), flush=True)

    report["checks"] = checks
    report["all_checks_passed"] = all(checks.values())
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "all_checks_passed": report["all_checks_passed"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
