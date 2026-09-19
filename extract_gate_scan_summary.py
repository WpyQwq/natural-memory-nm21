"""Extract the gate-scan summary block from a probe log (quoting-safe helper)."""

from __future__ import annotations

import json
import re
import sys

KEYS = [
    "cases_scanned",
    "target_record_retracted_or_absent",
    "target_record_active",
    "records_per_case",
    "active_records_min",
    "active_records_max",
    "cases_with_direct_records",
    "cases_with_address_hits",
    "cases_with_lexical_hits",
    "v2_gate_open_pct_deployed_router",
    "v2_gate_open_pct_random_router_copies",
    "stop_reasons",
    "legacy_prefix_used_pct",
]


def main() -> int:
    path = sys.argv[1]
    text = open(path, encoding="utf-8", errors="replace").read()
    decoder = json.JSONDecoder()
    found = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and "target_record_active" in obj:
            found.append(obj)
    if not found:
        print("no summary block with target_record_active found; tail follows")
        print(text[-1200:])
        return 1
    summary = found[-1]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    correct = 0
    total = 0
    for match in re.finditer(r'"correct": (true|false)', text):
        total += 1
        correct += 1 if match.group(1) == "true" else 0
    if total:
        print("per-case correct: %d/%d = %.2f%%" % (correct, total, 100.0 * correct / total))
    print()
    for key in KEYS:
        if key in summary:
            print("  %-44s = %s" % (key, json.dumps(summary[key], ensure_ascii=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
