"""Check how LoCoMo's adversarial questions carry evidence, before trusting the mapping."""
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else r"H:\Memory\V2_dpskw\data\net_locomo\locomo10.json")
data = json.loads(path.read_text(encoding="utf-8"))

with_evidence = Counter()
without_evidence = Counter()
keys_seen = Counter()
examples = []
for item in data:
    for qa in item["qa"]:
        cat = qa.get("category")
        keys_seen.update(qa.keys())
        ev = qa.get("evidence") or []
        if ev:
            with_evidence[cat] += 1
        else:
            without_evidence[cat] += 1
        if cat == 5 and len(examples) < 4:
            examples.append(qa)

print("qa field names:", dict(keys_seen))
print()
print(f"{'category':>9}{'with evidence':>15}{'without':>10}")
for cat in sorted(set(with_evidence) | set(without_evidence)):
    print(f"{cat:>9}{with_evidence[cat]:>15}{without_evidence[cat]:>10}")
print()
print("=== adversarial examples ===")
for qa in examples:
    print(json.dumps(qa, ensure_ascii=False, indent=1)[:600])
