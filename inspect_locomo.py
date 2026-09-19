"""Inspect the downloaded LoCoMo dataset: structure, categories, evidence format."""
import collections
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else r"H:\Memory\V2_dpskw\data\net_locomo\locomo10.json")
data = json.loads(path.read_text(encoding="utf-8"))
print("conversations:", len(data))
first = data[0]
print("top keys:", list(first.keys()))
conv = first["conversation"]
print("conversation keys:", list(conv.keys()))
for key in list(conv.keys())[:5]:
    value = conv[key]
    size = len(value) if hasattr(value, "__len__") else "-"
    print(f"  {key}: {type(value).__name__} len={size}")
    if isinstance(value, list) and value:
        print("     first:", json.dumps(value[0], ensure_ascii=False)[:300])

categories = collections.Counter()
n_qa = 0
turns = 0
sessions = 0
for item in data:
    c = item["conversation"]
    keys = [k for k in c if k.startswith("session_") and not k.endswith("date_time")]
    sessions += len(keys)
    turns += sum(len(c[k]) for k in keys)
    for q in item["qa"]:
        n_qa += 1
        categories[q.get("category")] += 1

print()
print(f"total QA: {n_qa}   sessions: {sessions}   turns: {turns}")
print("category distribution:", dict(sorted(categories.items())))

print()
print("=== one example per category ===")
seen = set()
for item in data:
    for q in item["qa"]:
        cat = q.get("category")
        if cat in seen:
            continue
        seen.add(cat)
        ev = q.get("evidence")
        print(f"[cat {cat}] Q={q['question']!r}")
        print(f"          answer={q['answer']!r}  evidence={ev}  adversarial={q.get('adversarial_answer')!r}")

print()
print("=== a conversation turn, verbatim ===")
sample = data[0]["conversation"]["session_1"][:3]
print(json.dumps(sample, ensure_ascii=False, indent=1)[:900])
