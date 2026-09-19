"""Reply-level diff between two realistic_v2 eval runs, per category."""
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    key = next(iter(data))
    body = data[key]
    rows = body if isinstance(body, list) else body.get("rows", [])
    return rows


def idx_keys(rows):
    idx = defaultdict(int)
    out = []
    for r in rows:
        c = r["category"]
        out.append((c, r["query"], idx[c]))
        idx[c] += 1
    return out


def main(a, b):
    ra, rb = load(a), load(b)
    ka, kb = idx_keys(ra), idx_keys(rb)
    ma = {k: r for k, r in zip(ka, ra)}
    mb = {k: r for k, r in zip(kb, rb)}
    same_reply = defaultdict(int)
    diff_reply = defaultdict(int)
    diff_examples = defaultdict(list)
    for k in ma:
        if k not in mb:
            continue
        if ma[k].get("reply") == mb[k].get("reply"):
            same_reply[k[0]] += 1
        else:
            diff_reply[k[0]] += 1
            if len(diff_examples[k[0]]) < 3:
                diff_examples[k[0]].append((k[1], ma[k].get("reply"), mb[k].get("reply")))
    cats = sorted(set(same_reply) | set(diff_reply))
    print(f"{'category':<20}{'same':>6}{'changed':>9}{'total':>7}")
    for c in cats:
        print(f"{c:<20}{same_reply[c]:>6}{diff_reply[c]:>9}{same_reply[c]+diff_reply[c]:>7}")
    print()
    for c in cats:
        if diff_reply[c]:
            print(f"=== {c}: {diff_reply[c]} replies changed ===")
            for q, x, y in diff_examples[c]:
                print(f"  Q: {q}")
                print(f"   old: {x!r}")
                print(f"   new: {y!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
