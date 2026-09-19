"""Per-category + per-case comparison of realistic_v2 end-to-end eval JSONs.

Handles two on-disk shapes:
  {"<router>": {"summary": {...}, "rows": [...]}}   (current)
  {"<router>": [ ...rows... ]}                       (older)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"H:\Memory\V2_dpskw")


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    key = next(iter(data))
    body = data[key]
    if isinstance(body, list):
        rows = body
        summary = None
    else:
        rows = body.get("rows", [])
        summary = body.get("summary")
    return rows, summary


def per_cat(rows):
    agg = defaultdict(lambda: {"n": 0, "ok": 0, "abst": 0, "wrong_abst": 0})
    for r in rows:
        c = r["category"]
        a = agg[c]
        a["n"] += 1
        if r.get("correct"):
            a["ok"] += 1
        if r.get("abstained"):
            a["abst"] += 1
        if r.get("wrongly_abstained"):
            a["wrong_abst"] += 1
    return agg


def pct(x, n):
    return 100.0 * x / n if n else 0.0


def key_of(r, idx):
    """Stable identity for a case row: category + query + index within category."""
    return (r["category"], r["query"], idx)


def main(paths):
    loaded = []
    for p in paths:
        rows, summary = load(p)
        loaded.append((p, rows, summary))

    cats = sorted({r["category"] for _, rows, _ in loaded for r in rows})
    agg = [per_cat(rows) for _, rows, _ in loaded]

    head = f"{'category':<20}" + "".join(f"{Path(p).stem[:20]:>22}" for p, _, _ in loaded)
    print(head)
    for c in cats:
        line = f"{c:<20}"
        for a in agg:
            n = a[c]["n"]
            line += f"{a[c]['ok']:>10}/{n:<3}{pct(a[c]['ok'], n):>7.2f}%"
        print(line)
    line = f"{'TOTAL':<20}"
    for a in agg:
        n = sum(v["n"] for v in a.values())
        ok = sum(v["ok"] for v in a.values())
        line += f"{ok:>10}/{n:<3}{pct(ok, n):>7.2f}%"
    print(line)

    # per-case flip analysis between baseline (first) and candidate (last)
    base_rows = loaded[0][1]
    cand_rows = loaded[-1][1]
    if len(base_rows) == len(cand_rows):
        idx = defaultdict(int)
        bk = []
        for r in base_rows:
            c = r["category"]
            bk.append(key_of(r, idx[c]))
            idx[c] += 1
        idx2 = defaultdict(int)
        ck = []
        for r in cand_rows:
            c = r["category"]
            ck.append(key_of(r, idx2[c]))
            idx2[c] += 1
        bmap = {k: r for k, r in zip(bk, base_rows)}
        cmap = {k: r for k, r in zip(ck, cand_rows)}
        only_b = [k for k in bmap if k not in cmap]
        only_c = [k for k in cmap if k not in bmap]
        print(f"\nkeys only in baseline: {len(only_b)}  only in candidate: {len(only_c)}")
        fixes, breaks = [], []
        for k in bmap:
            if k not in cmap:
                continue
            b, c = bmap[k], cmap[k]
            if not b.get("correct") and c.get("correct"):
                fixes.append((k, b, c))
            elif b.get("correct") and not c.get("correct"):
                breaks.append((k, b, c))
        fc = defaultdict(int)
        for k, _, _ in fixes:
            fc[k[0]] += 1
        bc = defaultdict(int)
        for k, _, _ in breaks:
            bc[k[0]] += 1
        print(f"FIXED  (wrong->right): {len(fixes)}  {dict(fc)}")
        print(f"BROKEN (right->wrong): {len(breaks)}  {dict(bc)}")
        for k, b, c in fixes:
            print(f"  [+] {k[0]:<18} {k[1][:34]:<34} {b.get('reply','')[:52]!r} -> {c.get('reply','')[:52]!r}")
        for k, b, c in breaks:
            print(f"  [-] {k[0]:<18} {k[1][:34]:<34} {b.get('reply','')[:52]!r} -> {c.get('reply','')[:52]!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
