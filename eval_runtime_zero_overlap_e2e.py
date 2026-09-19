"""Runtime-level end-to-end evaluation over the zero-overlap paraphrase eval set.

The 16-case router-critical suite is too small to separate real effects from one flipped
case (each case is 6.25% of the score).  This evaluator drives the *full* zero-overlap eval
split -- every episode writes all of its 24 same-shape candidate facts into the runtime,
then asks the paraphrased question -- so each case is ~1.3% at 80 cases and the recorded
rate is far less sensitive to a single example.

Scoring is deliberately conservative and code-based rather than wording-based:

* answerable episode: correct when the expected code appears in the reply;
* ``wrong_attribute``: a *different* candidate's code appears and the expected one does not;
* unknown episode: ``leaked`` when any candidate code appears (the runtime should decline).

Usage::

    python -m V2_dpskw.eval_runtime_zero_overlap_e2e --cases 64 --blends 0.0,0.5
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from .eval_end_to_end_memory import write_fact
from .eval_router_critical_e2e import answer
from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer


def load_episodes(path: Path, count: int, seed: int, only: str = "all") -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(seed)
    answerable = [row for row in rows if row.get("positive_indices")]
    unknown = [row for row in rows if not row.get("positive_indices")]
    rng.shuffle(answerable)
    rng.shuffle(unknown)
    if only == "unknown":
        return unknown[:count]
    if only == "answerable":
        return answerable[:count]
    # Keep the dataset's own proportion of abstention episodes.
    share = len(unknown) / max(1, len(rows))
    wanted_unknown = int(round(count * share))
    picked = answerable[: max(0, count - wanted_unknown)] + unknown[:wanted_unknown]
    rng.shuffle(picked)
    return picked[:count]


@torch.inference_mode()
def run_episodes(model, tokenizer, episodes, device, *, max_new_tokens: int, label: str) -> dict:
    rows = []
    started = time.perf_counter()
    for index, episode in enumerate(episodes, 1):
        # Prefer the dataset's own answer strings: they are the official scoring target
        # and they generalise beyond the "我的X是 VAL-…。" template, so the same evaluator
        # can run on the v6 mega episodes as well.
        acceptable = [str(item) for item in (episode["metadata"].get("acceptable") or [])]
        if not acceptable:
            acceptable = [str(episode["metadata"].get("answer", ""))]
        acceptable = [item for item in acceptable if item]
        # Every candidate's answer surface, used to detect "answered a different attribute".
        other_surfaces: list[str] = []
        for candidate in episode["candidates"]:
            for item in (candidate.get("acceptable") or []):
                other_surfaces.append(str(item))
            if not candidate.get("acceptable") and candidate.get("text"):
                other_surfaces.append(str(candidate["text"].split()[-1].rstrip("。")))
        expected = acceptable[0] if acceptable else ""
        model.reset_memory(batch_size=1, device=device)
        for candidate in episode["candidates"]:
            write_fact(model, tokenizer, candidate["text"], device)
        active = sum(1 for record in model.memory_os_v2.bank.records.values()
                     if record.status == "active")
        reply = answer(model, tokenizer, episode["query"], device, max_new_tokens)
        decision = model.runtime.v2_last_decisions[-1] if model.runtime.v2_last_decisions else {}
        lowered = reply.lower()
        hit_expected = any(item.lower() in lowered for item in acceptable)
        other_hits = [item for item in set(other_surfaces)
                      if item.lower() not in {a.lower() for a in acceptable}
                      and item.lower() in lowered]
        answerable = bool(episode.get("positive_indices"))
        rows.append({
            "id": episode["id"],
            "attribute": episode["metadata"].get("attribute", ""),
            "answerable": answerable,
            "expected": expected,
            "reply": reply[:140],
            "correct": bool(hit_expected) if answerable else False,
            "wrong_attribute": bool(answerable and not hit_expected and other_hits),
            "leaked": bool((not answerable) and other_hits),
            "records_written": len(episode["candidates"]),
            "records_active": active,
            "stop_reason": str(decision.get("stop_reason")),
            "records_selected": len(decision.get("record_ids") or []),
            "top_score": float(decision.get("top_score") or 0.0),
            "need_memory": bool(decision.get("need_memory")),
        })
        if index % 8 == 0:
            print(json.dumps({"label": label, "case": index, "total": len(episodes),
                              "accuracy_pct": round(100 * sum(r["correct"] for r in rows)
                                                    / max(1, len(rows)), 2)}), flush=True)

    answerable_rows = [row for row in rows if row["answerable"]]
    unknown_rows = [row for row in rows if not row["answerable"]]

    def score_band(block: list[dict]) -> dict:
        """Report the top-score distribution, so a separating threshold can be judged."""
        values = sorted(row["top_score"] for row in block)
        if not values:
            return {"n": 0}

        def pct(fraction: float) -> float:
            return values[min(len(values) - 1, int(fraction * len(values)))]

        return {
            "n": len(values),
            # Raw decision scores, NOT probabilities: the runtime's top_score is an
            # unbounded relevance value (observed up to ~851), so it must never be
            # rendered as a percentage.
            "min": round(values[0], 2),
            "p10": round(pct(0.10), 2),
            "p50": round(pct(0.50), 2),
            "p90": round(pct(0.90), 2),
            "max": round(values[-1], 2),
        }

    summary = {
        "router": label,
        "cases": len(rows),
        "answerable_cases": len(answerable_rows),
        "unknown_cases": len(unknown_rows),
        "accuracy_pct": 100 * sum(row["correct"] for row in answerable_rows) / max(1, len(answerable_rows)),
        "wrong_attribute_pct": 100 * sum(row["wrong_attribute"] for row in answerable_rows)
        / max(1, len(answerable_rows)),
        "unknown_leak_pct": 100 * sum(row["leaked"] for row in unknown_rows) / max(1, len(unknown_rows)),
        "mean_records_selected": sum(row["records_selected"] for row in rows) / max(1, len(rows)),
        "records_active_min": min((row["records_active"] for row in rows), default=0),
        "records_active_max": max((row["records_active"] for row in rows), default=0),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
        "top_score_answerable": score_band(answerable_rows),
        "top_score_unknown": score_band(unknown_rows),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", default="data/zero_overlap/eval.jsonl")
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--only", choices=("all", "answerable", "unknown"), default="all",
                        help="restrict the sample to one class of episode")
    parser.add_argument("--blends", default="0.0,0.5")
    parser.add_argument("--prior-scales", default="1.0",
                        help="comma list of PagedMemoryBankV2._record_scores prior scales; "
                             "1.0 is historical, 0.0 leaves ordering to the learned scorer")
    parser.add_argument("--router", default="checkpoints/router_replay_v7_v2_128/memory_router_v2.pt")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", default="runtime_zero_overlap_e2e.json")
    parser.add_argument("--markdown", default="runtime_zero_overlap_e2e.md")
    args = parser.parse_args()

    blends = [float(value) for value in args.blends.split(",") if value.strip()]
    episodes = load_episodes(Path(args.eval_file), args.cases, args.seed, args.only)
    print(json.dumps({"episodes": len(episodes),
                      "answerable": sum(1 for e in episodes if e.get("positive_indices")),
                      "candidate_count": len(episodes[0]["candidates"])}), flush=True)

    model_path = Path(args.package)
    memory_config = load_memory_config(model_path)
    model = load_qwen_dynamic(model_path, memory_config=memory_config, load_in_4bit=True,
                              max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    tokenizer = load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    router_path = Path(args.router)
    if router_path.exists():
        state = torch.load(router_path, map_location="cpu", weights_only=True)
        model.memory_router_v2.load_state_dict(state.get("router_state_dict", state), strict=True)
        model.memory_router_v2.to(device).eval()
        print(json.dumps({"router_swapped": str(router_path)}), flush=True)

    results = {}
    for prior_scale in [float(v) for v in args.prior_scales.split(",") if v.strip()]:
        model.memory_config.memory_prior_scale = prior_scale
        for blend in blends:
            model.memory_config.memory_record_router_blend = blend
            label = "prior=%.2f blend=%.2f" % (prior_scale, blend)
            block = run_episodes(model, tokenizer, episodes, device,
                                 max_new_tokens=args.max_new_tokens, label=label)
            results[label] = block
            print(json.dumps(block["summary"], ensure_ascii=False), flush=True)

    report = {
        "eval_file": args.eval_file,
        "episodes": len(episodes),
        "candidate_count": len(episodes[0]["candidates"]),
        "router": str(router_path),
        "results": {label: block["summary"] for label, block in results.items()},
        "rows": {label: block["rows"] for label, block in results.items()},
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 运行时端到端评测（零重叠改写，大样本）", "",
             "每条 episode 把全部 {0} 个同形候选事实写入运行时，再问改写后的问题。".format(
                 len(episodes[0]["candidates"])),
             "随机基线 1/{0} = {1:.2f}%。全部为百分比。".format(
                 len(episodes[0]["candidates"]), 100.0 / len(episodes[0]["candidates"])), "",
             "| blend | 用例 | 可回答 | 回答正确率 | 答成别的属性 | 未知泄漏率 | 平均选中记录 | 活跃记录 min/max |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for label, block in results.items():
        s = block["summary"]
        lines.append("| {0} | {1} | {2} | {3:.2f}% | {4:.2f}% | {5:.2f}% | {6:.4f} | {7}/{8} |".format(
            label, s["cases"], s["answerable_cases"], s["accuracy_pct"], s["wrong_attribute_pct"],
            s["unknown_leak_pct"], s["mean_records_selected"],
            s["records_active_min"], s["records_active_max"]))
    lines += ["", "## 检索最高分分布（原始分，非百分比；用于判断阈值能否分开『未知』与『可回答』）", "",
              "| 组 | n | min | p10 | p50 | p90 | max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, block in results.items():
        s = block["summary"]
        for key, nice in (("top_score_answerable", "可回答"), ("top_score_unknown", "不可回答")):
            band = s.get(key) or {}
            if band.get("n"):
                lines.append("| {0} · {1} | {2} | {3} | {4} | {5} | {6} | {7} |".format(
                    label, nice, band["n"], band["min"], band["p10"], band["p50"],
                    band["p90"], band["max"]))
    text = "\n".join(lines) + "\n"
    Path(args.markdown).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
