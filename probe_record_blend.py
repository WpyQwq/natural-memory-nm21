"""Dose-response for the record-ordering blend: router score vs packaged retriever.

``_score_semantic_memory_records`` normally orders every record that carries a
``semantic_key`` with the packaged ``text_retriever`` alone.  Measured on identical frozen
keys over the zero-overlap eval (250 answerable episodes, 24 same-shape candidates,
chance 4.17%) the two scorers differ sharply at Top-1:

    packaged text_retriever   23.20%
    frozen-key cosine         30.40%
    trained router (REPLAY)   59.60%

This probe runs the 16 router-critical end-to-end cases at several blend weights
(``memory_config.memory_record_router_blend``: 0.0 = historical behaviour, 1.0 = router
only) and reports answer accuracy and wrong-attribute rate for each, so the weight is
chosen from a measured curve instead of a guess.  The default stays 0.0.

Usage::

    python -m V2_dpskw.probe_record_blend --package qwen3_5_4b_natural_memory_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .eval_router_critical_e2e import build_cases, run_cases
from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--blends", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--router", default="checkpoints/router_replay_v7_v2_128/memory_router_v2.pt")
    parser.add_argument("--output", default="record_blend_dose_response.json")
    parser.add_argument("--markdown", default="record_blend_dose_response.md")
    args = parser.parse_args()

    blends = [float(value) for value in args.blends.split(",") if value.strip()]
    cases = build_cases(args.cases, 20260911)

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
        print(json.dumps({"router": str(router_path)}), flush=True)

    results = {}
    for blend in blends:
        model.memory_config.memory_record_router_blend = blend
        block = run_cases(model, tokenizer, cases, device, max_new_tokens=48,
                          label="blend=%.2f" % blend)
        summary = block["summary"]
        results["%.2f" % blend] = summary
        print(json.dumps({"blend": blend, **summary}, ensure_ascii=False), flush=True)

    report = {
        "package": str(model_path),
        "router": str(router_path),
        "cases": len(cases),
        "note": ("blend 0.00 is the historical behaviour (packaged retriever only); "
                 "1.00 is the trained router only"),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 记录排序混合权重 · 端到端剂量曲线", "",
             f"用例 {len(cases)} 条（零重叠改写，24 条同形候选）。"
             "`blend=0.00` 为历史行为（仅打包 retriever），`1.00` 为仅训练路由器。", "",
             "| blend | 回答正确率 | 答成别的属性 | 触发读取 | 平均选中记录 | 决策来源 |",
             "|---|---:|---:|---:|---:|---|"]
    for label, summary in results.items():
        lines.append(
            "| {0} | {1:.2f}% | {2:.2f}% | {3:.2f}% | {4:.4f} | {5} |".format(
                label, summary["accuracy_pct"], summary["wrong_attribute_pct"],
                summary["read_pct"], summary["mean_records_selected"],
                ", ".join("%s:%d" % (k, v) for k, v in sorted(summary["stop_reasons"].items())),
            )
        )
    text = "\n".join(lines) + "\n"
    Path(args.markdown).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
