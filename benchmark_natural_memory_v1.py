"""Full local comparison between Qwen3.5-4B and Natural Memory v1.

This is an engineering benchmark, not a claim of state-of-the-art performance.
It evaluates the same frozen Qwen3.5-4B backbone with and without the internal
memory path, using deterministic greedy decoding and locally generated cases.
The report includes general ability, extra math and reasoning cases, long
context retrieval, throughput, latency, VRAM, automatic write decisions,
conflict replacement, unknown-fact refusal, reset, and shard-backed restart.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .qwen_integration import (
    load_memory_config,
    load_qwen_base,
    load_qwen_dynamic,
    load_tokenizer,
)
from .stream_chat_qwen_memory import _chat_tensor, _memory_system_prefix, _write_turn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=r"H:\Memory")
    parser.add_argument(
        "--memory-model",
        default=r"H:\Memory\V2_dpskw\qwen3_5_4b_memory_merged_v13",
    )
    parser.add_argument(
        "--data",
        default=r"H:\Memory\V2_dpskw\data\comprehensive_general.jsonl",
    )
    parser.add_argument(
        "--output",
        default=r"H:\Memory\V2_dpskw\natural_memory_v1_full_benchmark.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--perf-repeats", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    keep = str(text).lower()
    for char in " \t\r\n`*_#，。！？、；：,.!?;:'\"（）()[]{}<>|\\/：":
        keep = keep.replace(char, "")
    return keep


def contains_answer(text: str, acceptable: list[str]) -> bool:
    normalized = normalize(text)
    for answer in acceptable:
        expected = normalize(str(answer))
        if not expected:
            continue
        if expected.isdigit() and len(expected) == 1:
            if any(
                normalized[index : index + 1] == expected
                and (index == 0 or not normalized[index - 1].isdigit())
                and (index + 1 == len(normalized) or not normalized[index + 1].isdigit())
                for index in range(len(normalized))
            ):
                return True
        elif expected in normalized:
            return True
    return False


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def prompt_inputs(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def input_token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False)
    return len(encoded["input_ids"])


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    dynamic: bool,
    max_new_tokens: int,
) -> tuple[str, int, float]:
    device = model._find_layer_device() if dynamic else model.get_input_embeddings().weight.device
    encoded = prompt_inputs(tokenizer, prompt, device)
    query = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    query_ids = query["input_ids"].to(device)
    query_mask = query.get("attention_mask")
    if query_mask is None:
        query_mask = torch.ones_like(query_ids)
    query_mask = query_mask.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if dynamic:
            kwargs.update(
                {
                    "update_memory": False,
                    "memory_query_input_ids": query_ids,
                    "memory_query_attention_mask": query_mask,
                }
            )
        output = model.generate(**encoded, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    prompt_len = int(encoded["input_ids"].shape[1])
    response_ids = output[0, prompt_len:]
    response = tokenizer.decode(response_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()
    return response, int(response_ids.numel()), elapsed


def make_math_cases() -> list[dict[str, Any]]:
    raw = [
        ("m01", "只输出最终整数：38 + 47 = ?", ["85"]),
        ("m02", "只输出最终整数：900 - 376 = ?", ["524"]),
        ("m03", "只输出最终整数：24 × 17 = ?", ["408"]),
        ("m04", "只输出最终整数：936 ÷ 18 = ?", ["52"]),
        ("m05", "只输出结果：2.75 + 3.6 = ?", ["6.35"]),
        ("m06", "只输出结果：7/8 - 1/4 = ?", ["5/8", "0.625"]),
        ("m07", "只输出整数：15 和 28 的最小公倍数是多少？", ["420"]),
        ("m08", "只输出百分数：480 的 12.5% 是多少？", ["60"]),
        ("m09", "只输出百分数：80 增长到 100，增长率是多少？", ["25%", "25"]),
        ("m10", "只输出结果：3 的 5 次方是多少？", ["243"]),
        ("m11", "只输出 x：4x + 7 = 31。", ["6"]),
        ("m12", "只输出 x：9x - 18 = 0。", ["2"]),
        ("m13", "只输出 x：2(x + 5) = 18。", ["4"]),
        ("m14", "只输出 x：x/3 + 4 = 9。", ["15"]),
        ("m15", "只输出 x：5x - 2 = 3x + 10。", ["6"]),
        ("m16", "只输出下一个数：5，10，20，40，？", ["80"]),
        ("m17", "只输出下一个数：3，6，11，18，27，？", ["38"]),
        ("m18", "只输出下一个数：1，4，9，16，？", ["25"]),
        ("m19", "只输出下一个数：2，3，5，8，12，？", ["17"]),
        ("m20", "只输出整数：阶乘 6! 等于多少？", ["720"]),
        ("m21", "只输出面积：长 12、宽 7 的矩形面积是多少？", ["84"]),
        ("m22", "只输出周长：边长为 9 的正方形周长是多少？", ["36"]),
        ("m23", "只输出面积：底 10、高 6 的三角形面积是多少？", ["30"]),
        ("m24", "只输出角度：一个三角形两个角是 35 度和 65 度，第三个角是多少？", ["80"]),
        ("m25", "只输出数量：3 件不同衬衫和 2 条不同裤子可以组成多少套穿搭？", ["6"]),
        ("m26", "只输出数量：从 5 个人中选 2 个人，有多少种选法？", ["10"]),
        ("m27", "只输出余数：17 除以 5 的余数是多少？", ["2"]),
        ("m28", "只输出结果：平均数 8、12、16、20 是多少？", ["14"]),
        ("m29", "只输出结果：一个商品原价 240 元，打八折后多少钱？", ["192"]),
        ("m30", "只输出结果：2.4 × 0.5 = ?", ["1.2"]),
    ]
    return [
        {"id": case_id, "category": "math", "prompt": prompt, "acceptable": answers}
        for case_id, prompt, answers in raw
    ]


def make_reasoning_cases() -> list[dict[str, Any]]:
    raw = [
        ("r01", "只输出名字：甲比乙早到，乙比丙早到，谁最后到？", ["丙"]),
        ("r02", "只输出名字：小李在小王左边，小王在小张左边，谁最右边？", ["小张"]),
        ("r03", "只输出结论：所有鸟都有翅膀，企鹅是鸟，所以企鹅有翅膀吗？", ["是"]),
        ("r04", "只输出结论：所有猫都是哺乳动物，鲸鱼是哺乳动物，所以鲸鱼是猫吗？", ["不是", "否"]),
        ("r05", "只输出结论：如果下雨就带伞。现在下雨了，要不要带伞？", ["要"]),
        ("r06", "只输出结论：只有持票者才能入场。小林没有票，他能入场吗？", ["不能", "不可以"]),
        ("r07", "只输出星期：今天是星期三，五天后是星期几？", ["星期一", "周一"]),
        ("r08", "只输出方向：你面向北，右转后面向哪个方向？", ["东"]),
        ("r09", "只输出方向：你面向东，左转后面向哪个方向？", ["北"]),
        ("r10", "只输出数量：盒子里有 4 个红球和 3 个蓝球，不看颜色拿出一个，至少有几个球？", ["1"]),
        ("r11", "只输出名字：甲不是第一，乙在甲前面，丙在乙后面，谁可能是第一？", ["乙"]),
        ("r12", "只输出结论：有些学生会游泳，小周是学生，能确定小周会游泳吗？", ["不能", "无法"]),
        ("r13", "只输出下一个数：1，2，4，7，11，？", ["16"]),
        ("r14", "只输出下一个数：81，27，9，3，？", ["1"]),
        ("r15", "只输出名字：红色比蓝色重，绿色比红色轻但比蓝色重，哪个最轻？", ["蓝色"]),
        ("r16", "只输出答案：苹果不是蔬菜，胡萝卜是蔬菜，香蕉是水果，哪个不是水果？", ["胡萝卜"]),
        ("r17", "只输出结论：如果 A 大于 B 且 B 大于 C，那么 A 大于 C 吗？", ["是"]),
        ("r18", "只输出结论：如果一个数能被 2 整除，它一定是偶数。14 能被 2 整除，它是偶数吗？", ["是"]),
        ("r19", "只输出名字：小赵比小钱高，小孙比小赵矮但比小钱高，谁最高？", ["小赵"]),
        ("r20", "只输出数量：一周中有几天的名字包含‘星’字？", ["7"]),
        ("r21", "只输出结论：没有鱼是鸟，金鱼是鱼，所以金鱼是鸟吗？", ["不是", "否"]),
        ("r22", "只输出顺序：春、夏、秋、冬之后又回到哪个季节？", ["春"]),
        ("r23", "只输出结论：所有密码都需要保密，这个字符串是密码，所以它需要保密吗？", ["是"]),
        ("r24", "只输出答案：小明有两个兄弟，每个兄弟都有一个姐姐，小明有几个姐姐？", ["1"]),
    ]
    return [
        {"id": case_id, "category": "reasoning", "prompt": prompt, "acceptable": answers}
        for case_id, prompt, answers in raw
    ]


def make_context_cases(tokenizer: Any) -> list[dict[str, Any]]:
    rng = random.Random(20260904)
    filler = (
        "这是一段与问题无关的背景说明。系统记录了版本号、构建时间、测试批次、"
        "设备温度、日志摘要和普通项目备注。这些文字只是干扰项，不包含目标答案。"
    )
    cases: list[dict[str, Any]] = []
    for target_tokens in (512, 2048, 4096, 8192):
        for position in ("early", "middle", "late"):
            answer = f"CTX{target_tokens}-{position.upper()}-{rng.randrange(100, 999)}"
            needle = f"目标记录：本次检索需要返回的项目编码是 {answer}。"
            chunks: list[str] = []
            while input_token_count(tokenizer, " ".join(chunks + [filler, needle])) < target_tokens:
                chunks.append(filler)
            if position == "early":
                material = " ".join([needle] + chunks)
            elif position == "middle":
                half = len(chunks) // 2
                material = " ".join(chunks[:half] + [needle] + chunks[half:])
            else:
                material = " ".join(chunks + [needle])
            prompt = (
                "请阅读下面的材料，只输出目标记录中的项目编码，不要解释。\n"
                "---材料开始---\n"
                f"{material}\n"
                "---材料结束---\n"
                "问题：目标记录中的项目编码是什么？"
            )
            cases.append(
                {
                    "id": f"ctx-{target_tokens}-{position}",
                    "category": f"context_{target_tokens}",
                    "prompt": prompt,
                    "acceptable": [answer],
                    "target_tokens": target_tokens,
                    "position": position,
                }
            )
    return cases


def evaluate_cases(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    *,
    dynamic: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    categories: dict[str, list[float]] = {}
    started = time.perf_counter()
    for case in cases:
        if dynamic:
            model.reset_memory(batch_size=1, device=model._find_layer_device())
        response, generated_tokens, elapsed = generate_answer(
            model,
            tokenizer,
            str(case["prompt"]),
            dynamic=dynamic,
            max_new_tokens=max_new_tokens,
        )
        passed = contains_answer(response, list(case["acceptable"]))
        category = str(case["category"])
        categories.setdefault(category, []).append(float(passed))
        rows.append(
            {
                "id": case["id"],
                "category": category,
                "prompt_tokens": input_token_count(tokenizer, str(case["prompt"])),
                "acceptable": case["acceptable"],
                "generated": response,
                "generated_tokens": generated_tokens,
                "seconds": elapsed,
                "passed": passed,
            }
        )
    total = sum(sum(values) for values in categories.values())
    return {
        "cases": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "overall_score": total / max(1, len(rows)),
        "categories": {
            category: {
                "count": len(values),
                "score": sum(values) / max(1, len(values)),
            }
            for category, values in sorted(categories.items())
        },
        "rows": rows,
    }


def device_snapshot(model: Any) -> dict[str, Any]:
    device = model._find_layer_device() if hasattr(model, "_find_layer_device") else model.get_input_embeddings().weight.device
    params = sum(parameter.numel() for parameter in model.parameters())
    result: dict[str, Any] = {
        "device": str(device),
        "parameter_count": int(params),
        "parameter_count_billion": params / 1e9,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_gb": properties.total_memory / 1024**3,
            }
        )
    return result


def measure_performance(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    *,
    dynamic: bool,
    max_new_tokens: int,
    repeats: int,
) -> dict[str, Any]:
    selected = [cases[0]]
    for wanted in (512, 2048, 4096):
        matching = [case for case in cases if case.get("target_tokens") == wanted]
        if matching:
            selected.append(matching[1])
    rows = []
    device = model._find_layer_device() if dynamic else model.get_input_embeddings().weight.device
    for case in selected:
        latencies = []
        generated = 0
        for _ in range(max(1, repeats)):
            if dynamic:
                model.reset_memory(batch_size=1, device=device)
            _, token_count, elapsed = generate_answer(
                model,
                tokenizer,
                str(case["prompt"]),
                dynamic=dynamic,
                max_new_tokens=max_new_tokens,
            )
            latencies.append(elapsed)
            generated += token_count
        rows.append(
            {
                "id": case["id"],
                "prompt_tokens": input_token_count(tokenizer, str(case["prompt"])),
                "median_seconds": statistics.median(latencies),
                "mean_seconds": statistics.mean(latencies),
                "tokens_per_second": generated / max(1e-9, sum(latencies)),
                "repeats": len(latencies),
            }
        )
    return {"rows": rows}


def memory_payload(model: Any) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {
        "memory_state": model.runtime.state.detach().cpu().clone(),
    }
    if model.memory_config.natural_language_memory:
        for name in (
            "text_token_ids",
            "text_token_mask",
            "text_slot_valid",
            "text_slot_keys",
            "text_slot_age",
            "text_write_counter",
            "text_key_token_ids",
            "text_key_token_mask",
        ):
            value = getattr(model.runtime, name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"runtime memory field is unavailable: {name}")
            payload[name] = value.detach().cpu().clone()
    return payload


def answer_memory_query(model: Any, tokenizer: Any, text: str, max_new_tokens: int) -> tuple[str, bool]:
    response, _, _ = generate_answer(
        model,
        tokenizer,
        text,
        dynamic=True,
        max_new_tokens=max_new_tokens,
    )
    return response, bool(model.runtime.text_prefix_used)


def run_memory_benchmark(
    model: Any,
    tokenizer: Any,
    package_path: str | Path,
    *,
    max_new_tokens: int,
    no_4bit: bool,
) -> dict[str, Any]:
    """Run memory tests and restore the user's original embedded state."""

    original = memory_payload(model)
    decision_rows = []
    positives = [
        "我叫林舟。",
        "我的常住城市是苏州。",
        "我最喜欢的水果是红富士苹果。",
        "我正在开发 Natural Memory v1 项目。",
        "以后请把代码默认写成 Python。",
        "我的常用时区是 Asia/Shanghai。",
        "这是我的长期偏好：使用简洁中文。",
        "这个项目的重要约束是不要修改原始 Qwen 权重。",
    ]
    negatives = [
        "我叫什么？",
        "帮我解释向量数据库是什么。",
        "你觉得今天的天气怎么样？",
        "请把 memory 翻译成中文。",
        "计算一下 17 × 19。",
        "如果我选择 GPU，会发生什么？",
        "我之前有没有提到我的城市？",
        "给我一个自然语言记忆方案。",
    ]
    try:
        for expected, items in ((True, positives), (False, negatives)):
            for text in items:
                model.reset_memory(batch_size=1, device=model._find_layer_device())
                changed = _write_turn(model, tokenizer, text, model._find_layer_device())
                decision_rows.append(
                    {
                        "text": text,
                        "expected_write": expected,
                        "actual_write": changed,
                        "write_probability": float(model.runtime.auto_memory_probability.mean())
                        if isinstance(model.runtime.auto_memory_probability, torch.Tensor)
                        else None,
                    }
                )
        tp = sum(row["expected_write"] and row["actual_write"] for row in decision_rows)
        tn = sum((not row["expected_write"]) and (not row["actual_write"]) for row in decision_rows)
        fp = sum((not row["expected_write"]) and row["actual_write"] for row in decision_rows)
        fn = sum(row["expected_write"] and (not row["actual_write"]) for row in decision_rows)

        model.reset_memory(batch_size=1, device=model._find_layer_device())
        first = "我的工作地点代号是NM-R7。"
        second = "我最喜欢的水果是青提。"
        replacement = "我的工作地点代号改为NM-K9。"
        writes = [
            {"text": first, "changed": _write_turn(model, tokenizer, first, model._find_layer_device())},
            {"text": second, "changed": _write_turn(model, tokenizer, second, model._find_layer_device())},
            {
                "text": replacement,
                "changed": _write_turn(model, tokenizer, replacement, model._find_layer_device()),
            },
        ]
        before_restart = {}
        for name, query, expected in (
            ("work_code", "我的工作地点代号是什么？", "NM-K9"),
            ("fruit", "我最喜欢的水果是什么？", "青提"),
            ("unknown", "我的血型是什么？如果没有记录，请明确说不知道。", "不知道"),
        ):
            response, prefix_used = answer_memory_query(model, tokenizer, query, max_new_tokens)
            before_restart[name] = {
                "query": query,
                "expected": expected,
                "response": response,
                "expected_found": expected in response,
                "prefix_used": prefix_used,
            }

        # Persist through ordinary natural-language turns.  The caller releases
        # this model before loading a fresh process/model for the restart test;
        # keeping that lifecycle outside this function avoids two 4-bit Qwen
        # backbones occupying the GPU at the same time.
        model.save_embedded_memory_weights(package_path)
        return {
            "automatic_write_decision": {
                "rows": decision_rows,
                "true_positive": int(tp),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "specificity": tn / max(1, tn + fp),
            },
            "natural_language_writes": writes,
            "before_restart": before_restart,
            "embedded_write_persisted": True,
        }
    except Exception:
        # Best-effort restoration if a test fails halfway through.
        model._load_persistent_memory_payload(original)
        model.save_embedded_memory_weights(package_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model)
    memory_config_hint = load_memory_config(args.memory_model)
    memory_variant = (
        "natural_memory_v2"
        if memory_config_hint.memory_version >= 2 or memory_config_hint.hierarchical_memory
        else "natural_memory_v1"
    )
    general = load_jsonl(args.data)
    math_cases = make_math_cases()
    reasoning_cases = make_reasoning_cases()
    context_cases = make_context_cases(tokenizer)
    all_cases = general + math_cases + reasoning_cases + context_cases
    use_4bit = not args.no_4bit
    report: dict[str, Any] = {
        "benchmark": f"Natural Memory {memory_variant.rsplit('_', 1)[-1]} full local comparison",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": 20260904,
        "base_model": str(Path(args.base_model).resolve()),
        "memory_model": str(Path(args.memory_model).resolve()),
        "data": str(Path(args.data).resolve()),
        "quantization": "4bit_nf4" if use_4bit else "none",
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "case_counts": {
            "general_existing": len(general),
            "math": len(math_cases),
            "reasoning": len(reasoning_cases),
            "context": len(context_cases),
            "total": len(all_cases),
        },
        "context_targets": sorted({case["target_tokens"] for case in context_cases}),
    }

    print(f"cases={len(all_cases)} quantization={report['quantization']}")
    print("loading Qwen3.5-4B baseline")
    started = time.perf_counter()
    baseline = load_qwen_base(args.base_model, load_in_4bit=use_4bit)
    baseline.eval()
    report["baseline"] = {
        "load_seconds": time.perf_counter() - started,
        "hardware": device_snapshot(baseline),
    }
    baseline_device = baseline.get_input_embeddings().weight.device
    if baseline_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(baseline_device)
    report["baseline"]["quality"] = evaluate_cases(
        baseline,
        tokenizer,
        all_cases,
        dynamic=False,
        max_new_tokens=args.max_new_tokens,
    )
    report["baseline"]["performance"] = measure_performance(
        baseline,
        tokenizer,
        context_cases,
        dynamic=False,
        max_new_tokens=args.max_new_tokens,
        repeats=args.perf_repeats,
    )
    if baseline_device.type == "cuda":
        report["baseline"]["peak_memory_allocated_gb"] = torch.cuda.max_memory_allocated(baseline_device) / 1024**3
        report["baseline"]["peak_memory_reserved_gb"] = torch.cuda.max_memory_reserved(baseline_device) / 1024**3
    release(baseline)

    print(f"loading {memory_variant} embedded package")
    started = time.perf_counter()
    dynamic = load_qwen_dynamic(args.memory_model, load_in_4bit=use_4bit)
    dynamic.eval()
    report[memory_variant] = {
        "load_seconds": time.perf_counter() - started,
        "hardware": device_snapshot(dynamic),
    }
    dynamic_device = dynamic._find_layer_device()
    if dynamic_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dynamic_device)
    report[memory_variant]["quality"] = evaluate_cases(
        dynamic,
        tokenizer,
        all_cases,
        dynamic=True,
        max_new_tokens=args.max_new_tokens,
    )
    report[memory_variant]["performance"] = measure_performance(
        dynamic,
        tokenizer,
        context_cases,
        dynamic=True,
        max_new_tokens=args.max_new_tokens,
        repeats=args.perf_repeats,
    )
    if dynamic_device.type == "cuda":
        report[memory_variant]["peak_memory_allocated_gb"] = torch.cuda.max_memory_allocated(dynamic_device) / 1024**3
        report[memory_variant]["peak_memory_reserved_gb"] = torch.cuda.max_memory_reserved(dynamic_device) / 1024**3

    original_embedded_payload = memory_payload(dynamic)
    memory_report = run_memory_benchmark(
        dynamic,
        tokenizer,
        args.memory_model,
        max_new_tokens=args.max_new_tokens,
        no_4bit=args.no_4bit,
    )
    # Release the first Qwen backbone before constructing the fresh model used
    # by the shard-backed restart check.  This is important on a 12 GB GPU.
    release(dynamic)
    dynamic = None
    restarted = None
    try:
        restarted = load_qwen_dynamic(args.memory_model, load_in_4bit=use_4bit)
        restarted.eval()
        after_restart = {}
        for name, query, expected in (
            ("work_code", "我的工作地点代号是什么？", "NM-K9"),
            ("fruit", "我最喜欢的水果是什么？", "青提"),
        ):
            response, prefix_used = answer_memory_query(restarted, tokenizer, query, args.max_new_tokens)
            after_restart[name] = {
                "query": query,
                "expected": expected,
                "response": response,
                "expected_found": expected in response,
                "prefix_used": prefix_used,
            }
        restarted.reset_memory(batch_size=1, device=restarted._find_layer_device())
        reset_response, reset_prefix = answer_memory_query(
            restarted,
            tokenizer,
            "我的工作地点代号是什么？",
            args.max_new_tokens,
        )
        reset_slots = int(restarted.runtime.text_slot_valid.sum().item())
        memory_report.update(
            {
                "after_restart_without_history_or_pt": after_restart,
                "restart_pass": all(item["expected_found"] for item in after_restart.values()),
                "reset": {
                    "response": reset_response,
                    "prefix_used": reset_prefix,
                    "valid_slots": reset_slots,
                    "cleared": reset_slots == 0 and not reset_prefix,
                },
            }
        )
    finally:
        if restarted is not None:
            # Restore the user's pre-benchmark state, so the benchmark itself
            # does not overwrite the active embedded memory snapshot.
            restarted._load_persistent_memory_payload(original_embedded_payload)
            restarted.save_embedded_memory_weights(args.memory_model)
            memory_report["state_restored"] = int(restarted.runtime.text_slot_valid.sum().item()) == int(
                original_embedded_payload["text_slot_valid"].sum().item()
            )
            release(restarted)
    report["memory"] = memory_report

    baseline_quality = report["baseline"]["quality"]
    dynamic_quality = report[memory_variant]["quality"]
    categories = sorted(
        set(baseline_quality["categories"]) & set(dynamic_quality["categories"])
    )
    category_deltas = {
        category: dynamic_quality["categories"][category]["score"]
        - baseline_quality["categories"][category]["score"]
        for category in categories
    }
    report["comparison"] = {
        "overall_delta": dynamic_quality["overall_score"] - baseline_quality["overall_score"],
        "category_deltas": category_deltas,
        "peak_memory_allocated_delta_gb": report[memory_variant].get("peak_memory_allocated_gb", 0.0)
        - report["baseline"].get("peak_memory_allocated_gb", 0.0),
        "load_seconds_delta": report[memory_variant]["load_seconds"]
        - report["baseline"]["load_seconds"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "baseline_score": baseline_quality["overall_score"],
        f"{memory_variant}_score": dynamic_quality["overall_score"],
        "overall_delta": report["comparison"]["overall_delta"],
        "memory_restart_pass": report["memory"]["restart_pass"],
        "memory_reset_pass": report["memory"]["reset"]["cleared"],
        "cases": len(all_cases),
        "output": str(output),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
