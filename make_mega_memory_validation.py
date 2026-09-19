"""Generate a large, deterministic validation set for KV-vs-memory parity.

Each case contains a complete fact episode, distractors, a query, and exact
acceptance metadata.  The default is 100,000 cases across ten categories.
This is a data generator only; model quality must be measured by the paired
teacher/student benchmark after generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
CATEGORIES = (
    "single_fact",
    "distractor_32",
    "distractor_128",
    "conflict_update",
    "multi_hop",
    "unknown_abstention",
    "random_position",
    "long_context",
    "paraphrase",
    "forget_correction",
)

ATTRIBUTES = (
    "常用语言",
    "工作地点",
    "最喜欢的水果",
    "项目代号",
    "常用时区",
    "提醒时间",
    "默认输出风格",
    "备用联系人",
)
UNKNOWN_MARKERS = ["不知道", "没有记录", "无法确定", "未找到相关信息"]


def _path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _value(rng: random.Random, prefix: str) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return f"{prefix}-" + "".join(rng.choice(alphabet) for _ in range(8))


def _fact(
    text: str,
    *,
    entity: str,
    attribute: str,
    value: str,
    should_write: bool = True,
    kind: str = "fact",
) -> dict[str, Any]:
    return {
        "text": text,
        "assistant": "好的，我会记录这条信息。",
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "should_write": should_write,
        "kind": kind,
    }


def _noise(rng: random.Random, subject: str, index: int) -> dict[str, Any]:
    attribute = f"干扰属性{index}"
    value = _value(rng, "N")
    return _fact(
        f"这是普通对话噪声：{subject}暂时提到一个无关编号{value}，不需要长期保存。",
        entity=subject,
        attribute=attribute,
        value=value,
        should_write=False,
        kind="noise",
    )


def _base_case(category: str, index: int, rng: random.Random) -> dict[str, Any]:
    subject = f"验证用户-{index:07d}"
    attribute = rng.choice(ATTRIBUTES)
    value = _value(rng, "V")
    facts: list[dict[str, Any]] = []
    query = ""
    acceptable: list[str] = []
    forbidden: list[str] = []
    metadata: dict[str, Any] = {}

    if category == "single_fact":
        facts = [_fact(f"请记住：我的{attribute}是{value}。", entity=subject, attribute=attribute, value=value)]
        query = f"跨对话后请回答：我的{attribute}是什么？"
        acceptable = [value]
    elif category in {"distractor_32", "distractor_128"}:
        count = 32 if category.endswith("32") else 128
        target = _fact(f"请记住：我的{attribute}是{value}。", entity=subject, attribute=attribute, value=value)
        facts = [_noise(rng, subject, item) for item in range(count)]
        facts.insert(rng.randrange(len(facts) + 1), target)
        query = f"在大量无关信息中，只读取我的{attribute}，它的值是什么？"
        acceptable = [value]
        metadata["distractor_count"] = count
    elif category == "conflict_update":
        old_value = _value(rng, "OLD")
        facts = [
            _fact(f"我的{attribute}是{old_value}。", entity=subject, attribute=attribute, value=old_value),
            _fact(
                f"更正一下：我的{attribute}已经改为{value}，旧值不要再使用。",
                entity=subject,
                attribute=attribute,
                value=value,
                kind="correction",
            ),
        ]
        query = f"我的{attribute}最新值是什么？"
        acceptable = [value]
        forbidden = [old_value]
    elif category == "multi_hop":
        project = f"项目-{_value(rng, 'P')}"
        person = f"成员-{_value(rng, 'M')}"
        code = _value(rng, "H")
        facts = [
            _fact(f"项目{project}的负责人是{person}。", entity=project, attribute="负责人", value=person),
            _fact(f"成员{person}的工作代号是{code}。", entity=person, attribute="工作代号", value=code),
        ]
        query = f"请通过项目负责人关系，找出项目{project}负责人的工作代号。"
        acceptable = [code]
        metadata["hop_count"] = 2
    elif category == "unknown_abstention":
        known_attribute = rng.choice(ATTRIBUTES)
        missing_attribute = f"不存在的个人属性-{index:07d}"
        known_value = _value(rng, "KNOWN")
        facts = [_fact(f"我的{known_attribute}是{known_value}。", entity=subject, attribute=known_attribute, value=known_value)]
        query = f"我的{missing_attribute}是什么？如果没有记录，请明确说不知道。"
        acceptable = UNKNOWN_MARKERS
        forbidden = [known_value]
        metadata["answerable"] = False
    elif category == "random_position":
        count = 64
        facts = [_noise(rng, subject, item) for item in range(count)]
        target = _fact(f"请记住：我的{attribute}是{value}。", entity=subject, attribute=attribute, value=value)
        position = rng.randrange(count + 1)
        facts.insert(position, target)
        query = f"随机位置事实测试：我的{attribute}是什么？"
        acceptable = [value]
        metadata["target_position"] = position
        metadata["distractor_count"] = count
    elif category == "long_context":
        count = 96
        facts = [_noise(rng, subject, item) for item in range(count)]
        target = _fact(f"请记住：我的{attribute}是{value}。", entity=subject, attribute=attribute, value=value)
        position = rng.randrange(count + 1)
        facts.insert(position, target)
        query = f"在长上下文压缩之后，检索我的{attribute}并回答。"
        acceptable = [value]
        metadata["target_position"] = position
        metadata["distractor_count"] = count
        metadata["synthetic_padding_tokens"] = 2048 + (index % 5) * 512
    elif category == "paraphrase":
        fact_templates = (
            f"个人资料更新：{attribute}这一栏填写为{value}。",
            f"以后涉及{attribute}时，请使用{value}这个值。",
            f"记录一下，我的{attribute}偏好/设置是{value}。",
        )
        query_templates = (
            f"我之前登记的{attribute}内容是什么？",
            f"关于{attribute}，你保存的用户信息是哪一个？",
            f"不要猜，回忆一下我在{attribute}上的设置。",
        )
        facts = [_fact(rng.choice(fact_templates), entity=subject, attribute=attribute, value=value)]
        query = rng.choice(query_templates)
        acceptable = [value]
    elif category == "forget_correction":
        old_value = _value(rng, "FORGET")
        facts = [
            _fact(f"请记住：我的{attribute}是{old_value}。", entity=subject, attribute=attribute, value=old_value),
            _fact(
                f"请删除关于我的{attribute}的记忆，不要再保留这个信息。",
                entity=subject,
                attribute=attribute,
                value=old_value,
                should_write=False,
                kind="forget",
            ),
        ]
        query = f"我的{attribute}是什么？如果已经删除，请回答没有记录。"
        acceptable = UNKNOWN_MARKERS
        forbidden = [old_value]
        metadata["answerable"] = False
    else:
        raise ValueError(category)

    return {
        "id": f"mega-{category}-{index:07d}",
        "category": category,
        "subject": subject,
        "facts": facts,
        "query": query,
        "acceptable": acceptable,
        "forbidden": forbidden,
        "metadata": metadata,
        "generator_version": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/mega_validation/memory_validation_100k.jsonl")
    parser.add_argument("--cases-per-category", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.cases_per_category < 1:
        raise SystemExit("--cases-per-category must be positive")
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    total = 0
    with output.open("w", encoding="utf-8") as handle:
        for category_index, category in enumerate(CATEGORIES):
            for index in range(args.cases_per_category):
                case_seed = args.seed + category_index * 1_000_003 + index * 97
                case = _base_case(category, index, random.Random(case_seed))
                raw = (json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                handle.write(raw.decode("utf-8"))
                digest.update(raw)
                counts[category] += 1
                total += 1
    manifest = {
        "format_version": 1,
        "generator": "make_mega_memory_validation.py",
        "generator_version": 1,
        "seed": args.seed,
        "categories": list(CATEGORIES),
        "cases_per_category": args.cases_per_category,
        "total_cases": total,
        "counts": dict(counts),
        "sha256": digest.hexdigest(),
        "teacher_student_protocol": {
            "teacher": "complete fact episode plus query in one full KV context",
            "student": "facts presented one turn at a time, memory read for query, no history replay",
            "primary_gate": "student answer agreement >= 0.95 * teacher answer accuracy",
        },
        "warning": "Synthetic stress validation; add redacted real conversations before production certification.",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
