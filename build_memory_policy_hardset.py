"""Build a larger hard-negative dataset for the automatic memory controller.

The normal bootstrap corpus contains mostly short, obvious examples.  This
hard set adds realistic negations, questions, replacement requests, and long
noise clauses so the write/forget heads are evaluated on decisions that are
easy to confuse with durable facts.
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
ATTRIBUTES = (
    "常用时区",
    "最喜欢的水果",
    "项目代号",
    "备用联系人",
    "默认输出风格",
    "工作区域",
    "提醒时间",
    "档案代号",
)

TRAIN_FACT_TEMPLATES = (
    "请记住：我的{attribute}是{value}。",
    "以后涉及{attribute}时，请使用{value}这个值。",
    "个人资料更新——我的{attribute}等于{value}，以后可能会问到。",
    "请把这条个人资料保存下来：我的{attribute}为{value}。",
    "我的{attribute}是{value}，这是需要长期保留的信息。",
)
EVAL_FACT_TEMPLATES = (
    "登记一下，我的{attribute}：{value}。",
    "将我的{attribute}记为{value}，后续请按这个资料回答。",
    "长期资料里新增一项：{attribute}={value}。",
    "请把我的{attribute}保存成{value}。",
)
TRAIN_NOISE_TEMPLATES = (
    "这是普通对话噪声：我暂时提到一个无关编号{value}，不需要长期保存。",
    "随口一提，编号{value}只是临时信息，请不要记住。",
    "请不要把这句话写入长期记忆：今天看到的临时编号是{value}。",
    "这只是一次性测试值{value}，不用保存，也不要据此推断个人资料。",
    "聊天中的无关内容：{value}；它不是我的个人事实。",
)
EVAL_NOISE_TEMPLATES = (
    "临时提到{value}，这不是需要保存的资料。",
    "忽略这个一次性编号{value}，不要将它写入记忆。",
    "普通闲聊内容：{value}，没有长期价值。",
    "不要记住{value}，它只是当前消息里的干扰项。",
)
TRAIN_QUERY_TEMPLATES = (
    "我的{attribute}是什么？",
    "只根据已经保存的资料，告诉我{attribute}。",
    "不要猜测，请读取记忆回答：我的{attribute}为？",
    "记忆中是否有我的{attribute}？",
)
EVAL_QUERY_TEMPLATES = (
    "跨对话后，我登记的{attribute}是哪一个？",
    "请从长期资料中查找我的{attribute}。",
    "之前保存的{attribute}内容是什么？",
)
TRAIN_FORGET_TEMPLATES = (
    "请删除关于我的{attribute}的记忆，不要再保留{value}。",
    "忘掉我的{attribute}，这条资料已经失效。",
    "撤销之前保存的{attribute}，以后不要再使用它。",
    "清除我的{attribute}记录；{value}不再有效。",
)
EVAL_FORGET_TEMPLATES = (
    "请移除长期记忆中的{attribute}，不要继续记住它。",
    "我的{attribute}已经作废，请忘记这项资料。",
    "撤回关于{attribute}的个人信息，不要再保留。",
)


def _value(rng: random.Random, prefix: str) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return f"{prefix}-" + "".join(rng.choice(alphabet) for _ in range(8))


def _row(
    *,
    group: str,
    index: int,
    text: str,
    write: float,
    forget: float,
    kind: str,
    attribute: str,
    value: str,
) -> dict[str, Any]:
    return {
        "id": f"{group}:{index}",
        "group_id": group,
        "text": text,
        "messages": [{"role": "user", "content": text}],
        "write_label": write,
        "forget_label": forget,
        "kind": kind,
        "source": "synthetic_memory_policy_hardset",
        "subject": "验证用户",
        "attribute": attribute,
        "value": value,
        "answer": "",
        "answerable": None,
    }


def _build_split(
    *,
    count: int,
    split: str,
    seed: int,
    fact_templates: tuple[str, ...],
    noise_templates: tuple[str, ...],
    query_templates: tuple[str, ...],
    forget_templates: tuple[str, ...],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for group_index in range(count):
        group = f"hard-{split}-{group_index:06d}"
        attribute = ATTRIBUTES[group_index % len(ATTRIBUTES)]
        fact_value = _value(rng, "FACT")
        noise_value = _value(rng, "NOISE")
        replacement_value = _value(rng, "NEW")
        rows.extend(
            (
                _row(
                    group=group,
                    index=0,
                    text=rng.choice(fact_templates).format(attribute=attribute, value=fact_value),
                    write=1.0,
                    forget=0.0,
                    kind="fact",
                    attribute=attribute,
                    value=fact_value,
                ),
                _row(
                    group=group,
                    index=1,
                    text=rng.choice(noise_templates).format(value=noise_value),
                    write=0.0,
                    forget=0.0,
                    kind="noise",
                    attribute="",
                    value=noise_value,
                ),
                _row(
                    group=group,
                    index=2,
                    text=rng.choice(query_templates).format(attribute=attribute),
                    write=0.0,
                    forget=0.0,
                    kind="query",
                    attribute=attribute,
                    value="",
                ),
                _row(
                    group=group,
                    index=3,
                    text=rng.choice(forget_templates).format(attribute=attribute, value=fact_value),
                    write=0.0,
                    forget=1.0,
                    kind="forget",
                    attribute=attribute,
                    value=fact_value,
                ),
                _row(
                    group=group,
                    index=4,
                    text=f"更正一下：我的{attribute}改为{replacement_value}，旧值不再有效。",
                    write=1.0,
                    # A replacement is a write/update, not a delete.  The
                    # runtime retires the matched old version and keeps the
                    # new fragment active.  Only an explicit forget request
                    # receives forget_label=1.
                    forget=0.0,
                    kind="replacement",
                    attribute=attribute,
                    value=replacement_value,
                ),
            )
        )
        # These rows are deliberately close to real conversation and are not
        # ordinary keyword negatives.  They teach the controller that a
        # question, a hypothetical, a quoted third-party claim, and a
        # temporary value must not become durable personal memory.
        rows.extend(
            (
                _row(
                    group=group,
                    index=5,
                    text=f"我想知道我的{attribute}是什么？",
                    write=0.0,
                    forget=0.0,
                    kind="question",
                    attribute=attribute,
                    value="",
                ),
                _row(
                    group=group,
                    index=6,
                    text=f"如果我的{attribute}改成{replacement_value}，会有什么影响？",
                    write=0.0,
                    forget=0.0,
                    kind="hypothetical",
                    attribute=attribute,
                    value=replacement_value,
                ),
                _row(
                    group=group,
                    index=7,
                    text=f"别人说我的{attribute}是{fact_value}，这不是我的个人资料，请不要记录。",
                    write=0.0,
                    forget=0.0,
                    kind="quoted_noise",
                    attribute=attribute,
                    value=fact_value,
                ),
                _row(
                    group=group,
                    index=8,
                    text=f"今天临时使用{noise_value}，只在本次对话有效，不要长期保存。",
                    write=0.0,
                    forget=0.0,
                    kind="temporary",
                    attribute="",
                    value=noise_value,
                ),
            )
        )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/production_memory_hard")
    parser.add_argument("--train-groups", type=int, default=4000)
    parser.add_argument("--eval-groups", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.train_groups < 1 or args.eval_groups < 1:
        raise SystemExit("group counts must be positive")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute() and not output_dir.exists():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _build_split(
        count=args.train_groups,
        split="train",
        seed=args.seed,
        fact_templates=TRAIN_FACT_TEMPLATES,
        noise_templates=TRAIN_NOISE_TEMPLATES,
        query_templates=TRAIN_QUERY_TEMPLATES,
        forget_templates=TRAIN_FORGET_TEMPLATES,
    )
    eval_rows = _build_split(
        count=args.eval_groups,
        split="eval",
        seed=args.seed + 1,
        fact_templates=EVAL_FACT_TEMPLATES,
        noise_templates=EVAL_NOISE_TEMPLATES,
        query_templates=EVAL_QUERY_TEMPLATES,
        forget_templates=EVAL_FORGET_TEMPLATES,
    )
    for name, rows in (("train", train_rows), ("eval", eval_rows)):
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "format_version": 1,
        "generator": "build_memory_policy_hardset.py",
        "seed": args.seed,
        "train_groups": args.train_groups,
        "eval_groups": args.eval_groups,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "labels": {
            "write_positive": sum(row["write_label"] >= 0.5 for row in train_rows),
            "forget_positive": sum(row["forget_label"] >= 0.5 for row in train_rows),
        },
        "warning": "Synthetic hard negatives; combine with redacted real conversations before production deployment.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
