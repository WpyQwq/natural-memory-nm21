"""Create deterministic train/eval data for the dynamic-memory benchmark."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ATTRIBUTES = ("驻地", "负责人", "维护日", "安全级别", "档案类别")
# Single-character answers remove shared prefixes and make free-generation
# exact match a meaningful associative-recall metric.
VALUES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

MEMORY_USER_TEMPLATES = (
    "记住这条资料：{subject}的{attribute}代号是{value}。",
    "请存储信息——对象{subject}的{attribute}为代号{value}。",
    "新事实：{subject}的{attribute}代号={value}。请记住。",
)
MEMORY_ASSISTANT_TEMPLATES = (
    "已记录：{subject}的{attribute}代号是{value}。",
    "好的，{subject}的{attribute}已记为代号{value}。",
    "收到，已经保存{subject}的{attribute}代号：{value}。",
)
QUERY_USER_TEMPLATES = (
    "查询：{subject}的{attribute}代号是什么？",
    "请问对象{subject}的{attribute}代号为？",
    "根据已记信息，{subject}的{attribute}代号是？",
)


def make_records(count: int, *, prefix: str, rng: random.Random) -> list[dict]:
    records = []
    for index in range(count):
        subject = f"{prefix}{index:04d}"
        attribute = ATTRIBUTES[index % len(ATTRIBUTES)]
        value = rng.choice(VALUES)
        fields = {"subject": subject, "attribute": attribute, "value": value}
        records.append(
            {
                "id": f"{prefix.lower()}-{index:04d}",
                "memory": [
                    {
                        "role": "user",
                        "content": rng.choice(MEMORY_USER_TEMPLATES).format(**fields),
                    },
                    {
                        "role": "assistant",
                        "content": rng.choice(MEMORY_ASSISTANT_TEMPLATES).format(**fields),
                    },
                ],
                "query": [
                    {
                        "role": "user",
                        "content": rng.choice(QUERY_USER_TEMPLATES).format(**fields),
                    },
                {"role": "assistant", "content": value},
                ],
                "subject": subject,
                "attribute": attribute,
                "answer": value,
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="V2_dpskw/data")
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.train_count < 1 or args.eval_count < 1:
        raise ValueError("train-count and eval-count must be positive")

    train = make_records(args.train_count, prefix="训练实体", rng=random.Random(args.seed))
    evaluation = make_records(args.eval_count, prefix="测试实体", rng=random.Random(args.seed + 1))
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "benchmark_train.jsonl", train)
    write_jsonl(output_dir / "benchmark_eval.jsonl", evaluation)
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_count": len(train),
                "eval_count": len(evaluation),
                "task": "random subject-to-code associative recall",
                "train_subject_prefix": "训练实体",
                "eval_subject_prefix": "测试实体",
                "answer_is_not_derived_from_subject": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"train={len(train)} path={output_dir / 'benchmark_train.jsonl'}")
    print(f"eval={len(evaluation)} path={output_dir / 'benchmark_eval.jsonl'}")


if __name__ == "__main__":
    main()
