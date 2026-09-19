"""Create train/eval streams for learned write, forget, and no-hallucination tests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ATTRIBUTES = ("常用语言", "备用联系人", "工作区域", "档案代号", "提醒时间")
VALUES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

FACT_USER = (
    "请记住：{subject}的{attribute}是代号{value}。",
    "把这条个人资料写入记忆：{subject}的{attribute}为{value}。",
    "个人事实更新——{subject}的{attribute}等于{value}，以后可能会问到。",
)
FACT_ASSISTANT = (
    "好的，这条资料已经记录。",
    "收到，我会保留这条个人事实。",
    "已保存。",
)
NOISE_USER = (
    "今天天气不错，随便聊聊。",
    "请给我一个简短的问候。",
    "这是一条不需要长期记忆的闲聊。",
)
NOISE_ASSISTANT = (
    "好的。",
    "明白。",
    "收到。",
)
QUERY_KNOWN = (
    "只根据已经保存的个人资料，{subject}的{attribute}是什么？",
    "不要猜测，请读取记忆回答：{subject}的{attribute}为？",
    "跨对话查询：请问{subject}的{attribute}代号是什么？",
)
QUERY_UNKNOWN = (
    "记忆中是否有{subject}的{attribute}？如果没有，请明确说不知道。",
    "请查询个人记忆：{subject}的{attribute}是什么？没有记录时不要猜。",
)


def _messages(user: str, assistant: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _fact_chunk(subject: str, attribute: str, value: str, rng: random.Random, *, forget: int = 0) -> dict:
    fields = {"subject": subject, "attribute": attribute, "value": value}
    return {
        "messages": _messages(rng.choice(FACT_USER).format(**fields), rng.choice(FACT_ASSISTANT)),
        "value": value,
        "write_label": 1.0,
        "forget_label": float(forget),
        "kind": "fact" if not forget else "replacement",
    }


def _noise_chunk(rng: random.Random) -> dict:
    return {
        "messages": _messages(rng.choice(NOISE_USER), rng.choice(NOISE_ASSISTANT)),
        "value": None,
        "write_label": 0.0,
        "forget_label": 0.0,
        "kind": "noise",
    }


def make_record(index: int, *, prefix: str, rng: random.Random) -> dict:
    subject = f"{prefix}{index:05d}"
    attribute = rng.choice(ATTRIBUTES)
    value = rng.choice(VALUES)
    mode = rng.random()
    chunks: list[dict] = []
    if mode < 0.20:
        chunks.append(_noise_chunk(rng))
        answer = "不知道。"
        query = rng.choice(QUERY_UNKNOWN).format(subject=subject, attribute=attribute)
        answerable = False
    elif mode < 0.45:
        old_value = rng.choice(tuple(item for item in VALUES if item != value))
        chunks.append(_fact_chunk(subject, attribute, old_value, rng))
        chunks.append(_noise_chunk(rng))
        chunks.append(_fact_chunk(subject, attribute, value, rng, forget=1))
        answer = value
        query = rng.choice(QUERY_KNOWN).format(subject=subject, attribute=attribute)
        answerable = True
    else:
        if rng.random() < 0.35:
            chunks.append(_noise_chunk(rng))
        chunks.append(_fact_chunk(subject, attribute, value, rng))
        if rng.random() < 0.35:
            chunks.append(_noise_chunk(rng))
        answer = value
        query = rng.choice(QUERY_KNOWN).format(subject=subject, attribute=attribute)
        answerable = True
    return {
        "id": f"{prefix.lower()}-{index:05d}",
        "memory_chunks": chunks,
        "query": _messages(query, answer),
        "subject": subject,
        "attribute": attribute,
        "answer": answer,
        "answerable": answerable,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="V2_dpskw/data/native_memory")
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    train = [
        make_record(i, prefix="训练用户", rng=random.Random(args.seed + i * 17))
        for i in range(args.train_count)
    ]
    evaluation = [
        make_record(i, prefix="评估用户", rng=random.Random(args.seed + 100000 + i * 17))
        for i in range(args.eval_count)
    ]
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "eval.jsonl", evaluation)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_count": len(train),
                "eval_count": len(evaluation),
                "task": "learned persistent memory with noise, unknowns, and replacement",
                "contains_write_labels": True,
                "contains_forget_labels": True,
                "answer_is_not_derived_from_subject": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"train={len(train)} path={output_dir / 'train.jsonl'}")
    print(f"eval={len(evaluation)} path={output_dir / 'eval.jsonl'}")


if __name__ == "__main__":
    main()
