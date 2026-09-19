"""Normalize conversation logs into a leak-resistant memory-policy dataset.

The runtime accepts many local data shapes because real users rarely keep
their chat exports in one format.  This command converts them to a small,
auditable JSONL schema without inventing labels.  It understands the current
``native_memory`` episode format, the streaming demo format, and a generic
format documented in the output manifest.

The bundled fallback files are bootstrap data for smoke tests.  A real user
corpus can be supplied with ``--source``/``--eval-source`` and receives the
same normalization and group-level split guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCES = (
    "data/native_memory/train.jsonl",
    "data/native_memory/eval.jsonl",
    "data/demo_stream.jsonl",
)
PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value


def _message_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages.strip()
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            role = str(message.get("role", "user"))
            parts.append(f"[{role}] {content.strip()}")
    return "\n".join(parts).strip()


def _user_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages.strip()
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
    return _message_text(messages)


def _explicit_split(path: Path, *, forced: str | None) -> str | None:
    if forced in {"train", "eval"}:
        return forced
    name = path.name.lower()
    if any(mark in name for mark in ("eval", "valid", "test")):
        return "eval"
    if "train" in name:
        return "train"
    return None


def _make_example(
    *,
    group_id: str,
    example_id: str,
    text: str,
    write_label: float,
    forget_label: float = 0.0,
    kind: str = "conversation",
    source: str,
    subject: str = "",
    attribute: str = "",
    value: Any = None,
    answer: str = "",
    answerable: bool | None = None,
    messages: Any = None,
) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    return {
        "id": example_id,
        "group_id": group_id,
        "text": text,
        "messages": messages if isinstance(messages, list) else [{"role": "user", "content": text}],
        "write_label": float(max(0.0, min(1.0, write_label))),
        "forget_label": float(max(0.0, min(1.0, forget_label))),
        "kind": kind,
        "source": source,
        "subject": str(subject or ""),
        "attribute": str(attribute or ""),
        "value": "" if value is None else str(value),
        "answer": str(answer or ""),
        "answerable": answerable,
    }


def normalize_record(record: dict[str, Any], *, source: str, line_number: int) -> list[dict[str, Any]]:
    """Convert one source record into labeled write/query decisions."""

    raw_id = str(record.get("id") or record.get("conversation_id") or f"line-{line_number}")
    group_id = f"{source}:{raw_id}"
    output: list[dict[str, Any]] = []

    chunks = record.get("memory_chunks")
    if isinstance(chunks, list):
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            messages = chunk.get("messages", [])
            item = _make_example(
                group_id=group_id,
                example_id=f"{raw_id}:memory:{index}",
                text=_user_text(messages) or str(chunk.get("text", "")),
                write_label=float(chunk.get("write_label", 1.0)),
                forget_label=float(chunk.get("forget_label", 0.0)),
                kind=str(chunk.get("kind", "fact")),
                source=source,
                subject=record.get("subject", ""),
                attribute=record.get("attribute", ""),
                value=chunk.get("value", record.get("value", "")),
                messages=messages,
            )
            if item is not None:
                output.append(item)
        query = record.get("query")
        query_text = _user_text(query)
        item = _make_example(
            group_id=group_id,
            example_id=f"{raw_id}:query",
            text=query_text,
            write_label=0.0,
            kind="query",
            source=source,
            subject=record.get("subject", ""),
            attribute=record.get("attribute", ""),
            answer=record.get("answer", ""),
            answerable=record.get("answerable"),
            messages=query if isinstance(query, list) else None,
        )
        if item is not None:
            output.append(item)
        return output

    memory = record.get("memory")
    if isinstance(memory, list):
        for index, item_messages in enumerate(memory):
            item = _make_example(
                group_id=group_id,
                example_id=f"{raw_id}:memory:{index}",
                text=_user_text(item_messages),
                write_label=1.0,
                kind="fact",
                source=source,
                messages=item_messages if isinstance(item_messages, list) else None,
            )
            if item is not None:
                output.append(item)
    query = record.get("query")
    if query is not None:
        item = _make_example(
            group_id=group_id,
            example_id=f"{raw_id}:query",
            text=_user_text(query),
            write_label=0.0,
            kind="query",
            source=source,
            answer=record.get("answer", ""),
            answerable=record.get("answerable"),
            messages=query if isinstance(query, list) else None,
        )
        if item is not None:
            output.append(item)

    event = record.get("memory_event")
    if not output and (record.get("text") is not None or record.get("messages") is not None):
        event = event if isinstance(event, dict) else {}
        item = _make_example(
            group_id=group_id,
            example_id=f"{raw_id}:turn",
            text=_user_text(record.get("messages")) or str(record.get("text", "")),
            write_label=float(event.get("write_label", event.get("write", record.get("write_label", 0.0)))),
            forget_label=float(event.get("forget_label", event.get("forget", record.get("forget_label", 0.0)))),
            kind=str(event.get("kind", record.get("kind", "conversation"))),
            source=source,
            subject=record.get("subject", ""),
            attribute=record.get("attribute", ""),
            value=record.get("value", ""),
            answer=record.get("answer", ""),
            answerable=record.get("answerable"),
            messages=record.get("messages"),
        )
        if item is not None:
            output.append(item)
    return output


def _split_for_group(group_id: str, explicit: str | None, *, eval_ratio: float) -> str:
    if explicit is not None:
        return explicit
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "eval" if value < eval_ratio else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="input JSONL; may be repeated")
    parser.add_argument("--eval-source", action="append", default=[], help="input JSONL forced into eval")
    parser.add_argument("--output-dir", default="data/production_memory")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.eval_ratio < 1.0:
        raise SystemExit("--eval-ratio must be between 0 and 1")

    source_paths = [_project_path(item) for item in (args.source or DEFAULT_SOURCES)]
    eval_paths = [_project_path(item) for item in args.eval_source]
    all_inputs = [(path, None) for path in source_paths] + [(path, "eval") for path in eval_paths]
    examples: list[tuple[str, dict[str, Any]]] = []
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str, float, float, str]] = set()
    for path, forced_split in all_inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        source = str(path)
        name_split = _explicit_split(path, forced=forced_split)
        for line_number, record in _read_jsonl(path):
            normalized = normalize_record(record, source=source, line_number=line_number)
            for item in normalized:
                dedupe_key = (
                    item["group_id"],
                    item["text"],
                    item["write_label"],
                    item["forget_label"],
                    item["kind"],
                )
                if dedupe_key in seen:
                    source_stats[source]["deduplicated"] += 1
                    continue
                seen.add(dedupe_key)
                split = _split_for_group(item["group_id"], name_split, eval_ratio=args.eval_ratio)
                examples.append((split, item))
                source_stats[source][split] += 1

    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    for split in ("train", "eval"):
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item_split, item in examples:
                if item_split == split:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    split_counts[split] += 1
    manifest = {
        "format_version": 1,
        "schema": {
            "text": "current turn presented to the write policy",
            "messages": "optional original chat messages",
            "write_label": "1 durable memory, 0 ordinary query/casual turn",
            "forget_label": "1 explicit correction/forget request",
            "group_id": "conversation/episode identity; never split across train and eval",
        },
        "bootstrap_data_warning": "Default files are local bootstrap/synthetic data; pass real exports with --source for production training.",
        "inputs": [str(path) for path, _ in all_inputs],
        "counts": dict(split_counts),
        "source_stats": {key: dict(value) for key, value in source_stats.items()},
        "dedupe_count": sum(value.get("deduplicated", 0) for value in source_stats.values()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
