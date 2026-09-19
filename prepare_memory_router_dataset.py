"""Prepare a leak-resistant, mixed-domain dataset for MemoryRouterV2.

The router is trained on *episodes*, not isolated labels.  Every episode has
one natural-language query, a bounded candidate set, one or more positive
memory records (or no positive for abstention), and a hop label.  The script
keeps train/eval groups disjoint, writes the evaluation file before training,
and records SHA-256 hashes in a manifest.

Local sources are intentionally supported first because they are reproducible
and already contain the project's real failure cases.  Public Hugging Face
sources can be added with ``--include-public`` or ``--hf-source``.  Network
failures are recorded in the manifest and never silently replaced by made-up
public data.

Recommended public sources for a later online refresh:

* HotpotQA: multi-hop open-domain QA;
* MuSiQue: compositional multi-hop QA;
* CodeSearchNet: natural-language/code retrieval;
* FEVER: evidence selection and unknown/unsupported claims;
* QReCC: conversational question rewriting and retrieval.

The generic Hugging Face adapter is deliberately tolerant of schema changes,
but every imported row is still auditable through ``source`` and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_TRAIN_SOURCES = (
    "data/benchmark_train.jsonl",
    "data/native_memory/train.jsonl",
    "data/production_memory/train.jsonl",
    "data/production_memory_hard_v2/train.jsonl",
)
DEFAULT_EVAL_SOURCES = (
    "data/benchmark_eval.jsonl",
    "data/native_memory/eval.jsonl",
    "data/production_memory/eval.jsonl",
    "data/production_memory_hard_v2/eval.jsonl",
    "data/mega_validation/smoke.jsonl",
)

PUBLIC_RECIPES = (
    {
        "name": "hotpotqa_train",
        "dataset_id": "hotpot_qa",
        "config": "distractor",
        "split": "train",
        "split_kind": "train",
        "task": "qa",
        "url": "https://huggingface.co/datasets/hotpot_qa",
    },
    {
        "name": "hotpotqa_validation",
        "dataset_id": "hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "split_kind": "eval",
        "task": "qa",
        "url": "https://huggingface.co/datasets/hotpot_qa",
    },
    {
        "name": "codesearchnet_python_train",
        "dataset_id": "code_search_net",
        "config": "python",
        "split": "train",
        "split_kind": "train",
        "task": "code",
        "url": "https://huggingface.co/datasets/code_search_net",
    },
    {
        "name": "codesearchnet_python_validation",
        "dataset_id": "code_search_net",
        "config": "python",
        "split": "validation",
        "split_kind": "eval",
        "task": "code",
        "url": "https://huggingface.co/datasets/code_search_net",
    },
    {
        "name": "fever_train",
        "dataset_id": "fever",
        "config": "v1.0",
        "split": "train",
        "split_kind": "train",
        "task": "evidence",
        "url": "https://huggingface.co/datasets/fever",
    },
    {
        "name": "fever_validation",
        "dataset_id": "fever",
        "config": "v1.0",
        "split": "labelled_dev",
        "split_kind": "eval",
        "task": "evidence",
        "url": "https://huggingface.co/datasets/fever",
    },
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    # Accept both forms when launched from H:\\Memory or from the package
    # directory itself: ``data/...`` and ``V2_dpskw/data/...``.
    if path.parts and path.parts[0].lower() == PROJECT_ROOT.name.lower():
        path = Path(*path.parts[1:])
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return PROJECT_ROOT / path


def _read_jsonl(path: Path, *, max_rows: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if max_rows and line_number > max_rows:
                break
            raw = raw.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", " ").split()).strip()


def _message_text(value: Any, *, last_user: bool = False) -> str:
    if isinstance(value, str):
        return _clean_text(value)
    if not isinstance(value, list):
        return ""
    messages = [item for item in value if isinstance(item, dict)]
    if last_user:
        for item in reversed(messages):
            if str(item.get("role", "")).lower() == "user":
                return _clean_text(item.get("content", ""))
    parts: list[str] = []
    for item in messages:
        content = _clean_text(item.get("content", ""))
        if content:
            role = str(item.get("role", "user"))
            parts.append(f"[{role}] {content}")
    return "\n".join(parts).strip()


def _first_string(value: Any) -> str:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        for item in value:
            result = _first_string(item)
            if result:
                return result
    if isinstance(value, dict):
        for key in ("text", "content", "answer", "value", "sentence", "paragraph"):
            result = _first_string(value.get(key))
            if result:
                return result
    return ""


def _stable_key(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:20]


@dataclass(frozen=True)
class MemoryItem:
    item_id: str
    text: str
    source: str
    group_id: str
    family: str
    kind: str = "memory"
    entity: str = ""
    attribute: str = ""

    def conflict_key(self) -> str:
        if self.entity and self.attribute:
            return f"{self.entity.strip().lower()}::{self.attribute.strip().lower()}"
        return ""

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "text": self.text,
            "source": self.source,
            "group_id": self.group_id,
            "family": self.family,
            "kind": self.kind,
            "entity": self.entity,
            "attribute": self.attribute,
        }


@dataclass
class RawEpisode:
    episode_id: str
    group_id: str
    source: str
    family: str
    query: str
    positive_ids: list[str]
    local_candidate_ids: list[str]
    need_memory: float
    hop: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Corpus:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, MemoryItem]] = {"train": {}, "eval": {}}
        self.episodes: dict[str, list[RawEpisode]] = {"train": [], "eval": []}
        self.stats: dict[str, Counter[str]] = defaultdict(Counter)

    def add_item(self, split: str, item: MemoryItem) -> None:
        self.items[split].setdefault(item.item_id, item)

    def add_episode(self, split: str, episode: RawEpisode) -> None:
        if not episode.query:
            return
        self.episodes[split].append(episode)
        self.stats[episode.source]["episodes"] += 1
        self.stats[episode.source]["positive"] += int(bool(episode.positive_ids))
        self.stats[episode.source]["unknown"] += int(not episode.positive_ids)


NON_MEMORY_KINDS = {
    "query",
    "question",
    "read",
    "memory_query",
    "hypothetical",
    "forget",
}


def _candidate_allowed(item: MemoryItem) -> bool:
    return item.kind.lower() not in NON_MEMORY_KINDS


def _make_item(
    *,
    source: str,
    family: str,
    group_id: str,
    raw_id: Any,
    text: str,
    kind: str = "memory",
    entity: Any = "",
    attribute: Any = "",
) -> MemoryItem | None:
    text = _clean_text(text)
    if not text:
        return None
    item_id = f"{family}:{_stable_key(source, group_id, raw_id, text)}"
    return MemoryItem(
        item_id=item_id,
        text=text,
        source=source,
        group_id=group_id,
        family=family,
        kind=str(kind or "memory"),
        entity=_clean_text(entity),
        attribute=_clean_text(attribute),
    )


def _add_benchmark_row(corpus: Corpus, split: str, source: str, row_number: int, row: dict[str, Any]) -> None:
    raw_id = row.get("id", f"line-{row_number}")
    family = "benchmark_qa"
    group_id = f"{family}:{raw_id}"
    memory = row.get("memory", [])
    memory_values = memory if isinstance(memory, list) else [memory]
    item_ids: list[str] = []
    for index, value in enumerate(memory_values):
        text = _message_text(value, last_user=True) or _message_text(value)
        item = _make_item(
            source=source,
            family=family,
            group_id=group_id,
            raw_id=f"{raw_id}:memory:{index}",
            text=text,
            kind="fact",
            entity=row.get("subject", ""),
            attribute=row.get("attribute", ""),
        )
        if item is not None:
            corpus.add_item(split, item)
            item_ids.append(item.item_id)
    query = _message_text(row.get("query"), last_user=True) or _message_text(row.get("query"))
    corpus.add_episode(
        split,
        RawEpisode(
            episode_id=f"{group_id}:query",
            group_id=group_id,
            source=source,
            family=family,
            query=query,
            positive_ids=item_ids,
            local_candidate_ids=item_ids,
            need_memory=float(bool(item_ids)),
            hop=min(3, max(1, len(item_ids))) if item_ids else 0,
            metadata={
                "answer": _clean_text(row.get("answer", "")),
                "subject": _clean_text(row.get("subject", "")),
                "attribute": _clean_text(row.get("attribute", "")),
                "task": "single_fact_retrieval",
            },
        ),
    )


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return _message_text(chunk.get("messages"), last_user=True) or _clean_text(chunk.get("text", ""))
    return _message_text(chunk, last_user=True) or _message_text(chunk)


def _add_native_row(corpus: Corpus, split: str, source: str, row_number: int, row: dict[str, Any]) -> None:
    raw_id = row.get("id", f"line-{row_number}")
    family = "native_memory"
    group_id = f"{family}:{raw_id}"
    chunks = row.get("memory_chunks", [])
    if not isinstance(chunks, list):
        chunks = []
    all_ids: list[str] = []
    durable: list[tuple[dict[str, Any], str]] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        text = _chunk_text(chunk)
        item = _make_item(
            source=source,
            family=family,
            group_id=group_id,
            raw_id=f"{raw_id}:chunk:{index}",
            text=text,
            kind=str(chunk.get("kind", "memory")),
            entity=row.get("subject", ""),
            attribute=row.get("attribute", ""),
        )
        if item is None:
            continue
        corpus.add_item(split, item)
        all_ids.append(item.item_id)
        kind = str(chunk.get("kind", "memory")).lower()
        if float(chunk.get("write_label", 0.0) or 0.0) >= 0.5 and kind not in {
            "noise",
            "temporary",
            "quoted_noise",
            "hypothetical",
            "forget",
        }:
            durable.append((chunk, item.item_id))
    query = _message_text(row.get("query"), last_user=True) or _message_text(row.get("query"))
    answerable = row.get("answerable")
    if answerable is False:
        positive_ids: list[str] = []
    else:
        answer = _clean_text(row.get("answer", ""))
        exact = [item_id for chunk, item_id in durable if answer and answer in _chunk_text(chunk)]
        positive_ids = exact[-1:] if exact else ([durable[-1][1]] if durable else [])
    corpus.add_episode(
        split,
        RawEpisode(
            episode_id=f"{group_id}:query",
            group_id=group_id,
            source=source,
            family=family,
            query=query,
            positive_ids=positive_ids,
            local_candidate_ids=all_ids,
            need_memory=float(bool(positive_ids)),
            hop=min(3, max(1, len(positive_ids))) if positive_ids else 0,
            metadata={
                "answer": _clean_text(row.get("answer", "")),
                "subject": _clean_text(row.get("subject", "")),
                "attribute": _clean_text(row.get("attribute", "")),
                "task": "write_replace_read",
            },
        ),
    )


def _add_normalized_policy_file(corpus: Corpus, split: str, source: str, rows: list[tuple[int, dict[str, Any]]]) -> None:
    family = "memory_policy"
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for line_number, row in rows:
        groups[f"{family}:{row.get('group_id', row.get('id', line_number))}"].append((line_number, row))
    for group_id, group_rows in groups.items():
        all_ids: list[str] = []
        durable: list[tuple[dict[str, Any], str]] = []
        for line_number, row in group_rows:
            item = _make_item(
                source=source,
                family=family,
                group_id=group_id,
                raw_id=row.get("id", line_number),
                text=row.get("text", ""),
                kind=row.get("kind", "memory"),
                entity=row.get("subject", ""),
                attribute=row.get("attribute", ""),
            )
            if item is None:
                continue
            corpus.add_item(split, item)
            all_ids.append(item.item_id)
            kind = str(row.get("kind", "memory")).lower()
            if float(row.get("write_label", 0.0) or 0.0) >= 0.5 and kind not in {
                "noise",
                "temporary",
                "quoted_noise",
                "hypothetical",
                "forget",
            }:
                durable.append((row, item.item_id))
        for line_number, row in group_rows:
            kind = str(row.get("kind", "")).lower()
            if kind not in {"query", "question", "read", "memory_query"}:
                continue
            attribute = _clean_text(row.get("attribute", ""))
            answer = _clean_text(row.get("answer", ""))
            exact = [item_id for item_row, item_id in durable if answer and answer in _clean_text(item_row.get("text", ""))]
            same_attribute = [
                item_id
                for item_row, item_id in durable
                if attribute and _clean_text(item_row.get("attribute", "")) == attribute
            ]
            positive_ids = exact[-1:] or same_attribute[-1:]
            if row.get("answerable") is False:
                positive_ids = []
            query = _clean_text(row.get("text", ""))
            corpus.add_episode(
                split,
                RawEpisode(
                    episode_id=f"{group_id}:{row.get('id', line_number)}",
                    group_id=group_id,
                    source=source,
                    family=family,
                    query=query,
                    positive_ids=positive_ids,
                    local_candidate_ids=all_ids,
                    need_memory=float(bool(positive_ids)),
                    hop=min(3, max(1, len(positive_ids))) if positive_ids else 0,
                    metadata={
                        "answer": answer,
                        "subject": _clean_text(row.get("subject", "")),
                        "attribute": attribute,
                        "task": "natural_language_policy",
                    },
                ),
            )


def _expand_evidence_chain(
    positive_ids: list[str],
    evidence: list[tuple[str, str, str]],
    *,
    limit: int = 3,
) -> tuple[list[str], int]:
    """Add the intermediate facts a multi-hop answer depends on.

    The generator labels only the fact containing the final value.  For a chain
    such as ``项目 P -> 负责人 M -> 工号 H`` the router also needs the fact that
    defines ``M``; without it "all evidence in Top-K" cannot be measured and a
    nominally correct answer would be ungrounded.
    """

    by_id = {item_id: (text, value) for item_id, text, value in evidence}
    selected = list(positive_ids)
    selected_set = set(selected)
    frontier = list(positive_ids)
    added = 0
    while frontier and len(selected) < limit:
        current_text = by_id.get(frontier.pop(0), ("", ""))[0]
        if not current_text:
            continue
        for item_id, _text, value in evidence:
            if item_id in selected_set or not value:
                continue
            if value in current_text:
                selected.append(item_id)
                selected_set.add(item_id)
                frontier.append(item_id)
                added += 1
                if len(selected) >= limit:
                    break
    return selected, added


def _add_mega_row(corpus: Corpus, split: str, source: str, row_number: int, row: dict[str, Any]) -> None:
    family = "mega_validation"
    raw_id = row.get("id", f"line-{row_number}")
    group_id = f"{family}:{raw_id}"
    facts = row.get("facts", [])
    if not isinstance(facts, list):
        facts = []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    # The mega generator marks unanswerable rows explicitly, and for those rows
    # ``acceptable`` holds abstention phrases ("不知道" / "没有记录") rather than
    # evidence values.  Treating those as answers made every abstention episode
    # answerable and then the ``not positive_ids`` fallback below labelled *all*
    # of its facts as positive, which silently destroyed the unknown-refusal and
    # multi-hop axes of the evaluation.
    row_answerable = bool(metadata.get("answerable", True))
    acceptable = (
        [_clean_text(value) for value in row.get("acceptable", []) if _clean_text(value)]
        if row_answerable
        else []
    )
    answerable = bool(acceptable)
    ids: list[str] = []
    positive_ids: list[str] = []
    evidence: list[tuple[str, str, str]] = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        text = _clean_text(fact.get("text", ""))
        item = _make_item(
            source=source,
            family=family,
            group_id=group_id,
            raw_id=f"{raw_id}:fact:{index}",
            text=text,
            kind=str(fact.get("kind", "fact")),
            entity=fact.get("entity", row.get("subject", "")),
            attribute=fact.get("attribute", ""),
        )
        if item is None:
            continue
        corpus.add_item(split, item)
        ids.append(item.item_id)
        evidence.append((item.item_id, text, _clean_text(fact.get("value", ""))))
        if answerable and any(value in text for value in acceptable):
            positive_ids.append(item.item_id)
    if answerable and not positive_ids:
        positive_ids = ids[:]
    chain_added = 0
    hop_count = metadata.get("hop_count")
    if positive_ids and hop_count:
        positive_ids, chain_added = _expand_evidence_chain(positive_ids, evidence)
    if positive_ids:
        hop = min(3, int(hop_count)) if isinstance(hop_count, (int, float)) and hop_count else min(3, max(1, len(positive_ids)))
    else:
        hop = 0
    corpus.add_episode(
        split,
        RawEpisode(
            episode_id=f"{group_id}:query",
            group_id=group_id,
            source=source,
            family=family,
            query=_clean_text(row.get("query", "")),
            positive_ids=positive_ids,
            local_candidate_ids=ids,
            need_memory=float(bool(positive_ids)),
            hop=hop,
            metadata={
                "acceptable": acceptable,
                "category": _clean_text(row.get("category", "")),
                "task": "stress_validation",
                "row_answerable": row_answerable,
                "hop_count": hop_count,
                "chain_added": chain_added,
            },
        ),
    )


def _support_titles(value: Any) -> set[str]:
    titles: set[str] = set()
    if isinstance(value, dict):
        raw_titles = value.get("title", value.get("titles", []))
        if isinstance(raw_titles, list):
            titles.update(_clean_text(item) for item in raw_titles if _clean_text(item))
        elif _clean_text(raw_titles):
            titles.add(_clean_text(raw_titles))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and item:
                title = _clean_text(item[0])
                if title:
                    titles.add(title)
            elif isinstance(item, dict):
                title = _clean_text(item.get("title", item.get("document", "")))
                if title:
                    titles.add(title)
    return titles


def _context_candidates(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    output: list[tuple[str, str, str]] = []
    context = row.get("context")
    if isinstance(context, dict) and isinstance(context.get("title"), list):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        for index, title in enumerate(titles):
            sentence_group = sentences[index] if index < len(sentences) else []
            text = " ".join(_clean_text(item) for item in sentence_group) if isinstance(sentence_group, list) else _clean_text(sentence_group)
            if text:
                output.append((_clean_text(title), text, f"context:{index}"))
    for key in ("contexts", "documents", "passages", "evidence", "search_results"):
        values = row.get(key)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if isinstance(value, dict):
                title = _clean_text(value.get("title", value.get("document", value.get("id", ""))))
                text = _first_string(value.get("text", value.get("content", value.get("passage", value.get("snippet", value)))))
            else:
                title = ""
                text = _first_string(value)
            if text:
                output.append((title, text, f"{key}:{index}"))
    # CodeSearchNet and similar code datasets do not call the fields contexts.
    code = _first_string(row.get("whole_func_string", row.get("code", row.get("function", ""))))
    docstring = _first_string(row.get("func_documentation_string", row.get("docstring", "")))
    if code:
        output.append((_clean_text(row.get("repository_name", "")), code, "code:0"))
    if not output and docstring and code:
        output.append(("", code, "code:0"))
    return output


def _qa_answer(row: dict[str, Any]) -> str:
    answers = row.get("answers")
    if isinstance(answers, dict):
        return _first_string(answers.get("text", answers.get("answer", answers)))
    return _first_string(row.get("answer", answers))


def _add_generic_qa_row(
    corpus: Corpus,
    split: str,
    source: str,
    family: str,
    row_number: int,
    row: dict[str, Any],
    task: str,
) -> None:
    query = _first_string(
        row.get(
            "question",
            row.get(
                "query",
                row.get(
                    "claim",
                    row.get(
                        "Question",
                        row.get(
                            "func_documentation_string",
                            row.get("docstring", row.get("documentation", "")),
                        ),
                    ),
                ),
            ),
        )
    )
    contexts = _context_candidates(row)
    if not query or not contexts:
        return
    raw_id = row.get("id", row.get("_id", row_number))
    group_id = f"{family}:{raw_id}"
    answer = _qa_answer(row)
    label = _clean_text(row.get("label", row.get("gold_label", ""))).upper()
    unsupported = label in {"NOT ENOUGH INFO", "NEI", "UNKNOWN", "UNANSWERABLE"} or row.get("answerable") is False
    support_titles = _support_titles(row.get("supporting_facts"))
    positive_ids: list[str] = []
    local_ids: list[str] = []
    for index, (title, text, context_id) in enumerate(contexts):
        item = _make_item(
            source=source,
            family=family,
            group_id=group_id,
            raw_id=f"{raw_id}:{context_id}:{index}",
            text=text,
            kind="code" if task == "code" else "document",
            entity=title,
            attribute=task,
        )
        if item is None:
            continue
        corpus.add_item(split, item)
        local_ids.append(item.item_id)
        if not unsupported and (
            (title and title in support_titles)
            or (answer and answer.lower() in text.lower())
            or (task == "code" and index == 0)
        ):
            positive_ids.append(item.item_id)
    if unsupported:
        positive_ids = []
    corpus.add_episode(
        split,
        RawEpisode(
            episode_id=f"{group_id}:query",
            group_id=group_id,
            source=source,
            family=family,
            query=query,
            positive_ids=list(dict.fromkeys(positive_ids)),
            local_candidate_ids=local_ids,
            need_memory=float(bool(positive_ids)),
            hop=min(3, max(1, len(positive_ids))) if positive_ids else 0,
            metadata={
                "answer": answer,
                "label": label,
                "task": task,
            },
        ),
    )


def _parse_local_file(corpus: Corpus, split: str, path: Path, *, max_rows: int) -> dict[str, Any]:
    source = path.as_posix()
    rows = list(_read_jsonl(path, max_rows=max_rows))
    if not rows:
        return {"source": source, "rows": 0, "kind": "empty"}
    first = rows[0][1]
    if "memory_chunks" in first:
        for line_number, row in rows:
            _add_native_row(corpus, split, source, line_number, row)
        kind = "native_memory"
    elif "memory" in first and "query" in first:
        for line_number, row in rows:
            _add_benchmark_row(corpus, split, source, line_number, row)
        kind = "benchmark"
    elif "facts" in first and "query" in first:
        for line_number, row in rows:
            _add_mega_row(corpus, split, source, line_number, row)
        kind = "mega_validation"
    elif {"group_id", "write_label", "kind"}.issubset(first):
        _add_normalized_policy_file(corpus, split, source, rows)
        kind = "normalized_policy"
    else:
        for line_number, row in rows:
            _add_generic_qa_row(corpus, split, source, "local_qa", line_number, row, "qa")
        kind = "generic_qa"
    return {"source": source, "rows": len(rows), "kind": kind, "split": split}


def _parse_hf_spec(value: str, *, default_split_kind: str = "train") -> dict[str, Any]:
    parts = value.split("|")
    if len(parts) < 2:
        raise ValueError("--hf-source syntax: DATASET_ID|SPLIT|CONFIG(optional)|SPLIT_KIND(optional)")
    dataset_id = parts[0].strip()
    split = parts[1].strip()
    config = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
    split_kind = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else default_split_kind
    return {
        "name": value,
        "dataset_id": dataset_id,
        "split": split,
        "config": config,
        "split_kind": split_kind,
        "task": "qa",
        "url": f"https://huggingface.co/datasets/{dataset_id}",
    }


def _load_hf_rows(spec: dict[str, Any], *, cache_dir: Path, max_rows: int, retries: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - dependency is optional
        return [], f"datasets import failed: {exc}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            dataset = load_dataset(
                spec["dataset_id"],
                spec.get("config"),
                split=spec["split"],
                cache_dir=str(cache_dir),
                trust_remote_code=False,
            )
            rows: list[dict[str, Any]] = []
            for index, row in enumerate(dataset):
                if max_rows and index >= max_rows:
                    break
                if isinstance(row, dict):
                    rows.append(row)
            return rows, None
        except Exception as exc:  # pragma: no cover - depends on network/source
            last_error = exc
            if attempt < retries:
                continue
    return [], f"load failed after {retries} attempts: {last_error}"


def _add_hf_source(corpus: Corpus, spec: dict[str, Any], *, cache_dir: Path, max_rows: int, retries: int) -> dict[str, Any]:
    rows, error = _load_hf_rows(spec, cache_dir=cache_dir, max_rows=max_rows, retries=retries)
    if error:
        return {**spec, "rows": 0, "episodes_before": len(corpus.episodes[spec["split_kind"]]), "error": error}
    split = "eval" if spec["split_kind"].lower() in {"eval", "validation", "test", "dev"} else "train"
    task = str(spec.get("task", "qa"))
    family = f"hf:{spec['dataset_id']}"
    for line_number, row in enumerate(rows, 1):
        _add_generic_qa_row(corpus, split, spec["name"], family, line_number, row, task)
    return {**spec, "rows": len(rows), "split": split, "error": None}


class _ExcludedPool(Sequence):
    """View of ``pool`` with a few known positions removed, without copying.

    The router builder used to materialise ``[item for item in pool if item not in
    blocked]`` for every episode.  With the mega-validation source the family pool
    holds ~2.66M items, so that copy cost ~2.1 s per episode and projected to ~47
    hours for the 80k mega episodes.  This view reproduces the materialised list
    exactly -- same ``len``, same item at every index, same iteration order -- so
    the episode RNG draw and therefore the generated dataset are bit-identical.
    """

    __slots__ = ("_pool", "_blocked")

    def __init__(self, pool: list, blocked: tuple[int, ...]) -> None:
        self._pool = pool
        self._blocked = blocked

    def __len__(self) -> int:
        return len(self._pool) - len(self._blocked)

    def __getitem__(self, index: int) -> Any:
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("pool index out of range")
        blocked = self._blocked
        if not blocked:
            return self._pool[index]
        # Smallest raw index whose filtered position is ``index``.
        low, high = index, index + len(blocked)
        while low < high:
            mid = (low + high) // 2
            if mid - bisect_right(blocked, mid) < index:
                low = mid + 1
            else:
                high = mid
        return self._pool[low]


class PoolPositionIndex:
    """Positions of the few items an episode may exclude from a shared pool.

    Only ids/conflict keys that are positives somewhere in the split are indexed,
    so the auxiliary dictionaries stay small while the per-episode work becomes
    O(number of excluded items) instead of O(pool size).
    """

    def __init__(self, needed_ids: frozenset[str], needed_conflicts: frozenset[str]) -> None:
        self._needed_ids = needed_ids
        self._needed_conflicts = needed_conflicts
        self._by_pool: dict[int, tuple[list, dict[str, int], dict[str, list[int]]]] = {}

    def _index_for(self, pool: list) -> tuple[dict[str, int], dict[str, list[int]]]:
        cached = self._by_pool.get(id(pool))
        if cached is not None:
            return cached[1], cached[2]
        by_id: dict[str, int] = {}
        by_conflict: dict[str, list[int]] = {}
        needed_ids = self._needed_ids
        needed_conflicts = self._needed_conflicts
        for position, item in enumerate(pool):
            if needed_ids and item.item_id in needed_ids:
                by_id[item.item_id] = position
            if needed_conflicts:
                conflict_key = item.conflict_key()
                if conflict_key and conflict_key in needed_conflicts:
                    by_conflict.setdefault(conflict_key, []).append(position)
        # Keep a reference to the pool so its id() cannot be recycled while cached.
        self._by_pool[id(pool)] = (pool, by_id, by_conflict)
        return by_id, by_conflict

    def view(self, pool: list, positive_ids: Iterable[str], blocked_conflicts: Iterable[str]) -> Sequence:
        by_id, by_conflict = self._index_for(pool)
        blocked: list[int] = []
        for item_id in positive_ids:
            position = by_id.get(item_id)
            if position is not None:
                blocked.append(position)
        for conflict_key in blocked_conflicts:
            positions = by_conflict.get(conflict_key)
            if positions:
                blocked.extend(positions)
        if not blocked:
            # Nothing to remove: the raw pool is already the materialised list.
            return pool
        blocked.sort()
        unique: list[int] = []
        previous = -1
        for position in blocked:
            if position != previous:
                unique.append(position)
                previous = position
        return _ExcludedPool(pool, tuple(unique))


def _source_candidates(
    split: str,
    episode: RawEpisode,
    all_items: dict[str, MemoryItem],
    *,
    candidate_count: int,
    seed: int,
    conflict_aware: bool = False,
    eligible_items: list[MemoryItem] | None = None,
    items_by_attribute: dict[str, list[MemoryItem]] | None = None,
    items_by_family: dict[str, list[MemoryItem]] | None = None,
    pool_index: "PoolPositionIndex | None" = None,
) -> dict[str, Any] | None:
    positive_set = set(episode.positive_ids)
    positive_items = [all_items[item_id] for item_id in episode.positive_ids if item_id in all_items]
    if episode.need_memory >= 0.5 and not positive_items:
        return None
    eligible = eligible_items if eligible_items is not None else [item for item in all_items.values() if _candidate_allowed(item)]
    attribute = str(episode.metadata.get("attribute", ""))
    # A positive episode must be identifiable from the information available
    # to the runtime router.  The old protocol mixed independent memory banks
    # into one candidate set; for generic subjects such as "验证用户", this
    # created several mutually exclusive values for the same attribute and
    # then forced the router to guess one of them.  In the real runtime,
    # superseded records are not active candidates, so exclude non-positive
    # records from the same entity/attribute conflict group.  Unknown
    # episodes intentionally keep these hard negatives because they teach
    # abstention.
    positive_conflicts = (
        {item.conflict_key() for item in positive_items if item.conflict_key()}
        if (conflict_aware and positive_set)
        else set()
    )
    blocked_conflicts = positive_conflicts if (conflict_aware and positive_set) else set()

    def is_blocked(item: MemoryItem) -> bool:
        """True when the old code would have removed this item from a pool."""

        if item.item_id in positive_set:
            return True
        conflict_key = item.conflict_key()
        return bool(conflict_key) and conflict_key in blocked_conflicts

    local_items = [
        all_items[item_id]
        for item_id in episode.local_candidate_ids
        if item_id in all_items
        and item_id not in positive_set
        and _candidate_allowed(all_items[item_id])
        and not is_blocked(all_items[item_id])
    ]
    if items_by_attribute is not None:
        attr_pool = items_by_attribute.get(attribute, [])
        same_attr = (
            pool_index.view(attr_pool, positive_set, blocked_conflicts)
            if pool_index is not None
            else [item for item in attr_pool if not is_blocked(item)]
        )
    else:
        same_attr = [item for item in eligible if not is_blocked(item) and attribute and item.attribute == attribute]
    if items_by_family is not None:
        family_pool = items_by_family.get(episode.family, [])
        same_family = (
            pool_index.view(family_pool, positive_set, blocked_conflicts)
            if pool_index is not None
            else [item for item in family_pool if not is_blocked(item)]
        )
    else:
        same_family = [item for item in eligible if not is_blocked(item) and item.family == episode.family]

    def allowed(item: MemoryItem) -> bool:
        if item.item_id in positive_set:
            return False
        conflict_key = item.conflict_key()
        return not conflict_key or conflict_key not in blocked_conflicts

    rng = random.Random(int(_stable_key(seed, split, episode.episode_id), 16) % (2**32))
    ordered: list[MemoryItem] = []
    seen: set[str] = set(positive_set)

    def add_pool(pool: Iterable[MemoryItem], limit: int = 0) -> None:
        if isinstance(pool, (list, _ExcludedPool)):
            total = len(pool)
        else:
            pool = list(pool)
            total = len(pool)
        draw = max(256, limit * 8) if limit else 0
        if draw and total > draw:
            # Avoid copying and shuffling the full global corpus for every
            # episode.  This keeps large synthetic banks close to O(k) per
            # episode while still drawing stable, reproducible negatives.
            # Sampling indices is identical to the old
            # ``rng.sample(range(len(values)), draw)`` draw.
            values = [pool[index] for index in rng.sample(range(total), draw)]
        else:
            values = list(pool)
            rng.shuffle(values)
        count = 0
        for item in values:
            if not allowed(item) or item.item_id in seen:
                continue
            ordered.append(item)
            seen.add(item.item_id)
            count += 1
            if limit and count >= limit:
                break

    # Preserve conflict and same-domain negatives before random background.
    add_pool(local_items, max(2, candidate_count // 4))
    add_pool(same_attr, max(2, candidate_count // 4))
    add_pool(same_family, max(2, candidate_count // 2))
    add_pool(eligible, max(2, candidate_count))
    candidates = positive_items + ordered[: max(0, candidate_count - len(positive_items))]
    if not candidates:
        return None
    rng.shuffle(candidates)
    positive_indices = [index for index, item in enumerate(candidates) if item.item_id in positive_set]
    if episode.need_memory >= 0.5 and not positive_indices:
        return None
    return {
        "id": episode.episode_id,
        "group_id": episode.group_id,
        "source": episode.source,
        "family": episode.family,
        "query": episode.query,
        "candidates": [item.as_json() for item in candidates],
        "positive_indices": positive_indices,
        "positive_index": positive_indices[0] if positive_indices else -1,
        "need_memory": float(episode.need_memory),
        "hop": int(episode.hop),
        "metadata": episode.metadata,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_split(
    corpus: Corpus,
    split: str,
    output_path: Path,
    *,
    candidate_count: int,
    seed: int,
    max_episodes: int,
    conflict_aware: bool = False,
) -> tuple[int, Counter[str]]:
    counters: Counter[str] = Counter()
    eligible_items = [item for item in corpus.items[split].values() if _candidate_allowed(item)]
    items_by_attribute: dict[str, list[MemoryItem]] = defaultdict(list)
    items_by_family: dict[str, list[MemoryItem]] = defaultdict(list)
    for item in eligible_items:
        if item.attribute:
            items_by_attribute[item.attribute].append(item)
        items_by_family[item.family].append(item)
    # Only ids/conflict keys that are positives somewhere need position lookups,
    # which keeps the lazy pool views cheap even for a 2.6M-item family pool.
    needed_ids: set[str] = set()
    needed_conflicts: set[str] = set()
    for episode in corpus.episodes[split]:
        for item_id in episode.positive_ids:
            needed_ids.add(item_id)
            if conflict_aware:
                item = corpus.items[split].get(item_id)
                if item is not None:
                    conflict_key = item.conflict_key()
                    if conflict_key:
                        needed_conflicts.add(conflict_key)
    pool_index = PoolPositionIndex(frozenset(needed_ids), frozenset(needed_conflicts))
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in corpus.episodes[split]:
            if max_episodes and written >= max_episodes:
                break
            row = _source_candidates(
                split,
                episode,
                corpus.items[split],
                candidate_count=candidate_count,
                seed=seed,
                conflict_aware=conflict_aware,
                eligible_items=eligible_items,
                items_by_attribute=items_by_attribute,
                items_by_family=items_by_family,
                pool_index=pool_index,
            )
            if row is None:
                counters["dropped"] += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            counters["kept"] += 1
            counters["unknown"] += int(not row["positive_indices"])
            counters[f"family:{row['family']}"] += 1
    counters["written"] = written
    return written, counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/router_training")
    parser.add_argument("--train-source", action="append", default=None)
    parser.add_argument("--eval-source", action="append", default=None)
    parser.add_argument("--include-public", action="store_true", help="attempt the built-in public Hugging Face recipes")
    parser.add_argument("--hf-source", action="append", default=[], help="DATASET_ID|SPLIT|CONFIG(optional)|SPLIT_KIND(optional)")
    parser.add_argument("--hf-cache-dir", default="data/_hf_cache")
    parser.add_argument("--hf-max-rows", type=int, default=0)
    parser.add_argument("--hf-retries", type=int, default=3)
    parser.add_argument("--source-max-rows", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-eval-episodes", type=int, default=0)
    parser.add_argument(
        "--conflict-aware",
        action="store_true",
        help="exclude non-positive records from the same entity/attribute conflict group for answerable episodes",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.candidate_count < 2:
        raise SystemExit("--candidate-count must be at least 2")
    if args.hf_retries < 1:
        raise SystemExit("--hf-retries must be positive")

    random.seed(args.seed)
    corpus = Corpus()
    source_reports: list[dict[str, Any]] = []
    train_sources = args.train_source if args.train_source is not None else list(DEFAULT_TRAIN_SOURCES)
    eval_sources = args.eval_source if args.eval_source is not None else list(DEFAULT_EVAL_SOURCES)

    for split, sources in (("train", train_sources), ("eval", eval_sources)):
        for raw_path in sources:
            path = _resolve_path(raw_path)
            if not path.exists():
                raise FileNotFoundError(path)
            report = _parse_local_file(corpus, split, path, max_rows=args.source_max_rows)
            source_reports.append(report)

    hf_specs: list[dict[str, Any]] = []
    if args.include_public:
        hf_specs.extend(PUBLIC_RECIPES)
    hf_specs.extend(_parse_hf_spec(value) for value in args.hf_source)
    hf_cache_dir = _resolve_path(args.hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    public_reports: list[dict[str, Any]] = []
    for spec in hf_specs:
        public_reports.append(
            _add_hf_source(
                corpus,
                spec,
                cache_dir=hf_cache_dir,
                max_rows=args.hf_max_rows,
                retries=args.hf_retries,
            )
        )

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    train_count, train_stats = _write_split(
        corpus,
        "train",
        train_path,
        candidate_count=args.candidate_count,
        seed=args.seed,
        max_episodes=args.max_train_episodes,
        conflict_aware=args.conflict_aware,
    )
    eval_count, eval_stats = _write_split(
        corpus,
        "eval",
        eval_path,
        candidate_count=args.candidate_count,
        seed=args.seed + 1,
        max_episodes=args.max_eval_episodes,
        conflict_aware=args.conflict_aware,
    )

    train_groups = {row.group_id for row in corpus.episodes["train"]}
    eval_groups = {row.group_id for row in corpus.episodes["eval"]}
    overlap = sorted(train_groups & eval_groups)
    if overlap:
        raise RuntimeError(f"train/eval group leakage detected: {overlap[:5]}")
    manifest = {
        "format_version": 3 if args.conflict_aware else 2,
        "generator": "prepare_memory_router_dataset.py",
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "schema": {
            "query": "natural-language routing query",
            "candidates": "bounded memory records with text and provenance",
            "positive_indices": "one or more supporting memory records; empty means abstain",
            "need_memory": "1 if evidence is required and present, 0 for unknown/unsupported queries",
            "hop": "0 for abstention, otherwise number of supporting records clipped to router max_hops",
            "group_id": "conversation or QA episode identity; no group may cross train/eval",
        },
        "files": {
            "train": {"path": str(train_path), "episodes": train_count, "sha256": _sha256(train_path)},
            "eval": {"path": str(eval_path), "episodes": eval_count, "sha256": _sha256(eval_path)},
        },
        "counts": {
            "train_groups": len(train_groups),
            "eval_groups": len(eval_groups),
            "train_episodes": train_count,
            "eval_episodes": eval_count,
            "train_unknown": train_stats["unknown"],
            "eval_unknown": eval_stats["unknown"],
            "train_candidates": sum(len(row["candidates"]) for _, row in _read_jsonl(train_path)),
            "eval_candidates": sum(len(row["candidates"]) for _, row in _read_jsonl(eval_path)),
        },
        "local_sources": source_reports,
        "public_sources": public_reports,
        "public_catalog": list(PUBLIC_RECIPES),
        "split_stats": {"train": dict(train_stats), "eval": dict(eval_stats)},
        "leakage_check": {"group_overlap": len(overlap), "passed": not overlap},
        "conflict_policy": {
            "enabled": bool(args.conflict_aware),
            "answerable_positive_conflicts_excluded": bool(args.conflict_aware),
            "unknown_hard_conflicts_retained": bool(args.conflict_aware),
        },
        "evaluation_policy": "eval.jsonl is generated and hashed before router training; the trainer refuses an optional hash mismatch.",
        "warnings": [
            "Local generated memory-policy files are useful hard negatives but are not public-human chat data.",
            "Add redacted real user traces only after consent and PII removal.",
            "Public download failures are recorded; failed sources contribute zero rows.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "eval.sha256").write_text(manifest["files"]["eval"]["sha256"] + "\n", encoding="ascii")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
