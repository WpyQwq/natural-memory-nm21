"""Build (query, records, answerable) training pairs for the open-vocabulary gate.

The coverage gate shipped with NM2.1 is a *closed* 24-class attribute classifier
plus a self-gating rule that disables the gate entirely unless the bank already
holds 90% of that closed vocabulary.  That rule is why the gate does nothing on
any new attribute space: on the realistic eval set the bank holds 5 attributes the
head never saw, so the gate reports "no opinion" and the runtime fabricates an
answer for all 25 unknown-attribute questions (unknown-refusal 52.00%, and the 13
refusals that do happen come from the model's own hedging, not from the gate).

This script produces the data for a gate that is open-vocabulary by construction:
it does not ask "which of my 24 attributes is this?", it asks "does any record in
the bank actually match what this question is about?".  Labels come from the
corpus' own attribute bookkeeping, and the train/eval attribute families are
disjoint, so a head that scores well has generalised rather than memorised.

Each episode yields two examples over the *same* query and the *same* distractor
records:

* POSITIVE -- the answering fact is among the records (label 1)
* NEGATIVE -- the answering facts are removed, distractors remain (label 0)

That pairing is what makes the task learnable at all: the two examples differ
only in whether a matching record is present, so a head cannot win by recognising
the question or the distractors.

Output: a single ``.pt`` with the tensors the trainer needs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.make_realistic_memory_data import ATTRIBUTE_POOL

HIDDEN = 2560
FACTS_PER_EPISODE = 6


def build_frame_regexes() -> dict[str, list[re.Pattern]]:
    out: dict[str, list[re.Pattern]] = {}
    for name, frames, _questions in ATTRIBUTE_POOL:
        pats = []
        for frame in frames:
            parts = [re.escape(p) for p in re.split(r"\{[sv]\}", frame)]
            pats.append(re.compile("".join(p if i == 0 else r"(.+?)" + p for i, p in enumerate(parts)) + r"\Z"))
        out[name] = pats
    return out


FRAME_RE = build_frame_regexes()


def build_question_regexes() -> dict[str, list[re.Pattern]]:
    """Patterns for each family's question templates.

    `unknown_attribute` episodes carry no ``metadata.attribute`` (the generator
    omits it for that category), so the asked attribute has to be recovered from
    the question itself to be able to build the "answer present" contrast.
    """
    out: dict[str, list[re.Pattern]] = {}
    for name, _frames, questions in ATTRIBUTE_POOL:
        pats = []
        for question in questions:
            parts = [re.escape(p) for p in re.split(r"\{q\}", question)]
            pats.append(re.compile("".join(p if i == 0 else r"(.*?)" + p for i, p in enumerate(parts)) + r"\Z"))
        out[name] = pats
    return out


QUESTION_RE = build_question_regexes()


def attribute_of_question(query: str) -> str | None:
    query = query.strip()
    for name, pats in QUESTION_RE.items():
        if any(p.search(query) for p in pats):
            return name
    return None


def belongs_to(text: str, attribute: str) -> bool:
    text = text.strip()
    return any(p.search(text) for p in FRAME_RE.get(attribute, []))


def value_slot(text: str, attribute: str):
    for p in FRAME_RE.get(attribute, []):
        m = p.search(text.strip())
        if m:
            groups = [g for g in m.groups() if g]
            return groups[-1].strip() if groups else None
    return None


class FeatureBank:
    def __init__(self, directory: Path):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.hidden = int(manifest["hidden_size"])
        self.index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        self.features = np.memmap(directory / manifest["bank"], dtype=np.float16, mode="r",
                                  shape=(int(manifest["text_count"]), self.hidden))

    def row(self, text: str):
        return self.index.get(hashlib.sha1(text.encode("utf-8", "replace")).hexdigest())

    def key(self, row: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.features[row], dtype=np.float32))


def load_episodes(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def family_of(text: str) -> str | None:
    """Which attribute family a fact text belongs to, by frame shape."""
    for name, pats in FRAME_RE.items():
        if any(p.search(text.strip()) for p in pats):
            return name
    return None


def write_set(episode: dict, *, include_answer: bool) -> list[str]:
    """Record set for one variant, mirroring the harness write order.

    POSITIVE variant: distractors first, answering facts reserved last -- the same
    order the harness now uses, so the answer is the freshest evidence.

    NEGATIVE variant: only distractors that provably belong to a *different*
    attribute family.  Keeping same-attribute distractors would make the label
    wrong: for `update_conflict` the distractors are stale values of the very
    attribute being asked about, so the bank does still hold the asked attribute
    and refusing would be the wrong behaviour to teach.
    """
    candidates = [str(c.get("text", "")) for c in (episode.get("candidates") or []) if isinstance(c, dict)]
    positives = [candidates[i] for i in (episode.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
    attribute = str((episode.get("metadata") or {}).get("attribute", ""))
    if not attribute:
        attribute = attribute_of_question(str(episode.get("query", ""))) or ""
    other = [c for c in candidates
             if c not in positives and family_of(c) is not None and family_of(c) != attribute]
    if include_answer:
        room = max(0, FACTS_PER_EPISODE - len(positives))
        return other[:room] + positives
    return other[:FACTS_PER_EPISODE]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path(r"H:\Memory\nm_cache\nm_realistic_v2\feature_cache"))
    parser.add_argument("--corpus", type=Path, default=Path("data/realistic_v2/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/memory_answerability_head/pairs_train.pt"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)
    bank = FeatureBank(args.bank)

    query_rows, record_rows, labels, meta = [], [], [], []
    missing = {"query": 0, "record": 0, "episodes": 0}
    added_positive_for_unknown = 0
    failed_unknown = 0
    skipped_empty = 0
    frame_check_disagreements = 0
    per_category: dict[str, dict] = {}

    for episode in load_episodes(args.corpus):
        metadata = episode.get("metadata") or {}
        attribute = str(metadata.get("attribute", ""))
        category = str(metadata.get("category", ""))
        stats = per_category.setdefault(category, {"examples": 0, "positives": 0, "negatives": 0})
        candidates = [str(c.get("text", "")) for c in (episode.get("candidates") or []) if isinstance(c, dict)]
        positives = [candidates[i] for i in (episode.get("positive_indices") or []) if 0 <= int(i) < len(candidates)]
        query = str(episode.get("query", ""))

        qrow = bank.row(query)
        if qrow is None:
            missing["query"] += 1
            continue

        # Label from the corpus' own bookkeeping.  Whether a record belongs to the
        # asked attribute is then checked by frame matching as a *diagnostic* only:
        # the generator phrases some categories outside their family templates
        # ("near_miss" builds its facts from a literal format string), so frame
        # matching is not a sound supervision source.
        if positives:
            variants = [(True, 1.0), (False, 0.0)]
        else:
            asked = attribute or (attribute_of_question(query) or "")
            frames = [f for name, f, _q in ATTRIBUTE_POOL if name == asked]
            synthetic = None
            if frames:
                value = str((metadata.get("acceptable") or ["42"])[0])
                synthetic = frames[0][0].format(s="我", v=value)
                added_positive_for_unknown += 1
            else:
                failed_unknown += 1
            variants = [(True, 1.0), (False, 0.0)]
            positives = [synthetic] if synthetic else []

        for include_answer, label in variants:
            if include_answer and not positives:
                continue
            records = write_set(episode, include_answer=include_answer)
            if not records:
                skipped_empty += 1
                continue
            rows = [bank.row(t) for t in records]
            kept = [(t, r) for t, r in zip(records, rows) if r is not None]
            if len(kept) < len(rows):
                missing["record"] += len(rows) - len(kept)
            if not kept:
                missing["episodes"] += 1
                continue
            texts = [t for t, _ in kept]
            if label > 0.5 and not any(belongs_to(t, attribute) for t in texts):
                frame_check_disagreements += 1
            query_rows.append(qrow)
            record_rows.append(torch.tensor([r for _, r in kept], dtype=torch.long))
            labels.append(label)
            stats["examples"] += 1
            stats["positives" if label > 0.5 else "negatives"] += 1
            meta.append({"attribute": attribute, "category": category, "query": query})

    labels_t = torch.tensor(labels, dtype=torch.float32)
    print(json.dumps({
        "examples": len(labels),
        "positives": int((labels_t > 0.5).sum()),
        "negatives": int((labels_t < 0.5).sum()),
        "synthetic_positives_for_unknown": added_positive_for_unknown,
        "unknown_episodes_without_recoverable_attribute": failed_unknown,
        "variants_skipped_for_having_no_records": skipped_empty,
        "frame_check_disagreements_on_labeled_positives": frame_check_disagreements,
        "missing": missing,
    }, ensure_ascii=False))
    print(f"{'category':<20}{'examples':>10}{'pos':>7}{'neg':>7}")
    for name in sorted(per_category):
        s = per_category[name]
        print(f"{name:<20}{s['examples']:>10}{s['positives']:>7}{s['negatives']:>7}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "query_rows": torch.tensor(query_rows, dtype=torch.long),
        "record_rows": record_rows,
        "labels": labels_t,
        "meta": meta,
        "hidden": bank.hidden,
    }, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
