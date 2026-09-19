"""Generate zero-lexical-overlap paraphrase episodes for router training.

The end-to-end test in ``E2E_FINDINGS.md`` located exactly one generalisation gap:
queries that share **no distinctive characters** with the fact they refer to.  The
shipped v6 paraphrase data still shares the attribute wording
("喜欢的水果" vs "喜欢的水果"), so the router learned lexical+semantic matching and
is roughly a coin flip on genuinely semantic paraphrases (top-k=1: 43.75% of
answers cited the wrong attribute).

This generator builds training episodes for that condition and *verifies the
property programmatically* instead of trusting hand-written intent:

* every query/fact pair is checked for shared distinctive characters (a small
  stop-word set such as 我/的/是 is ignored, matching how the runtime's lexical
  paths treat non-distinctive terms);
* candidates are 31 facts with the **same sentence shape but a different
  attribute**, so the only way to succeed is semantic attribute matching;
* values are random codes that appear in exactly one fact, so evidence mixing is
  detectable;
* a share of episodes are unanswerable (the paraphrase refers to an attribute that
  is absent from the candidates) to keep the abstention behaviour intact.

Usage::

    python -m V2_dpskw.make_zero_overlap_paraphrase_data ^
        --output-dir data/zero_overlap --train-episodes 1200 --eval-episodes 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

#: (attribute, [paraphrases that avoid the attribute's distinctive characters])
#:
#: Three variants per attribute: the first two are the training queries and the third
#: is held out for evaluation, so no query *string* is shared between the splits and
#: the eval measures generalisation to an unseen paraphrase of a known attribute
#: rather than memorisation of the training strings.
ATTRIBUTE_PARAPHRASES: list[tuple[str, list[str]]] = [
    ("常住城市", ["我平时待得最久的地儿是哪里？", "我长期落脚的地儿在哪儿？",
                  "我待得最久的那块地儿是哪里？"]),
    ("出生城市", ["我老家在哪儿？", "我小时候成长的地儿是哪里？",
                  "我小时候待的地儿是哪里？"]),
    ("办公城市", ["我每天上班要去哪儿？", "我干活儿的地儿是哪儿？",
                  "我白天上班的地儿在哪儿？"]),
    ("档案标识", ["系统派给我的那串字符是什么？", "他们给我的那串字段是多少？",
                  "系统分给我的那串字符是啥？"]),
    ("常用编辑器", ["我写代码靠什么？", "我做开发时靠什么？",
                    "我写代码靠哪个软件？"]),
    ("默认语言", ["我平时讲哪种话？", "我跟人交流用哪门话？",
                  "我跟人交流讲哪种话？"]),
    ("通勤方式", ["我早上怎么去公司？", "我每天路上靠什么？",
                  "我每天如何到公司？"]),
    ("主管姓名", ["谁带我？", "我归谁带？", "谁是我上级？"]),
    ("工位楼层", ["我坐在哪儿，第几间？", "我上班待的格子是哪个？",
                  "我坐在哪一间房？"]),
    ("团队名称", ["我属于哪个小组？", "我在哪个组干活？", "我归在哪个小组里？"]),
    ("邮箱域名", ["别人给我发信要写哪个后缀？", "我收信地址的后半段是什么？",
                  "收信时我地址的后半段是啥？"]),
    ("手机尾号", ["我随身那台设备的末几位数字是啥？", "那串数字的最后几位是啥？",
                  "我随身那台设备的末端数字是啥？"]),
    ("项目代号", ["我正在做的那个工程叫什么？", "我手上那摊活儿叫什么？",
                  "我手上那摊活儿叫啥？"]),
    ("入职年份", ["我什么时候开始在这儿干活的？", "我从何时起在这家公司上班？",
                  "我何时开始在这家公司上班？"]),
    ("紧急联系人姓氏", ["出事时该喊谁来？", "万一出岔子该找哪一户？",
                        "万一出岔子该找哪一个？"]),
    ("午餐偏好", ["我白天那顿想吃什么？", "我白天那顿想尝什么味道？",
                  "我白天那顿打算吃啥？"]),
    ("运动习惯", ["我平时怎么锻炼？", "我靠什么门路保持体力？",
                  "我每天都在练些什么？"]),
    ("阅读工具", ["我看电子书靠什么？", "我看书靠什么设备？",
                  "我翻书时依赖什么？"]),
    ("起床时间", ["我每天几点醒？", "我早上几点睁眼？", "我闹钟设在哪一刻？"]),
    ("咖啡口味", ["我早上那杯要什么风格？", "我早上那杯想喝哪种？",
                  "我早上那杯偏好哪种豆子？"]),
    ("宿舍楼号", ["我住在哪一栋？", "我睡觉的地儿是第几栋？",
                  "我睡觉的地儿是第几排？"]),
    ("课程名称", ["我最近在学什么？", "我报的那门学的是什么？",
                  "我最近报的那门教的是什么？"]),
    ("客户名称", ["我在跟哪家公司打交道？", "我服务的那家叫什么？",
                  "我服务的对象是哪家？"]),
    ("设备型号", ["我用的那台是哪种款？", "我拿的那台是哪种档？",
                  "我使用的机器是哪一档？"]),
]

#: Characters that do not count as distinctive evidence for the lexical paths.
STOP_CHARS = set("我的了是在有个吗？。！，、你他她它和与及为以及把被这那哪些什么哪儿哪儿")


def distinctive(text: str) -> set[str]:
    return {char for char in text if char.strip() and char not in STOP_CHARS and not char.isascii()}


def overlap_ratio(query: str, fact: str) -> float:
    fact_chars = distinctive(fact)
    if not fact_chars:
        return 0.0
    return len(distinctive(query) & fact_chars) / len(fact_chars)


def code_for(rng: random.Random, attribute: str) -> str:
    prefix = "VAL"
    return "%s-%s" % (prefix, "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8)))


def fact_for(attribute: str, code: str) -> str:
    return "我的%s是 %s。" % (attribute, code)


def audit_variants() -> tuple[dict[str, list[str]], list[dict], list[dict]]:
    """Verify zero distinctive-character overlap with each paraphrase's OWN fact.

    The property that makes this dataset meaningful is that the query shares no
    distinctive character with *the fact it refers to*, so lexical matching cannot
    succeed and only semantic attribute matching can.  Overlap with an unrelated
    candidate is deliberately tolerated: picking that candidate answers the question
    **wrongly**, so such facts act as traps that make the task harder rather than
    easier.  Only own-fact overlap (or an attribute left with no usable query at all)
    is fatal, and every violation is reported at once.
    """
    facts = {name: distinctive(fact_for(name, "VAL-ABCDEFGH")) for name, _ in ATTRIBUTE_PARAPHRASES}
    variants: dict[str, list[str]] = {}
    violations: list[dict] = []
    traps: list[dict] = []
    for attribute, paraphrases in ATTRIBUTE_PARAPHRASES:
        chars = distinctive(fact_for(attribute, "VAL-ABCDEFGH"))
        survivors: list[str] = []
        for paraphrase in paraphrases:
            query_chars = distinctive(paraphrase)
            shared_own = sorted(query_chars & chars)
            if shared_own:
                violations.append({"attribute": attribute, "paraphrase": paraphrase,
                                   "shared_with_own_fact": shared_own})
                continue
            survivors.append(paraphrase)
            hits = sorted(name for name, other in facts.items()
                          if name != attribute and query_chars & other)
            if hits:
                traps.append({"attribute": attribute, "paraphrase": paraphrase,
                              "lexically_matches": hits})
        variants[attribute] = survivors
    unusable = sorted(name for name, options in variants.items() if not options)
    if violations or unusable:
        raise SystemExit(json.dumps(
            {"attributes_without_usable_paraphrase": unusable, "own_fact_violations": violations},
            ensure_ascii=False, indent=2))
    return variants, violations, traps


def build_episodes(count: int, seed: int, *, variants: dict[str, list[str]],
                   candidate_count: int, unknown_ratio: float) -> list[dict]:
    rng = random.Random(seed)
    attributes = [name for name, _ in ATTRIBUTE_PARAPHRASES]
    episodes: list[dict] = []
    for index in range(count):
        target = rng.choice(attributes)
        paraphrase = rng.choice(variants[target])
        code = code_for(rng, target)
        fact = fact_for(target, code)
        ratio = overlap_ratio(paraphrase, fact)
        if ratio > 0.0:
            raise SystemExit(
                f"paraphrase for {target!r} shares distinctive characters with its fact "
                f"(ratio {ratio:.2f}): {paraphrase!r} vs {fact!r}"
            )
        unknown = rng.random() < unknown_ratio
        if unknown:
            # Refer to an attribute that is deliberately absent from the candidates.
            absent = rng.choice([name for name in attributes if name != target])
            absent_paraphrase = rng.choice(variants[absent])
            query = absent_paraphrase
            positives: list[int] = []
        else:
            query = paraphrase
            positives = [0]
        pool = [name for name in attributes if name != (absent if unknown else target)]
        rng.shuffle(pool)
        candidates = [{"text": fact, "kind": "fact", "entity": "user", "attribute": target}]
        for other in pool[: candidate_count - 1]:
            candidates.append({
                "text": fact_for(other, code_for(rng, other)),
                "kind": "fact", "entity": "user", "attribute": other,
            })
        rng.shuffle(candidates)
        if not unknown:
            positives = [next(i for i, c in enumerate(candidates) if c["attribute"] == target)]
        episodes.append({
            "id": "zero-overlap-%06d" % index,
            "group_id": "zero-overlap-%06d" % index,
            "source": "zero_overlap_paraphrase",
            "family": "zero_overlap_paraphrase",
            "query": query,
            "candidates": candidates,
            "positive_indices": positives,
            "positive_index": positives[0] if positives else -1,
            "need_memory": 1.0 if positives else 0.0,
            "hop": 1 if positives else 0,
            "metadata": {
                "category": "zero_overlap_paraphrase",
                "attribute": target,
                "answer": code,
                "acceptable": [code],
                "answerable": not unknown,
            },
        })
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/zero_overlap")
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--unknown-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    # Guarantee, not hope: every paraphrase is checked against its own fact up front
    # (all violations reported at once), then the finished corpus is re-verified.
    variants, violations, traps = audit_variants()
    attribute_count = len(ATTRIBUTE_PARAPHRASES)
    candidate_count = min(args.candidate_count, attribute_count)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_variants = {name: options[:-1] for name, options in variants.items()}
    eval_variants = {name: options[-1:] for name, options in variants.items()}
    train_queries = {q for options in train_variants.values() for q in options}
    eval_queries = {q for options in eval_variants.values() for q in options}
    shared_queries = sorted(train_queries & eval_queries)
    if shared_queries:
        raise SystemExit(json.dumps({"query_strings_shared_between_splits": shared_queries},
                                    ensure_ascii=False, indent=2))
    train = build_episodes(args.train_episodes, args.seed, variants=train_variants,
                           candidate_count=candidate_count, unknown_ratio=args.unknown_ratio)
    evaluation = build_episodes(args.eval_episodes, args.seed + 1, variants=eval_variants,
                                candidate_count=candidate_count, unknown_ratio=args.unknown_ratio)

    def write(path: Path, rows: list[dict]) -> str:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    train_sha = write(out / "train.jsonl", train)
    eval_sha = write(out / "eval.jsonl", evaluation)

    # Independent verification on the written corpus: the query must share no
    # distinctive character with its own target fact, while overlap with unrelated
    # candidates is counted as dataset hardness (those candidates are wrong answers).
    worst_target = 0.0
    worst_distractor = 0.0
    trapped_rows = 0
    for row in train + evaluation:
        target_text = None
        if row["positive_indices"]:
            target_text = row["candidates"][row["positive_index"]]["text"]
            worst_target = max(worst_target, overlap_ratio(row["query"], target_text))
        row_worst = 0.0
        for index, candidate in enumerate(row["candidates"]):
            if index == row["positive_index"]:
                continue
            row_worst = max(row_worst, overlap_ratio(row["query"], candidate["text"]))
        if row_worst > 0.0:
            trapped_rows += 1
        worst_distractor = max(worst_distractor, row_worst)
    manifest = {
        "generator": "make_zero_overlap_paraphrase_data.py",
        "purpose": "train the router on paraphrase queries with no distinctive-character overlap",
        "attributes": attribute_count,
        "usable_paraphrases": sum(len(options) for options in variants.values()),
        "candidate_count": candidate_count,
        "unknown_ratio": args.unknown_ratio,
        "query_string_split": {
            "train": sorted(train_queries),
            "eval": sorted(eval_queries),
            "shared_query_strings": shared_queries,
            "disjoint": not shared_queries,
            "note": ("eval queries are paraphrases the router never saw, so the eval "
                     "measures generalisation to an unseen phrasing, not memorisation"),
        },
        "train": {"episodes": len(train), "sha256": train_sha,
                  "unknown": sum(1 for r in train if not r["positive_indices"])},
        "eval": {"episodes": len(evaluation), "sha256": eval_sha,
                 "unknown": sum(1 for r in evaluation if not r["positive_indices"])},
        "target_overlap_check": {
            "max_query_target_overlap_ratio": worst_target,
            "passed": worst_target == 0.0,
        },
        "hardness": {
            "paraphrases_lexically_matching_an_unrelated_attribute": len(traps),
            "episodes_with_at_least_one_lexical_distractor":
                round(100.0 * trapped_rows / max(1, len(train) + len(evaluation)), 2),
            "max_query_distractor_overlap_ratio": worst_distractor,
        },
        "trap_examples": traps[:12],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if worst_target > 0.0:
        raise SystemExit("overlap check failed: a query shares distinctive characters with its own target fact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
