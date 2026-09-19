"""Build a *realistic-shaped* memory evaluation corpus.

Why: every number this project has produced so far rests on a template corpus -- one
sentence shape ("我的X是 VAL-…。"), a single entity (user), 24 attributes and no updates,
conflicts, multi-hop chains or noise.  Those numbers are internally consistent and
reproducible, but none of them predict behaviour on text that looks like a real user's.
This generator produces the missing population.

Eight categories, each generating the same episode schema the existing harnesses consume
(``query`` / ``candidates[].text`` / ``positive_indices`` / ``metadata.acceptable``):

  multi_entity        同一属性在多个实体上出现，问题只问其中一个
  alias_paraphrase    提问与事实无字面重叠（换一种说法）
  update_conflict     同一实体+属性写过两次，正确答案是较新那个，旧值是强干扰
  multi_hop           答案需要串联两条事实（两条都是正例）
  near_miss           候选属性名高度相近（手机尾号 / 座机尾号）
  noise_context       库里混入大量无关闲聊，问题仍然可回答
  long_fact           事实是长句，值嵌在句子中间
  unknown_attribute   问一个从未写过的属性（不可回答，必须拒答）

Usage::

    python -m V2_dpskw.make_realistic_memory_eval --output data/realistic_eval.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

#: entity -> (subject phrase used inside facts, alias list used by questions)
ENTITIES = [
    ("user", ["我", "我本人"], ["我", "我自己"]),
    ("zhangsan", ["张三", "我们组的张三"], ["张三", "我们组那位张三"]),
    ("liworks", ["李工", "运维的李工"], ["李工", "运维那位李工"]),
    ("proj_alpha", ["项目 Alpha", "Alpha 项目"], ["Alpha", "Alpha 项目"]),
    ("client_beta", ["客户 Beta", "Beta 客户"], ["Beta", "Beta 客户"]),
]

#: attribute -> (fact frames, question frames without lexical overlap)
ATTRIBUTES = [
    ("值班电话", ["{s}的值班电话是 {v}。", "{s}留的值班号码为 {v}。"],
     ["{q}有急事按哪个号找人？", "{q}轮班时该拨什么号码？"]),
    ("常驻机房", ["{s}常驻机房在 {v}。", "{s}平时待的机房是 {v}。"],
     ["{q}平时在哪间屋子干活？", "{q}固定在哪个位置办公？"]),
    ("发布窗口", ["{s}的发布窗口是 {v}。", "{s}固定 {v} 做上线。"],
     ["{q}一般什么时候做上线？", "{q}挑哪个时段变更？"]),
    ("接口版本号", ["{s}对接的接口版本号是 {v}。", "{s}目前跑的是 {v} 版本接口。"],
     ["{q}现在对接的是哪一版？", "{q}联调用的那个版本是什么？"]),
    ("告警阈值", ["{s}的告警阈值是 {v}。", "{s}超过 {v} 就报警。"],
     ["{q}到多少会触发报警？", "{q}哪条线一破就叫？"]),
    ("应急联系人", ["{s}的应急联系人是 {v}。", "{s}出事找 {v}。"],
     ["{q}出问题该喊谁？", "{q}紧急情况下找谁接手？"]),
    ("主库地址", ["{s}的主库地址是 {v}。", "{s}主库落在 {v}。"],
     ["{q}数据落在哪一个地址？", "{q}写库连的是哪里？"]),
    ("缓存容量", ["{s}的缓存容量是 {v}。", "{s}缓存开到 {v}。"],
     ["{q}内存开到多大？", "{q}缓冲配了多少？"]),
]

#: attribute pairs that are deliberately easy to confuse
NEAR_MISS = [
    ("手机尾号", "座机尾号"),
    ("办公城市", "常住城市"),
    ("工位楼层", "宿舍楼层"),
    ("项目代号", "客户代号"),
]

_VALUES = ["分机 8821", "B3-204", "每周三 22:00", "v3.14.2", "91.5%", "王工 138****6621",
           "10.20.3.7:5432", "64GB", "A-7719", "夜间 02:00", "v2.8.0", "78%",
           "李工 139****3344", "10.20.9.1:6379", "128GB", "B-3310"]

_ALIAS_FACTS = [
    ("我平时待得最久的那座城是 {v}。", "我常年落脚在哪儿？"),
    ("我每天睡醒最早看的那个数字是 {v}。", "我那个数字是多少？"),
    ("我手上那台机器是 {v}。", "我用的那台是什么款？"),
    ("我每天早上灌下去的那杯偏爱 {v}。", "我早上那杯是什么口味？"),
]

_QUESTION_POOL = ["{q}这个是什么？", "{q}对应的值是多少？"]


def _value(rng: random.Random) -> str:
    return rng.choice(_VALUES)


def _episode(index: int, category: str, query: str, candidates: list[str],
             positive_texts: list[str], acceptable: list[str], *, hop: int = 1,
             extra: dict | None = None) -> dict:
    positives = [position for position, text in enumerate(candidates) if text in positive_texts]
    metadata = {
        "category": category,
        "answerable": bool(positives),
        "acceptable": acceptable if positives else [],
        "hop_count": hop if positives else 0,
        "answer": acceptable[0] if acceptable and positives else "",
    }
    if extra:
        metadata.update(extra)
    return {
        "id": "realistic-%s-%05d" % (category, index),
        "group_id": "realistic-%s-%05d" % (category, index),
        "source": "realistic_memory_eval",
        "family": category,
        "query": query,
        "candidates": [{"text": text, "kind": "fact", "entity": "user", "attribute": ""}
                       for text in candidates],
        "positive_indices": positives,
        "positive_index": positives[0] if positives else -1,
        "need_memory": 1.0 if positives else 0.0,
        "hop": hop if positives else 0,
        "metadata": metadata,
    }


def build(count_per_category: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    episodes: list[dict] = []
    counter = 0

    for _ in range(count_per_category):
        # --- multi_entity: same attribute on several entities, one is asked about -----
        attribute, fact_frames, question_frames = rng.choice(ATTRIBUTES)
        speakers = rng.sample(ENTITIES, 3)
        target = rng.choice(speakers)
        candidates, values = [], {}
        for key, subjects, aliases in speakers:
            value = _value(rng)
            values[key] = value
            candidates.append(rng.choice(fact_frames).format(s=subjects[0], v=value))
        rng.shuffle(candidates)
        fact = rng.choice(fact_frames).format(s=target[1][0], v=values[target[0]])
        # Rebuild so the target's own phrasing is present verbatim among candidates.
        candidates = []
        for key, subjects, aliases in speakers:
            frame = fact_frames[0] if key == target[0] else fact_frames[1 % len(fact_frames)]
            candidates.append(frame.format(s=subjects[0], v=values[key]))
        rng.shuffle(candidates)
        fact = [c for c in candidates if values[target[0]] in c][0]
        query = rng.choice(question_frames).format(q=rng.choice(target[2]))
        episodes.append(_episode(counter, "multi_entity", query, candidates, [fact],
                                 [values[target[0]]], extra={"attribute": attribute}))
        counter += 1

        # --- alias_paraphrase: fact and question share no distinctive characters -------
        # The target fact is the ONLY fact written in its own frame; every distractor uses
        # a different attribute's frame, so the question identifies exactly one value.
        # (An earlier version wrote the same frame with five different values, which is not
        # a paraphrase test at all but a self-contradictory bank.)
        target_index = rng.randrange(len(ATTRIBUTES))
        template, question = rng.choice(_ALIAS_FACTS)
        value = _value(rng)
        fact = template.format(v=value)
        others = [ATTRIBUTES[i][1][0].format(s="我", v=_value(rng))
                  for i in range(len(ATTRIBUTES)) if i != target_index][:4]
        candidates = [fact] + others
        rng.shuffle(candidates)
        episodes.append(_episode(counter, "alias_paraphrase", question, candidates, [fact], [value]))
        counter += 1

        # --- update_conflict: older value is the strong distractor ---------------------
        attribute, fact_frames, question_frames = rng.choice(ATTRIBUTES)
        old_value, new_value = _value(rng), _value(rng)
        while new_value == old_value:
            new_value = _value(rng)
        old_fact = fact_frames[0].format(s="我", v=old_value)
        new_fact = fact_frames[1 % len(fact_frames)].format(s="我", v=new_value)
        candidates = [old_fact, new_fact]
        candidates += [fact_frames[0].format(s="我", v=_value(rng)) for _ in range(3)]
        rng.shuffle(candidates)
        query = rng.choice(question_frames).format(q="我")
        episodes.append(_episode(counter, "update_conflict", query, candidates, [new_fact],
                                 [new_value], extra={"attribute": attribute,
                                                     "superseded_value": old_value}))
        counter += 1

        # --- multi_hop: the answer needs two chained facts, and is a *unique* token ------
        # The answer must not be recoverable by blending: every chain points at a distinct
        # code, so only following the right chain yields the expected token.
        code = "REF-%s" % "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        answer = "值班人-7741"
        others = ["值班人-%04d" % rng.randrange(1000, 9999) for _ in range(3)]
        first = "张三负责的那个项目的内部编号是 %s。" % code
        second = "编号 %s 对应的值班人是 %s。" % (code, answer)
        decoy_codes = ["REF-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
                                        for _ in range(6)) for _ in others]
        decoys = ["编号 %s 对应的值班人是 %s。" % (dc, name)
                  for dc, name in zip(decoy_codes, others)]
        candidates = [first, second] + decoys
        rng.shuffle(candidates)
        query = "负责那个项目的同事对应的值班人是谁？"
        episodes.append(_episode(counter, "multi_hop", query, candidates, [first, second],
                                 [answer], hop=2))
        counter += 1

        # --- near_miss: the question refers to the attribute without naming it ----------
        # Naming the attribute outright ("我的手机尾号是什么") is not a hard test; the
        # question here paraphrases it so the near-miss decoy is genuinely in play.
        left, right = rng.choice(NEAR_MISS)
        value, decoy = _value(rng), _value(rng)
        candidates = ["我的%s是 %s。" % (left, value), "我的%s是 %s。" % (right, decoy)]
        candidates += ["我的%s是 %s。" % (name, _value(rng))
                       for name, _, _ in rng.sample(ATTRIBUTES, 2)]
        rng.shuffle(candidates)
        ask_left = rng.random() < 0.5
        target_value = value if ask_left else decoy
        target_fact = [c for c in candidates if target_value in c][0]
        query = ("我随身那台设备的末几位数字是啥？" if ask_left
                 else "我家座机最后几位数字是啥？")
        episodes.append(_episode(counter, "near_miss", query, candidates, [target_fact],
                                 [target_value], extra={"attribute": left if ask_left else right}))
        counter += 1

        # --- noise_context: lots of unrelated chatter, question still answerable -------
        attribute, fact_frames, question_frames = rng.choice(ATTRIBUTES)
        value = _value(rng)
        fact = fact_frames[0].format(s="我", v=value)
        noise = ["这是普通对话噪声：验证用户-%06d 暂时提到一个无关编号 N-%s，不需要长期保存。"
                 % (rng.randrange(10 ** 6), "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
                                                    for _ in range(8)))
                 for _ in range(8)]
        candidates = [fact] + noise
        rng.shuffle(candidates)
        query = rng.choice(question_frames).format(q="我")
        episodes.append(_episode(counter, "noise_context", query, candidates, [fact], [value]))
        counter += 1

        # --- long_fact: the value sits mid-sentence in a long fact ---------------------
        attribute, fact_frames, question_frames = rng.choice(ATTRIBUTES)
        value = _value(rng)
        long_fact = ("关于%s这件事，之前散会时我们临时定了下来，当时讨论得比较久，最后确认%s，"
                     "后续如果还有调整会在群里同步，暂时先按这个执行。" % (attribute, value))
        others = [("关于%s这件事，会上也提过一嘴，但那次的说法是%s，后来没有正式确认过。"
                   % (attribute, _value(rng))) for _ in range(3)]
        candidates = [long_fact] + others
        rng.shuffle(candidates)
        query = rng.choice(question_frames).format(q="我")
        episodes.append(_episode(counter, "long_fact", query, candidates, [long_fact], [value]))
        counter += 1

        # --- unknown_attribute: the asked attribute was never written ------------------
        # Candidates must be written in *their own* frames (an earlier version reused the
        # asked attribute's frame, so the "unknown" attribute was in fact present and the
        # question was answerable -- the 0.00% it produced measured nothing).
        attribute, fact_frames, question_frames = rng.choice(ATTRIBUTES)
        written = [(name, frames) for name, frames, _ in ATTRIBUTES if name != attribute]
        picked = rng.sample(written, 5)
        candidates = [frames[0].format(s="我", v=_value(rng)) for _, frames in picked]
        query = rng.choice(question_frames).format(q="我")
        episodes.append(_episode(counter, "unknown_attribute", query, candidates, [], []))
        counter += 1

    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/realistic_eval.jsonl")
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    episodes = build(args.per_category, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    counts: dict[str, dict] = {}
    for row in episodes:
        category = row["metadata"]["category"]
        bucket = counts.setdefault(category, {"episodes": 0, "answerable": 0, "unknown": 0,
                                              "max_candidates": 0})
        bucket["episodes"] += 1
        bucket["answerable" if row["positive_indices"] else "unknown"] += 1
        bucket["max_candidates"] = max(bucket["max_candidates"], len(row["candidates"]))
    manifest = {
        "generator": "make_realistic_memory_eval.py",
        "purpose": "realistic-shaped memory evaluation corpus (multi-entity, aliases, updates, "
                   "multi-hop, near-miss, noise, long facts, unknowns)",
        "per_category": args.per_category,
        "seed": args.seed,
        "episodes": len(episodes),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "categories": counts,
        "note": ("episodes use the schema consumed by eval_end_to_end_memory "
                 "(query / candidates[].text / positive_indices / metadata.acceptable)"),
    }
    (output.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
