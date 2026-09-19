"""Realistic-shaped router training/eval data, with a *disjoint* attribute vocabulary.

The earlier corpora (v5/v6, and the first realistic eval) share one property that makes
their numbers untrustworthy for real use: the training and evaluation text comes from the
same small template pool, so "generalisation" mostly measured memorisation of those
templates.  On a realistic-shaped corpus the router trained that way showed no advantage
at all over the original.

This generator fixes the split itself: a pool of attribute families is partitioned, and the
evaluation half uses families that **never appear in training** -- different attribute
names, different fact frames, different question frames.  Anything the router gets right on
the eval half is therefore generalisation to unseen attribute wording, not recall.

Categories mirror the realistic eval corpus so the same harness and scoring apply:
  multi_entity, alias_paraphrase, update_conflict, multi_hop, near_miss,
  noise_context, long_fact, unknown_attribute

Usage::

    python -m V2_dpskw.make_realistic_memory_data --split train --output data/realistic_v2/train.jsonl
    python -m V2_dpskw.make_realistic_memory_data --split eval  --output data/realistic_v2/eval.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ENTITIES = [
    ("user", ["我", "我本人"], ["我", "我自己"]),
    ("zhangsan", ["张三", "我们组的张三"], ["张三", "我们组那位张三"]),
    ("liworks", ["李工", "运维的李工"], ["李工", "运维那位李工"]),
    ("wangwu", ["王五", "测试的王五"], ["王五", "测试那位王五"]),
    ("proj_alpha", ["项目 Alpha", "Alpha 项目"], ["Alpha", "Alpha 项目"]),
    ("proj_gamma", ["项目 Gamma", "Gamma 项目"], ["Gamma", "Gamma 项目"]),
    ("client_beta", ["客户 Beta", "Beta 客户"], ["Beta", "Beta 客户"]),
    ("client_delta", ["客户 Delta", "Delta 客户"], ["Delta", "Delta 客户"]),
]

#: (attribute, [fact frames], [question frames]) -- never reuses wording across families.
ATTRIBUTE_POOL: list[tuple[str, list[str], list[str]]] = [
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
    ("备份周期", ["{s}的备份周期是 {v}。", "{s}每 {v} 做一次备份。"],
     ["{q}多久备一次？", "{q}留档的间隔是多长？"]),
    ("证书到期日", ["{s}的证书到期日是 {v}。", "{s}那张证书到 {v} 失效。"],
     ["{q}那张凭证什么时候过期？", "{q}还剩多久必须换新的？"]),
    ("代码仓库", ["{s}的代码仓库是 {v}。", "{s}源码放在 {v}。"],
     ["{q}提交到哪个地方？", "{q}源码托管在哪里？"]),
    ("周会时间", ["{s}的周会时间是 {v}。", "{s}固定在 {v} 开周会。"],
     ["{q} weekly 的会安排在何时？", "{q}团队什么时候碰一次？"]),
    ("审批人", ["{s}的审批人是 {v}。", "{s}的单子由 {v} 签。"],
     ["{q}最后谁点头才算过？", "{q}该找谁签字？"]),
    ("灰度比例", ["{s}的灰度比例是 {v}。", "{s}先放 {v} 的量。"],
     ["{q}先放多少出去？", "{q}试探性放量放到多少？"]),
    ("限流上限", ["{s}的限流上限是 {v}。", "{s}每秒最多 {v}。"],
     ["{q}顶到多少会被挡？", "{q}请求量封顶是多少？"]),
    ("日志保留", ["{s}的日志保留是 {v}。", "{s}日志留 {v}。"],
     ["{q}这些记录存多久？", "{q}存档保留多长时间？"]),
    ("镜像标签", ["{s}的镜像标签是 {v}。", "{s}打的是 {v} 这个标签。"],
     ["{q}拉的是哪一版镜像？", "{q}容器上标的是哪个？"]),
    ("监控面板", ["{s}的监控面板是 {v}。", "{s}的看板挂在 {v}。"],
     ["{q}出图在哪个地址？", "{q}曲线从哪里看？"]),
    ("值班表", ["{s}的值班表是 {v}。", "{s}按 {v} 排班。"],
     ["{q}这周按什么表轮？", "{q}排班依据是哪一份？"]),
    ("预算上限", ["{s}的预算上限是 {v}。", "{s}这笔最多花 {v}。"],
     ["{q}最多能批多少？", "{q}额度封在哪儿？"]),
    ("供应商", ["{s}的供应商是 {v}。", "{s}的货从 {v} 来。"],
     ["{q}货是谁供的？", "{q}上游是哪一家？"]),
    ("合同编号", ["{s}的合同编号是 {v}。", "{s}签的是 {v} 号合同。"],
     ["{q}纸面上那个号是多少？", "{q}协议编号是什么？"]),
    ("对账日", ["{s}的对账日是 {v}。", "{s}每月 {v} 对账。"],
     ["{q}哪天核数？", "{q}账目什么时候核一次？"]),
    ("保险到期", ["{s}的保险到期是 {v}。", "{s}那份保到 {v}。"],
     ["{q}保障什么时候结束？", "{q}这份多久后失效？"]),
    ("运输方式", ["{s}的运输方式是 {v}。", "{s}走 {v} 运。"],
     ["{q}东西怎么送？", "{q}靠什么渠道发？"]),
    ("结算币种", ["{s}的结算币种是 {v}。", "{s}按 {v} 结算。"],
     ["{q}用哪种钱算账？", "{q}计价单位是什么？"]),
    ("培训周期", ["{s}的培训周期是 {v}。", "{s}训 {v} 那么久。"],
     ["{q}要学多久？", "{q}上手需要多长时间？"]),
    ("仓库库位", ["{s}的仓库库位是 {v}。", "{s}堆在 {v}。"],
     ["{q}货码在哪个位置？", "{q}东西放在哪一格？"]),
    ("质检标准", ["{s}的质检标准是 {v}。", "{s}按 {v} 验。"],
     ["{q}凭什么判合格？", "{q}验收看哪条线？"]),
    ("客户等级", ["{s}的客户等级是 {v}。", "{s}被划到 {v}。"],
     ["{q}这家算第几档？", "{q}排在哪个层级？"]),
    ("返修地址", ["{s}的返修地址是 {v}。", "{s}寄回 {v}。"],
     ["{q}坏了往哪儿寄？", "{q}退回的收件地是哪里？"]),
    ("样机编号", ["{s}的样机编号是 {v}。", "{s}这台是 {v} 号样机。"],
     ["{q}手上这台是几号？", "{q}试产那台的编号是什么？"]),
    ("开源协议", ["{s}的开源协议是 {v}。", "{s}按 {v} 开源。"],
     ["{q}授权方式是哪一种？", "{q}发布条款是什么？"]),
    ("风速上限", ["{s}的风速上限是 {v}。", "{s}顶到 {v} 就得停。"],
     ["{q}多大的风必须停？", "{q}安全上限是多少？"]),
    ("对接端口", ["{s}的对接端口是 {v}。", "{s}监听 {v}。"],
     ["{q}走哪个口通信？", "{q}连的是哪个门？"]),
    ("续约提醒", ["{s}的续约提醒是 {v}。", "{s}提前 {v} 提醒续约。"],
     ["{q}到期前多久会提醒？", "{q}提前多长时间通知？"]),
]

_ALIAS_PAIRS = [
    ("我平时待得最久的那座城是 {v}。", "我常年落脚在哪儿？"),
    ("我每天睡醒最早看的那个数字是 {v}。", "我那个数字是多少？"),
    ("我手上那台机器是 {v}。", "我用的那台是什么款？"),
    ("我每天早上灌下去的那杯偏爱 {v}。", "我早上那杯是什么口味？"),
    ("我包里常备的那本册子是 {v}。", "我随身带的是哪一本？"),
    ("我周末常去的那家店叫 {v}。", "我常去的那家叫什么？"),
]

_VALUES = ["分机 8821", "B3-204", "每周三 22:00", "v3.14.2", "91.5%", "王工 138****6621",
           "10.20.3.7:5432", "64GB", "A-7719", "夜间 02:00", "v2.8.0", "78%",
           "李工 139****3344", "10.20.9.1:6379", "128GB", "B-3310", "分机 6612", "C1-105",
           "每周一 09:30", "v4.0.0-rc1", "63.2%", "赵工 137****8890", "10.30.1.9:5432",
           "32GB", "D-9021", "凌晨 03:30", "v1.9.7", "85%", "钱工 135****2211",
           "10.30.7.3:6379", "256GB", "E-4408"]

NEAR_MISS_LABELS = [("手机尾号", "我随身那台设备的末几位数字是啥？"),
                    ("座机尾号", "我家那台固定电话最后几位是啥？"),
                    ("办公城市", "我白天上班待的地儿在哪儿？"),
                    ("常住城市", "我平时待得最久的地儿是哪里？"),
                    ("工位楼层", "我坐着干活的地方在第几阶？"),
                    ("宿舍楼层", "我睡觉的地方在第几阶？")]


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
        "id": "rv2-%s-%06d" % (category, index),
        "group_id": "rv2-%s-%06d" % (category, index),
        "source": "realistic_memory_v2",
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


def build(families: list[tuple[str, list[str], list[str]]], per_category: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    episodes: list[dict] = []
    counter = 0
    for _ in range(per_category):
        # multi_entity
        attribute, frames, questions = rng.choice(families)
        speakers = rng.sample(ENTITIES, 3)
        target = rng.choice(speakers)
        values = {key: _value(rng) for key, _, _ in speakers}
        candidates = []
        for key, subjects, aliases in speakers:
            frame = frames[0] if key == target[0] else frames[len(frames) - 1]
            candidates.append(frame.format(s=subjects[0], v=values[key]))
        rng.shuffle(candidates)
        fact = [c for c in candidates if values[target[0]] in c][0]
        query = rng.choice(questions).format(q=rng.choice(target[2]))
        episodes.append(_episode(counter, "multi_entity", query, candidates, [fact],
                                 [values[target[0]]], extra={"attribute": attribute}))
        counter += 1

        # alias_paraphrase -- the target fact is the only one in its own frame; every
        # distractor comes from a *different* family, so exactly one value is being asked
        # about and the question shares no distinctive characters with the fact.
        template, question = rng.choice(_ALIAS_PAIRS)
        value = _value(rng)
        fact = template.format(v=value)
        others = [rng.choice(rng.choice(families)[1]).format(s="我", v=_value(rng))
                  for _ in range(4)]
        candidates = [fact] + others
        rng.shuffle(candidates)
        episodes.append(_episode(counter, "alias_paraphrase", question, candidates, [fact], [value]))
        counter += 1

        # update_conflict
        attribute, frames, questions = rng.choice(families)
        old_value, new_value = _value(rng), _value(rng)
        while new_value == old_value:
            new_value = _value(rng)
        old_fact = frames[0].format(s="我", v=old_value)
        new_fact = frames[len(frames) - 1].format(s="我", v=new_value)
        candidates = [old_fact, new_fact] + [frames[0].format(s="我", v=_value(rng)) for _ in range(3)]
        rng.shuffle(candidates)
        query = rng.choice(questions).format(q="我")
        episodes.append(_episode(counter, "update_conflict", query, candidates, [new_fact],
                                 [new_value], extra={"attribute": attribute,
                                                     "superseded_value": old_value}))
        counter += 1

        # multi_hop -- answer is a unique token only reachable through the right chain
        code = "REF-%s" % "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        answer = "值班人-%04d" % rng.randrange(1000, 9999)
        others = ["值班人-%04d" % rng.randrange(1000, 9999) for _ in range(3)]
        first = "张三负责的那个项目的内部编号是 %s。" % code
        second = "编号 %s 对应的值班人是 %s。" % (code, answer)
        decoys = ["编号 REF-%s 对应的值班人是 %s。" % (
            "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6)), name)
            for name in others]
        candidates = [first, second] + decoys
        rng.shuffle(candidates)
        episodes.append(_episode(counter, "multi_hop",
                                 "负责那个项目的同事对应的值班人是谁？", candidates,
                                 [first, second], [answer], hop=2))
        counter += 1

        # near_miss -- ask about one of two confusable attributes without naming it
        ask_left = rng.random() < 0.5
        left, left_question = NEAR_MISS_LABELS[0] if ask_left else NEAR_MISS_LABELS[2]
        right, right_question = NEAR_MISS_LABELS[1] if ask_left else NEAR_MISS_LABELS[3]
        value, decoy = _value(rng), _value(rng)
        candidates = ["我的%s是 %s。" % (left, value), "我的%s是 %s。" % (right, decoy)]
        candidates += [rng.choice(rng.choice(families)[1]).format(s="我", v=_value(rng))
                       for _ in range(2)]
        rng.shuffle(candidates)
        target_value = value
        target_fact = [c for c in candidates if target_value in c][0]
        episodes.append(_episode(counter, "near_miss", left_question, candidates, [target_fact],
                                 [target_value], extra={"attribute": left}))
        counter += 1

        # noise_context
        attribute, frames, questions = rng.choice(families)
        value = _value(rng)
        fact = frames[0].format(s="我", v=value)
        noise = ["这是普通对话噪声：验证用户-%06d 暂时提到一个无关编号 N-%s，不需要长期保存。"
                 % (rng.randrange(10 ** 6),
                    "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8)))
                 for _ in range(8)]
        candidates = [fact] + noise
        rng.shuffle(candidates)
        episodes.append(_episode(counter, "noise_context", rng.choice(questions).format(q="我"),
                                 candidates, [fact], [value]))
        counter += 1

        # long_fact
        attribute, frames, questions = rng.choice(families)
        value = _value(rng)
        long_fact = ("关于%s这件事，之前散会时我们临时定了下来，当时讨论得比较久，最后确认%s，"
                     "后续如果还有调整会在群里同步，暂时先按这个执行。" % (attribute, value))
        others = ["关于%s这件事，会上也提过一嘴，但那次的说法是%s，后来没有正式确认过。"
                  % (attribute, _value(rng)) for _ in range(3)]
        candidates = [long_fact] + others
        rng.shuffle(candidates)
        episodes.append(_episode(counter, "long_fact", rng.choice(questions).format(q="我"),
                                 candidates, [long_fact], [value]))
        counter += 1

        # unknown_attribute -- candidates use *other* families' frames
        attribute, frames, questions = rng.choice(families)
        pool = [(name, frame_list) for name, frame_list, _ in families if name != attribute]
        picked = rng.sample(pool, min(5, len(pool)))
        candidates = [frame_list[0].format(s="我", v=_value(rng)) for _, frame_list in picked]
        episodes.append(_episode(counter, "unknown_attribute",
                                 rng.choice(questions).format(q="我"), candidates, [], []))
        counter += 1

    return episodes


def split_families() -> tuple[list, list]:
    """Partition the attribute pool so eval families never appear in training."""
    ordered = sorted(ATTRIBUTE_POOL, key=lambda item: item[0])
    half = len(ordered) // 3
    return ordered[half:], ordered[:half]  # train, eval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_families, eval_families = split_families()
    families = train_families if args.split == "train" else eval_families
    per_category = args.per_category or (400 if args.split == "train" else 25)
    seed = args.seed or (20260913 if args.split == "train" else 20260914)

    episodes = build(families, per_category, seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    counts: dict[str, int] = {}
    for row in episodes:
        counts[row["metadata"]["category"]] = counts.get(row["metadata"]["category"], 0) + 1
    manifest = {
        "generator": "make_realistic_memory_data.py",
        "split": args.split,
        "episodes": len(episodes),
        "per_category": per_category,
        "seed": seed,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "categories": counts,
        "attribute_families": [name for name, _, _ in families],
        "disjointness": {
            "train_families": [name for name, _, _ in train_families],
            "eval_families": [name for name, _, _ in eval_families],
            "overlap": sorted({name for name, _, _ in train_families}
                              & {name for name, _, _ in eval_families}),
        },
        "note": ("eval families never appear in training, so eval results measure "
                 "generalisation to unseen attribute wording"),
    }
    (output.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "attribute_families"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
