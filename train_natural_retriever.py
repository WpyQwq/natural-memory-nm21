"""Train the small internal query-to-text-memory retrieval head."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer


ATTRIBUTES = {
    "name": {
        "facts": (
            "我的名字是{value}。",
            "请记住，我叫{value}。",
            "用户姓名记录为{value}。",
            "我叫{value}",
            "我是{value}",
            "叫我{value}就行",
            "姓名：{value}",
            "用户叫{value}",
        ),
        "queries": (
            "我叫什么名字？",
            "请告诉我已经记录的姓名。",
            "你还记得我的名字吗？",
            "我叫什么？",
            "我的姓名是什么？",
            "怎么称呼我？",
            "我的名字呢？",
            "你记得我叫什么吗？",
            "我在这里登记的名字是什么？",
            "我在这里叫什么",
            "你这里记录的我叫什么",
            "我在你这边叫什么",
            "你这边怎么称呼我",
            "系统里记录的我的名字是什么",
        ),
        "holdout_queries": (
            "我是谁？",
            "你知道我是谁吗？",
            "你还记得我是谁吗？",
            "我在你这里叫什么？",
        ),
    },
    "project": {
        "facts": (
            "我正在开发{value}项目。",
            "请记住，我当前负责的项目是{value}。",
            "我的当前项目名称是{value}。",
            "我做的是{value}",
            "项目：{value}",
            "最近在搞{value}",
            "我手头做{value}",
        ),
        "queries": (
            "我正在开发什么项目？",
            "请查询我当前的项目。",
            "我之前说过正在做什么吗？",
            "我最近在做什么？",
            "我手头的项目叫什么？",
            "我在忙哪个项目？",
            "我最近搞的是什么？",
            "手上的活是什么项目？",
        ),
        "holdout_queries": (
            "我现在主要做什么？",
            "我最近在开发哪一个东西？",
            "之前提到的项目是什么？",
        ),
    },
    "plan": {
        "facts": (
            "我计划在{value}完成这件事。",
            "请记住我的计划：{value}。",
            "我的下一步安排是{value}。",
            "计划：{value}",
            "我打算{value}",
            "准备在{value}完成",
            "安排：{value}",
            "待办：{value}",
        ),
        "queries": (
            "我接下来的计划是什么？",
            "请查询我记录过的安排。",
            "我之前说过下一步要做什么？",
            "我下一步准备做什么？",
            "我安排在什么时候完成？",
            "我接下来怎么安排？",
            "我的安排是什么？",
            "我有什么计划？",
        ),
        "holdout_queries": (
            "我接下来打算怎么安排？",
            "我已经计划好的事情是什么？",
        ),
    },
    "constraint": {
        "facts": (
            "这个项目的重要约束是{value}。",
            "请记住这个开发约束：{value}。",
            "以后处理这个项目时必须遵守：{value}。",
            "要求：{value}",
            "必须{value}",
            "别忘了：{value}",
        ),
        "queries": (
            "这个项目的重要约束是什么？",
            "请查询我记录的开发约束。",
            "之前说过这个项目需要遵守什么吗？",
            "这个项目有什么限制？",
            "开发时需要注意哪条规则？",
            "有哪些要求不能忘？",
        ),
        "holdout_queries": (
            "这个项目有哪些不能违反的要求？",
            "我之前定下的开发规则是什么？",
        ),
    },
    "work_code": {
        "facts": (
            "我的工作地点代号是{value}。",
            "请记住，我的工作地点代号为{value}。",
            "以后如果问到工作地点，请记住代号{value}。",
            "工作地点：{value}",
            "地点编号{value}",
            "我在{value}办公",
            "办公地点：{value}",
            "工作地点编号：{value}",
        ),
        "queries": (
            "我的工作地点代号是什么？",
            "请告诉我已经记录的工作地点代号。",
            "我之前说过的工作地点代号是多少？",
            "工作地点对应哪个代号？",
            "我工作的地方编号是什么？",
            "我办公地点是哪儿？",
            "我在哪办公？",
            "我在哪工作？",
            "工作地点是哪儿？",
        ),
        "holdout_queries": (
            "我的办公地点编号是多少？",
            "我在哪个工作地点？",
        ),
    },
    "fruit": {
        "facts": (
            "我最喜欢的水果是{value}。",
            "请记住我的水果偏好：我喜欢{value}。",
            "我的个人偏好是最喜欢吃{value}。",
            "我爱吃{value}",
            "水果偏好：{value}",
            "我喜欢{value}",
        ),
        "queries": (
            "我最喜欢吃什么水果？",
            "请查询我记录过的水果偏好。",
            "我之前告诉你的水果喜好是什么？",
            "我平时爱吃哪种水果？",
            "我的水果口味偏好是什么？",
            "我爱吃哪种？",
        ),
        "holdout_queries": (
            "我喜欢吃哪一类水果？",
            "哪种水果是我的首选？",
        ),
    },
    "pet": {
        "facts": (
            "我养的宠物名字叫{value}。",
            "请记住，我的宠物是{value}。",
            "我的宠物信息：名字是{value}。",
            "宠物是{value}",
            "我养了{value}",
            "宠物：{value}",
        ),
        "queries": (
            "我养的宠物叫什么名字？",
            "请查询我的宠物姓名。",
            "你记得我的宠物是谁吗？",
            "我家的宠物叫什么？",
            "我的宠物是哪一只？",
            "我养的是什么？",
        ),
        "holdout_queries": (
            "我养了什么动物？",
            "我的宠物信息是什么？",
        ),
    },
    "editor": {
        "facts": (
            "我平时最常用的编辑器是{value}。",
            "记住我的开发工具偏好：编辑器使用{value}。",
            "我的编程编辑器偏好为{value}。",
            "我用{value}写代码",
            "编辑器：{value}",
            "开发工具是{value}",
            "代码用{value}",
            "工具：{value}",
            "编程工具：{value}",
            "写代码用{value}",
        ),
        "queries": (
            "我最常用哪个编辑器？",
            "请查询我的开发工具偏好。",
            "我的编程编辑器是什么？",
            "我平时用什么编辑器？",
            "我习惯用哪款开发工具？",
            "我写代码用什么？",
            "我用什么写代码？",
            "我平时写程序用什么工具？",
            "我编程时使用什么工具？",
            "写代码用的是哪款工具？",
            "我用什么工具编程？",
        ),
        "holdout_queries": (
            "我写代码通常使用什么工具？",
            "我常用的 IDE 是哪个？",
        ),
    },
    "city": {
        "facts": (
            "我现在长期居住在{value}。",
            "请记住，我的常住城市是{value}。",
            "我的个人资料显示常住地为{value}。",
            "我住在{value}",
            "常住地：{value}",
            "我在{value}生活",
        ),
        "queries": (
            "我的常住城市是哪里？",
            "请查询我的居住地。",
            "我平时住在哪座城市？",
            "我长期住在哪里？",
            "我的居住城市是什么？",
            "我现在住哪儿？",
        ),
        "holdout_queries": (
            "我现在定居在哪儿？",
            "我的长期住址城市是哪座？",
        ),
    },
    "timezone": {
        "facts": (
            "我的常用时区是{value}。",
            "请把我的时区偏好记为{value}。",
            "个人资料：我的时区设置为{value}。",
            "时区：{value}",
            "我在{value}时区",
            "本地时区是{value}",
        ),
        "queries": (
            "我的常用时区是什么？",
            "请查询我的时区设置。",
            "我使用哪个时区？",
            "我平时按哪个时区生活？",
            "我的时间设置是哪一个时区？",
            "我所在的时区是什么？",
        ),
        "holdout_queries": (
            "我的本地时间属于哪个时区？",
            "我应该使用什么时区？",
        ),
    },
}

VALUES = {
    "name": ("林浩", "小明", "周宁", "陈雪", "Alice", "Zoe", "X7"),
    "project": ("星火记忆", "自然语言记忆", "Qwen 架构实验", "个人助手"),
    "plan": ("下周五", "今晚八点", "本周末", "明天上午"),
    "constraint": ("不要删除用户数据", "使用简洁中文", "保持原版能力", "优先保证可恢复"),
    "work_code": ("R7", "K9", "蓝鲸-47", "M2"),
    "fruit": ("红富士苹果", "阳光玫瑰葡萄", "海南芒果", "脆甜梨"),
    "pet": ("豆包", "团子", "可可", "雪球"),
    "editor": ("VS Code", "Neovim", "PyCharm", "Emacs"),
    "city": ("上海", "成都", "深圳", "杭州"),
    "timezone": ("Asia/Shanghai", "UTC+8", "Europe/London", "America/Los_Angeles"),
}

HARD_NEGATIVE_QUERIES = (
    "你是谁",
    "你叫什么",
    "请介绍你自己",
    "你能做什么",
    "你的名字是什么",
    "今天天气怎么样",
    "帮我写一段代码",
    "Python是什么",
    "解释一下这个概念",
)

REDTEAM_NEGATIVE_QUERIES = (
    "如果我的{attribute}改成另一个值，会发生什么？",
    "别人说我的{attribute}是这个值，但那不是我的资料。",
    "这次临时提到{attribute}，不用保存。",
    "我想了解{attribute}这个概念，不是查询我的个人资料。",
    "我的不存在的{attribute}是什么？如果没有登记就说不知道。",
    "从来没有登记过的{attribute}是什么？不要从别的字段推断。",
)


def make_redteam_records(seed: int, count: int = 960) -> list[dict[str, object]]:
    """Add adversarial pairs that look semantically close but are not reads.

    The original corpus mostly contrasted one attribute with another.  These
    examples specifically target the failure mode called out in the review:
    a paraphrase, quotation, or hypothetical sentence can receive a confident
    retrieval score even though it should not select a personal memory.
    """

    rng = random.Random(seed + 101)
    attributes = list(ATTRIBUTES)
    rows: list[dict[str, object]] = []
    for index in range(count // 2):
        attribute = rng.choice(attributes)
        value = f"RT-{index:05d}"
        fact = rng.choice(ATTRIBUTES[attribute]["facts"]).format(value=value)
        positive_queries = (
            f"我之前登记的{attribute}是哪一个？",
            f"只读取长期资料，告诉我{attribute}。",
            f"跨会话后，我保存的{attribute}是什么？",
            f"不要猜测，回忆我的{attribute}设置。",
        )
        rows.append(
            {
                "fact": fact,
                "query": rng.choice(positive_queries),
                "label": 1.0,
                "attribute": attribute,
            }
        )
        rows.append(
            {
                "fact": fact,
                "query": rng.choice(REDTEAM_NEGATIVE_QUERIES).format(attribute=attribute),
                "label": 0.0,
                "attribute": f"{attribute}->redteam_negative",
            }
        )
    rng.shuffle(rows)
    return rows


def make_redteam_holdout(seed: int, count: int = 240) -> list[dict[str, object]]:
    """Generate held-out adversarial paraphrases with unseen values."""

    rng = random.Random(seed + 109)
    attributes = list(ATTRIBUTES)
    rows: list[dict[str, object]] = []
    for index in range(count):
        attribute = attributes[index % len(attributes)]
        value = f"HELDOUT-{rng.randrange(10**8):08d}"
        fact = f"我的{attribute}是{value}。"
        if index % 4 == 0:
            query = f"跨对话后，之前存下来的{attribute}是哪一个？"
            label = 1.0
        elif index % 4 == 1:
            query = f"如果把我的{attribute}改掉，应该如何规划？"
            label = 0.0
        elif index % 4 == 2:
            query = f"别人提到我的{attribute}，但请不要把这句话当作我的资料。"
            label = 0.0
        else:
            query = f"我的不存在的{attribute}是什么？如果没有登记就说不知道。"
            label = 0.0
        rows.append({"fact": fact, "query": query, "label": label, "attribute": attribute})
    return rows


def _chat_ids(tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    ids = encoded["input_ids"]
    mask = encoded.get("attention_mask", torch.ones_like(ids))
    return ids, mask


def _plain_ids(tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"]
    mask = encoded.get("attention_mask", torch.ones_like(ids))
    return ids, mask


def _training_query_candidates(attribute: str) -> list[str]:
    queries = list(ATTRIBUTES[attribute]["queries"])
    queries.extend(
        query.rstrip("？?。！!，,")
        for query in ATTRIBUTES[attribute]["queries"]
        if query.rstrip("？?。！!，,")
    )
    return list(dict.fromkeys(queries))


def make_pairs(seed: int, count: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    keys = list(ATTRIBUTES)
    positive_records: list[dict[str, object]] = []

    # Cover every short/long key form and every train-time query form before
    # falling back to random samples.  This prevents the optimizer from
    # seeing a mostly easy subset of the cross-product.
    for attribute in keys:
        for value in VALUES[attribute][:2]:
            for fact_template in ATTRIBUTES[attribute]["facts"]:
                for query in _training_query_candidates(attribute):
                    positive_records.append(
                        {
                            "fact": fact_template.format(value=value),
                            "query": query,
                            "label": 1.0,
                            "attribute": attribute,
                        }
                    )

    while len(positive_records) < count:
        attribute = rng.choice(keys)
        value = rng.choice(VALUES[attribute])
        positive_records.append(
            {
                "fact": rng.choice(ATTRIBUTES[attribute]["facts"]).format(value=value),
                "query": rng.choice(_training_query_candidates(attribute)),
                "label": 1.0,
                "attribute": attribute,
            }
        )
    rng.shuffle(positive_records)
    positive_records = positive_records[:count]

    pairs: list[dict[str, object]] = []
    for positive in positive_records:
        pairs.append(positive)
        attribute = str(positive["attribute"])
        negative_attribute = rng.choice([item for item in keys if item != attribute])
        negative_query_pool = _training_query_candidates(negative_attribute)
        if rng.random() < 0.5:
            negative_query_pool = list(negative_query_pool) + list(HARD_NEGATIVE_QUERIES)
        negative_query = rng.choice(negative_query_pool)
        pairs.append(
            {
                "fact": positive["fact"],
                "query": negative_query,
                "label": 0.0,
                "attribute": f"{attribute}->{negative_attribute}",
            }
        )
    rng.shuffle(pairs)
    return pairs


def make_holdout_queries(seed: int) -> list[dict[str, object]]:
    """Create query paraphrases that are never used during optimization."""

    rng = random.Random(seed + 17)
    records: list[dict[str, object]] = []
    for attribute, definition in ATTRIBUTES.items():
        value = rng.choice(VALUES[attribute])
        fact = str(definition["facts"][0]).format(value=value)
        for query in definition.get("holdout_queries", ()):
            records.append({"fact": fact, "query": query, "attribute": attribute})
    rng.shuffle(records)
    return records


def make_short_fact_holdout() -> list[dict[str, object]]:
    """Stress-test colloquial, short and previously unseen fact strings.

    The values are deliberately different from ``VALUES``.  This checks that
    the retriever recognizes the attribute-bearing language around a value,
    instead of memorizing names, cities or project identifiers.
    """

    return [
        {"fact": "我叫Wpy", "query": "你知道我是谁吗", "attribute": "name"},
        {"fact": "我叫Wpy", "query": "我是谁", "attribute": "name"},
        {"fact": "我做的是量子账本", "query": "我最近在忙什么项目", "attribute": "project"},
        {"fact": "准备在周三完成", "query": "我的待办安排是什么", "attribute": "plan"},
        {"fact": "必须保留原始数据", "query": "有什么要求必须遵守", "attribute": "constraint"},
        {"fact": "地点编号Z3", "query": "我在哪儿办公", "attribute": "work_code"},
        {"fact": "我喜欢白桃", "query": "我爱吃什么", "attribute": "fruit"},
        {"fact": "我养了阿福", "query": "家里养的是什么动物", "attribute": "pet"},
        {"fact": "我用Cursor写代码", "query": "我编程时用哪个工具", "attribute": "editor"},
        {"fact": "我住在苏州", "query": "我人住哪座城", "attribute": "city"},
        {"fact": "时区：Asia/Tokyo", "query": "本地采用哪个时区", "attribute": "timezone"},
    ]


def make_hard_negative_records() -> list[dict[str, object]]:
    """Build explicit non-memory queries for every representative key form."""

    records: list[dict[str, object]] = []
    for attribute, definition in ATTRIBUTES.items():
        key_templates = (definition["facts"][0], definition["facts"][3])
        for value in VALUES[attribute][:2]:
            for fact_template in key_templates:
                fact = fact_template.format(value=value)
                for query in HARD_NEGATIVE_QUERIES:
                    records.append(
                        {
                            "fact": fact,
                            "query": query,
                            "label": 0.0,
                            "attribute": attribute,
                        }
                    )
    return records


def make_update_positive_records(seed: int, count: int = 640) -> list[dict[str, object]]:
    """Create same-attribute fact-to-fact update pairs."""

    rng = random.Random(seed + 31)
    attributes = list(ATTRIBUTES)
    records: list[dict[str, object]] = []
    for index in range(count):
        attribute = rng.choice(attributes)
        first_value = f"UPD-OLD-{index:05d}"
        second_value = f"UPD-NEW-{index:05d}"
        definition = ATTRIBUTES[attribute]
        first = rng.choice(definition["facts"]).format(value=first_value)
        second = rng.choice(definition["facts"]).format(value=second_value)
        records.append(
            {
                "fact": first,
                "query": second,
                "label": 1.0,
                "attribute": attribute,
            }
        )
    return records


def make_update_negative_records(seed: int, count: int = 640) -> list[dict[str, object]]:
    """Create same-entity, different-attribute hard negatives.

    Both sides intentionally use first-person language.  The only durable
    distinction is the attribute, which is exactly what the hot-slot update
    decision must learn instead of collapsing every personal fragment.
    """

    rng = random.Random(seed + 37)
    attributes = list(ATTRIBUTES)
    records: list[dict[str, object]] = []
    for index in range(count):
        left, right = rng.sample(attributes, 2)
        left_definition = ATTRIBUTES[left]
        right_definition = ATTRIBUTES[right]
        fact = rng.choice(left_definition["facts"]).format(value=f"NEG-L-{index:05d}")
        query = rng.choice(right_definition["facts"]).format(value=f"NEG-R-{index:05d}")
        records.append(
            {
                "fact": fact,
                "query": query,
                "label": 0.0,
                "attribute": f"{right}!={left}",
            }
        )
    return records


@torch.inference_mode()
def encode_records(model, tokenizer, records, *, batch_size: int, device):
    """Encode pairs without padding-induced representation drift.

    The Qwen linear-attention path is not perfectly invariant to right-padded
    batches.  Runtime retrieval encodes one query/fact at a time, so training
    must use the same effective sequence shape.  Grouping by exact token
    length keeps batching efficient while guaranteeing that every row is
    unpadded and therefore matches single-example inference.
    """

    query_rows: list[torch.Tensor] = []
    key_rows: list[torch.Tensor] = []
    for record in records:
        # Runtime retrieval receives the raw user query, not a chat-template
        # wrapped prompt.  Keeping this protocol identical is essential:
        # otherwise a retriever can score its offline test set well while
        # failing on the actual restart path.
        q_ids, _ = _plain_ids(tokenizer, str(record["query"]))
        k_ids, _ = _plain_ids(tokenizer, str(record["fact"]))
        query_rows.append(q_ids[0])
        key_rows.append(k_ids[0])

    def encode_without_padding(rows: list[torch.Tensor]) -> torch.Tensor:
        vectors: list[Optional[torch.Tensor]] = [None] * len(rows)
        groups: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(int(row.numel()), []).append(index)
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                length = rows[selected[0]].numel()
                batch = torch.stack([rows[index] for index in selected]).to(device)
                mask = torch.ones((len(selected), length), dtype=torch.long, device=device)
                encoded = model._encode_model_key(batch, mask).cpu()
                for row_index, vector in zip(selected, encoded):
                    vectors[row_index] = vector
        if any(vector is None for vector in vectors):
            raise RuntimeError("failed to encode every memory record")
        return torch.stack([vector for vector in vectors if vector is not None])

    return encode_without_padding(query_rows), encode_without_padding(key_rows)


@torch.inference_mode()
def evaluate_retriever(model, query_vectors, key_vectors, query_attributes, key_attributes) -> dict[str, float]:
    """Measure attribute retrieval on held-out paraphrases."""

    scores = model.text_retriever(
        query_vectors,
        key_vectors.unsqueeze(0).expand(query_vectors.shape[0], -1, -1),
    )
    key_attributes = list(key_attributes)
    positive_mask = torch.tensor(
        [[query_attribute == key_attribute for key_attribute in key_attributes]
         for query_attribute in query_attributes],
        dtype=torch.bool,
        device=scores.device,
    )
    positive_scores = scores.masked_fill(~positive_mask, torch.finfo(scores.dtype).min).max(dim=1).values
    negative_scores = scores.masked_fill(positive_mask, torch.finfo(scores.dtype).min).max(dim=1).values
    positive_probabilities = torch.sigmoid(positive_scores)
    negative_probabilities = torch.sigmoid(negative_scores)
    predicted = scores.argmax(dim=1).detach().cpu().tolist()
    predicted_attributes = [key_attributes[index] for index in predicted]
    accuracy = sum(
        predicted_attribute == query_attribute
        for predicted_attribute, query_attribute in zip(predicted_attributes, query_attributes)
    ) / max(1, len(query_attributes))
    return {
        "accuracy": float(accuracy),
        "positive_score_mean": float(positive_scores.mean().detach().cpu()),
        "negative_score_mean": float(negative_scores.mean().detach().cpu()),
        "margin_mean": float((positive_scores - negative_scores).mean().detach().cpu()),
        "positive_probability_min": float(positive_probabilities.min().detach().cpu()),
        "negative_probability_max": float(negative_probabilities.max().detach().cpu()),
        "positive_threshold_recall": float((positive_probabilities >= 0.5).float().mean().detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--base-adapter", default="V2_dpskw/qwen_memory_adapter_native_v3")
    parser.add_argument(
        "--output-adapter",
        default="V2_dpskw/qwen_memory_adapter_natural_controller_v1",
    )
    parser.add_argument("--steps", type=int, default=2200)
    parser.add_argument("--pair-count", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    config = load_memory_config(args.base_adapter)
    config.natural_language_memory = True
    config.persistent_memory = False
    config.direct_logit_scale = 0.0
    model = load_qwen_dynamic(
        args.model_path,
        memory_config=config,
        load_in_4bit=not args.no_4bit,
    )
    model.load_memory_adapter(args.base_adapter, strict=True)
    model.eval()
    if model.text_retriever is None:
        raise RuntimeError("natural-language retriever was not created")
    model.text_retriever.train()
    device = model._find_layer_device()

    pairs = make_pairs(args.seed, args.pair_count)
    training_records = [pair for pair in pairs if float(pair["label"]) == 1.0]
    query_tensor, key_tensor = encode_records(
        model,
        tokenizer,
        training_records,
        batch_size=args.batch_size,
        device=device,
    )
    query_tensor = query_tensor.to(device)
    key_tensor = key_tensor.to(device)
    training_attributes = [str(record["attribute"]) for record in training_records]
    attribute_names = list(ATTRIBUTES)
    attribute_indices = {
        attribute: [index for index, item in enumerate(training_attributes) if item == attribute]
        for attribute in attribute_names
    }
    if any(not indices for indices in attribute_indices.values()):
        raise RuntimeError("training data did not cover every memory attribute")

    holdout_records = make_holdout_queries(args.seed)
    holdout_query_tensor, holdout_key_tensor = encode_records(
        model,
        tokenizer,
        holdout_records,
        batch_size=args.batch_size,
        device=device,
    )
    holdout_query_tensor = holdout_query_tensor.to(device)
    holdout_key_tensor = holdout_key_tensor.to(device)
    holdout_attributes = [str(record["attribute"]) for record in holdout_records]
    short_holdout_records = make_short_fact_holdout()
    short_holdout_query_tensor, short_holdout_key_tensor = encode_records(
        model,
        tokenizer,
        short_holdout_records,
        batch_size=args.batch_size,
        device=device,
    )
    short_holdout_query_tensor = short_holdout_query_tensor.to(device)
    short_holdout_key_tensor = short_holdout_key_tensor.to(device)
    short_holdout_attributes = [str(record["attribute"]) for record in short_holdout_records]
    hard_negative_records = make_hard_negative_records()
    hard_negative_query_tensor, hard_negative_key_tensor = encode_records(
        model,
        tokenizer,
        hard_negative_records,
        batch_size=args.batch_size,
        device=device,
    )
    hard_negative_query_tensor = hard_negative_query_tensor.to(device)
    hard_negative_key_tensor = hard_negative_key_tensor.to(device)
    update_positive_records = make_update_positive_records(args.seed)
    update_positive_query_tensor, update_positive_key_tensor = encode_records(
        model,
        tokenizer,
        update_positive_records,
        batch_size=args.batch_size,
        device=device,
    )
    update_positive_query_tensor = update_positive_query_tensor.to(device)
    update_positive_key_tensor = update_positive_key_tensor.to(device)
    update_negative_records = make_update_negative_records(args.seed)
    update_negative_query_tensor, update_negative_key_tensor = encode_records(
        model,
        tokenizer,
        update_negative_records,
        batch_size=args.batch_size,
        device=device,
    )
    update_negative_query_tensor = update_negative_query_tensor.to(device)
    update_negative_key_tensor = update_negative_key_tensor.to(device)
    redteam_records = make_redteam_records(args.seed)
    redteam_query_tensor, redteam_key_tensor = encode_records(
        model,
        tokenizer,
        redteam_records,
        batch_size=args.batch_size,
        device=device,
    )
    redteam_query_tensor = redteam_query_tensor.to(device)
    redteam_key_tensor = redteam_key_tensor.to(device)
    redteam_labels = torch.tensor(
        [float(record["label"]) for record in redteam_records],
        dtype=torch.float32,
        device=device,
    )
    redteam_holdout_records = make_redteam_holdout(args.seed)
    redteam_holdout_query_tensor, redteam_holdout_key_tensor = encode_records(
        model,
        tokenizer,
        redteam_holdout_records,
        batch_size=args.batch_size,
        device=device,
    )
    redteam_holdout_query_tensor = redteam_holdout_query_tensor.to(device)
    redteam_holdout_key_tensor = redteam_holdout_key_tensor.to(device)
    redteam_holdout_labels = torch.tensor(
        [float(record["label"]) for record in redteam_holdout_records],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.text_retriever.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed + 1)
    for step in range(1, args.steps + 1):
        selected_indices = [
            rng.choice(attribute_indices[attribute]) for attribute in attribute_names
        ]
        selected_indices.extend(
            rng.randrange(query_tensor.shape[0])
            for _ in range(max(0, args.batch_size - len(selected_indices)))
        )
        indices = torch.tensor(
            selected_indices[: args.batch_size],
            dtype=torch.long,
            device=device,
        )
        batch_queries = query_tensor[indices]
        batch_keys = key_tensor[indices]
        batch_attributes = [training_attributes[index] for index in indices.detach().cpu().tolist()]
        pair_logits = model.text_retriever(
            batch_queries,
            batch_keys.unsqueeze(0).expand(batch_queries.shape[0], -1, -1),
        )
        positive_mask = torch.tensor(
            [[left == right for right in batch_attributes] for left in batch_attributes],
            dtype=torch.bool,
            device=device,
        )
        positive_logsum = torch.logsumexp(
            pair_logits.masked_fill(~positive_mask, torch.finfo(pair_logits.dtype).min),
            dim=1,
        )
        contrastive_loss = -(positive_logsum - torch.logsumexp(pair_logits, dim=1)).mean()

        negative_indices = [
            rng.choice(
                attribute_indices[
                    rng.choice([name for name in attribute_names if name != attribute])
                ]
            )
            for attribute in batch_attributes
        ]
        negative_keys = key_tensor[torch.tensor(negative_indices, dtype=torch.long, device=device)]
        positive_logits = pair_logits.diagonal()
        negative_logits = model.text_retriever(batch_queries, negative_keys)
        hard_negative_logits = model.text_retriever(
            hard_negative_query_tensor,
            hard_negative_key_tensor,
        )
        bce_logits = torch.cat((positive_logits, negative_logits), dim=0)
        bce_labels = torch.cat(
            (
                torch.ones_like(positive_logits),
                torch.zeros_like(negative_logits),
            ),
            dim=0,
        )
        classification_loss = F.binary_cross_entropy_with_logits(bce_logits, bce_labels)
        hard_negative_loss = F.binary_cross_entropy_with_logits(
            hard_negative_logits,
            torch.zeros_like(hard_negative_logits),
        )
        update_indices = torch.tensor(
            [rng.randrange(update_positive_query_tensor.shape[0]) for _ in range(args.batch_size)],
            dtype=torch.long,
            device=device,
        )
        update_positive_logits = model.text_retriever(
            update_positive_query_tensor[update_indices],
            update_positive_key_tensor[update_indices],
        )
        update_negative_logits = model.text_retriever(
            update_negative_query_tensor[update_indices],
            update_negative_key_tensor[update_indices],
        )
        update_pair_loss = F.binary_cross_entropy_with_logits(
            torch.cat((update_positive_logits, update_negative_logits), dim=0),
            torch.cat(
                (
                    torch.ones_like(update_positive_logits),
                    torch.zeros_like(update_negative_logits),
                ),
                dim=0,
            ),
        )
        redteam_indices = torch.tensor(
            [rng.randrange(redteam_query_tensor.shape[0]) for _ in range(args.batch_size)],
            dtype=torch.long,
            device=device,
        )
        redteam_logits = model.text_retriever(
            redteam_query_tensor[redteam_indices],
            redteam_key_tensor[redteam_indices],
        )
        redteam_loss = F.binary_cross_entropy_with_logits(
            redteam_logits,
            redteam_labels[redteam_indices],
        )
        loss = (
            contrastive_loss
            + 0.5 * classification_loss
            + 0.75 * hard_negative_loss
            + 1.25 * update_pair_loss
            + 1.00 * redteam_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.text_retriever.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            with torch.inference_mode():
                predictions = pair_logits.argmax(dim=1)
                batch_accuracy = sum(
                    batch_attributes[index] == attribute
                    for index, attribute in zip(
                        predictions.detach().cpu().tolist(), batch_attributes
                    )
                ) / max(1, len(batch_attributes))
                holdout_stats = evaluate_retriever(
                    model,
                    holdout_query_tensor,
                    holdout_key_tensor,
                    holdout_attributes,
                    holdout_attributes,
                )
                short_holdout_stats = evaluate_retriever(
                    model,
                    short_holdout_query_tensor,
                    short_holdout_key_tensor,
                    short_holdout_attributes,
                    short_holdout_attributes,
                )
                update_positive_probability = torch.sigmoid(update_positive_logits).mean()
                update_negative_probability = torch.sigmoid(update_negative_logits).mean()
            print(
                f"step={step} loss={float(loss.detach()):.5f} "
                f"batch_attribute_accuracy={batch_accuracy:.3f} "
                f"holdout_accuracy={holdout_stats['accuracy']:.3f} "
                f"holdout_margin={holdout_stats['margin_mean']:.3f} "
                f"short_fact_accuracy={short_holdout_stats['accuracy']:.3f} "
                f"short_fact_margin={short_holdout_stats['margin_mean']:.3f} "
                f"update_pos={float(update_positive_probability):.3f} "
                f"update_neg={float(update_negative_probability):.3f} "
                f"redteam_loss={float(redteam_loss.detach()):.5f} "
                f"short_fact_threshold_recall={short_holdout_stats['positive_threshold_recall']:.3f}"
            )

    model.text_retriever.eval()
    model._text_retriever_ready = True
    model.memory_config.persistent_memory = False
    output_dir = Path(args.output_adapter)
    model.save_memory_adapter(output_dir)
    stats = {
        "steps": args.steps,
        "pair_count": len(pairs),
        "positive_training_records": len(training_records),
        "hard_negative_records": len(hard_negative_records),
        "update_positive_records": len(update_positive_records),
        "update_negative_records": len(update_negative_records),
        "redteam_records": len(redteam_records),
        "redteam_holdout_records": len(redteam_holdout_records),
        "holdout_records": len(holdout_records),
        "source_adapter": str(args.base_adapter),
        "retriever": "qwen_hidden_pair_mlp_multisample_contrastive_with_hard_negatives",
        "holdout": evaluate_retriever(
            model,
            holdout_query_tensor,
            holdout_key_tensor,
            holdout_attributes,
            holdout_attributes,
        ),
        "short_fact_holdout": evaluate_retriever(
            model,
            short_holdout_query_tensor,
            short_holdout_key_tensor,
            short_holdout_attributes,
            short_holdout_attributes,
        ),
        "update_pair_holdout": {
            "positive_probability_mean": float(
                torch.sigmoid(
                    model.text_retriever(update_positive_query_tensor, update_positive_key_tensor)
                ).mean().detach().cpu()
            ),
            "negative_probability_mean": float(
                torch.sigmoid(
                    model.text_retriever(update_negative_query_tensor, update_negative_key_tensor)
                ).mean().detach().cpu()
            ),
        },
        "redteam_holdout": {
            "positive_probability_mean": float(
                torch.sigmoid(
                    model.text_retriever(
                        redteam_holdout_query_tensor[redteam_holdout_labels >= 0.5],
                        redteam_holdout_key_tensor[redteam_holdout_labels >= 0.5],
                    )
                ).mean().detach().cpu()
            ),
            "negative_probability_mean": float(
                torch.sigmoid(
                    model.text_retriever(
                        redteam_holdout_query_tensor[redteam_holdout_labels < 0.5],
                        redteam_holdout_key_tensor[redteam_holdout_labels < 0.5],
                    )
                ).mean().detach().cpu()
            ),
        },
    }
    (output_dir / "retriever_training.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
