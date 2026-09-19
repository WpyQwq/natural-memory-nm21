"""Integrate dynamic memory into a local Qwen3.5 checkpoint.

The original Hugging Face checkpoint is loaded unchanged. Decoder layers are
wrapped after loading, and only the separate memory module is trainable by
default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
import json
import hashlib
import inspect
import logging
import os
import re
import time
import weakref

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .memory_os_v2 import (
    KVBudgetManagerV2,
    MemoryOSV2,
    MemoryRouterV2,
    PagedMemoryBankV2,
    memory_record_to_dict,
)
from .tiered_memory_store_v2 import TieredMemoryStoreV2

logger = logging.getLogger(__name__)


# Qwen's local tokenizer already contains this unused FIM control token.  It
# gives us a reset signal without changing the base vocabulary or resizing a
# quantized embedding table.  Callers may still pass any other token id.
DEFAULT_MEMORY_RESET_TOKEN = "<|fim_prefix|>"


def split_memory_candidates(text: str, *, max_candidates: int = 8) -> list[str]:
    """Split one user turn into bounded fact-sized write candidates.

    This is only a write-side segmentation aid.  The model's internal policy
    still decides whether each candidate is durable; the read path remains
    entirely inside the model-owned memory bank.
    """

    text = " ".join(str(text).strip().split())
    if not text:
        return []
    # A comma is not a safe write boundary.  In natural conversation it is
    # common for the second half of a sentence to negate or qualify the
    # first half (for example, "mentioned X, but it is not important").
    # Splitting there would make the policy see a misleading fragment and can
    # permanently store noise.  Keep clauses intact and only split at hard
    # sentence boundaries.
    parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])\s*", text)
        if part.strip()
    ]
    if len(parts) <= 1:
        return [text]
    if len(parts) <= max_candidates:
        return parts
    # Preserve all text when a long paragraph exceeds the safety bound.  The
    # exact sequence can still be stored in one slot without silent loss.
    return [text]


def looks_like_question(text: str) -> bool:
    """Reject question-shaped turns from the automatic durable-write path."""

    normalized = " ".join(str(text).strip().split())
    if not normalized:
        return False
    if any(mark in normalized for mark in ("？", "?")):
        return True
    return normalized.startswith(
        (
            "请问",
            "什么",
            "为什么",
            "怎么",
            "如何",
            "能不能",
            "是否",
            "有没有",
            "如果",
            "帮我",
            "解释",
        )
    )


def infer_memory_metadata(text: str) -> dict[str, str | bool]:
    """Extract conservative versioning metadata from an explicit user fact.

    This is not the memory decision and it never invents a value.  The learned
    policy still decides whether a turn is durable.  The small parser only
    supplies an entity/attribute/value triple when the user used an explicit
    ``X is Y``-style statement, allowing the version ledger to retire stale
    values instead of treating every new sentence as an unrelated fragment.
    """

    normalized = " ".join(str(text).strip().split())
    if not normalized:
        return {"kind": "empty", "should_write": False}
    forget = any(mark in normalized for mark in ("删除", "忘记", "忘掉", "清除", "撤回", "作废"))
    if forget:
        attribute = ""
        match = re.search(r"我的(.{1,24}?)(?:的记忆|记录|信息|资料)", normalized)
        if match:
            attribute = match.group(1).strip(" ：:，,。.!！？")
        if not attribute:
            match = re.search(r"(?:忘记|忘掉|删除|清除|撤回|作废)(?:我的|关于我的)?(.{1,24}?)(?:吧|了|。|！|！|$)", normalized)
            if match:
                attribute = match.group(1).strip(" ：:，,。.!！？")
        return {
            "kind": "forget",
            "should_write": False,
            "entity": "user",
            "attribute": attribute,
            "value": "",
        }
    if looks_like_question(normalized):
        return {"kind": "query", "should_write": False}

    # Keep the operator list intentionally explicit.  If no unambiguous
    # operator is present, the text remains an unstructured candidate and the
    # model's policy may still decide to store it as an episode.
    #
    # The bare colon must not match when a single ASCII letter precedes it, because that
    # colon belongs to a Windows drive letter.  Measured: 「记住：我的评测报告放在
    # E:\deepseek\artifacts。」 uses 放在 rather than 是, so the parser fell through to the
    # colon and split on the drive letter itself, producing attribute 「评测报告放在 E」 and
    # value 「\deepseek\artifacts」 -- a fact filed under a key that can never match the same
    # fact written by the agent (评测报告路径), so the two could never version each other and
    # the stale value stayed active.  A colon after CJK text is a real separator and is kept.
    operator = r"(?:改为|改成|更新为|设置为|变为|是|为|叫|(?<![A-Za-z])[:：]|=)"
    patterns = (
        rf"我的(?P<attribute>.{{1,24}}?){operator}(?P<value>.{{1,80}}?)(?:，|,|。|！|!|；|;|$)",
        rf"(?P<attribute>常用时区|时区|工作地点|办公地点|常住城市|居住地|最喜欢的水果|水果偏好|项目代号|工作代号|提醒时间|默认输出风格|备用联系人|编辑器|开发工具|语言|名字|姓名)\s*{operator}\s*(?P<value>.{{1,80}}?)(?:，|,|。|！|!|；|;|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is None:
            continue
        attribute = match.group("attribute").strip(" ：:，,。.!！？")
        value = match.group("value").strip(" ：:，,。.!！？")
        if attribute and value:
            return {
                "kind": "correction" if any(mark in normalized for mark in ("更正", "更新", "改为", "改成", "变为")) else "fact",
                "should_write": True,
                "entity": "user",
                "attribute": attribute,
                "value": value,
            }
    return {"kind": "episode", "should_write": True}


def format_memory_evidence(
    text: str,
    *,
    entity: str = "",
    attribute: str = "",
    value: str = "",
) -> str:
    """Render a compact, explicit evidence card for the model reader.

    The router stores structured fields separately, but Qwen can only use what
    is present in the retrieved token prefix.  Keeping the raw fragment and
    the verified value together prevents a successful retrieval from becoming
    a generation-time refusal when the original fragment was terse or split.
    """

    # Keep the proven v2 order for normal records.  Putting metadata before
    # the factual fragment made Qwen spend its short answer budget explaining
    # labels instead of copying the answer, and caused a measurable regression
    # on the dirty-corpus protocol.  The structured fields remain available
    # after the source fragment for audit and exact phrase reuse.
    lines = ["【长期记忆证据】", f"事实：{str(text).strip()}"]
    if str(entity).strip():
        lines.append(f"实体：{str(entity).strip()}")
    if str(attribute).strip():
        lines.append(f"属性：{str(attribute).strip()}")
    if str(value).strip():
        lines.append(f"已确认值：{str(value).strip()}")
    answer_fields = [
        field.strip()
        for field in (str(entity), str(attribute), str(value))
        if field.strip()
    ]
    if answer_fields:
        # Put a compact copy target next to the fact so a short generation
        # budget is spent answering instead of explaining retrieval metadata.
        lines.append(f"可直接复述的关键短语：{'；'.join(answer_fields)}")
    lines.append(
        "回答要求：第一句先直接回答问题，并原样复述上面的关键短语；"
        "需要多个字段时一次性列出，不要先解释检索过程；证据不足就明确说不知道。"
    )
    return "\n".join(lines)


#: Score an existing record must reach before a turn with no parseable attribute may inherit its
#: conflict key.
#:
#: Deliberately an absolute score, NOT the router's ``score_margin``.  The margin is
#: ``top - second`` over *all* returned records, and the automatic layer inserts a near-duplicate
#: record for every turn it stores, so the margin collapses on real traffic: measured with one
#: stored fact plus the previous turn's episode, the margin fell from 8.591 to 0.1523 while the
#: identified record's own score stayed 8.591.  A margin gate therefore blocks exactly the case it
#: was meant to allow.
#:
#: Measured clusters instead (only the old record present, real write-path timing):
#:   * same-attribute restatement -- identified record scores 8.591 / 8.6325 / 8.64
#:   * unrelated turn              -- best score 0.682 / 0.7421, or the router abstains (0)
#: The floor sits in the empty middle of that gap.
_INHERIT_SCORE_FLOOR = 5.0


def memory_origin(text: str) -> str:
    """Hash the user turn a record came from.

    Records carry this so a forget request can retire everything one turn told the
    model, not just the single record whose key happened to match.  It is derived
    from the turn text rather than a counter so the automatic write path and any
    explicit write issued while serving that turn independently compute the same
    value and nothing has to be shared between them.

    Blank text yields an empty origin, never the hash of the empty string: a shared
    "empty" origin would make every text-less record one group, and forgetting any
    one of them would retract all of them.
    """

    normalized = str(text or "").strip()
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def memory_fact_sentence(text: str) -> str:
    """Return the utterance a stored evidence card was built from.

    Records hold a rendered card (``【长期记忆证据】 …``) whose ``事实`` line is the
    original utterance.  A rebuild has to reuse that sentence rather than nest a card
    inside itself.
    """

    body = str(text or "").strip()
    if not body.startswith("【长期记忆证据】"):
        return body
    for line in body.splitlines():
        if line.startswith("事实："):
            return line[len("事实："):].strip()
    return body


def compose_corrected_evidence(
    old: dict[str, Any],
    *,
    entity: Optional[str] = None,
    attribute: Optional[str] = None,
    value: Optional[str] = None,
) -> Optional[str]:
    """Rebuild a record's evidence card so it agrees with the corrected fields.

    What the model reads is the record's *stored text*, not its structured fields, so a
    correction that only supplies a new value used to leave the card -- and its token
    cache -- still asserting the old one.  Measured: a correction to ``value=王五`` reported
    success, bumped the version and superseded the old record, while every injected card
    read ``已确认值：Wpy`` and the model answered ``Wpy``.  Silent, because the bank looked
    correct from the outside.

    Returns ``None`` when there is nothing to rebuild (no field supplied, or the record
    carries no structured identity to render), which keeps the old behaviour for records
    this cannot describe better than they already describe themselves.
    """

    if entity is None and attribute is None and value is None:
        return None
    next_entity = str(old.get("entity") or "") if entity is None else str(entity)
    next_attribute = str(old.get("attribute") or "") if attribute is None else str(attribute)
    next_value = str(old.get("value") or "") if value is None else str(value)
    if not (next_entity.strip() and next_attribute.strip() and next_value.strip()):
        return None
    # The 事实 line is what the model actually copies from, so leaving the superseded value
    # in it would keep the answer stale even after 已确认值 was corrected.  Substitute the
    # old value in place, which preserves the sentence's natural phrasing; the model does
    # the same thing on its own path ("我的名字是 Wpy" -> "我的名字是 王五"), and the original
    # utterance is still kept in the record's evidence for audit.
    sentence = memory_fact_sentence(str(old.get("text") or ""))
    old_value = str(old.get("value") or "").strip()
    if value is not None and old_value and old_value in sentence:
        sentence = sentence.replace(old_value, next_value)
    elif value is not None:
        sentence = f"{next_attribute}是 {next_value}"
    return format_memory_evidence(
        sentence,
        entity=next_entity,
        attribute=next_attribute,
        value=next_value,
    )


def _looks_like_grounded_memory_query(text: str) -> bool:
    """Identify questions whose factual answer must come from saved state.

    The guard is intentionally narrower than "all questions".  Ordinary
    general-knowledge chat must keep the base Qwen behavior, while questions
    about the user, this project, the repository, or prior decisions must not
    be completed from generic model knowledge when the memory reader found no
    supporting record.
    """

    normalized = " ".join(str(text).strip().split()).lower()
    if not normalized:
        return False
    grounded_markers = (
        "我的",
        "我之前",
        "之前告诉",
        "当前项目",
        "这个项目",
        "本项目",
        "仓库",
        "代码库",
        "readme",
        "记忆",
        "历史上下文",
        "重启后",
        "原版对照",
        "当前模型",
        "这套模型",
        "长期记忆",
    )
    return any(marker in normalized for marker in grounded_markers)


@dataclass
class QwenMemoryConfig:
    memory_slots: int = 16
    memory_dim: int = 512
    layer_indices: Optional[tuple[int, ...]] = None
    read_scale: float = 1.0
    write_scale: float = 1.0
    mode: str = "residual"
    blend_init: float = 0.0
    direct_logit_scale: float = 0.0
    write_token_offset: Optional[int] = None
    broadcast_write: bool = False
    raw_token_write: bool = False
    raw_logit_scale: float = 0.0
    native_mode: bool = False
    persistent_memory: bool = False
    reset_token_id: Optional[int] = None
    summary_pooling: bool = True
    # Natural-language episodic memory keeps the exact token sequence of a
    # selected write inside the model-owned memory state.  The model retrieves
    # and injects these tokens internally; callers never need to replay chat
    # history or assemble a prompt by hand.
    natural_language_memory: bool = False
    text_memory_tokens: int = 256
    text_memory_top_k: int = 2
    text_memory_threshold: float = 0.40
    text_memory_write_threshold: float = 0.5
    text_memory_replace_threshold: float = 0.35
    # A slightly more permissive boundary for recognizing a same-attribute
    # update. Unrelated fragments still remain independent V2 records.
    text_memory_update_overlap_threshold: float = 0.30
    # A learned semantic score alone must be very strong before it is allowed
    # to retire an older fragment. A false update destroys future recall of
    # the independent fragment, so this is stricter than the read threshold.
    text_memory_semantic_update_threshold: float = 0.95
    text_memory_key_tokens: int = 128
    # Weight of the trained ``memory_router_v2`` pair score when ordering records that
    # carry a semantic key; the remainder goes to the packaged ``text_retriever``.
    #
    # Default 0.5, set from a **fresh-process** measurement (the earlier in-process
    # comparison was order-confounded and had to be retracted).  C segment, 48 episodes
    # (40 answerable + 8 abstention), 24 same-shape candidates, prior scale 1.0:
    #   blend 0.00 -> answerable 65.00%, wrong-attribute 35.00%   (historical)
    #   blend 0.50 -> answerable 70.00%, wrong-attribute 27.50%
    #   blend 1.00 -> answerable 62.50%, wrong-attribute 37.50%
    # so the two scorers are complementary and 0.5 is the optimum, not a monotone artefact.
    memory_record_router_blend: float = 0.5
    # Coverage gate: refuse a question whose asked-about attribute the bank does not hold.
    # Measured on the zero-overlap eval set: unknown refusal 100.00%, known false refusal
    # 0.00% (tp=50 fn=0 fp=0 tn=250), versus a 75.00% leak without it.  Off by default so a
    # package opts in explicitly, together with the head file it should load.
    memory_coverage_gate: bool = False
    memory_attribute_head: str = ""
    # Fraction of the head's vocabulary the bank must actually populate before the coverage
    # gate is allowed to judge.  0.9 keeps it on the domain the head was trained for and
    # off elsewhere; see the note in setup_attribute_coverage.
    memory_coverage_vocabulary_fraction: float = 0.9
    # Scale applied to every additive prior in ``PagedMemoryBankV2._record_scores``.
    # 1.0 is the historical behaviour; 0.0 leaves record ordering entirely to the learned
    # scorer.  Exposed because those priors are 83-88% of the summed score while the
    # learned term is 12-17%, so this is the knob that can actually change which records
    # reach the model (changing the learned ranker alone measurably cannot).
    memory_prior_scale: float = 1.0
    text_memory_overlap_threshold: float = 0.22
    automatic_memory: bool = False
    auto_memory_threshold: float = 0.35
    # v2 adds a separately trained natural-language forget decision.  Version
    # 1 keeps the old one-logit policy byte-compatible with existing adapters.
    automatic_memory_policy_version: int = 1
    auto_forget_threshold: float = 0.50
    # Natural Memory v2: compact addressed pages sit beside the v1 hot text
    # bank.  V2 is opt-in so existing checkpoints remain byte-compatible.
    memory_version: int = 1
    hierarchical_memory: bool = False
    memory_router_dim: int = 128
    memory_router_heads: int = 8
    memory_page_capacity: int = 32
    # 32K pages x 32 records/page gives a one-million-record address space.
    # Pages are allocated lazily; this is a capacity, not a startup tensor.
    memory_max_pages: int = 32768
    memory_hot_pages: int = 8
    memory_top_k_pages: int = 4
    memory_top_k_records: int = 8
    memory_max_hops: int = 3
    memory_coarse_index_bits: int = 20
    memory_v2_read_threshold: float = 0.65
    memory_v2_write_threshold: float = 0.50
    # The learned route must have a calibrated margin before a paraphrase
    # without exact token overlap is allowed into the model context.  Keeping
    # this opt-in preserves compatibility with older adapters.
    memory_min_read_margin: float = 0.0
    memory_require_evidence: bool = False
    # ``embedded`` keeps the portable user snapshot inside the safetensors
    # package. ``tiered`` additionally opens a durable page store so cold
    # records can leave RAM while the same model-owned reader remains active.
    memory_storage_mode: str = "embedded"
    memory_storage_path: Optional[str] = None
    memory_resident_pages: int = 256
    # Embedded mode keeps every record in process RAM.  Only this bounded hot
    # cache is copied to the execution device; it is not a second store.
    memory_gpu_cache_records: int = 256
    memory_gpu_cache_tokens: int = 131072
    # Keep a large VRAM safety margin for the model weights, activations and
    # KV cache.  Memory records that do not fit remain in system RAM.
    memory_gpu_cache_reserve_mb: int = 2048
    memory_gpu_cache_adaptive: bool = True
    # HF generation can keep the active cache on CPU while the current layer
    # is executing.  This is opt-in because it trades GPU memory for PCIe
    # traffic and is therefore not always faster on short prompts.
    kv_offload: bool = False
    kv_cache_implementation: Optional[str] = None
    kv_offload_only_non_sliding: bool = True
    # When a prompt exceeds the hot-window budget, archive the old prefix as
    # V2 context records and run Qwen only on the recent window.  The reader
    # remains model-owned; callers do not have to assemble a retrieval prompt.
    auto_compact_context: bool = True
    context_chunk_tokens: int = 512
    context_archive_max_records: int = 1_048_576
    kv_budget_tokens: int = 32768
    kv_hard_max_tokens: int = 131072
    kv_compaction_trigger: float = 0.90
    kv_keep_recent_tokens: int = 8192

    def __post_init__(self) -> None:
        if self.mode not in {"residual", "blend", "replace"}:
            raise ValueError("mode must be one of: residual, blend, replace")
        if not 0.0 <= self.blend_init <= 1.0:
            raise ValueError("blend_init must be between 0 and 1")
        if self.direct_logit_scale < 0.0:
            raise ValueError("direct_logit_scale must be non-negative")
        if self.write_token_offset is not None and self.write_token_offset < 1:
            raise ValueError("write_token_offset must be positive")
        if self.raw_logit_scale < 0.0:
            raise ValueError("raw_logit_scale must be non-negative")
        if self.text_memory_tokens < 1:
            raise ValueError("text_memory_tokens must be positive")
        if self.text_memory_top_k < 1:
            raise ValueError("text_memory_top_k must be positive")
        if not 0.0 <= self.text_memory_threshold <= 1.0:
            raise ValueError("text_memory_threshold must be between 0 and 1")
        if not 0.0 <= self.text_memory_write_threshold <= 1.0:
            raise ValueError("text_memory_write_threshold must be between 0 and 1")
        if not -1.0 <= self.text_memory_replace_threshold <= 1.0:
            raise ValueError("text_memory_replace_threshold must be between -1 and 1")
        if not -1.0 <= self.text_memory_update_overlap_threshold <= 1.0:
            raise ValueError("text_memory_update_overlap_threshold must be between -1 and 1")
        if not 0.0 <= self.text_memory_semantic_update_threshold <= 1.0:
            raise ValueError("text_memory_semantic_update_threshold must be between 0 and 1")
        if self.text_memory_key_tokens < 1:
            raise ValueError("text_memory_key_tokens must be positive")
        if not 0.0 <= self.text_memory_overlap_threshold <= 1.0:
            raise ValueError("text_memory_overlap_threshold must be between 0 and 1")
        if not 0.0 <= self.auto_memory_threshold <= 1.0:
            raise ValueError("auto_memory_threshold must be between 0 and 1")
        if self.automatic_memory_policy_version < 1:
            raise ValueError("automatic_memory_policy_version must be positive")
        if not 0.0 <= self.auto_forget_threshold <= 1.0:
            raise ValueError("auto_forget_threshold must be between 0 and 1")
        if self.memory_version < 1:
            raise ValueError("memory_version must be positive")
        if self.memory_router_heads < 1:
            raise ValueError("memory_router_heads must be positive")
        if self.memory_router_dim < 8 or self.memory_router_dim % self.memory_router_heads != 0:
            raise ValueError("memory_router_dim must be divisible by memory_router_heads")
        if self.memory_page_capacity < 1 or self.memory_max_pages < 1:
            raise ValueError("memory page limits must be positive")
        if self.memory_top_k_pages < 1 or self.memory_top_k_records < 1:
            raise ValueError("memory top-k limits must be positive")
        if self.memory_max_hops < 1:
            raise ValueError("memory_max_hops must be positive")
        if not 0.0 <= self.memory_v2_read_threshold <= 1.0:
            raise ValueError("memory_v2_read_threshold must be between 0 and 1")
        if not 0.0 <= self.memory_v2_write_threshold <= 1.0:
            raise ValueError("memory_v2_write_threshold must be between 0 and 1")
        if self.memory_min_read_margin < 0.0:
            raise ValueError("memory_min_read_margin must be non-negative")
        if self.memory_storage_mode not in {"embedded", "tiered"}:
            raise ValueError("memory_storage_mode must be embedded or tiered")
        if self.memory_resident_pages < self.memory_hot_pages:
            raise ValueError("memory_resident_pages must be >= memory_hot_pages")
        if self.memory_gpu_cache_records < 0 or self.memory_gpu_cache_tokens < 0:
            raise ValueError("memory GPU cache limits must be non-negative")
        if self.memory_gpu_cache_reserve_mb < 0:
            raise ValueError("memory_gpu_cache_reserve_mb must be non-negative")
        if self.kv_cache_implementation is not None and not str(self.kv_cache_implementation).strip():
            raise ValueError("kv_cache_implementation must be non-empty when provided")
        if self.context_chunk_tokens < 1:
            raise ValueError("context_chunk_tokens must be positive")
        if self.context_archive_max_records < 1:
            raise ValueError("context_archive_max_records must be positive")
        if not 1 <= self.kv_budget_tokens <= self.kv_hard_max_tokens:
            raise ValueError("kv_budget_tokens must be inside [1, kv_hard_max_tokens]")

    def resolved_layers(self, num_hidden_layers: int) -> tuple[int, ...]:
        if self.layer_indices is not None:
            layers = tuple(sorted(set(self.layer_indices)))
        else:
            stride = max(1, num_hidden_layers // 4)
            layers = tuple(min(num_hidden_layers - 1, stride * i - 1) for i in range(1, 5))
            layers = tuple(sorted(set(layers)))
        if not layers or min(layers) < 0 or max(layers) >= num_hidden_layers:
            raise ValueError(f"layer_indices must be inside [0, {num_hidden_layers})")
        return layers


def load_memory_config(adapter_dir: str | Path) -> QwenMemoryConfig:
    """Load the architecture/configuration metadata saved with an adapter."""

    metadata_path = Path(adapter_dir) / "memory_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved = metadata.get("memory_config", {})
    saved_layers = metadata.get("layer_indices", saved.get("layer_indices"))
    return QwenMemoryConfig(
        memory_slots=int(saved.get("memory_slots", 16)),
        memory_dim=int(saved.get("memory_dim", 512)),
        layer_indices=tuple(saved_layers) if saved_layers is not None else None,
        read_scale=float(saved.get("read_scale", 1.0)),
        write_scale=float(saved.get("write_scale", 1.0)),
        mode=str(saved.get("mode", "residual")),
        blend_init=float(saved.get("blend_init", 0.0)),
        direct_logit_scale=float(saved.get("direct_logit_scale", 0.0)),
        write_token_offset=(
            int(saved["write_token_offset"]) if saved.get("write_token_offset") is not None else None
        ),
        broadcast_write=bool(saved.get("broadcast_write", False)),
        raw_token_write=bool(saved.get("raw_token_write", False)),
        raw_logit_scale=float(saved.get("raw_logit_scale", 0.0)),
        native_mode=bool(saved.get("native_mode", False)),
        persistent_memory=bool(saved.get("persistent_memory", False)),
        reset_token_id=(int(saved["reset_token_id"]) if saved.get("reset_token_id") is not None else None),
        summary_pooling=bool(saved.get("summary_pooling", True)),
        natural_language_memory=bool(saved.get("natural_language_memory", False)),
        text_memory_tokens=int(saved.get("text_memory_tokens", 256)),
        text_memory_top_k=int(saved.get("text_memory_top_k", 2)),
        text_memory_threshold=float(saved.get("text_memory_threshold", 0.40)),
        text_memory_write_threshold=float(saved.get("text_memory_write_threshold", 0.5)),
        text_memory_replace_threshold=float(saved.get("text_memory_replace_threshold", 0.35)),
        text_memory_update_overlap_threshold=float(
            saved.get("text_memory_update_overlap_threshold", 0.30)
        ),
        text_memory_semantic_update_threshold=float(
            saved.get("text_memory_semantic_update_threshold", 0.95)
        ),
        text_memory_key_tokens=int(saved.get("text_memory_key_tokens", 128)),
        memory_record_router_blend=float(saved.get("memory_record_router_blend", 0.5)),
        memory_coverage_gate=bool(saved.get("memory_coverage_gate", False)),
        memory_attribute_head=str(saved.get("memory_attribute_head", "")),
        memory_coverage_vocabulary_fraction=float(
            saved.get("memory_coverage_vocabulary_fraction", 0.9)),
        memory_prior_scale=float(saved.get("memory_prior_scale", 1.0)),
        text_memory_overlap_threshold=float(saved.get("text_memory_overlap_threshold", 0.22)),
        automatic_memory=bool(saved.get("automatic_memory", False)),
        auto_memory_threshold=float(saved.get("auto_memory_threshold", 0.35)),
        automatic_memory_policy_version=int(saved.get("automatic_memory_policy_version", 1)),
        auto_forget_threshold=float(saved.get("auto_forget_threshold", 0.50)),
        memory_version=int(saved.get("memory_version", 1)),
        hierarchical_memory=bool(saved.get("hierarchical_memory", False)),
        memory_router_dim=int(saved.get("memory_router_dim", 128)),
        memory_router_heads=int(saved.get("memory_router_heads", 8)),
        memory_page_capacity=int(saved.get("memory_page_capacity", 32)),
        memory_max_pages=int(saved.get("memory_max_pages", 32768)),
        memory_hot_pages=int(saved.get("memory_hot_pages", 8)),
        memory_top_k_pages=int(saved.get("memory_top_k_pages", 4)),
        memory_top_k_records=int(saved.get("memory_top_k_records", 8)),
        memory_max_hops=int(saved.get("memory_max_hops", 3)),
        memory_coarse_index_bits=int(saved.get("memory_coarse_index_bits", 20)),
        memory_v2_read_threshold=float(saved.get("memory_v2_read_threshold", 0.50)),
        memory_v2_write_threshold=float(saved.get("memory_v2_write_threshold", 0.50)),
        memory_min_read_margin=float(saved.get("memory_min_read_margin", 0.0)),
        memory_require_evidence=bool(saved.get("memory_require_evidence", False)),
        memory_storage_mode=str(saved.get("memory_storage_mode", "embedded")),
        memory_storage_path=(str(saved["memory_storage_path"]) if saved.get("memory_storage_path") else None),
        memory_resident_pages=int(saved.get("memory_resident_pages", 256)),
        memory_gpu_cache_records=int(saved.get("memory_gpu_cache_records", 256)),
        memory_gpu_cache_tokens=int(saved.get("memory_gpu_cache_tokens", 131072)),
        memory_gpu_cache_reserve_mb=int(saved.get("memory_gpu_cache_reserve_mb", 2048)),
        memory_gpu_cache_adaptive=bool(saved.get("memory_gpu_cache_adaptive", True)),
        kv_offload=bool(saved.get("kv_offload", False)),
        kv_cache_implementation=(
            str(saved["kv_cache_implementation"])
            if saved.get("kv_cache_implementation") is not None
            else None
        ),
        kv_offload_only_non_sliding=bool(saved.get("kv_offload_only_non_sliding", True)),
        auto_compact_context=bool(saved.get("auto_compact_context", True)),
        context_chunk_tokens=int(saved.get("context_chunk_tokens", 512)),
        context_archive_max_records=int(saved.get("context_archive_max_records", 1_048_576)),
        kv_budget_tokens=int(saved.get("kv_budget_tokens", 32768)),
        kv_hard_max_tokens=int(saved.get("kv_hard_max_tokens", 131072)),
        kv_compaction_trigger=float(saved.get("kv_compaction_trigger", 0.90)),
        kv_keep_recent_tokens=int(saved.get("kv_keep_recent_tokens", 8192)),
    )


def resolve_memory_reset_token(tokenizer: Any, token: str = DEFAULT_MEMORY_RESET_TOKEN) -> int:
    """Resolve a tokenizer control token suitable for clearing memory."""

    token_id = tokenizer.convert_tokens_to_ids(token)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or (unknown_id is not None and int(token_id) == int(unknown_id)):
        raise ValueError(
            f"reset token {token!r} is not present in this tokenizer; pass --reset-token-id instead"
        )
    return int(token_id)


class QwenDynamicMemory(nn.Module):
    """A compact float32 key-value memory connected to Qwen hidden states."""

    def __init__(self, hidden_size: int, config: QwenMemoryConfig) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        dim = config.memory_dim
        self.query = nn.Linear(hidden_size, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.read_out = nn.Linear(dim, hidden_size, bias=False)
        self.read_gate = nn.Linear(hidden_size, 1)
        self.slot_keys = nn.Parameter(torch.randn(config.memory_slots, dim) / dim**0.5)
        self.write_key = nn.Linear(hidden_size, dim, bias=False)
        self.write_value = nn.Linear(hidden_size, dim, bias=False)
        self.write_gate = nn.Linear(hidden_size, config.memory_slots)
        self.apply(self._init_weights)
        self.last_read_address: Optional[Tensor] = None
        self.last_read_relevance: Optional[Tensor] = None
        self.last_write_address: Optional[Tensor] = None
        self.last_write_probability: Optional[Tensor] = None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initial_state(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.zeros(
            batch_size,
            self.config.memory_slots,
            self.config.memory_dim,
            device=device,
            dtype=self.slot_keys.dtype,
        )

    def read(self, hidden_states: Tensor, memory: Tensor) -> Tensor:
        work_dtype = self.slot_keys.dtype
        hidden = hidden_states.to(work_dtype)
        memory = memory.to(device=hidden.device, dtype=work_dtype)
        q = self.query(hidden)
        k = self.key(memory)
        v = self.value(memory)
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.config.memory_dim**0.5
        address = scores.softmax(dim=-1)
        retrieved = torch.matmul(address, v)
        retrieved = self.read_out(retrieved)
        gate = torch.sigmoid(self.read_gate(hidden))
        self.last_read_address = address
        self.last_read_relevance = gate * address.max(dim=-1, keepdim=True).values
        return (self.config.read_scale * gate * retrieved).to(hidden_states.dtype)

    def update(
        self,
        hidden_states: Tensor,
        memory: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        work_dtype = self.slot_keys.dtype
        hidden = hidden_states.to(work_dtype)
        memory = memory.to(device=hidden.device, dtype=work_dtype)
        if self.config.write_token_offset is None:
            summary = hidden[:, -1]
        else:
            if hidden.shape[1] < self.config.write_token_offset:
                raise ValueError("write_token_offset exceeds the memory sequence length")
            summary = hidden[:, -self.config.write_token_offset]
        proposal = self.write_value(summary)
        if self.config.broadcast_write:
            strength = torch.ones(
                hidden.shape[0],
                self.config.memory_slots,
                device=hidden.device,
                dtype=work_dtype,
            )
            address = strength / float(self.config.memory_slots)
        else:
            address = (self.write_key(summary) @ self.slot_keys.t()).softmax(dim=-1)
            strength = torch.sigmoid(self.write_gate(summary)) * address
        self.last_write_address = address
        self.last_write_probability = strength.sum(dim=-1, keepdim=True)
        strength = (self.config.write_scale * strength).unsqueeze(-1)
        proposal = proposal[:, None, :].expand(-1, self.config.memory_slots, -1)
        return memory + strength * (proposal - memory)


class NaturalLanguageRetriever(nn.Module):
    """Trainable pair scorer for query-to-episodic-memory retrieval."""

    def __init__(self, hidden_size: int, projection_size: int = 256) -> None:
        super().__init__()
        self.query_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.pair_scorer = nn.Sequential(
            nn.Linear(projection_size * 4, projection_size),
            nn.SiLU(),
            nn.Linear(projection_size, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, query: Tensor, keys: Tensor) -> Tensor:
        query = F.normalize(query.float(), dim=-1)
        keys = F.normalize(keys.float(), dim=-1)
        query_projection = self.query_projection(query)
        key_projection = self.key_projection(keys)
        if key_projection.ndim == 3:
            query_projection = query_projection.unsqueeze(1)
        features = torch.cat(
            (
                query_projection.expand_as(key_projection),
                key_projection,
                query_projection.expand_as(key_projection) * key_projection,
                (query_projection.expand_as(key_projection) - key_projection).abs(),
            ),
            dim=-1,
        )
        return self.pair_scorer(features).squeeze(-1)


class AutomaticMemoryPolicy(nn.Module):
    """High-recall controller for deciding whether a user turn is durable."""

    def __init__(self, hidden_size: int, projection_size: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.LayerNorm(projection_size),
            nn.SiLU(),
            nn.Linear(projection_size, projection_size // 2),
            nn.SiLU(),
        )
        self.importance = nn.Linear(projection_size // 2, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.importance(self.encoder(hidden.float())).squeeze(-1)


class AutomaticMemoryPolicyV2(AutomaticMemoryPolicy):
    """Write/forget policy trained on the same natural-language turn state."""

    def __init__(self, hidden_size: int, projection_size: int = 256) -> None:
        super().__init__(hidden_size, projection_size)
        self.forget = nn.Linear(projection_size // 2, 1)
        nn.init.normal_(self.forget.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.forget.bias)

    def _features(self, hidden: Tensor) -> Tensor:
        return self.encoder(hidden.float())

    def forward(self, hidden: Tensor) -> Tensor:
        return self.importance(self._features(hidden)).squeeze(-1)

    def forget_logits(self, hidden: Tensor) -> Tensor:
        return self.forget(self._features(hidden)).squeeze(-1)


class NativeQwenDynamicMemory(QwenDynamicMemory):
    """Learned write/forget controller for model-owned persistent memory."""

    def __init__(self, hidden_size: int, config: QwenMemoryConfig) -> None:
        super().__init__(hidden_size, config)
        dim = config.memory_dim
        self.summary_score = nn.Linear(hidden_size, 1, bias=False)
        self.write_decision = nn.Linear(hidden_size, 1)
        # Forgetting must depend on both the incoming candidate and what is
        # already stored.  A hidden-only gate can learn "this sentence looks
        # like a fact", but it cannot learn "this fact conflicts with the
        # value in the addressed slot".
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_size + (2 * dim), dim),
            nn.SiLU(),
            nn.Linear(dim, config.memory_slots),
        )
        nn.init.normal_(self.summary_score.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.write_decision.weight, mean=0.0, std=0.02)
        for module in self.forget_gate:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.write_decision.bias, -1.5)
        nn.init.constant_(self.forget_gate[-1].bias, -2.0)
        self.last_write_probability: Optional[Tensor] = None
        self.last_forget_probability: Optional[Tensor] = None
        # Keep the semantic summary available to the automatic policy.  The
        # older ``last_write_representation`` is optimized for the memory
        # value path and may discard distinctions such as negation or a
        # question-shaped turn.
        self.last_write_summary: Optional[Tensor] = None
        self.last_write_representation: Optional[Tensor] = None

    def _summary(self, hidden_states: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        work_dtype = self.slot_keys.dtype
        hidden = hidden_states.to(work_dtype)
        if self.config.write_token_offset is not None:
            if hidden.shape[1] < self.config.write_token_offset:
                raise ValueError("write_token_offset exceeds the memory sequence length")
            return hidden[:, -self.config.write_token_offset]
        if not self.config.summary_pooling:
            return hidden[:, -1]
        scores = self.summary_score(hidden).squeeze(-1)
        if attention_mask is not None and attention_mask.ndim == 2:
            # During cached generation the decoder may expose only the newest
            # token while the attention mask still covers the full sequence.
            # Align the mask to the score sequence instead of assuming both
            # lengths are identical.
            score_length = scores.shape[1]
            mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
            if mask.shape[1] > score_length:
                mask = mask[:, -score_length:]
            elif mask.shape[1] < score_length:
                pad = torch.ones(
                    mask.shape[0],
                    score_length - mask.shape[1],
                    device=mask.device,
                    dtype=mask.dtype,
                )
                mask = torch.cat((mask, pad), dim=1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)

    def update(
        self,
        hidden_states: Tensor,
        memory: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        work_dtype = self.slot_keys.dtype
        memory = memory.to(device=hidden_states.device, dtype=work_dtype)
        summary = self._summary(hidden_states, attention_mask)
        self.last_write_summary = summary
        proposal = self.write_value(summary)
        self.last_write_representation = self.read_out(self.value(proposal))
        address = (self.write_key(summary) @ self.slot_keys.t()).softmax(dim=-1)
        slot_probability = torch.sigmoid(self.write_gate(summary))
        write_probability = torch.sigmoid(self.write_decision(summary))
        self.last_write_address = address
        matched_memory = torch.sum(address.unsqueeze(-1) * memory, dim=1)
        forget_features = torch.cat((summary, proposal, matched_memory), dim=-1)
        forget_probability = torch.sigmoid(self.forget_gate(forget_features))
        strength = self.config.write_scale * write_probability * slot_probability * address
        erase = strength * forget_probability
        proposal = proposal[:, None, :]
        updated = memory * (1.0 - erase.unsqueeze(-1))
        updated = updated + strength.unsqueeze(-1) * (proposal - updated)
        self.last_write_probability = write_probability
        self.last_forget_probability = forget_probability
        return updated


class _MemoryRuntime:
    def __init__(self, memory: QwenDynamicMemory) -> None:
        self.memory = memory
        self.state: Optional[Tensor] = None
        self.read_enabled = True
        self.update_enabled = True
        self.last_read: Optional[Tensor] = None
        self.raw_memory: Optional[Tensor] = None
        self.input_ids: Optional[Tensor] = None
        self.attention_mask: Optional[Tensor] = None
        self.output_embeddings: Optional[nn.Module] = None
        self.reset_mask: Optional[Tensor] = None
        self.text_token_ids: Optional[Tensor] = None
        self.text_token_mask: Optional[Tensor] = None
        self.text_slot_valid: Optional[Tensor] = None
        self.text_slot_keys: Optional[Tensor] = None
        self.text_slot_age: Optional[Tensor] = None
        self.text_write_counter: Optional[Tensor] = None
        self.text_key_token_ids: Optional[Tensor] = None
        self.text_key_token_mask: Optional[Tensor] = None
        self.text_last_written_slot: Optional[Tensor] = None
        self.text_read_slots: Optional[Tensor] = None
        self.text_read_relevance: Optional[Tensor] = None
        self.text_read_overlap: Optional[Tensor] = None
        # One read is performed at the request boundary.  Generation then
        # proceeds through the normal Qwen decode loop; this timer lets the
        # service benchmark separate routing/reranking from decode cost.
        self.text_read_seconds: float = 0.0
        self.text_prefix_tokens: int = 0
        self.v2_query_key: Optional[Tensor] = None
        self.v2_last_decisions: list[Any] = []
        self.text_prefix_used: bool = False
        self.text_guard_token_ids: Optional[Tensor] = None
        self.text_guard_token_mask: Optional[Tensor] = None
        self.text_guard_used: bool = False
        self.v2_no_evidence: bool = False
        self.auto_memory_probability: Optional[Tensor] = None
        self.auto_memory_forget_probability: Optional[Tensor] = None
        self.context_compaction: Optional[dict[str, Any]] = None
        # Persistent checkpoints are the durable user state.  Evaluation,
        # branch execution and a new temporary conversation must be able to
        # reset only their working state without mutating that checkpoint.
        self.use_persistent_state: bool = True


#: Attribute paths that expose a decoder layer stack, in resolution order.
DECODER_LAYER_PATHS = (
    "model.language_model.layers",  # Qwen3.5 / Gemma3-VL style multimodal backbones
    "model.layers",                 # Llama / Qwen2 / Qwen3 / Mistral / OLMoE / Phi / Cohere
    "language_model.layers",
    "layers",
    "model.decoder.layers",         # OPT / Bart
    "decoder.layers",
    "transformer.h",                # GPT-2 style stacks
    "gpt_neox.layers",
)


def resolve_text_config(config: Any) -> Any:
    """Return the text sub-config for multimodal *and* text-only backbones.

    Qwen3.5 carries ``config.text_config``; a plain ``LlamaForCausalLM`` does not,
    so reading the attribute directly raised AttributeError and blocked the whole
    embedding path for text-only architectures.
    """

    if hasattr(config, "get_text_config"):
        try:
            return config.get_text_config(decoder=True)
        except Exception:
            pass
    return getattr(config, "text_config", config)


def resolve_decoder_layers(model: nn.Module) -> tuple[str, Any]:
    """Return ``(attribute_path, layer_stack)`` for the backbone being adapted.

    The memory surgery swaps entries of this stack, so it has to be found for each
    architecture: Qwen3.5/Gemma3-VL keep the text decoder under
    ``model.language_model``, Llama/Qwen2/Qwen3/Mistral/OLMoE/Phi/Cohere expose
    ``model.layers``, OPT/Bart use ``model.decoder.layers`` and GPT-2 style stacks
    use ``transformer.h``.  Everything else in the memory stack (router, pager,
    retriever, policy, text bank) is architecture independent and needs only
    ``hidden_size``.
    """

    for path in DECODER_LAYER_PATHS:
        current: Any = model
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if (
            current is not None
            and hasattr(current, "__len__")
            and len(current) > 0
            and hasattr(current[0], "forward")
        ):
            return path, current
    raise AttributeError(
        "could not locate the decoder layer stack; tried: " + ", ".join(DECODER_LAYER_PATHS)
    )


class MemoryLayerAdapter(nn.Module):
    """Wrap one Qwen decoder layer and optionally replace its token mixer."""

    def __init__(
        self,
        inner: nn.Module,
        runtime: _MemoryRuntime,
        *,
        read: bool,
        write: bool,
        mode: str = "residual",
        blend_init: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"residual", "blend", "replace"}:
            raise ValueError("mode must be one of: residual, blend, replace")
        self.inner = inner
        self._runtime_ref = weakref.ref(runtime)
        self.read_enabled = read
        self.write_enabled = write
        self.mode = mode
        # Two decoder-layer calling conventions exist across architectures:
        # Gemma/Qwen3.5-style layers take ``position_embeddings`` as the second
        # positional parameter, while Llama/Qwen2/Qwen3/Mistral/OLMoE/Phi/Cohere
        # style layers take ``attention_mask`` there.  Forwarding position_embeddings
        # positionally therefore raised "got multiple values for argument
        # 'attention_mask'" and broke the memory path on that whole family, so the
        # convention is detected once and the matching call form is used.
        inner_parameters = inspect.signature(inner.forward).parameters
        parameter_names = list(inner_parameters)
        self._inner_parameter_names = parameter_names
        self._inner_accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in inner_parameters.values()
        )
        self._position_embeddings_position = (
            parameter_names.index("position_embeddings") if "position_embeddings" in parameter_names else None
        )
        self._forward_convention = (
            "positional"
            if self._position_embeddings_position == 1
            else ("keyword" if self._position_embeddings_position is not None else "absent")
        )
        if mode == "blend":
            blend_init = min(max(float(blend_init), 1e-4), 1.0 - 1e-4)
            logit = torch.logit(torch.tensor(blend_init, dtype=torch.float32))
            inner_device = next(
                (parameter.device for parameter in inner.parameters() if parameter.device.type != "meta"),
                None,
            )
            if inner_device is not None:
                logit = logit.to(inner_device)
            self.blend_logit = nn.Parameter(logit)

    def _call_inner(
        self,
        hidden_states: Tensor,
        position_embeddings: Any,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        past_key_values: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Call the wrapped decoder layer with its own argument convention.

        ``position_embeddings`` is passed positionally only when the layer really
        declares it as the second positional parameter (Qwen3.5/Gemma style), so the
        existing behaviour is bit-for-bit unchanged for those models; otherwise it is
        passed by keyword, which is what Llama/Qwen2/Qwen3/Mistral/OLMoE-style layers
        expect.  Layers without such a parameter (older architectures that compute
        rotary internally) are called without it.
        """

        if self._forward_convention == "positional":
            # Qwen3.5 / Gemma / Mixtral style: the historical call form, unchanged.
            return self.inner(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        # Architecture-agnostic form: pass exactly the arguments this layer
        # declares.  Parameter *names* differ too, not just positions -- for
        # example Ling/Bailing uses ``past_key_value`` (singular) while
        # Llama/Qwen use ``past_key_values``, and some layers have no ``**kwargs``
        # to absorb the difference.
        available: dict[str, Any] = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_values,
            "past_key_values": past_key_values,
            "position_embeddings": position_embeddings,
        }
        arguments = {
            name: available[name]
            for name in self._inner_parameter_names
            if name in available
        }
        if self._inner_accepts_kwargs:
            arguments.update(kwargs)
        return self.inner(hidden_states, **arguments)

    def _original_token_mixer(
        self,
        normalized_hidden: Tensor,
        position_embeddings: Any,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        past_key_values: Any,
        kwargs: dict[str, Any],
    ) -> Tensor:
        layer_type = getattr(self.inner, "layer_type", None)
        if layer_type == "linear_attention":
            output = self.inner.linear_attn(
                hidden_states=normalized_hidden,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
            return output[0] if isinstance(output, tuple) else output
        if layer_type == "full_attention":
            output = self.inner.self_attn(
                hidden_states=normalized_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            return output[0] if isinstance(output, tuple) else output
        raise TypeError(f"unsupported Qwen layer_type for surgery: {layer_type!r}")

    def _prefetch_cache_layer(self, past_key_values: Any) -> None:
        """Make an offloaded Qwen hybrid-cache layer ready before it is read."""

        if past_key_values is None or not getattr(past_key_values, "offloading", False):
            return
        component = getattr(self.inner, "self_attn", None)
        if component is None:
            component = getattr(self.inner, "linear_attn", None)
        layer_index = getattr(component, "layer_idx", None)
        if layer_index is None or not hasattr(past_key_values, "prefetch"):
            return
        only_non_sliding = bool(getattr(past_key_values, "only_non_sliding", True))
        parameter = next(component.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        past_key_values.prefetch(int(layer_index), only_non_sliding)
        # The stock offload helper uses a separate CUDA stream.  Qwen3.5
        # reads linear-attention state before it calls ``update_*``; make the
        # current layer's ownership synchronous here so a CPU state can never
        # reach a CUDA convolution or attention kernel.
        if 0 <= int(layer_index) < len(past_key_values.layers):
            cache_layer = past_key_values.layers[int(layer_index)]
            for name in ("keys", "values", "conv_states", "recurrent_states"):
                value = getattr(cache_layer, name, None)
                if isinstance(value, Tensor) and value.device != device:
                    setattr(cache_layer, name, value.to(device=device, non_blocking=False))

    def _memory_read(self, hidden_states: Tensor, runtime: _MemoryRuntime) -> Tensor:
        if runtime.read_enabled and self.read_enabled and runtime.state is not None:
            output = runtime.memory.read(hidden_states, runtime.state)
        else:
            output = torch.zeros_like(hidden_states)
        if self.write_enabled:
            runtime.last_read = output
        return output

    def _surgical_forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Any,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        past_key_values: Any,
        kwargs: dict[str, Any],
        runtime: _MemoryRuntime,
    ) -> Tensor:
        residual = hidden_states
        normalized_hidden = self.inner.input_layernorm(hidden_states)
        memory_output = self._memory_read(normalized_hidden, runtime)

        if self.mode == "replace":
            token_mixer = memory_output
        else:
            original_output = self._original_token_mixer(
                normalized_hidden,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_values,
                kwargs,
            )
            mix = torch.sigmoid(self.blend_logit).to(dtype=original_output.dtype)
            # A zero/empty memory must be an exact no-op.  This prevents a
            # trained blend coefficient from attenuating the original Qwen
            # token mixer on ordinary prompts that have no user memory.
            if runtime.state is None or not runtime.read_enabled:
                # Exact zero in the forward pass, but retain a surrogate
                # gradient so the blend parameter remains trainable in unit
                # tests and future calibration runs.
                mix = mix - mix.detach()
            else:
                memory_activity = runtime.state.detach().abs().mean()
                if float(memory_activity) <= 1e-6:
                    mix = mix - mix.detach()
            token_mixer = (1.0 - mix) * original_output + mix * memory_output

        hidden_states = residual + token_mixer
        residual = hidden_states
        hidden_states = self.inner.post_attention_layernorm(hidden_states)
        hidden_states = self.inner.mlp(hidden_states)
        return residual + hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Any = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any:
        runtime = self._runtime_ref()
        if runtime is None:
            return self._call_inner(
                hidden_states, position_embeddings, attention_mask, position_ids, past_key_values, kwargs
            )

        self._prefetch_cache_layer(past_key_values)
        # Natural-language memory is supplied as an internal text prefix. In
        # that mode the surgical residual path is intentionally disabled for
        # the generation call, so execute the original layer directly. This
        # avoids rebuilding layernorm/token-mixer arithmetic at every token
        # on the four memory-instrumented layers.
        if not runtime.read_enabled and not runtime.update_enabled:
            return self._call_inner(
                hidden_states, position_embeddings, attention_mask, position_ids, past_key_values, kwargs
            )
        if self.mode == "residual":
            hidden_states = hidden_states + self._memory_read(hidden_states, runtime)
            output = self._call_inner(
                hidden_states, position_embeddings, attention_mask, position_ids, past_key_values, kwargs
            )
        else:
            output = self._surgical_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_values,
                kwargs,
                runtime,
            )
        hidden_output = output[0] if isinstance(output, tuple) else output
        if runtime.update_enabled and self.write_enabled and runtime.state is not None:
            updated_state = runtime.memory.update(
                hidden_output,
                runtime.state,
                attention_mask=runtime.attention_mask,
            )
            if runtime.reset_mask is not None:
                reset_mask = runtime.reset_mask.to(device=updated_state.device, dtype=torch.bool)
                updated_state = torch.where(
                    reset_mask[:, None, None],
                    torch.zeros_like(updated_state),
                    updated_state,
                )
                runtime.reset_mask = None
            runtime.state = updated_state
            if (
                runtime.memory.config.raw_token_write
                and not runtime.read_enabled
                and runtime.input_ids is not None
            ):
                if runtime.output_embeddings is None:
                    raise RuntimeError("raw_token_write requires output embeddings")
                offset = runtime.memory.config.write_token_offset or 1
                if runtime.attention_mask is not None and runtime.attention_mask.ndim == 2:
                    last_valid = runtime.attention_mask.sum(dim=-1).to(torch.long) - 1
                    positions = (last_valid - (offset - 1)).clamp_min(0)
                    token_ids = runtime.input_ids.gather(1, positions[:, None]).squeeze(1)
                else:
                    token_ids = runtime.input_ids[:, -offset]
                # Store the output-projection row, not the input embedding row.
                # Qwen checkpoints are allowed to untie these matrices, and the
                # generation-time pointer is consumed through output_embeddings.
                runtime.raw_memory = runtime.output_embeddings.weight[token_ids].detach()
        return output


class QwenDynamicMemoryOutput:
    """Proxy for a normal HF output with one added field: ``memory``."""

    def __init__(self, base_output: Any, memory: Optional[Tensor]) -> None:
        self.base_output = base_output
        self.memory = memory

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_output, name)

    def __getitem__(self, key: Any) -> Any:
        return self.base_output[key]


class QwenDynamicMemoryModel(nn.Module):
    """Qwen3.5 with caller-owned persistent dynamic memory state."""

    def __init__(
        self,
        base_model: nn.Module,
        memory_config: Optional[QwenMemoryConfig] = None,
        *,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        text_config = resolve_text_config(base_model.config)
        self.memory_config = memory_config or QwenMemoryConfig()
        self.layer_indices = self.memory_config.resolved_layers(text_config.num_hidden_layers)
        memory_type = NativeQwenDynamicMemory if self.memory_config.native_mode else QwenDynamicMemory
        self.memory = memory_type(text_config.hidden_size, self.memory_config)
        self.text_retriever = (
            NaturalLanguageRetriever(text_config.hidden_size)
            if self.memory_config.natural_language_memory
            else None
        )
        policy_type = (
            AutomaticMemoryPolicyV2
            if self.memory_config.automatic_memory_policy_version >= 2
            else AutomaticMemoryPolicy
        )
        self.memory_policy = (
            policy_type(text_config.hidden_size)
            if self.memory_config.natural_language_memory and self.memory_config.automatic_memory
            else None
        )
        self.memory_router_v2: Optional[MemoryRouterV2] = None
        self.memory_os_v2: Optional[MemoryOSV2] = None
        self._memory_router_v2_ready = False
        if self.memory_config.hierarchical_memory or self.memory_config.memory_version >= 2:
            self.memory_router_v2 = MemoryRouterV2(
                text_config.hidden_size,
                router_dim=self.memory_config.memory_router_dim,
                num_heads=self.memory_config.memory_router_heads,
                max_hops=self.memory_config.memory_max_hops,
            )
            self.memory_os_v2 = self._new_memory_os_v2(text_config.hidden_size)
        self._text_retriever_ready = False
        self._memory_policy_ready = False
        self.runtime = _MemoryRuntime(self.memory)
        self.runtime.output_embeddings = self.base_model.get_output_embeddings()
        self._persistent_memory: Optional[Tensor] = None
        self.register_buffer("persistent_memory", torch.empty(0), persistent=True)
        self.register_buffer(
            "persistent_text_token_ids",
            torch.empty(0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "persistent_text_token_mask",
            torch.empty(0, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "persistent_text_slot_valid",
            torch.empty(0, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "persistent_text_slot_keys",
            torch.empty(0),
            persistent=True,
        )
        self.register_buffer(
            "persistent_text_slot_age",
            torch.empty(0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "persistent_text_write_counter",
            torch.empty(0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "persistent_text_key_token_ids",
            torch.empty(0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "persistent_text_key_token_mask",
            torch.empty(0, dtype=torch.bool),
            persistent=True,
        )

        if freeze_backbone:
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)

        self.decoder_layer_path, decoder_layers = resolve_decoder_layers(self.base_model)
        self._memory_adapters: list[MemoryLayerAdapter] = []
        for layer_index in self.layer_indices:
            layer = decoder_layers[layer_index]
            adapter = MemoryLayerAdapter(
                layer,
                self.runtime,
                read=True,
                write=layer_index == self.layer_indices[-1],
                mode=self.memory_config.mode,
                blend_init=self.memory_config.blend_init,
            )
            decoder_layers[layer_index] = adapter
            self._memory_adapters.append(adapter)

        self.memory.to(self._find_layer_device())
        if self.memory_router_v2 is not None:
            self.memory_router_v2.to(self._find_layer_device())
        if self.text_retriever is not None:
            self.text_retriever.to(self._find_layer_device())
        if self.memory_policy is not None:
            self.memory_policy.to(self._find_layer_device())
        if self.memory_config.persistent_memory:
            self.persistent_memory = self.memory.initial_state(
                1,
                device=self._find_layer_device(),
            )
            self._persistent_memory = self.persistent_memory.detach()
        if hasattr(self.base_model.config, "use_cache"):
            self.base_model.config.use_cache = False
        if hasattr(self.base_model.config, "text_config"):
            self.base_model.config.text_config.use_cache = False

    def _new_memory_os_v2(self, hidden_size: int) -> MemoryOSV2:
        """Create a V2 store from the configured page and candidate limits."""

        if self.memory_router_v2 is None:
            raise RuntimeError("cannot create V2 memory without a router")
        tier_store = None
        if self.memory_config.memory_storage_mode == "tiered":
            if not self.memory_config.memory_storage_path:
                raise ValueError("tiered memory requires memory_storage_path")
            tier_store = TieredMemoryStoreV2(
                self.memory_config.memory_storage_path,
                key_dim=self.memory_config.memory_router_dim,
                page_capacity=self.memory_config.memory_page_capacity,
            )
        bank = PagedMemoryBankV2(
            hidden_size,
            page_capacity=self.memory_config.memory_page_capacity,
            max_pages=self.memory_config.memory_max_pages,
            hot_pages=self.memory_config.memory_hot_pages,
            top_k_pages=self.memory_config.memory_top_k_pages,
            top_k_records=self.memory_config.memory_top_k_records,
            max_hops=self.memory_config.memory_max_hops,
            router=self.memory_router_v2,
            key_dim=self.memory_config.memory_router_dim,
            coarse_index_bits=self.memory_config.memory_coarse_index_bits,
            tier_store=tier_store,
            max_resident_pages=self.memory_config.memory_resident_pages,
            runtime_device=self._find_layer_device(),
            gpu_cache_records=self.memory_config.memory_gpu_cache_records,
            gpu_cache_tokens=self.memory_config.memory_gpu_cache_tokens,
            gpu_cache_reserve_mb=self.memory_config.memory_gpu_cache_reserve_mb,
            gpu_cache_adaptive=self.memory_config.memory_gpu_cache_adaptive,
            record_scorer=self._score_semantic_memory_records,
        )
        prior_scale = float(getattr(self.memory_config, "memory_prior_scale", 1.0) or 0.0)
        if prior_scale != 1.0:
            bank.prior_weights = {name: weight * prior_scale
                                  for name, weight in bank.prior_weights.items()}
        return MemoryOSV2(
            hidden_size,
            router=self.memory_router_v2,
            bank=bank,
            attribute_coverage=getattr(self, "_attribute_coverage", None),
            kv_budget=KVBudgetManagerV2(
                max_tokens=self.memory_config.kv_budget_tokens,
                hard_max_tokens=self.memory_config.kv_hard_max_tokens,
                compaction_trigger=self.memory_config.kv_compaction_trigger,
                keep_recent_tokens=self.memory_config.kv_keep_recent_tokens,
            ),
            read_threshold=self.memory_config.memory_v2_read_threshold,
            write_threshold=self.memory_config.memory_v2_write_threshold,
            record_scorer=self._score_semantic_memory_records,
            min_read_margin=self.memory_config.memory_min_read_margin,
            require_evidence=self.memory_config.memory_require_evidence,
        )

    @torch.no_grad()
    def setup_attribute_coverage(self, package_dir: Optional[Path] = None) -> None:
        """Load the attribute-coverage head when the config enables the coverage gate.

        The gate refuses a question whose asked-about attribute the bank does not hold.
        It is the only mechanism measured to work for that case: over nine score-geometry
        features the best fitted head reaches AUC 0.61 (and ~0.5 for three of four
        scorers), while the attribute head identifies the asked-about attribute on unseen
        paraphrases at 100.00% and the coverage check then yields 100.00% refusal with
        0.00% false refusal on the zero-overlap eval set.
        """

        self._attribute_coverage = None
        self._attribute_head_meta = None
        if not bool(getattr(self.memory_config, "memory_coverage_gate", False)):
            return
        configured = str(getattr(self.memory_config, "memory_attribute_head", "") or "").strip()
        if not configured:
            raise FileNotFoundError(
                "memory_coverage_gate is enabled but memory_attribute_head is empty")
        candidate = Path(configured)
        if not candidate.is_absolute() and package_dir is not None:
            candidate = Path(package_dir) / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"attribute head not found: {candidate}")
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        weight = payload["weight"].float()
        bias = payload["bias"].float()
        attributes = [str(name) for name in payload["attributes"]]

        head_vocabulary = {name.strip().lower() for name in attributes}
        min_coverage = float(getattr(self.memory_config, "memory_coverage_vocabulary_fraction", 0.9))
        coverage_stats = {"applicable": 0, "bypassed": 0}

        def coverage(query_key: Tensor):
            """Return (attribute, probability), or None when the gate must stand down.

            The head is a *closed* vocabulary: it can only answer with one of the
            attributes it was trained on, so on a bank whose attributes lie outside that
            vocabulary it would predict something absent and refuse every question.
            Measured on the v6 general suite (attributes such as "备用联系人", which the
            head never saw) that collapsed answerable accuracy to 0.00%.  The gate is
            therefore applied only while every attribute the bank actually holds is inside
            the head's vocabulary; otherwise it reports no opinion and the runtime behaves
            exactly as it did without the gate.
            """

            # Resolve the bank dynamically: ``clear_hierarchical_memory`` replaces
            # ``self.memory_os_v2`` on every reset, so a reference captured here would go
            # stale after the first case and the gate would silently never apply.
            live = self.memory_os_v2
            active = live.bank.active_by_conflict if live is not None else {}
            present = {key.split("::", 1)[1] for key in active if "::" in key}
            # Require the bank to actually populate the head's vocabulary, not merely to be
            # a subset of it.  A handful of incidental in-vocabulary attributes is what
            # made the gate fire on the general suite and refuse 2.50% of answerable
            # questions (answerable 100.00% -> 92.50%); demanding broad coverage keeps the
            # gate on the domain it was trained for and off everywhere else.
            if not present or len(present & head_vocabulary) < min_coverage * len(head_vocabulary):
                coverage_stats["bypassed"] += 1
                return None
            coverage_stats["applicable"] += 1
            key = query_key.reshape(-1).float()
            key = key / (key.norm() + 1e-6)
            logits = weight.t() @ key.to(weight.device) + bias
            probability = float(torch.softmax(logits, dim=-1).max().item())
            return attributes[int(logits.argmax().item())], probability

        self._attribute_coverage = coverage
        # Bind the gate to the live bank as well.  ``memory_os_v2`` is constructed during
        # model initialisation (``_new_memory_os_v2``), which happens *before* this head is
        # loaded, so the bank captured ``attribute_coverage=None`` and the gate was loaded
        # but never consulted.  Without this rebind the whole NM2.1 refusal mechanism is
        # inert for any already-constructed bank (observed via the control-plane selftest:
        # configured=True, loaded=True, bound_to_bank=False).
        if getattr(self, "memory_os_v2", None) is not None:
            self.memory_os_v2.attribute_coverage = coverage
        self._attribute_head_meta = {
            "path": str(candidate),
            "attributes": len(attributes),
            "vocabulary": sorted(head_vocabulary),
            "coverage_stats": coverage_stats,
        }

    def _score_semantic_memory_records(self, query_key: Tensor, candidate_keys: Tensor) -> Tensor:
        """Exact-rerank only the bounded page candidates with the trained reader.

        The coarse page index still uses ``MemoryRouterV2``.  This callback is
        deliberately restricted to records already admitted by that index, so
        learned natural-language matching improves recall without turning a
        million-record bank into a full attention operation.
        """

        if self.text_retriever is None or not self._text_retriever_ready:
            # NOTE (reverted experiment): routing this branch through
            # ``memory_router_v2`` instead of raw-key cosine was tried and is measurably
            # inert -- the record-level score is produced by the retriever, which is
            # baked into the merged package, so ``_text_retriever_ready`` is True and
            # this branch never executes in the shipped configuration.  Raw-key cosine
            # is measurably weaker than the trained router (30.40% vs 59.60% Top-1 on
            # the zero-overlap eval), so replacing it remains a plausible improvement,
            # but it must be verified in a package that actually lacks the retriever
            # before being shipped.  Left as-is for now.
            return F.cosine_similarity(
                query_key.reshape(1, -1).float(),
                candidate_keys.to(device=query_key.device).float(),
                dim=-1,
            )
        # The benchmark and the CPU-resident memory bank may provide query
        # keys on CPU, while the trained reranker normally lives on CUDA.
        # Keep the transfer local to this bounded exact-rerank step and return
        # scores to the caller's device so the rest of the memory OS remains
        # device-agnostic.
        retriever_device = next(self.text_retriever.parameters()).device
        query_for_model = query_key.to(device=retriever_device)
        candidates_for_model = candidate_keys.to(device=retriever_device)
        scores = self.text_retriever(
            query_for_model.reshape(1, -1),
            candidates_for_model.reshape(1, -1, candidates_for_model.shape[-1]),
        )
        retriever_scores = torch.sigmoid(scores[0]).detach().to(device=query_key.device)

        # Optional blend with the trained router's pair score.  Measured on the
        # zero-overlap eval set (250 answerable episodes, 24 same-shape candidates,
        # chance 4.17%) on *identical* frozen keys, the two scorers differ sharply at
        # Top-1: the packaged retriever ranks the correct fact first in 23.20% of cases
        # while the trained router reaches 59.60% (frozen-key cosine: 30.40%).  The
        # weight defaults to 0.0, i.e. exactly the historical behaviour, so enabling it
        # is an explicit, measurable decision rather than a silent change.
        blend = float(getattr(self.memory_config, "memory_record_router_blend", 0.0) or 0.0)
        router = self.memory_router_v2
        if (
            blend > 0.0
            and router is not None
            and self._memory_router_v2_ready
            and int(query_key.shape[-1]) == int(router.hidden_size)
            and int(candidate_keys.shape[-1]) == int(router.hidden_size)
        ):
            with torch.no_grad():
                router_device = next(router.parameters()).device
                projected = router.encode_key(
                    candidate_keys.to(router_device).reshape(-1, candidate_keys.shape[-1])
                )
                router_logits, _ = router.projected_scores(
                    query_key.to(router_device).reshape(1, -1),
                    projected.reshape(1, projected.shape[0], projected.shape[-1]),
                )
                router_scores = torch.sigmoid(router_logits[0]).detach().to(device=query_key.device)
            return (1.0 - blend) * retriever_scores + blend * router_scores
        return retriever_scores

    def _ensure_text_memory(self, batch_size: int, *, device: torch.device) -> None:
        """Create the fixed-size model-owned natural-language memory bank."""

        if not self.memory_config.natural_language_memory:
            return
        shape = (
            batch_size,
            self.memory_config.memory_slots,
            self.memory_config.text_memory_tokens,
        )
        if tuple(self.persistent_text_token_ids.shape) != shape:
            self.persistent_text_token_ids = torch.zeros(
                shape,
                dtype=torch.long,
                device=device,
            )
            self.persistent_text_token_mask = torch.zeros(
                shape,
                dtype=torch.bool,
                device=device,
            )
            self.persistent_text_slot_valid = torch.zeros(
                batch_size,
                self.memory_config.memory_slots,
                dtype=torch.bool,
                device=device,
            )
            hidden_size = self.memory.hidden_size
            self.persistent_text_slot_keys = torch.zeros(
                batch_size,
                self.memory_config.memory_slots,
                hidden_size,
                dtype=torch.float32,
                device=device,
            )
            self.persistent_text_slot_age = torch.full(
                (batch_size, self.memory_config.memory_slots),
                -1,
                dtype=torch.long,
                device=device,
            )
            self.persistent_text_write_counter = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )
            key_shape = (
                batch_size,
                self.memory_config.memory_slots,
                self.memory_config.text_memory_key_tokens,
            )
            self.persistent_text_key_token_ids = torch.zeros(
                key_shape,
                dtype=torch.long,
                device=device,
            )
            self.persistent_text_key_token_mask = torch.zeros(
                key_shape,
                dtype=torch.bool,
                device=device,
            )

    def _bind_text_memory(self, batch_size: int, *, device: torch.device) -> None:
        """Bind the current text bank to the per-call runtime."""

        if not self.memory_config.natural_language_memory:
            self.runtime.text_token_ids = None
            self.runtime.text_token_mask = None
            self.runtime.text_slot_valid = None
            return
        if self.memory_config.persistent_memory and self.runtime.use_persistent_state:
            self._ensure_text_memory(1, device=device)
        else:
            self._ensure_text_memory(batch_size, device=device)
        if not (self.memory_config.persistent_memory and self.runtime.use_persistent_state):
            shape = (
                batch_size,
                self.memory_config.memory_slots,
                self.memory_config.text_memory_tokens,
            )
            key_shape = (
                batch_size,
                self.memory_config.memory_slots,
                self.memory_config.text_memory_key_tokens,
            )
            if (
                self.runtime.text_token_ids is None
                or tuple(self.runtime.text_token_ids.shape) != shape
                or self.runtime.text_key_token_ids is None
                or tuple(self.runtime.text_key_token_ids.shape) != key_shape
            ):
                self.runtime.text_token_ids = torch.zeros(shape, dtype=torch.long, device=device)
                self.runtime.text_token_mask = torch.zeros(shape, dtype=torch.bool, device=device)
                self.runtime.text_slot_valid = torch.zeros(
                    batch_size, self.memory_config.memory_slots, dtype=torch.bool, device=device
                )
                self.runtime.text_slot_keys = torch.zeros(
                    batch_size, self.memory_config.memory_slots, self.memory.hidden_size,
                    dtype=torch.float32, device=device
                )
                self.runtime.text_slot_age = torch.full(
                    (batch_size, self.memory_config.memory_slots), -1, dtype=torch.long, device=device
                )
                self.runtime.text_write_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
                self.runtime.text_key_token_ids = torch.zeros(key_shape, dtype=torch.long, device=device)
                self.runtime.text_key_token_mask = torch.zeros(key_shape, dtype=torch.bool, device=device)
            return
        if self.persistent_text_token_ids.shape[0] == batch_size:
            self.runtime.text_token_ids = self.persistent_text_token_ids
            self.runtime.text_token_mask = self.persistent_text_token_mask
            self.runtime.text_slot_valid = self.persistent_text_slot_valid
            self.runtime.text_slot_keys = self.persistent_text_slot_keys
            self.runtime.text_slot_age = self.persistent_text_slot_age
            self.runtime.text_write_counter = self.persistent_text_write_counter
            self.runtime.text_key_token_ids = self.persistent_text_key_token_ids
            self.runtime.text_key_token_mask = self.persistent_text_key_token_mask
            return
        if self.persistent_text_token_ids.shape[0] == 1:
            self.runtime.text_token_ids = self.persistent_text_token_ids.expand(
                batch_size, -1, -1
            ).clone()
            self.runtime.text_token_mask = self.persistent_text_token_mask.expand(
                batch_size, -1, -1
            ).clone()
            self.runtime.text_slot_valid = self.persistent_text_slot_valid.expand(
                batch_size, -1
            ).clone()
            self.runtime.text_slot_keys = self.persistent_text_slot_keys.expand(
                batch_size, -1, -1
            ).clone()
            self.runtime.text_slot_age = self.persistent_text_slot_age.expand(
                batch_size, -1
            ).clone()
            self.runtime.text_write_counter = self.persistent_text_write_counter.expand(
                batch_size
            ).clone()
            self.runtime.text_key_token_ids = self.persistent_text_key_token_ids.expand(
                batch_size, -1, -1
            ).clone()
            self.runtime.text_key_token_mask = self.persistent_text_key_token_mask.expand(
                batch_size, -1, -1
            ).clone()
            return
        raise ValueError(
            "natural-language memory batch does not match the loaded persistent bank"
        )

    def _clear_text_memory(self, mask: Optional[Tensor] = None) -> None:
        if self.runtime.text_token_ids is None:
            return
        if mask is None:
            self.runtime.text_token_ids.zero_()
            self.runtime.text_token_mask.zero_()
            self.runtime.text_slot_valid.zero_()
            self.runtime.text_slot_keys.zero_()
            self.runtime.text_slot_age.fill_(-1)
            self.runtime.text_write_counter.zero_()
            self.runtime.text_key_token_ids.zero_()
            self.runtime.text_key_token_mask.zero_()
            self.runtime.text_last_written_slot = None
            self.runtime.text_prefix_used = False
            return
        mask = mask.to(device=self.runtime.text_token_ids.device, dtype=torch.bool)
        self.runtime.text_token_ids[mask] = 0
        self.runtime.text_token_mask[mask] = False
        self.runtime.text_slot_valid[mask] = False
        self.runtime.text_slot_keys[mask] = 0
        self.runtime.text_slot_age[mask] = -1
        self.runtime.text_write_counter[mask] = 0
        self.runtime.text_key_token_ids[mask] = 0
        self.runtime.text_key_token_mask[mask] = False
        self.runtime.text_last_written_slot = None

    @torch.no_grad()
    def _encode_model_key(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode a memory key with the frozen original Qwen representation."""

        previous_read = self.runtime.read_enabled
        previous_update = self.runtime.update_enabled
        self.runtime.read_enabled = False
        self.runtime.update_enabled = False
        try:
            embedding_layer = self.base_model.get_input_embeddings()
            input_ids = input_ids.to(embedding_layer.weight.device)
            attention_mask = attention_mask.to(input_ids.device)
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = output.hidden_states[-1].float()
            weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            key = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return F.normalize(key, dim=-1)
        finally:
            self.runtime.read_enabled = previous_read
            self.runtime.update_enabled = previous_update

    @torch.no_grad()
    def compact_context_for_kv(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        archive: bool = True,
        chunk_tokens: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Archive the old prefix and return a bounded hot-context window.

        This is the model-side boundary between the exact working KV and the
        lossy long-term memory path.  It never sends the current token to all
        pages: each archived chunk gets one address, and future reads still
        use the V2 coarse-page -> exact-record -> Top-K route.

        The method accepts already-tokenized input because it is also used by
        generation.  The archived token sequence is kept losslessly inside a
        V2 record; its semantic address is produced by the frozen Qwen hidden
        representation.  A caller may disable ``archive`` for measurement,
        in which case the method only reports the retention plan.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        budget = self.memory_os_v2.kv_budget if self.memory_os_v2 is not None else None
        if (
            self.memory_os_v2 is None
            or not self.memory_config.auto_compact_context
            or budget is None
        ):
            return input_ids, attention_mask, {
                "compacted": False,
                "reason": "disabled",
                "archived_records": 0,
                "original_tokens": [int(item) for item in attention_mask.sum(dim=-1).tolist()],
                "retained_tokens": [int(item) for item in attention_mask.sum(dim=-1).tolist()],
            }

        chunk_size = int(chunk_tokens or self.memory_config.context_chunk_tokens)
        if chunk_size < 1:
            raise ValueError("chunk_tokens must be positive")
        hot_limit = int(budget.max_tokens)
        pending: list[dict[str, Any]] = []
        pending_previous: list[Optional[int]] = []
        retained: list[Tensor] = []
        original_lengths: list[int] = []
        retained_lengths: list[int] = []

        for batch_index in range(input_ids.shape[0]):
            valid = input_ids[batch_index][attention_mask[batch_index].to(dtype=torch.bool)]
            original_lengths.append(int(valid.numel()))
            if valid.numel() <= hot_limit:
                retained.append(valid)
                retained_lengths.append(int(valid.numel()))
                continue

            archived_prefix = valid[:-hot_limit]
            retained_window = valid[-hot_limit:]
            retained.append(retained_window)
            retained_lengths.append(int(retained_window.numel()))
            previous_index: Optional[int] = None
            for start in range(0, int(archived_prefix.numel()), chunk_size):
                end = min(start + chunk_size, int(archived_prefix.numel()))
                if len(pending) >= self.memory_config.context_archive_max_records:
                    raise RuntimeError(
                        "context archive record limit reached; increase "
                        "context_archive_max_records before compacting this input"
                    )
                chunk = archived_prefix[start:end].detach()
                chunk_mask = torch.ones_like(chunk, dtype=torch.bool)
                key = self._encode_model_key(chunk.unsqueeze(0), chunk_mask.unsqueeze(0))[0]
                pending.append(
                    {
                        "text": f"context_chunk:{batch_index}:{start}:{end}",
                        "key": key,
                        "summary": key,
                        "memory_type": "context_chunk",
                        # Context chunks are ordered evidence, not competing
                        # values of one fact.  Leaving the fact-conflict
                        # fields empty prevents ordinary version resolution
                        # from superseding every earlier chunk in a document.
                        "entity": "",
                        "attribute": "",
                        "value": "",
                        "importance": 0.55,
                        "confidence": 0.80,
                        "source": "kv_compaction",
                        "evidence": [
                            f"batch:{batch_index}",
                            f"token_range:{start}:{end}",
                            f"token_count:{end - start}",
                        ],
                        "token_ids": chunk,
                        "token_mask": chunk_mask,
                        "trusted": True,
                        "force": True,
                    }
                )
                pending_previous.append(previous_index)
                previous_index = len(pending) - 1

        results: list[tuple[Any, str]] = []
        if pending and archive:
            results = self.memory_os_v2.write_batch(pending)
            # Add a forward chain after the atomic insert.  This gives the
            # multi-hop reader a deterministic path through adjacent chunks
            # without making page allocation depend on future record IDs.
            for result_index, previous_index in enumerate(pending_previous):
                if previous_index is None:
                    continue
                record = results[result_index][0]
                previous_record = results[previous_index][0]
                if previous_record.record_id not in record.related_ids:
                    record.related_ids.append(previous_record.record_id)
                    self.memory_os_v2.bank._store_record(record)

        max_retained = max(retained_lengths, default=0)
        compacted_ids = torch.zeros(
            (input_ids.shape[0], max_retained), dtype=input_ids.dtype, device=input_ids.device
        )
        compacted_mask = torch.zeros(
            (input_ids.shape[0], max_retained), dtype=attention_mask.dtype, device=input_ids.device
        )
        for batch_index, values in enumerate(retained):
            if values.numel() == 0:
                continue
            compacted_ids[batch_index, : values.numel()] = values.to(input_ids.device)
            compacted_mask[batch_index, : values.numel()] = 1
        return compacted_ids, compacted_mask, {
            "compacted": any(original > retained for original, retained in zip(original_lengths, retained_lengths)),
            "archived_records": len(results),
            "archived_tokens": sum(original - retained for original, retained in zip(original_lengths, retained_lengths)),
            "original_tokens": original_lengths,
            "retained_tokens": retained_lengths,
            "hot_limit": hot_limit,
            "archive_requested": bool(archive),
        }

    @torch.no_grad()
    def _rank_v2_text_matches(
        self,
        query_key: Tensor,
        query_token_ids: Tensor,
    ) -> list[tuple[float, int, Any]]:
        """Rank active V2 text records without materializing their payloads."""

        if self.memory_os_v2 is None:
            return []
        active = [
            record
            for record in self.memory_os_v2.bank.records.values()
            if record.status == "active"
        ]
        if not active:
            return []
        keys = torch.stack([record.key for record in active], dim=0)
        semantic_keys = [record.semantic_key for record in active]
        if (
            self.text_retriever is not None
            and self._text_retriever_ready
            and all(isinstance(value, Tensor) for value in semantic_keys)
        ):
            # Use the same learned pair scorer that governs hot-slot updates.
            # The generic bank cosine score is useful for coarse addressing but
            # cannot distinguish a same-entity, different-attribute fragment.
            scores = torch.sigmoid(
                self.text_retriever(
                    query_key.reshape(1, -1),
                    torch.stack(
                        [value for value in semantic_keys if isinstance(value, Tensor)],
                        dim=0,
                    ).to(query_key.device, dtype=query_key.dtype),
                )
            ).reshape(-1).cpu()
        else:
            scores = self.memory_os_v2.bank._score_candidates(query_key, keys).reshape(-1).cpu()
        query_unique = torch.unique(query_token_ids.detach().reshape(-1).cpu())
        ranked: list[tuple[float, int, Any]] = []
        for index, record in enumerate(active):
            shared = 0
            if record.token_ids is not None and query_unique.numel() > 0:
                own = torch.unique(record.token_ids.detach().reshape(-1).cpu())
                shared = int(torch.isin(query_unique, own).sum().item())
            ranked.append((float(scores[index].item()), shared, record))
        ranked.sort(key=lambda item: (item[0], item[1], item[2].last_access), reverse=True)
        return ranked

    @torch.no_grad()
    def _forget_text_memory_by_metadata(self, metadata: dict[str, str | bool]) -> bool:
        """Retract the active version addressed by an explicit attribute.

        Token-overlap matching is a useful fallback for free-form language, but
        a deletion request often shares only the attribute name with the old
        fact.  Once the write path extracted a conservative ``user::attribute``
        key, use the version ledger directly so a forget request cannot leave
        a stale value active merely because the wording was paraphrased.
        """

        if self.memory_os_v2 is None:
            return False
        entity = str(metadata.get("entity", "")).strip().lower()
        attribute = str(metadata.get("attribute", "")).strip().lower()
        if not entity or not attribute:
            return False
        conflict_key = f"{entity}::{attribute}"
        record_id = self.memory_os_v2.bank.active_by_conflict.get(conflict_key)
        if not record_id:
            return False
        record = self.memory_os_v2.bank.records.get(record_id)
        if record is None or record.status != "active":
            return False
        legacy_slot = record.slot_index
        for item in record.evidence:
            match = re.fullmatch(r"legacy_slot:(-?\d+)", str(item))
            if match is not None:
                legacy_slot = int(match.group(1))
                break
        self._clear_legacy_memory_slot(legacy_slot)
        self.memory_os_v2.retract_record(record.record_id)
        # The turn that stated this fact may have stated others alongside it (a contact
        # and their extension, say).  Forgetting "the emergency contact information" has
        # to retire the whole turn's records, otherwise the sibling stays readable and
        # gets volunteered later.
        self.memory_os_v2.retract_origin(record.origin)
        return True

    @torch.no_grad()
    def _forget_text_memory_by_key(
        self,
        key_input_ids: Tensor,
        key_attention_mask: Tensor,
    ) -> Tensor:
        """Erase the addressed natural-language record for a learned forget turn."""

        erased = torch.zeros(
            key_input_ids.shape[0], dtype=torch.bool, device=key_input_ids.device
        )
        if (
            self.runtime.text_slot_valid is None
            or self.runtime.text_key_token_ids is None
            or self.runtime.text_key_token_mask is None
            or not bool(self.runtime.text_slot_valid.any())
        ):
            return erased
        query_keys = self._encode_model_key(key_input_ids, key_attention_mask)
        for batch_index in range(key_input_ids.shape[0]):
            valid_slots = self.runtime.text_slot_valid[batch_index]
            if not bool(valid_slots.any()):
                continue
            query_ids = key_input_ids[batch_index][key_attention_mask[batch_index].bool()]
            query_ids = query_ids[: self.memory_config.text_memory_key_tokens]
            old_ids = self.runtime.text_key_token_ids[batch_index]
            old_mask = self.runtime.text_key_token_mask[batch_index]
            shared = (
                (old_ids[:, :, None] == query_ids[None, None, :])
                & old_mask[:, :, None]
                & valid_slots[:, None, None]
            ).any(dim=-1).sum(dim=-1).float()
            old_lengths = old_mask.sum(dim=-1).float()
            overlap = shared / torch.sqrt(
                old_lengths * float(max(1, query_ids.numel()))
            ).clamp_min(1.0)
            overlap = overlap.masked_fill(~valid_slots, -1.0)
            best_overlap, best_slot = overlap.max(dim=-1)
            best_score = best_overlap
            if self.text_retriever is not None and self._text_retriever_ready:
                slot_keys = F.normalize(
                    self.runtime.text_slot_keys[batch_index].to(query_keys.device, dtype=query_keys.dtype),
                    dim=-1,
                )
                learned = torch.sigmoid(
                    self.text_retriever(query_keys[batch_index].unsqueeze(0), slot_keys.unsqueeze(0))[0]
                ).masked_fill(~valid_slots.to(query_keys.device), -1.0)
                learned_score, learned_slot = learned.max(dim=-1)
                if float(learned_score) >= float(best_score):
                    best_score = learned_score
                    best_slot = learned_slot
            if float(best_score) < self.memory_config.text_memory_overlap_threshold and not (
                self.text_retriever is not None
                and self._text_retriever_ready
                and float(best_score) >= 0.65
            ):
                continue
            slot = int(best_slot.item())
            self._clear_legacy_memory_slot(slot)
            if self.memory_os_v2 is not None and self.runtime.use_persistent_state:
                ranked = self._rank_v2_text_matches(
                    query_keys[batch_index],
                    query_ids,
                )
                for score, shared, record in ranked:
                    # The hot-bank matcher has already located the target.
                    # V2 records created as independent fragments have no
                    # slot index, so use the frozen-Qwen address plus token
                    # evidence to retract only the matching old record.
                    same_hot_slot = record.slot_index == slot
                    score_threshold = (
                        self.memory_config.text_memory_semantic_update_threshold
                        if self.text_retriever is not None and self._text_retriever_ready
                        else 0.60
                    )
                    if (
                        same_hot_slot
                        or shared >= 3
                        or (shared >= 2 and score >= score_threshold)
                    ):
                        legacy_slot = record.slot_index
                        for item in record.evidence:
                            match = re.fullmatch(r"legacy_slot:(-?\d+)", str(item))
                            if match is not None:
                                legacy_slot = int(match.group(1))
                                break
                        self._clear_legacy_memory_slot(legacy_slot)
                        self.memory_os_v2.retract_record(record.record_id)
                        # Retire the rest of the turn that produced this record too; see
                        # the note in ``_forget_text_memory_by_metadata``.
                        self.memory_os_v2.retract_origin(record.origin)
                        if not same_hot_slot and score < 0.98:
                            break
            erased[batch_index] = True
        return erased

    def _write_text_memory(
        self,
        input_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        text_input_ids: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        key_input_ids: Optional[Tensor] = None,
        key_attention_mask: Optional[Tensor] = None,
        storage_input_ids: Optional[Tensor] = None,
        storage_attention_mask: Optional[Tensor] = None,
        force_write: bool = False,
        memory_text: Optional[str] = None,
    ) -> None:
        """Commit one conversational write into the selected internal slot.

        The learned controller decides whether a turn is a durable fact and
        which slot it belongs to.  The exact token sequence is then copied into
        the model-owned bank so multi-token names, values and punctuation are
        not lossy-compressed into a single output token.
        """

        # The bank is consumed as an internal prefix before the next Qwen
        # chat turn.  Store the valid system-message form for generation;
        # keep the exact user fact separately in ``text_key_token_ids`` for
        # duplicate detection and retrieval overlap.  A raw fact placed
        # before ``<|im_start|>user`` is not a reliable conditioning channel.
        storage_ids = (
            text_input_ids
            if text_input_ids is not None
            else (storage_input_ids if storage_input_ids is not None else input_ids)
        )
        storage_mask = (
            text_attention_mask
            if text_input_ids is not None
            else (storage_attention_mask if storage_input_ids is not None else attention_mask)
        )
        key_ids = key_input_ids if key_input_ids is not None else storage_ids
        key_mask = key_attention_mask if key_input_ids is not None else storage_mask
        if (
            not self.memory_config.natural_language_memory
            or storage_ids is None
            or self.runtime.text_token_ids is None
            or self.runtime.text_slot_keys is None
            or self.runtime.text_key_token_ids is None
        ):
            return
        self.runtime.text_last_written_slot = torch.full(
            (storage_ids.shape[0],),
            -1,
            dtype=torch.long,
            device=self.runtime.text_slot_valid.device,
        )
        address = self.memory.last_write_address
        write_probability = self.memory.last_write_probability
        if address is None or key_ids is None:
            return
        if write_probability is None:
            write_probability = address.max(dim=-1, keepdim=True).values
        if force_write:
            should_write = torch.ones(
                storage_ids.shape[0], dtype=torch.bool, device=storage_ids.device
            )
        elif (
            self.memory_config.automatic_memory
            and self._memory_policy_ready
            and self.runtime.auto_memory_probability is not None
        ):
            # High recall is intentional: the policy is trained to recognize
            # durable user information.  It is the final automatic decision;
            # the older native write gate is not OR-ed here because it was
            # trained for continuous-state updates and can be over-eager on
            # question-shaped inputs.
            should_write = self.runtime.auto_memory_probability >= self.memory_config.auto_memory_threshold
        else:
            should_write = write_probability.squeeze(-1) >= self.memory_config.text_memory_write_threshold
        metadata = infer_memory_metadata(memory_text or "")
        record_text = format_memory_evidence(
            memory_text or "",
            entity=str(metadata.get("entity", "")),
            attribute=str(metadata.get("attribute", "")),
            value=str(metadata.get("value", "")),
        )
        explicit_forget = metadata.get("kind") == "forget"
        if explicit_forget:
            # A direct user deletion request is a hard safety instruction.  It
            # bypasses the learned forget threshold, but it does not create a
            # new memory record and therefore cannot poison future retrieval.
            metadata_erased = self._forget_text_memory_by_metadata(metadata)
            forget_key_mask = (
                key_mask
                if key_mask is not None
                else torch.ones_like(key_ids, dtype=torch.bool)
            )
            # Run the bounded hot-bank matcher as well.  A V2 record can be an
            # independent fragment with ``slot_index=-1`` while the legacy
            # compatibility copy still occupies a hot slot; structured
            # retraction alone must not leave that copy readable.
            self._forget_text_memory_by_key(key_ids, forget_key_mask)
            should_write = torch.zeros_like(should_write, dtype=torch.bool)
        forget_probability = self.runtime.auto_memory_forget_probability
        if (
            not force_write
            and not explicit_forget
            and forget_probability is not None
            and bool((forget_probability >= self.memory_config.auto_forget_threshold).any())
        ):
            forget_mask = forget_probability >= self.memory_config.auto_forget_threshold
            forget_key_mask = (
                key_mask
                if key_mask is not None
                else torch.ones_like(key_ids, dtype=torch.bool)
            )
            erased = self._forget_text_memory_by_key(key_ids, forget_key_mask)
            should_write = should_write & ~(forget_mask & erased)
        if not force_write and memory_text:
            if looks_like_question(memory_text):
                should_write = torch.zeros_like(should_write, dtype=torch.bool)
        if storage_mask is None:
            storage_mask = torch.ones_like(storage_ids, dtype=torch.bool)
        else:
            storage_mask = storage_mask.to(device=storage_ids.device, dtype=torch.bool)
        if key_mask is None:
            key_mask = torch.ones_like(key_ids, dtype=torch.bool)
        else:
            key_mask = key_mask.to(device=key_ids.device, dtype=torch.bool)
        text_keys = self._encode_model_key(key_ids, key_mask)
        max_tokens = self.memory_config.text_memory_tokens
        for batch_index in range(storage_ids.shape[0]):
            if not bool(should_write[batch_index]):
                continue
            valid_ids = storage_ids[batch_index][storage_mask[batch_index]]
            if valid_ids.numel() == 0:
                continue
            valid_key_ids = key_ids[batch_index][key_mask[batch_index]]
            valid_key_ids = valid_key_ids[: self.memory_config.text_memory_key_tokens].detach().to(
                device=self.runtime.text_key_token_ids.device,
                dtype=torch.long,
            )
            valid_ids = valid_ids[:max_tokens].detach().to(
                device=self.runtime.text_token_ids.device,
                dtype=torch.long,
            )
            key = text_keys[batch_index]
            valid_slots = self.runtime.text_slot_valid[batch_index]
            lexical_similarity = torch.tensor(-1.0, device=key.device)
            learned_similarity = None
            learned_best = None
            if bool(valid_slots.any()):
                old_key_ids = self.runtime.text_key_token_ids[batch_index]
                old_key_mask = self.runtime.text_key_token_mask[batch_index]
                equal = old_key_ids[:, :, None] == valid_key_ids[None, None, :]
                overlap_count = (
                    equal
                    & old_key_mask[:, :, None]
                    & torch.ones(
                        1,
                        1,
                        valid_key_ids.numel(),
                        dtype=torch.bool,
                        device=old_key_ids.device,
                    )
                ).any(dim=-1).sum(dim=-1).float()
                old_lengths = old_key_mask.sum(dim=-1).float()
                similarities = overlap_count / torch.sqrt(
                    old_lengths * float(max(1, valid_key_ids.numel()))
                ).clamp_min(1.0)
                masked_similarities = similarities.masked_fill(
                    ~valid_slots,
                    torch.finfo(similarities.dtype).min,
                )
                best_similarity, best_slot = masked_similarities.max(dim=-1)
                lexical_similarity = best_similarity
                # Exact token equality is a stronger duplicate signal than
                # any learned semantic score.  It makes repeated natural
                # language facts idempotent, even when the retriever has not
                # seen that exact name/value during training.
                key_length = valid_key_ids.numel()
                exact_match = (
                    (old_key_ids[:, :key_length] == valid_key_ids[None, :])
                    | (~old_key_mask[:, :key_length])
                ).all(dim=-1) & valid_slots & (
                    old_key_mask.sum(dim=-1) == valid_key_ids.numel()
                )
                exact_slots = exact_match.nonzero(as_tuple=False).flatten()
                if exact_slots.numel() > 0:
                    best_similarity = torch.tensor(1.0, device=key.device)
                    best_slot = exact_slots[0]
                if self.text_retriever is not None and self._text_retriever_ready:
                    if exact_slots.numel() == 0:
                        learned_similarity = torch.sigmoid(
                            self.text_retriever(
                                key.unsqueeze(0),
                                self.runtime.text_slot_keys[batch_index].unsqueeze(0),
                            )[0]
                        )
                        learned_similarity = learned_similarity.masked_fill(
                            ~valid_slots,
                            torch.finfo(learned_similarity.dtype).min,
                        )
                        learned_best, learned_slot = learned_similarity.max(dim=-1)
                    # Shared words such as "我" and "请记住" are common in
                    # many facts.  Once the retriever is trained, use its
                    # semantic same-attribute score for replacement so a new
                    # fruit fact cannot evict a work-location fact merely
                    # because both contain the same pronouns.
                        if float(learned_best) >= 0.65:
                            best_similarity = learned_best
                            best_slot = learned_slot
                        # Keep the exact token-overlap score when the learned
                        # retriever is uncertain.  Overwriting it with -1
                        # made obvious same-attribute updates look unrelated
                        # merely because the paraphrase was outside the
                        # retriever's training distribution.
            else:
                best_similarity = torch.tensor(-1.0, device=key.device)
                best_slot = torch.tensor(0, dtype=torch.long, device=key.device)

            # Updating/retracting is intentionally harder than reading. The
            # old code used the maximum of lexical and learned similarity for
            # both operations, so related facts such as "project -> member"
            # and "member -> work code" could collapse into one record.
            #
            # A similarity score alone is NOT sufficient authorisation to
            # retire an existing record: it must first be structurally true
            # that the two texts describe the *same* fact, i.e. the same
            # ``entity::attribute`` pair that the conflict ledger already
            # tracks.  Measured on this write path, the packaged retriever
            # assigns >= 0.95 to genuinely unrelated attributes (10 of 190
            # unrelated fact pairs, max 0.9985), and acting on that score
            # retired 8 of 20 distinct facts -- including the record the
            # question needed -- before any reader could run.  Genuine
            # same-attribute updates are still versioned by
            # ``PagedMemoryBankV2.write`` through ``active_by_conflict``,
            # which keeps the previous version as ``superseded``.
            candidate_entity = str(metadata.get("entity", "")).strip().lower()
            candidate_attribute = str(metadata.get("attribute", "")).strip().lower()
            candidate_conflict_key = (
                f"{candidate_entity}::{candidate_attribute}"
                if candidate_entity and candidate_attribute
                else ""
            )
            active_conflict_keys = (
                self.memory_os_v2.bank.active_by_conflict
                if self.memory_os_v2 is not None
                else {}
            )
            same_fact_key = bool(candidate_conflict_key) and candidate_conflict_key in active_conflict_keys
            # A rephrased update ("我源码放在 X" for a stored "我的代码仓库是 Y") parses to no
            # attribute at all, so the structural key cannot authorise anything and the stale
            # value stayed active -- measured on the realistic corpus: 25/25 update episodes
            # failed to supersede, and the model then answered the superseded value.
            #
            # The fix keeps the two paths strictly separated: a high-confidence semantic match
            # may only *inherit the attribute* of the matched record so that ``bank.write``
            # versions it (old record becomes ``superseded``: history preserved, still auditable,
            # no longer injected).  It must never authorise ``retract_record``, which destroys --
            # that path stays structural-only, which is what keeps 20/20 write survival.
            inherited_conflict_key = ""
            if (
                not candidate_conflict_key
                and bool(valid_slots.any())
                and learned_best is not None
                and float(learned_best) >= self.memory_config.text_memory_semantic_update_threshold
                and self.memory_os_v2 is not None
            ):
                matched_slot = int(best_slot.item())
                for record in self.memory_os_v2.bank.records.values():
                    if record.slot_index == matched_slot and record.conflict_key():
                        inherited_conflict_key = record.conflict_key()
                        break
                if inherited_conflict_key:
                    inherited_entity, _, inherited_attribute = inherited_conflict_key.partition("::")
                    candidate_entity = inherited_entity
                    candidate_attribute = inherited_attribute
                    same_fact_key = inherited_conflict_key in active_conflict_keys
            # The block above matches the record to inherit from by *slot index*, so it can only
            # reach records the legacy slot bank wrote itself.  Records written through the memory
            # API are independent paged items with ``slot_index = -1``: measured, a fact the agent
            # stored could never be versioned by the automatic layer, so restating it ("改了：…",
            # which parses to no attribute at all) left the old record active and still ranking
            # first (old 8.598 vs new 8.399).
            #
            # TWO gates, both of which the first attempt lacked:
            #   * ``not candidate_conflict_key`` -- a structural parse is better evidence than the
            #     reader's nearest neighbour, and overriding one is exactly what caused an observed
            #     over-supersede: the turn 「我的项目代号是 蓝鲸-47」 parsed to 项目代号, the block
            #     below rewrote it to 默认语言 (the only record then stored), and the language fact
            #     was retired -- after which 「我的默认语言是什么」 answered 蓝鲸-47.  The
            #     pre-existing slot block carries the same guard for the same reason.
            #   * a score floor on the record itself.  Asked about a *different* attribute the
            #     reader still says need_memory (that flag means "memory is relevant", not "this is
            #     that record"), so the gate must be the identified record's own score.  The
            #     router's margin was the first attempt and it cannot work here: the automatic layer
            #     stores a near-duplicate every turn, which collapses top-minus-second -- measured,
            #     one stored fact plus the previous turn's episode dropped the margin from 8.591 to
            #     0.1523 while the identified record's own score stayed 8.591.  Inheritance can only
            #     mark a record ``superseded`` (history kept, still auditable), never retract it.
            if (
                not inherited_conflict_key
                and not candidate_conflict_key
                and self.memory_os_v2 is not None
                and self._memory_router_v2_ready
            ):
                turn_entity = str(metadata.get("entity", "") or "").strip().lower()
                try:
                    read_records, read_decision = self.memory_os_v2.read(
                        query_key=key.reshape(1, -1),
                        query_text=memory_text or "",
                        query_token_ids=key_input_ids[batch_index],
                        top_k_pages=self.memory_config.memory_top_k_pages,
                        top_k_records=self.memory_config.memory_top_k_records,
                        max_hops=1,
                    )
                except Exception:  # a diagnostic-grade read must never break a write
                    read_records, read_decision = [], None
                if read_decision is not None and bool(read_decision.need_memory):
                    scores = list(getattr(read_decision, "record_scores", None) or [])
                    best_key, best_score = "", 0.0
                    for position, match in enumerate(read_records):
                        match_key = match.conflict_key()
                        if not match_key:
                            continue
                        if turn_entity and match_key.partition("::")[0] != turn_entity:
                            continue
                        match_score = (
                            float(scores[position])
                            if position < len(scores)
                            else float(read_decision.top_score)
                        )
                        if match_score > best_score:
                            best_key, best_score = match_key, match_score
                    if best_key and best_score >= _INHERIT_SCORE_FLOOR:
                        inherited_conflict_key = best_key
                if inherited_conflict_key:
                    inherited_entity, _, inherited_attribute = inherited_conflict_key.partition("::")
                    candidate_entity = inherited_entity
                    candidate_attribute = inherited_attribute
                    same_fact_key = inherited_conflict_key in active_conflict_keys
            # An exact token-identical rewrite is structurally the same text and
            # stays idempotent without any similarity authorisation.
            confirmed_update = exact_slots.numel() > 0 if bool(valid_slots.any()) else False
            if not confirmed_update and same_fact_key:
                if learned_best is not None:
                    confirmed_update = float(learned_best) >= (
                        self.memory_config.text_memory_semantic_update_threshold
                    )
                else:
                    confirmed_update = float(lexical_similarity) >= (
                        self.memory_config.text_memory_update_overlap_threshold
                    )
            if confirmed_update:
                slot = int(best_slot.item())
            else:
                free_slots = (~valid_slots).nonzero(as_tuple=False).flatten()
                if free_slots.numel() > 0:
                    slot = int(free_slots[0].item())
                else:
                    slot = int(self.runtime.text_slot_age[batch_index].argmin().item())
            self.runtime.text_token_ids[batch_index, slot].zero_()
            self.runtime.text_token_mask[batch_index, slot].zero_()
            self.runtime.text_token_ids[batch_index, slot, : valid_ids.numel()].copy_(valid_ids)
            self.runtime.text_token_mask[batch_index, slot, : valid_ids.numel()] = True
            self.runtime.text_slot_valid[batch_index, slot] = True
            self.runtime.text_slot_keys[batch_index, slot].copy_(key.to(self.runtime.text_slot_keys.device))
            self.runtime.text_key_token_ids[batch_index, slot].zero_()
            self.runtime.text_key_token_mask[batch_index, slot].zero_()
            self.runtime.text_key_token_ids[batch_index, slot, : valid_key_ids.numel()].copy_(valid_key_ids)
            self.runtime.text_key_token_mask[batch_index, slot, : valid_key_ids.numel()] = True
            self.runtime.text_write_counter[batch_index] += 1
            self.runtime.text_slot_age[batch_index, slot] = self.runtime.text_write_counter[batch_index]
            self.runtime.text_last_written_slot[batch_index] = slot
            if self.memory_os_v2 is not None and self.runtime.use_persistent_state:
                confidence = float(write_probability[batch_index].reshape(-1)[0].item())
                if force_write:
                    confidence = 1.0
                # The 16-slot bank is only the hot cache.  New, unrelated
                # conversational fragments must remain independent V2
                # records instead of superseding whatever happened to be in
                # the same hot slot.  Reuse a V2 slot index only when the
                # hot-bank matcher already established that this is a
                # duplicate/update of an existing fact.
                v2_slot_index = (
                    slot
                    if confirmed_update
                    else -1
                )
                if v2_slot_index >= 0 and self.memory_os_v2 is not None:
                    # A semantic replacement should retire the matching
                    # active record, while unrelated fragments remain active
                    # even if they once occupied the same hot-cache slot.
                    # Only records that carry the *same* entity/attribute may be
                    # retired here; anything else is an unrelated fact that
                    # merely shares a hot slot or scored high on lexical overlap
                    # (every fact shares the 我的/是 template tokens, so the
                    # ``shared >= 2`` test discriminates nothing).
                    for score, shared, record in self._rank_v2_text_matches(
                        key,
                        valid_key_ids,
                    ):
                        if record.conflict_key() != candidate_conflict_key:
                            continue
                        if record.slot_index == slot or (
                            score >= self.memory_config.text_memory_semantic_update_threshold
                            and shared >= 2
                        ):
                            self.memory_os_v2.retract_record(record.record_id)
                            if record.slot_index != slot and score < 0.98:
                                break
                self.memory_os_v2.write(
                    text=record_text or f"memory_slot:{slot}",
                    key=key,
                    summary=key,
                    memory_type="episodic_text",
                    # The identity may have been *inherited* from a semantically matched
                    # record above, so a rephrased update still versions the fact it replaces.
                    entity=candidate_entity or str(metadata.get("entity", "")),
                    attribute=candidate_attribute or str(metadata.get("attribute", "")),
                    value=str(metadata.get("value", "")),
                    importance=max(confidence, 0.5),
                    confidence=confidence,
                    source=(
                        "explicit"
                        if force_write
                        else ("automatic_correction" if metadata.get("kind") == "correction" else "automatic")
                    ),
                    evidence=[
                        f"token_count:{int(valid_ids.numel())}",
                        f"legacy_slot:{slot}",
                        f"input_sha1:{hashlib.sha1((record_text or '').encode('utf-8')).hexdigest()[:16]}",
                        f"policy_write:{float(write_probability[batch_index].reshape(-1)[0].item()):.4f}",
                        *([f"inherited_key:{inherited_conflict_key}"] if inherited_conflict_key else []),
                    ],
                    slot_index=v2_slot_index,
                    token_ids=valid_ids,
                    token_mask=torch.ones_like(valid_ids, dtype=torch.bool),
                    semantic_key=key,
                    trusted=force_write or confidence >= self.memory_config.memory_v2_write_threshold,
                    force=force_write,
                    # Everything this turn wrote shares one origin, so a later explicit
                    # forget can retire the turn's records as a unit.
                    origin=memory_origin(memory_text or record_text or ""),
                )

    def _probe_text_retrieval(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Compute query-dependent similarity against internal text keys."""

        if (
            not self.memory_config.natural_language_memory
            or self.runtime.text_slot_valid is None
            or self.runtime.text_slot_keys is None
            or self.runtime.text_key_token_ids is None
            or self.runtime.text_key_token_mask is None
            or not bool(self.runtime.text_slot_valid.any())
        ):
            return None, None
        query_key = self._encode_model_key(input_ids, attention_mask)
        self.runtime.v2_query_key = query_key.detach()
        slot_keys = F.normalize(
            self.runtime.text_slot_keys.to(device=query_key.device, dtype=query_key.dtype),
            dim=-1,
        )
        if self.text_retriever is not None and self._text_retriever_ready:
            learned_logits = self.text_retriever(query_key, slot_keys)
            dense_similarity = torch.sigmoid(learned_logits)
        else:
            dense_similarity = torch.einsum("bh,bsh->bs", query_key, slot_keys)
        # Dense similarity alone is too permissive for short Chinese queries:
        # common tokens such as "我" and "的" can make unrelated memories
        # look relevant.  Combine it with exact token overlap from the
        # internally stored key, yielding a conservative hybrid score.
        key_token_ids = self.runtime.text_key_token_ids.to(device=input_ids.device)
        key_token_mask = self.runtime.text_key_token_mask.to(device=input_ids.device, dtype=torch.bool)
        query_token_ids = input_ids.to(device=key_token_ids.device)
        query_token_mask = attention_mask.to(device=key_token_ids.device, dtype=torch.bool)
        equal = query_token_ids[:, None, :, None] == key_token_ids[:, :, None, :]
        equal &= query_token_mask[:, None, :, None] & key_token_mask[:, :, None, :]
        shared_tokens = equal.any(dim=-1).sum(dim=-1).float()
        query_lengths = query_token_mask.sum(dim=-1, keepdim=True).float()
        key_lengths = key_token_mask.sum(dim=-1).float()
        overlap = shared_tokens / torch.sqrt(query_lengths * key_lengths).clamp_min(1.0)
        if self.text_retriever is not None and self._text_retriever_ready:
            scores = 0.85 * dense_similarity + 0.15 * overlap.clamp(0.0, 1.0)
        else:
            dense_score = (dense_similarity + 1.0).clamp(0.0, 2.0) * 0.5
            scores = 0.4 * dense_score + 0.6 * overlap.clamp(0.0, 1.0)
        self.runtime.text_read_overlap = overlap
        scores = scores.masked_fill(
            ~self.runtime.text_slot_valid.to(device=scores.device),
            torch.finfo(scores.dtype).min,
        )
        return scores, scores.max(dim=-1).values

    def configure_memory_grounding_guard(
        self,
        tokenizer: Any,
        content: Optional[str] = None,
    ) -> None:
        """Install the model-owned no-evidence instruction once per process.

        The guard is tokenized at setup time, then inserted by the model's
        own memory reader only when a grounded user/project question has no
        supporting record.  Callers do not decide whether to refuse; they
        only provide the tokenizer needed to prepare the model-local tokens.
        """

        instruction = content or (
            "当前没有检索到能直接支持这个问题的已保存长期证据。"
            "如果问题询问用户、项目、仓库或历史事实，只能回答没有记录或无法确认；"
            "禁止用常识、相似内容或训练语料补全，不要编造事实。"
        )
        full = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": "__memory_query_boundary__"},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        input_ids = full["input_ids"]
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        positions = (input_ids[0] == int(im_start)).nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            raise RuntimeError("could not locate the system/user memory boundary")
        end = int(positions[1].item())
        self.runtime.text_guard_token_ids = input_ids[:, :end].detach().cpu()
        self.runtime.text_guard_token_mask = torch.ones(
            (1, end), dtype=torch.long
        )

    def _guard_text_prefix(self, input_ids: Tensor) -> tuple[Optional[Tensor], Optional[Tensor], int]:
        guard_ids = self.runtime.text_guard_token_ids
        guard_mask = self.runtime.text_guard_token_mask
        if guard_ids is None or guard_mask is None or guard_ids.numel() == 0:
            return None, None, 0
        guard_ids = guard_ids.to(device=input_ids.device, dtype=input_ids.dtype)
        guard_mask = guard_mask.to(device=input_ids.device)
        self.runtime.text_guard_used = True
        self.runtime.text_prefix_tokens = int(guard_ids.shape[1])
        return guard_ids, guard_mask, int(guard_ids.shape[1])

    def _build_text_prefix(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        query_text: str = "",
    ) -> tuple[Optional[Tensor], Optional[Tensor], int]:
        """Retrieve text memory and build an internal prefix for Qwen."""

        # The query key belongs to this retrieval call.  Do not let an exact
        # address route's zero key leak into a later ambiguous query.
        self.runtime.v2_query_key = None
        fast_address_route = bool(
            self.memory_os_v2 is not None
            and self._memory_router_v2_ready
            and query_text
            and self.memory_os_v2.bank.has_explicit_address(query_text)
        )
        # V2 already owns an address/page index. For explicit addresses,
        # running the legacy 16-slot hidden-state probe adds a full Qwen
        # encoder pass without improving the routing decision. Keep the old
        # probe for ambiguous queries and legacy checkpoints.
        if fast_address_route:
            address, relevance = None, None
        else:
            address, relevance = self._probe_text_retrieval(input_ids, attention_mask)
        self.runtime.text_read_slots = None
        self.runtime.v2_last_decisions = []
        self.runtime.text_read_relevance = relevance
        self.runtime.text_prefix_used = False
        self.runtime.text_prefix_tokens = 0
        self.runtime.text_guard_used = False
        self.runtime.v2_no_evidence = False
        if (
            self.memory_os_v2 is not None
            and self._memory_router_v2_ready
            and self.runtime.v2_query_key is None
        ):
            if fast_address_route:
                self.runtime.v2_query_key = torch.zeros(
                    input_ids.shape[0],
                    self.memory.hidden_size,
                    dtype=torch.float32,
                    device=input_ids.device,
                )
            else:
                self.runtime.v2_query_key = self._encode_model_key(input_ids, attention_mask).detach()

        # V2 is the scalable address path.  It first selects a bounded set of
        # pages through the coarse index, then reranks only records in those
        # pages.  The legacy fixed bank remains a compatibility fallback for
        # old checkpoints that do not contain a trained V2 router.
        if (
            self.memory_os_v2 is not None
            and self._memory_router_v2_ready
            and self.runtime.v2_query_key is not None
        ):
            v2_prefix_parts: list[Tensor] = []
            v2_prefix_masks: list[Tensor] = []
            v2_slots: list[list[int]] = []
            v2_records_found = False
            coverage_refused = False
            for batch_index in range(input_ids.shape[0]):
                records, decision = self.memory_os_v2.read(
                    query_key=self.runtime.v2_query_key[batch_index],
                    query_text=query_text,
                    query_token_ids=input_ids[batch_index],
                    top_k_pages=self.memory_config.memory_top_k_pages,
                    top_k_records=self.memory_config.memory_top_k_records,
                    max_hops=self.memory_config.memory_max_hops,
                )
                self.runtime.v2_last_decisions.append(asdict(decision))
                if str(decision.stop_reason) in ("attribute_not_covered", "no_attribute_recognised"):
                    coverage_refused = True
                v2_records_found = v2_records_found or bool(records)
                ids_parts: list[Tensor] = []
                slot_list: list[int] = []
                for record in records:
                    ids: Optional[Tensor] = None
                    if record.token_ids is not None:
                        cached_ids, cached_mask = self.memory_os_v2.bank.gpu_record_payload(record)
                        if cached_ids is not None:
                            if cached_mask is not None and cached_mask.numel() == cached_ids.numel():
                                ids = cached_ids[cached_mask]
                            else:
                                ids = cached_ids
                    elif (
                        self.runtime.text_token_ids is not None
                        and self.runtime.text_token_mask is not None
                        and 0 <= record.slot_index < self.runtime.text_token_ids.shape[1]
                    ):
                        slot = record.slot_index
                        slot_mask = self.runtime.text_token_mask[batch_index, slot]
                        ids = self.runtime.text_token_ids[batch_index, slot][slot_mask].detach().cpu()
                    if ids is not None and ids.numel() > 0:
                        ids_parts.append(ids)
                        if record.slot_index >= 0:
                            slot_list.append(record.slot_index)
                if ids_parts:
                    v2_prefix_parts.append(torch.cat(ids_parts, dim=0))
                    v2_prefix_masks.append(torch.ones_like(v2_prefix_parts[-1], dtype=torch.bool))
                else:
                    v2_prefix_parts.append(torch.zeros(0, dtype=torch.long, device=input_ids.device))
                    v2_prefix_masks.append(torch.zeros(0, dtype=torch.bool, device=input_ids.device))
                v2_slots.append(slot_list)
            v2_prefix_length = max((part.numel() for part in v2_prefix_parts), default=0)
            if v2_prefix_length > 0:
                prefix_ids = torch.zeros(
                    input_ids.shape[0], v2_prefix_length, dtype=input_ids.dtype, device=input_ids.device
                )
                prefix_mask = torch.zeros(
                    input_ids.shape[0], v2_prefix_length, dtype=attention_mask.dtype, device=input_ids.device
                )
                for batch_index, (ids, mask) in enumerate(zip(v2_prefix_parts, v2_prefix_masks)):
                    prefix_ids[batch_index, : ids.numel()] = ids.to(input_ids.device)
                    prefix_mask[batch_index, : mask.numel()] = mask.to(attention_mask.dtype)
                self.runtime.text_read_slots = torch.tensor(
                    [slots + [-1] * max(0, self.memory_config.memory_top_k_records - len(slots)) for slots in v2_slots],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                self.runtime.text_prefix_used = True
                self.runtime.text_prefix_tokens = int(v2_prefix_length)
                return prefix_ids, prefix_mask, v2_prefix_length
            if (
                not v2_records_found
                and _looks_like_grounded_memory_query(query_text)
            ):
                self.runtime.v2_no_evidence = True
                return self._guard_text_prefix(input_ids)
            if coverage_refused:
                # The coverage gate refused because the bank does not hold the attribute
                # this question asks about.  Returning the guard prefix here is essential:
                # falling through would hand the question to the legacy 16-slot path, which
                # injects an unrelated memory anyway and lets the model answer.  Measured
                # without this short-circuit, the abstention leak only fell from 75.00% to
                # 62.50%; with it the gate is able to actually abstain.
                self.runtime.v2_no_evidence = True
                return self._guard_text_prefix(input_ids)

        if address is None or self.runtime.text_slot_valid is None:
            return None, None, 0

        valid = self.runtime.text_slot_valid
        scores = address.masked_fill(~valid, torch.finfo(address.dtype).min)
        top_k = min(self.memory_config.text_memory_top_k, scores.shape[-1])
        top_scores, top_slots = scores.topk(top_k, dim=-1)
        selected = top_scores >= self.memory_config.text_memory_threshold
        # If exactly one stored memory exists, its relevance still has to pass
        # the threshold; this prevents unrelated questions from receiving an
        # arbitrary memory snippet and reduces hallucinated personal facts.
        if relevance is not None:
            selected &= relevance[:, None] >= self.memory_config.text_memory_threshold
        if (
            self.runtime.text_read_overlap is not None
            and not (self.text_retriever is not None and self._text_retriever_ready)
        ):
            selected_overlap = self.runtime.text_read_overlap.gather(1, top_slots)
            selected &= selected_overlap >= self.memory_config.text_memory_overlap_threshold
        self.runtime.text_read_slots = top_slots
        if not bool(selected.any()):
            return None, None, 0
        self.runtime.text_prefix_used = True

        bank_ids = self.runtime.text_token_ids
        bank_mask = self.runtime.text_token_mask
        if bank_ids is None or bank_mask is None:
            return None, None, 0
        prefix_parts: list[Tensor] = []
        prefix_masks: list[Tensor] = []
        for batch_index in range(input_ids.shape[0]):
            ids_parts: list[Tensor] = []
            mask_parts: list[Tensor] = []
            for rank in range(top_k):
                if not bool(selected[batch_index, rank]):
                    continue
                slot = int(top_slots[batch_index, rank].item())
                slot_mask = bank_mask[batch_index, slot]
                ids_parts.append(bank_ids[batch_index, slot][slot_mask])
                mask_parts.append(torch.ones_like(ids_parts[-1], dtype=torch.bool))
            if ids_parts:
                prefix_parts.append(torch.cat(ids_parts, dim=0))
                prefix_masks.append(torch.cat(mask_parts, dim=0))
            else:
                prefix_parts.append(torch.zeros(0, dtype=torch.long, device=input_ids.device))
                prefix_masks.append(torch.zeros(0, dtype=torch.bool, device=input_ids.device))
        prefix_length = max((part.numel() for part in prefix_parts), default=0)
        if prefix_length == 0:
            return None, None, 0
        prefix_ids = torch.zeros(
            input_ids.shape[0], prefix_length, dtype=input_ids.dtype, device=input_ids.device
        )
        prefix_mask = torch.zeros(
            input_ids.shape[0], prefix_length, dtype=attention_mask.dtype, device=input_ids.device
        )
        for batch_index, (ids, mask) in enumerate(zip(prefix_parts, prefix_masks)):
            prefix_ids[batch_index, : ids.numel()] = ids.to(input_ids.device)
            prefix_mask[batch_index, : mask.numel()] = mask.to(attention_mask.dtype)
        self.runtime.text_prefix_tokens = int(prefix_length)
        return prefix_ids, prefix_mask, prefix_length

    def _find_layer_device(self) -> torch.device:
        for index in self.layer_indices:
            layer = resolve_decoder_layers(self.base_model)[1][index]
            # In blend mode the adapter owns a trainable scalar. Inspect the
            # original layer first so a CPU-created scalar cannot mislead the
            # device choice for input_ids and the memory module.
            layer_for_device = getattr(layer, "inner", layer)
            for parameter in layer_for_device.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
        return self.base_model.get_input_embeddings().weight.device

    @property
    def trainable_parameters(self):
        parameters = [parameter for parameter in self.memory.parameters() if parameter.requires_grad]
        if self.text_retriever is not None:
            parameters.extend(
                parameter for parameter in self.text_retriever.parameters() if parameter.requires_grad
            )
        if self.memory_policy is not None:
            parameters.extend(
                parameter for parameter in self.memory_policy.parameters() if parameter.requires_grad
            )
        if self.memory_router_v2 is not None:
            parameters.extend(
                parameter for parameter in self.memory_router_v2.parameters() if parameter.requires_grad
            )
        for adapter in self._memory_adapters:
            blend_logit = getattr(adapter, "blend_logit", None)
            if blend_logit is not None and blend_logit.requires_grad:
                parameters.append(blend_logit)
        return iter(parameters)

    @torch.no_grad()
    def read_hierarchical_memory(
        self,
        query_key: Tensor,
        *,
        query_text: str = "",
        query_token_ids: Optional[Tensor] = None,
        top_k_pages: Optional[int] = None,
        top_k_records: Optional[int] = None,
        max_hops: Optional[int] = None,
    ) -> tuple[list[Any], Any]:
        """Read V2 memory through its bounded page and record router."""

        if self.memory_os_v2 is None:
            raise RuntimeError("hierarchical memory is disabled in memory_config")
        return self.memory_os_v2.read(
            query_key=query_key,
            query_text=query_text,
            query_token_ids=query_token_ids,
            top_k_pages=top_k_pages,
            top_k_records=top_k_records,
            max_hops=max_hops,
        )

    @torch.no_grad()
    def write_hierarchical_memory(self, **kwargs: Any) -> tuple[Any, str]:
        """Write one versioned V2 record owned by the model checkpoint."""

        if self.memory_os_v2 is None:
            raise RuntimeError("hierarchical memory is disabled in memory_config")
        return self.memory_os_v2.write(**kwargs)

    def memory_v2_stats(self) -> dict[str, Any]:
        if self.memory_os_v2 is None:
            return {"enabled": False}
        output = dict(self.memory_os_v2.stats())
        output["enabled"] = True
        output["router_ready"] = self._memory_router_v2_ready
        return output

    def list_memory_records(
        self,
        *,
        query_text: str = "",
        status: str = "active",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return a safe, tensor-free view of model-owned memory records."""

        if self.memory_os_v2 is None:
            return []
        return [
            memory_record_to_dict(record)
            for record in self.memory_os_v2.list_records(
                query_text=query_text,
                status=status,
                limit=limit,
                offset=offset,
            )
        ]

    def get_memory_record(self, record_id: str) -> dict[str, Any]:
        if self.memory_os_v2 is None:
            raise KeyError(record_id)
        record = self.memory_os_v2.bank._resolve_record(record_id)
        return memory_record_to_dict(record)

    def _clear_legacy_memory_slot(self, slot_index: int) -> None:
        """Remove a V1 compatibility copy when a V2 record is edited/retracted."""

        if slot_index < 0:
            return
        fields = (
            ("persistent_text_token_ids", "text_token_ids", 0),
            ("persistent_text_token_mask", "text_token_mask", 0),
            ("persistent_text_slot_valid", "text_slot_valid", 0),
            ("persistent_text_slot_keys", "text_slot_keys", 0),
            ("persistent_text_slot_age", "text_slot_age", -1),
            ("persistent_text_key_token_ids", "text_key_token_ids", 0),
            ("persistent_text_key_token_mask", "text_key_token_mask", 0),
        )
        for persistent_name, runtime_name, fill_value in fields:
            persistent = getattr(self, persistent_name, None)
            if isinstance(persistent, Tensor) and persistent.ndim >= 2 and slot_index < persistent.shape[1]:
                persistent[:, slot_index].fill_(fill_value)
            runtime = getattr(self.runtime, runtime_name, None)
            if isinstance(runtime, Tensor) and runtime.ndim >= 2 and slot_index < runtime.shape[1]:
                runtime[:, slot_index].fill_(fill_value)

    @torch.no_grad()
    def edit_memory_record(
        self,
        record_id: str,
        *,
        text: Optional[str] = None,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        value: Optional[str] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        evidence: Optional[list[str]] = None,
        token_ids: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
    ) -> dict[str, Any]:
        """Edit memory by creating a versioned successor, never overwriting history."""

        if self.memory_os_v2 is None:
            raise KeyError(record_id)
        old = self.memory_os_v2.bank._resolve_record(record_id)
        key = None
        if token_ids is not None:
            ids = token_ids.detach().reshape(1, -1).to(self._find_layer_device())
            mask = (
                token_mask.detach().reshape(1, -1).to(self._find_layer_device(), dtype=torch.long)
                if token_mask is not None
                else torch.ones_like(ids)
            )
            key = self._encode_model_key(ids, mask)[0]
        self._clear_legacy_memory_slot(old.slot_index)
        record = self.memory_os_v2.edit_record(
            record_id,
            text=text,
            key=key,
            summary=key,
            entity=entity,
            attribute=attribute,
            value=value,
            importance=importance,
            confidence=confidence,
            evidence=evidence,
            token_ids=token_ids,
            token_mask=token_mask,
        )
        return memory_record_to_dict(record)

    @torch.no_grad()
    def retract_memory_record(self, record_id: str) -> dict[str, Any]:
        """Retract a record and its compatibility-slot copy without erasing history."""

        if self.memory_os_v2 is None:
            raise KeyError(record_id)
        record = self.memory_os_v2.bank._resolve_record(record_id)
        self._clear_legacy_memory_slot(record.slot_index)
        self.memory_os_v2.retract_record(record_id)
        return memory_record_to_dict(record)

    def audit_memory(self) -> dict[str, Any]:
        if self.memory_os_v2 is None:
            return {"healthy": True, "issues": [], "issue_count": 0, "stats": {"enabled": False}}
        return self.memory_os_v2.audit()

    def export_memory_records(self, *, limit: int = 10000, offset: int = 0) -> dict[str, Any]:
        records = self.list_memory_records(status="all", limit=limit, offset=offset)
        return {
            "format_version": 1,
            "records": records,
            "offset": offset,
            "limit": limit,
            "returned": len(records),
            "stats": self.memory_v2_stats(),
        }

    def flush_memory_storage(self) -> None:
        """Flush the optional warm/cold tier without rewriting model weights."""

        if self.memory_os_v2 is not None:
            self.memory_os_v2.flush_storage()

    def close_memory_storage(self) -> None:
        """Close the optional durable page store before process shutdown."""

        if self.memory_os_v2 is not None:
            self.memory_os_v2.close_storage()

    def clear_hierarchical_memory(self) -> None:
        """Clear the durable V2 page store while keeping router weights."""

        if self.memory_router_v2 is None:
            return
        if self.memory_os_v2 is not None and self.memory_os_v2.bank.tier_store is not None:
            old_store = self.memory_os_v2.bank.tier_store
            old_store.clear()
            old_store.close()
        self.memory_os_v2 = self._new_memory_os_v2(self.memory.hidden_size)

    def reset_memory(self, batch_size: Optional[int] = None, *, device: Optional[torch.device] = None) -> None:
        """Explicitly clear the durable memory state.

        Use :meth:`reset_runtime_memory` for a temporary conversation or an
        evaluation case.  Keeping these operations separate prevents a test
        or a branch from silently deleting a user's embedded checkpoint.
        """

        self.runtime.use_persistent_state = True
        self.clear_hierarchical_memory()
        if self.memory_config.persistent_memory:
            device = device or self._find_layer_device()
            batch_size = batch_size or 1
            self.persistent_memory = self.memory.initial_state(batch_size, device=device)
            self._persistent_memory = self.persistent_memory.detach()
            if self.memory_config.natural_language_memory:
                self._ensure_text_memory(batch_size, device=device)
                self._clear_text_memory()
            self.runtime.state = self._persistent_memory
            self.runtime.last_read = None
            self.runtime.auto_memory_probability = None
            self.runtime.raw_memory = None
            self.runtime.reset_mask = None
            self.runtime.text_token_ids = self.persistent_text_token_ids
            self.runtime.text_token_mask = self.persistent_text_token_mask
            self.runtime.text_slot_valid = self.persistent_text_slot_valid
            self.runtime.text_slot_keys = self.persistent_text_slot_keys
            self.runtime.text_slot_age = self.persistent_text_slot_age
            self.runtime.text_write_counter = self.persistent_text_write_counter
            self.runtime.text_key_token_ids = self.persistent_text_key_token_ids
            self.runtime.text_key_token_mask = self.persistent_text_key_token_mask
            self.runtime.text_last_written_slot = None
            return
        if batch_size is None:
            self._persistent_memory = None
            self.runtime.state = None
            self.runtime.last_read = None
            self.runtime.auto_memory_probability = None
            self.runtime.raw_memory = None
            self.runtime.text_token_ids = None
            self.runtime.text_token_mask = None
            self.runtime.text_slot_valid = None
            self.runtime.text_slot_keys = None
            self.runtime.text_slot_age = None
            self.runtime.text_write_counter = None
            self.runtime.text_key_token_ids = None
            self.runtime.text_key_token_mask = None
            self.runtime.text_last_written_slot = None
            return
        device = device or self._find_layer_device()
        self._persistent_memory = self.memory.initial_state(batch_size, device=device)
        if self.memory_config.natural_language_memory:
            self._ensure_text_memory(batch_size, device=device)
            self._clear_text_memory()
        self.runtime.state = self._persistent_memory
        self.runtime.last_read = None
        self.runtime.auto_memory_probability = None
        self.runtime.raw_memory = None
        self.runtime.text_token_ids = self.persistent_text_token_ids
        self.runtime.text_token_mask = self.persistent_text_token_mask
        self.runtime.text_slot_valid = self.persistent_text_slot_valid
        self.runtime.text_slot_keys = self.persistent_text_slot_keys
        self.runtime.text_slot_age = self.persistent_text_slot_age
        self.runtime.text_write_counter = self.persistent_text_write_counter
        self.runtime.text_key_token_ids = self.persistent_text_key_token_ids
        self.runtime.text_key_token_mask = self.persistent_text_key_token_mask
        self.runtime.text_last_written_slot = None

    def reset_runtime_memory(
        self,
        batch_size: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset only the active conversation, preserving durable state.

        In persistent mode this creates an ephemeral empty bank and prevents
        the generation path from rebinding the embedded user checkpoint.  It
        is used by evaluation, memory branches and temporary sessions.
        """

        device = device or self._find_layer_device()
        batch_size = batch_size or 1
        self.runtime.use_persistent_state = False
        self.runtime.state = self.memory.initial_state(batch_size, device=device)
        self._persistent_memory = self.runtime.state.detach()
        self.runtime.last_read = None
        self.runtime.auto_memory_probability = None
        self.runtime.raw_memory = None
        self.runtime.v2_query_key = None
        self.runtime.v2_last_decisions = []
        self.runtime.reset_mask = None
        self.runtime.text_last_written_slot = None
        if self.memory_config.natural_language_memory:
            self._bind_text_memory(batch_size, device=device)
            self._clear_text_memory()

    def _reset_mask(self, input_ids: Optional[Tensor]) -> Optional[Tensor]:
        token_id = self.memory_config.reset_token_id
        if token_id is None or input_ids is None:
            return None
        mask = (input_ids == token_id).any(dim=-1)
        return mask if bool(mask.any()) else None

    def save_runtime_memory(
        self,
        path: str | Path,
        memory_state: Optional[Tensor] = None,
    ) -> Path:
        """Persist one user's runtime memory, without saving model weights."""

        state = memory_state if memory_state is not None else self.runtime.state
        if state is None:
            state = self._persistent_memory
        if state is None:
            raise ValueError("no runtime memory state is available to save")
        if state.ndim != 3 or tuple(state.shape[1:]) != (
            self.memory_config.memory_slots,
            self.memory_config.memory_dim,
        ):
            raise ValueError(f"unexpected runtime memory shape: {tuple(state.shape)}")

        raw_memory = self.runtime.raw_memory
        if raw_memory is not None:
            raw_memory = raw_memory.detach().cpu()
        payload = {
            "format_version": 1,
            "memory_state": state.detach().cpu(),
            "raw_memory": raw_memory,
            "model_hidden_size": self.memory.hidden_size,
            "memory_config": asdict(self.memory_config),
            "layer_indices": list(self.layer_indices),
        }
        if self.memory_config.natural_language_memory and self.runtime.text_token_ids is not None:
            payload.update(
                {
                    "text_token_ids": self.runtime.text_token_ids.detach().cpu(),
                    "text_token_mask": self.runtime.text_token_mask.detach().cpu(),
                    "text_slot_valid": self.runtime.text_slot_valid.detach().cpu(),
                    "text_slot_keys": self.runtime.text_slot_keys.detach().cpu(),
                    "text_slot_age": self.runtime.text_slot_age.detach().cpu(),
                    "text_write_counter": self.runtime.text_write_counter.detach().cpu(),
                    "text_key_token_ids": self.runtime.text_key_token_ids.detach().cpu(),
                    "text_key_token_mask": self.runtime.text_key_token_mask.detach().cpu(),
                }
            )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        return output_path

    def load_runtime_memory(
        self,
        path: str | Path,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Load a user's runtime memory into this freshly created model."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if isinstance(payload, Tensor):
            state = payload
            raw_memory = None
        elif isinstance(payload, dict):
            state = payload.get("memory_state", payload.get("state"))
            raw_memory = payload.get("raw_memory")
        else:
            state = None
            raw_memory = None
        if not isinstance(state, Tensor):
            raise ValueError("runtime memory file does not contain a tensor memory_state")
        expected_shape = (
            self.memory_config.memory_slots,
            self.memory_config.memory_dim,
        )
        if state.ndim != 3 or tuple(state.shape[1:]) != expected_shape:
            raise ValueError(
                f"runtime memory shape {tuple(state.shape)} does not match {expected_shape}"
            )
        saved_hidden_size = payload.get("model_hidden_size") if isinstance(payload, dict) else None
        if saved_hidden_size is not None and int(saved_hidden_size) != self.memory.hidden_size:
            raise ValueError(
                f"runtime memory hidden size {saved_hidden_size} does not match {self.memory.hidden_size}"
            )

        target_device = device or self._find_layer_device()
        state = state.to(device=target_device, dtype=self.memory.slot_keys.dtype)
        self._persistent_memory = state.detach()
        if self.memory_config.persistent_memory:
            self.persistent_memory = self._persistent_memory
        self.runtime.state = self._persistent_memory
        self.runtime.last_read = None
        self.runtime.raw_memory = None
        if self.memory_config.natural_language_memory:
            token_ids = payload.get("text_token_ids") if isinstance(payload, dict) else None
            token_mask = payload.get("text_token_mask") if isinstance(payload, dict) else None
            slot_valid = payload.get("text_slot_valid") if isinstance(payload, dict) else None
            slot_keys = payload.get("text_slot_keys") if isinstance(payload, dict) else None
            slot_age = payload.get("text_slot_age") if isinstance(payload, dict) else None
            write_counter = payload.get("text_write_counter") if isinstance(payload, dict) else None
            key_token_ids = payload.get("text_key_token_ids") if isinstance(payload, dict) else None
            key_token_mask = payload.get("text_key_token_mask") if isinstance(payload, dict) else None
            if not all(
                isinstance(item, Tensor)
                for item in (
                    token_ids,
                    token_mask,
                    slot_valid,
                    slot_keys,
                    slot_age,
                    write_counter,
                    key_token_ids,
                    key_token_mask,
                )
            ):
                raise ValueError("natural-language memory file is missing its indexed text bank")
            expected_text_shape = (
                state.shape[0],
                self.memory_config.memory_slots,
                self.memory_config.text_memory_tokens,
            )
            if tuple(token_ids.shape) != expected_text_shape or tuple(token_mask.shape) != expected_text_shape:
                raise ValueError("natural-language text bank shape does not match memory config")
            if tuple(slot_valid.shape) != expected_text_shape[:2]:
                raise ValueError("natural-language slot validity shape does not match memory config")
            if tuple(slot_keys.shape) != (state.shape[0], self.memory_config.memory_slots, self.memory.hidden_size):
                raise ValueError("natural-language slot key shape does not match memory config")
            if tuple(slot_age.shape) != expected_text_shape[:2] or tuple(write_counter.shape) != (state.shape[0],):
                raise ValueError("natural-language slot age shape does not match memory config")
            expected_key_shape = (
                state.shape[0],
                self.memory_config.memory_slots,
                self.memory_config.text_memory_key_tokens,
            )
            if tuple(key_token_ids.shape) != expected_key_shape or tuple(key_token_mask.shape) != expected_key_shape:
                raise ValueError("natural-language key token shape does not match memory config")
            self.persistent_text_token_ids = token_ids.to(device=target_device, dtype=torch.long)
            self.persistent_text_token_mask = token_mask.to(device=target_device, dtype=torch.bool)
            self.persistent_text_slot_valid = slot_valid.to(device=target_device, dtype=torch.bool)
            self.persistent_text_slot_keys = slot_keys.to(device=target_device, dtype=torch.float32)
            self.persistent_text_slot_age = slot_age.to(device=target_device, dtype=torch.long)
            self.persistent_text_write_counter = write_counter.to(device=target_device, dtype=torch.long)
            self.persistent_text_key_token_ids = key_token_ids.to(device=target_device, dtype=torch.long)
            self.persistent_text_key_token_mask = key_token_mask.to(device=target_device, dtype=torch.bool)
            self.runtime.text_token_ids = self.persistent_text_token_ids
            self.runtime.text_token_mask = self.persistent_text_token_mask
            self.runtime.text_slot_valid = self.persistent_text_slot_valid
            self.runtime.text_slot_keys = self.persistent_text_slot_keys
            self.runtime.text_slot_age = self.persistent_text_slot_age
            self.runtime.text_write_counter = self.persistent_text_write_counter
            self.runtime.text_key_token_ids = self.persistent_text_key_token_ids
            self.runtime.text_key_token_mask = self.persistent_text_key_token_mask
        if raw_memory is not None:
            if not isinstance(raw_memory, Tensor):
                raise ValueError("runtime memory raw_memory must be a tensor or null")
            if raw_memory.ndim == 1:
                raw_memory = raw_memory.unsqueeze(0)
            if tuple(raw_memory.shape) != (state.shape[0], self.memory.hidden_size):
                raise ValueError(f"unexpected raw_memory shape: {tuple(raw_memory.shape)}")
            output_weight = self.base_model.get_output_embeddings().weight
            self.runtime.raw_memory = raw_memory.to(
                device=output_weight.device,
                dtype=output_weight.dtype,
            ).detach()
        return self._persistent_memory

    def save_memory_adapter(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.memory.state_dict(), path / "memory.pt")
        if self.text_retriever is not None and self._text_retriever_ready:
            torch.save(self.text_retriever.state_dict(), path / "text_retriever.pt")
        if self.memory_policy is not None and self._memory_policy_ready:
            torch.save(self.memory_policy.state_dict(), path / "memory_policy.pt")
        metadata = {
            "hidden_size": self.memory.hidden_size,
            "memory_config": asdict(self.memory_config),
            "layer_indices": list(self.layer_indices),
            "text_retriever_ready": self._text_retriever_ready,
            "memory_policy_ready": self._memory_policy_ready,
        }
        (path / "memory_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        surgery_state = {
            str(index): adapter.blend_logit.detach().cpu()
            for index, adapter in zip(self.layer_indices, self._memory_adapters)
            if hasattr(adapter, "blend_logit")
        }
        torch.save({"blend_logits": surgery_state}, path / "surgery.pt")
        if self.memory_config.persistent_memory and self.persistent_memory.numel() > 0:
            payload = {
                "format_version": 1,
                "memory_state": self.persistent_memory.detach().cpu(),
            }
            if self.memory_config.natural_language_memory:
                payload.update(
                    {
                        "text_token_ids": self.persistent_text_token_ids.detach().cpu(),
                        "text_token_mask": self.persistent_text_token_mask.detach().cpu(),
                        "text_slot_valid": self.persistent_text_slot_valid.detach().cpu(),
                        "text_slot_keys": self.persistent_text_slot_keys.detach().cpu(),
                        "text_slot_age": self.persistent_text_slot_age.detach().cpu(),
                        "text_write_counter": self.persistent_text_write_counter.detach().cpu(),
                        "text_key_token_ids": self.persistent_text_key_token_ids.detach().cpu(),
                        "text_key_token_mask": self.persistent_text_key_token_mask.detach().cpu(),
                    }
                )
            torch.save(
                payload,
                path / "persistent_memory.pt",
            )

    def save_persistent_memory_checkpoint(self, output_dir: str | Path) -> None:
        """Save controller weights and the current user memory together.

        This produces a compact model-owned memory checkpoint.  The large
        frozen Qwen shards remain untouched; the learned reader/controller and
        the user's current memory are stored in the adapter package.
        """

        state = self.runtime.state if self.runtime.state is not None else self._persistent_memory
        if state is None:
            raise ValueError("no runtime memory state is available to checkpoint")
        was_persistent = self.memory_config.persistent_memory
        self.memory_config.persistent_memory = True
        self.persistent_memory = state.detach().clone()
        self._persistent_memory = self.persistent_memory.detach()
        if self.memory_config.natural_language_memory:
            text_fields = (
                self.runtime.text_token_ids,
                self.runtime.text_token_mask,
                self.runtime.text_slot_valid,
                self.runtime.text_slot_keys,
                self.runtime.text_slot_age,
                self.runtime.text_write_counter,
                self.runtime.text_key_token_ids,
                self.runtime.text_key_token_mask,
            )
            if not all(isinstance(item, Tensor) for item in text_fields):
                raise ValueError("natural-language memory bank is not initialized")
            target_device = self._find_layer_device()
            self.persistent_text_token_ids = text_fields[0].detach().clone().to(target_device, dtype=torch.long)
            self.persistent_text_token_mask = text_fields[1].detach().clone().to(target_device, dtype=torch.bool)
            self.persistent_text_slot_valid = text_fields[2].detach().clone().to(target_device, dtype=torch.bool)
            self.persistent_text_slot_keys = text_fields[3].detach().clone().to(target_device, dtype=torch.float32)
            self.persistent_text_slot_age = text_fields[4].detach().clone().to(target_device, dtype=torch.long)
            self.persistent_text_write_counter = text_fields[5].detach().clone().to(target_device, dtype=torch.long)
            self.persistent_text_key_token_ids = text_fields[6].detach().clone().to(target_device, dtype=torch.long)
            self.persistent_text_key_token_mask = text_fields[7].detach().clone().to(target_device, dtype=torch.bool)
        try:
            self.save_memory_adapter(output_dir)
            path = Path(output_dir)
            metadata_path = path / "memory_config.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["memory_config"]["persistent_memory"] = True
            metadata["memory_config"]["checkpoint_contains_user_memory"] = True
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        finally:
            self.memory_config.persistent_memory = was_persistent

    def load_memory_adapter(self, adapter_dir: str | Path, *, strict: bool = True) -> None:
        path = Path(adapter_dir)
        state = torch.load(path / "memory.pt", map_location=self._find_layer_device(), weights_only=True)
        self.memory.load_state_dict(state, strict=strict)
        metadata_ready = None
        policy_ready = None
        metadata_path = path / "memory_config.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict) and "text_retriever_ready" in metadata:
                metadata_ready = bool(metadata["text_retriever_ready"])
            if isinstance(metadata, dict) and "memory_policy_ready" in metadata:
                policy_ready = bool(metadata["memory_policy_ready"])
        if metadata_ready and self.text_retriever is None:
            raise ValueError(
                "adapter contains a trained text retriever but natural_language_memory is disabled"
            )
        retriever_path = path / "text_retriever.pt"
        if self.text_retriever is not None and metadata_ready and not retriever_path.exists():
            raise FileNotFoundError(
                f"adapter declares a trained text retriever but is missing {retriever_path}"
            )
        should_load_retriever = (
            self.text_retriever is not None
            and retriever_path.exists()
            and metadata_ready is not False
        )
        if should_load_retriever:
            retriever_state = torch.load(
                retriever_path,
                map_location=self._find_layer_device(),
                weights_only=True,
            )
            self.text_retriever.load_state_dict(retriever_state, strict=True)
            self._text_retriever_ready = True
        policy_path = path / "memory_policy.pt"
        if policy_ready and self.memory_policy is None:
            raise ValueError(
                "adapter contains an automatic memory policy but automatic_memory is disabled"
            )
        if self.memory_policy is not None and policy_ready is not None:
            if policy_ready and not policy_path.exists():
                raise FileNotFoundError(
                    f"adapter declares an automatic memory policy but is missing {policy_path}"
                )
            if policy_ready:
                policy_state = torch.load(
                    policy_path,
                    map_location=self._find_layer_device(),
                    weights_only=True,
                )
                self.memory_policy.load_state_dict(policy_state, strict=strict)
                self._memory_policy_ready = True
        surgery_path = path / "surgery.pt"
        if surgery_path.exists():
            surgery_state = torch.load(surgery_path, map_location="cpu", weights_only=True)
            blend_logits = surgery_state.get("blend_logits", {})
            for index, adapter in zip(self.layer_indices, self._memory_adapters):
                if hasattr(adapter, "blend_logit") and str(index) in blend_logits:
                    adapter.blend_logit.data.copy_(blend_logits[str(index)].to(adapter.blend_logit.device))
        persistent_path = path / "persistent_memory.pt"
        if persistent_path.exists():
            payload = torch.load(persistent_path, map_location="cpu", weights_only=True)
            state = payload.get("memory_state", payload.get("state")) if isinstance(payload, dict) else payload
            if not isinstance(state, Tensor):
                raise ValueError("persistent_memory.pt does not contain a tensor memory_state")
            expected_shape = (self.memory_config.memory_slots, self.memory_config.memory_dim)
            if state.ndim != 3 or tuple(state.shape[1:]) != expected_shape:
                raise ValueError(
                    f"persistent memory shape {tuple(state.shape)} does not match {expected_shape}"
                )
            target_device = self._find_layer_device()
            self.memory_config.persistent_memory = True
            self.persistent_memory = state.to(
                device=target_device,
                dtype=self.memory.slot_keys.dtype,
            )
            self._persistent_memory = self.persistent_memory.detach()
            self.runtime.state = self._persistent_memory
            if self.memory_config.natural_language_memory:
                token_ids = payload.get("text_token_ids") if isinstance(payload, dict) else None
                token_mask = payload.get("text_token_mask") if isinstance(payload, dict) else None
                slot_valid = payload.get("text_slot_valid") if isinstance(payload, dict) else None
                slot_keys = payload.get("text_slot_keys") if isinstance(payload, dict) else None
                slot_age = payload.get("text_slot_age") if isinstance(payload, dict) else None
                write_counter = payload.get("text_write_counter") if isinstance(payload, dict) else None
                key_token_ids = payload.get("text_key_token_ids") if isinstance(payload, dict) else None
                key_token_mask = payload.get("text_key_token_mask") if isinstance(payload, dict) else None
                if not all(
                    isinstance(item, Tensor)
                    for item in (
                        token_ids,
                        token_mask,
                        slot_valid,
                        slot_keys,
                        slot_age,
                        write_counter,
                        key_token_ids,
                        key_token_mask,
                    )
                ):
                    raise ValueError("persistent adapter is missing its indexed natural-language bank")
                expected_text_shape = (
                    state.shape[0],
                    self.memory_config.memory_slots,
                    self.memory_config.text_memory_tokens,
                )
                if tuple(token_ids.shape) != expected_text_shape or tuple(token_mask.shape) != expected_text_shape:
                    raise ValueError("persistent text bank shape does not match memory config")
                if tuple(slot_valid.shape) != expected_text_shape[:2]:
                    raise ValueError("persistent slot validity shape does not match memory config")
                if tuple(slot_keys.shape) != (state.shape[0], self.memory_config.memory_slots, self.memory.hidden_size):
                    raise ValueError("persistent slot key shape does not match memory config")
                if tuple(slot_age.shape) != expected_text_shape[:2] or tuple(write_counter.shape) != (state.shape[0],):
                    raise ValueError("persistent slot age shape does not match memory config")
                expected_key_shape = (
                    state.shape[0],
                    self.memory_config.memory_slots,
                    self.memory_config.text_memory_key_tokens,
                )
                if tuple(key_token_ids.shape) != expected_key_shape or tuple(key_token_mask.shape) != expected_key_shape:
                    raise ValueError("persistent key token shape does not match memory config")
                self.persistent_text_token_ids = token_ids.to(device=target_device, dtype=torch.long)
                self.persistent_text_token_mask = token_mask.to(device=target_device, dtype=torch.bool)
                self.persistent_text_slot_valid = slot_valid.to(device=target_device, dtype=torch.bool)
                self.persistent_text_slot_keys = slot_keys.to(device=target_device, dtype=torch.float32)
                self.persistent_text_slot_age = slot_age.to(device=target_device, dtype=torch.long)
                self.persistent_text_write_counter = write_counter.to(device=target_device, dtype=torch.long)
                self.persistent_text_key_token_ids = key_token_ids.to(device=target_device, dtype=torch.long)
                self.persistent_text_key_token_mask = key_token_mask.to(device=target_device, dtype=torch.bool)
                self.runtime.text_token_ids = self.persistent_text_token_ids
                self.runtime.text_token_mask = self.persistent_text_token_mask
                self.runtime.text_slot_valid = self.persistent_text_slot_valid
                self.runtime.text_slot_keys = self.persistent_text_slot_keys
                self.runtime.text_slot_age = self.persistent_text_slot_age
                self.runtime.text_write_counter = self.persistent_text_write_counter
                self.runtime.text_key_token_ids = self.persistent_text_key_token_ids
                self.runtime.text_key_token_mask = self.persistent_text_key_token_mask

    @torch.no_grad()
    def _load_persistent_memory_payload(self, payload: dict[str, Any]) -> None:
        """Load a runtime/persistent memory payload shared by .pt and safetensors."""

        state = payload.get("memory_state", payload.get("state"))
        if not isinstance(state, Tensor):
            raise ValueError("persistent memory payload does not contain a tensor memory_state")
        expected_shape = (self.memory_config.memory_slots, self.memory_config.memory_dim)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected_shape:
            raise ValueError(
                f"persistent memory shape {tuple(state.shape)} does not match {expected_shape}"
            )
        target_device = self._find_layer_device()
        self.memory_config.persistent_memory = True
        self.runtime.use_persistent_state = True
        self.persistent_memory = state.to(device=target_device, dtype=self.memory.slot_keys.dtype)
        self._persistent_memory = self.persistent_memory.detach()
        self.runtime.state = self._persistent_memory
        if not self.memory_config.natural_language_memory:
            return

        fields = {
            "persistent_text_token_ids": ("text_token_ids", torch.long),
            "persistent_text_token_mask": ("text_token_mask", torch.bool),
            "persistent_text_slot_valid": ("text_slot_valid", torch.bool),
            "persistent_text_slot_keys": ("text_slot_keys", torch.float32),
            "persistent_text_slot_age": ("text_slot_age", torch.long),
            "persistent_text_write_counter": ("text_write_counter", torch.long),
            "persistent_text_key_token_ids": ("text_key_token_ids", torch.long),
            "persistent_text_key_token_mask": ("text_key_token_mask", torch.bool),
        }
        if not all(isinstance(payload.get(key), Tensor) for key, _ in fields.values()):
            raise ValueError("persistent memory payload is missing its indexed natural-language bank")
        expected_text_shape = (
            state.shape[0],
            self.memory_config.memory_slots,
            self.memory_config.text_memory_tokens,
        )
        expected_key_shape = (
            state.shape[0],
            self.memory_config.memory_slots,
            self.memory_config.text_memory_key_tokens,
        )
        expected_shapes = {
            "text_token_ids": expected_text_shape,
            "text_token_mask": expected_text_shape,
            "text_slot_valid": expected_text_shape[:2],
            "text_slot_keys": (state.shape[0], self.memory_config.memory_slots, self.memory.hidden_size),
            "text_slot_age": expected_text_shape[:2],
            "text_write_counter": (state.shape[0],),
            "text_key_token_ids": expected_key_shape,
            "text_key_token_mask": expected_key_shape,
        }
        for key, _ in fields.values():
            if tuple(payload[key].shape) != expected_shapes[key]:
                raise ValueError(f"persistent field {key} shape does not match memory config")
        for attribute, (key, dtype) in fields.items():
            setattr(self, attribute, payload[key].to(device=target_device, dtype=dtype))
        runtime_fields = {
            "text_token_ids": self.persistent_text_token_ids,
            "text_token_mask": self.persistent_text_token_mask,
            "text_slot_valid": self.persistent_text_slot_valid,
            "text_slot_keys": self.persistent_text_slot_keys,
            "text_slot_age": self.persistent_text_slot_age,
            "text_write_counter": self.persistent_text_write_counter,
            "text_key_token_ids": self.persistent_text_key_token_ids,
            "text_key_token_mask": self.persistent_text_key_token_mask,
        }
        for attribute, value in runtime_fields.items():
            setattr(self.runtime, attribute, value)

    def _export_memory_os_v2_checkpoint(self) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Split a V2 bank into tensors plus JSON-safe checkpoint metadata."""

        if self.memory_os_v2 is None:
            return {}, {}
        payload = self.memory_os_v2.export_payload()
        tensors: dict[str, Tensor] = {}
        metadata = dict(payload)

        def pack_record(item: dict[str, Any], prefix: str) -> dict[str, Any]:
            item = dict(item)
            for field in ("key", "summary", "semantic_key", "token_ids", "token_mask"):
                value = item.pop(field, None)
                if isinstance(value, Tensor):
                    name = f"dynamic_memory.v2.{prefix}.{field}"
                    tensors[name] = value.detach().cpu().contiguous()
                    item[f"{field}_ref"] = name
            return item

        metadata["records"] = [
            pack_record(item, f"records.{index}")
            for index, item in enumerate(payload.get("records", []))
        ]
        metadata["quarantine"] = [
            pack_record(item, f"quarantine.{index}")
            for index, item in enumerate(payload.get("quarantine", []))
        ]
        metadata["pages"] = []
        for index, item in enumerate(payload.get("pages", [])):
            item = dict(item)
            for field in ("key", "summary"):
                value = item.pop(field, None)
                if isinstance(value, Tensor):
                    name = f"dynamic_memory.v2.pages.{index}.{field}"
                    tensors[name] = value.detach().cpu().contiguous()
                    item[f"{field}_ref"] = name
            metadata["pages"].append(item)
        return tensors, metadata

    def _load_memory_os_v2_checkpoint(
        self,
        metadata_text: str,
        weights: dict[str, Tensor],
    ) -> None:
        """Rehydrate the V2 page store from safetensors metadata and tensors."""

        if self.memory_router_v2 is None:
            return
        payload = json.loads(metadata_text)

        def unpack_record(item: dict[str, Any]) -> dict[str, Any]:
            item = dict(item)
            for field in ("key", "summary", "semantic_key", "token_ids", "token_mask"):
                ref = item.pop(f"{field}_ref", None)
                item[field] = weights.get(ref) if ref else None
            return item

        payload["records"] = [unpack_record(item) for item in payload.get("records", [])]
        payload["quarantine"] = [unpack_record(item) for item in payload.get("quarantine", [])]
        unpacked_pages = []
        for item in payload.get("pages", []):
            item = dict(item)
            for field in ("key", "summary"):
                ref = item.pop(f"{field}_ref", None)
                item[field] = weights.get(ref) if ref else None
            unpacked_pages.append(item)
        payload["pages"] = unpacked_pages
        tier_store = None
        if self.memory_config.memory_storage_mode == "tiered":
            if not self.memory_config.memory_storage_path:
                raise ValueError("tiered memory requires memory_storage_path")
            tier_store = TieredMemoryStoreV2(
                self.memory_config.memory_storage_path,
                key_dim=self.memory_config.memory_router_dim,
                page_capacity=self.memory_config.memory_page_capacity,
            )
        self.memory_os_v2 = MemoryOSV2.from_payload(
            payload,
            router=self.memory_router_v2,
            tier_store=tier_store,
            max_resident_pages=self.memory_config.memory_resident_pages,
            runtime_device=self._find_layer_device(),
            gpu_cache_records=self.memory_config.memory_gpu_cache_records,
            gpu_cache_tokens=self.memory_config.memory_gpu_cache_tokens,
            gpu_cache_reserve_mb=self.memory_config.memory_gpu_cache_reserve_mb,
            gpu_cache_adaptive=self.memory_config.memory_gpu_cache_adaptive,
            record_scorer=self._score_semantic_memory_records,
            min_read_margin=self.memory_config.memory_min_read_margin,
            require_evidence=self.memory_config.memory_require_evidence,
        )
        # The package configuration is authoritative for runtime safety and
        # capacity gates. Older snapshots may contain looser values.
        self.memory_os_v2.read_threshold = self.memory_config.memory_v2_read_threshold
        self.memory_os_v2.write_threshold = self.memory_config.memory_v2_write_threshold
        self.memory_os_v2.min_read_margin = self.memory_config.memory_min_read_margin
        self.memory_os_v2.require_evidence = self.memory_config.memory_require_evidence
        bank = self.memory_os_v2.bank
        bank.record_scorer = self._score_semantic_memory_records
        bank.page_capacity = self.memory_config.memory_page_capacity
        bank.max_pages = self.memory_config.memory_max_pages
        bank.hot_pages = self.memory_config.memory_hot_pages
        bank.top_k_pages = self.memory_config.memory_top_k_pages
        bank.top_k_records = self.memory_config.memory_top_k_records
        bank.max_hops = self.memory_config.memory_max_hops
        bank._rebuild_coarse_index(self.memory_config.memory_coarse_index_bits)
        # The package configuration is authoritative for runtime safety gates.
        # Older snapshots may contain a looser threshold in their serialized
        # payload; do not let that silently re-enable noisy recalls after an
        # upgrade.
        self.memory_os_v2.read_threshold = self.memory_config.memory_v2_read_threshold
        self.memory_os_v2.write_threshold = self.memory_config.memory_v2_write_threshold
        self.memory_os_v2.min_read_margin = self.memory_config.memory_min_read_margin
        self.memory_os_v2.require_evidence = self.memory_config.memory_require_evidence

    @torch.no_grad()
    def load_embedded_memory_weights(self, merged_dir: str | Path) -> None:
        """Load memory modules and optional user state from one safetensors package."""

        path = Path(merged_dir)
        manifest = json.loads((path / "memory_merge.json").read_text(encoding="utf-8"))
        from safetensors import safe_open

        weights_path = path / str(manifest.get("memory_weights", "memory.safetensors"))
        tensor_prefix = str(manifest.get("tensor_prefix", "dynamic_memory."))
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            file_metadata = handle.metadata() or {}
            weights = {
                key: handle.get_tensor(key)
                for key in handle.keys()
                if key.startswith(tensor_prefix)
            }

        memory_prefix = f"{tensor_prefix}memory."
        memory_state = {
            key.removeprefix(memory_prefix): value
            for key, value in weights.items()
            if key.startswith(memory_prefix)
        }
        if not memory_state:
            raise ValueError(f"merged memory file has no {memory_prefix}* tensors: {weights_path}")
        self.memory.load_state_dict(memory_state, strict=True)

        retriever_prefix = f"{tensor_prefix}text_retriever."
        retriever_state = {
            key.removeprefix(retriever_prefix): value
            for key, value in weights.items()
            if key.startswith(retriever_prefix)
        }
        if retriever_state:
            if self.text_retriever is None:
                raise ValueError("merged package contains a text retriever but it is disabled")
            self.text_retriever.load_state_dict(retriever_state, strict=True)
            self._text_retriever_ready = True

        policy_prefix = f"{tensor_prefix}memory_policy."
        policy_state = {
            key.removeprefix(policy_prefix): value
            for key, value in weights.items()
            if key.startswith(policy_prefix)
        }
        if policy_state:
            if self.memory_policy is None:
                raise ValueError("merged package contains a memory policy but it is disabled")
            # A v2 candidate may be loaded on top of an older embedded package:
            # the old shard has only the write head, while the candidate adds
            # a trained forget head.  The adapter load immediately afterwards
            # is strict and supplies the complete v2 policy.
            self.memory_policy.load_state_dict(
                policy_state,
                strict=self.memory_config.automatic_memory_policy_version < 2,
            )
            self._memory_policy_ready = True

        router_prefix = f"{tensor_prefix}memory_router_v2."
        router_state = {
            key.removeprefix(router_prefix): value
            for key, value in weights.items()
            if key.startswith(router_prefix)
        }
        if router_state:
            if self.memory_router_v2 is None:
                raise ValueError("merged package contains a V2 router but hierarchical memory is disabled")
            self.memory_router_v2.load_state_dict(router_state, strict=True)
            self._memory_router_v2_ready = True

        for index, adapter in zip(self.layer_indices, self._memory_adapters):
            key = f"{tensor_prefix}blend_logits.{index}"
            if hasattr(adapter, "blend_logit") and key in weights:
                adapter.blend_logit.data.copy_(weights[key].to(adapter.blend_logit.device))

        persistent_prefix = f"{tensor_prefix}persistent."
        persistent_payload = {
            key.removeprefix(persistent_prefix): value
            for key, value in weights.items()
            if key.startswith(persistent_prefix)
        }
        if persistent_payload:
            self._load_persistent_memory_payload(persistent_payload)
        v2_payload = file_metadata.get("memory_os_v2_payload")
        if v2_payload and self.memory_os_v2 is not None:
            self._load_memory_os_v2_checkpoint(v2_payload, weights)

    @torch.no_grad()
    def save_embedded_memory_weights(self, merged_dir: str | Path) -> None:
        """Atomically write the current memory snapshot back into the main shard.

        The merged package keeps the original Qwen shard as the base payload
        and appends ``dynamic_memory.*`` tensors to the same safetensors file.
        This intentionally rewrites the package's second shard, so callers
        should use it as a deliberate weight-backed persistence mode rather
        than on every token.
        """

        path = Path(merged_dir)
        manifest = json.loads((path / "memory_merge.json").read_text(encoding="utf-8"))
        from safetensors.torch import save_file
        from .merge_memory_weights import _write_merged_shard

        memory_weights_name = str(
            manifest.get("memory_weights", "model.safetensors-00002-of-00002.safetensors")
        )
        target_shard = path / memory_weights_name
        if not target_shard.exists():
            raise FileNotFoundError(f"merged package target shard is missing: {target_shard}")

        tensors: dict[str, Tensor] = {}
        tensors.update(
            {
                f"dynamic_memory.memory.{key}": value.detach().cpu().contiguous()
                for key, value in self.memory.state_dict().items()
            }
        )
        if self.text_retriever is not None and self._text_retriever_ready:
            tensors.update(
                {
                    f"dynamic_memory.text_retriever.{key}": value.detach().cpu().contiguous()
                    for key, value in self.text_retriever.state_dict().items()
                }
            )
        if self.memory_router_v2 is not None:
            tensors.update(
                {
                    f"dynamic_memory.memory_router_v2.{key}": value.detach().cpu().contiguous()
                    for key, value in self.memory_router_v2.state_dict().items()
                }
            )
        if self.memory_policy is not None and self._memory_policy_ready:
            tensors.update(
                {
                    f"dynamic_memory.memory_policy.{key}": value.detach().cpu().contiguous()
                    for key, value in self.memory_policy.state_dict().items()
                }
            )
        for index, adapter in zip(self.layer_indices, self._memory_adapters):
            if hasattr(adapter, "blend_logit"):
                tensors[f"dynamic_memory.blend_logits.{index}"] = (
                    adapter.blend_logit.detach().cpu().contiguous()
                )

        if self.memory_config.persistent_memory and self.persistent_memory.numel() > 0:
            state = self.persistent_memory
        else:
            state = self.runtime.state if self.runtime.state is not None else self._persistent_memory
        if state is None:
            raise ValueError("cannot write embedded memory without a runtime memory state")
        tensors["dynamic_memory.persistent.memory_state"] = state.detach().cpu().contiguous()
        if self.memory_config.natural_language_memory:
            if self.memory_config.persistent_memory and self.persistent_text_token_ids.numel() > 0:
                runtime_fields = {
                    "text_token_ids": self.persistent_text_token_ids,
                    "text_token_mask": self.persistent_text_token_mask,
                    "text_slot_valid": self.persistent_text_slot_valid,
                    "text_slot_keys": self.persistent_text_slot_keys,
                    "text_slot_age": self.persistent_text_slot_age,
                    "text_write_counter": self.persistent_text_write_counter,
                    "text_key_token_ids": self.persistent_text_key_token_ids,
                    "text_key_token_mask": self.persistent_text_key_token_mask,
                }
            else:
                runtime_fields = {
                    "text_token_ids": self.runtime.text_token_ids,
                    "text_token_mask": self.runtime.text_token_mask,
                    "text_slot_valid": self.runtime.text_slot_valid,
                    "text_slot_keys": self.runtime.text_slot_keys,
                    "text_slot_age": self.runtime.text_slot_age,
                    "text_write_counter": self.runtime.text_write_counter,
                    "text_key_token_ids": self.runtime.text_key_token_ids,
                    "text_key_token_mask": self.runtime.text_key_token_mask,
                }
            if not all(isinstance(value, Tensor) for value in runtime_fields.values()):
                raise ValueError("cannot write embedded memory without a complete text memory bank")
            tensors.update(
                {
                    f"dynamic_memory.persistent.{key}": value.detach().cpu().contiguous()
                    for key, value in runtime_fields.items()
                }
            )

        v2_metadata: dict[str, Any] = {}
        if self.memory_os_v2 is not None:
            v2_tensors, v2_metadata = self._export_memory_os_v2_checkpoint()
            tensors.update(v2_tensors)

        extra_path = path / f".{memory_weights_name}.memory-update.tmp"
        staging_path = path / f".{memory_weights_name}.merged-update.tmp"

        def _sync_model_index(shard_name: str) -> None:
            """Keep the HF shard index aligned with the embedded memory shard."""

            index_path = path / "model.safetensors.index.json"
            if not index_path.exists():
                return
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                return
            changed = False
            for key, value in list(weight_map.items()):
                if str(key).startswith("dynamic_memory.") and value != shard_name:
                    weight_map[key] = shard_name
                    changed = True
            if changed:
                index_path.write_text(
                    json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
                )

        try:
            metadata = {"format": "qwen_dynamic_memory_embedded_v2"}
            if v2_metadata:
                metadata["memory_os_v2_payload"] = json.dumps(
                    v2_metadata, ensure_ascii=False, separators=(",", ":")
                )
            save_file(tensors, str(extra_path), metadata=metadata)
            if str(manifest.get("shard_mode", "combined")) == "extra_only":
                # The extra-only layout keeps the official Qwen shards
                # untouched.  Windows may keep the currently loaded shard
                # open, so rotate to a new small shard when replacement is
                # denied instead of requiring the chat process to exit.
                try:
                    os.replace(extra_path, target_shard)
                except PermissionError:
                    rotated_name = f"{target_shard.stem}.runtime-{time.time_ns()}.safetensors"
                    rotated_path = path / rotated_name
                    os.replace(extra_path, rotated_path)
                    manifest["memory_weights"] = rotated_name
                    manifest["checkpoint_contains_user_memory"] = True
                    (path / "memory_merge.json").write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                _sync_model_index(str(manifest.get("memory_weights", memory_weights_name)))
                return
            base_dir = Path(str(manifest["base_model"]))
            base_shard = base_dir / memory_weights_name
            if not base_shard.exists():
                raise FileNotFoundError(f"merged package base shard is missing: {base_shard}")
            _write_merged_shard(base_shard, extra_path, staging_path)
            os.replace(staging_path, target_shard)
            _sync_model_index(memory_weights_name)
        finally:
            if extra_path.exists():
                extra_path.unlink()
            if staging_path.exists():
                staging_path.unlink()

    def forward(
        self,
        *args: Any,
        memory_state: Optional[Tensor] = None,
        update_memory: bool = True,
        read_memory: bool = True,
        detach_memory: bool = False,
        return_memory: bool = True,
        memory_text_input_ids: Optional[Tensor] = None,
        memory_text_attention_mask: Optional[Tensor] = None,
        memory_key_input_ids: Optional[Tensor] = None,
        memory_key_attention_mask: Optional[Tensor] = None,
        memory_query_input_ids: Optional[Tensor] = None,
        memory_query_attention_mask: Optional[Tensor] = None,
        memory_storage_input_ids: Optional[Tensor] = None,
        memory_storage_attention_mask: Optional[Tensor] = None,
        force_memory_write: bool = False,
        memory_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        batch_size = input_ids.shape[0] if input_ids is not None else None
        reset_mask = self._reset_mask(input_ids)
        if reset_mask is not None and self.memory_os_v2 is not None:
            # The reset token is a model-level durable-memory clear signal,
            # not merely a transient recurrent-state reset.
            self.clear_hierarchical_memory()

        if memory_state is not None:
            state = memory_state
        elif self.memory_config.persistent_memory and self.persistent_memory.numel() > 0:
            state = self.persistent_memory
        elif self._persistent_memory is not None:
            state = self._persistent_memory
        elif batch_size is not None:
            state = self.memory.initial_state(batch_size, device=self._find_layer_device())
        else:
            state = None
        if state is not None and batch_size is not None and state.shape[0] != batch_size:
            if self.memory_config.persistent_memory and state.shape[0] == 1:
                state = state.expand(batch_size, -1, -1).clone()
            else:
                raise ValueError(f"memory batch {state.shape[0]} does not match input batch {batch_size}")
        if reset_mask is not None and state is not None:
            state = state.clone()
            state[reset_mask] = 0

        if batch_size is not None:
            self._bind_text_memory(batch_size, device=self._find_layer_device())
            if reset_mask is not None:
                self._clear_text_memory(reset_mask)

        self.runtime.state = state
        self.runtime.read_enabled = read_memory
        self.runtime.update_enabled = update_memory
        self.runtime.last_read = None
        self.runtime.reset_mask = reset_mask
        self.runtime.input_ids = input_ids
        self.runtime.attention_mask = kwargs.get("attention_mask")
        kwargs.setdefault("use_cache", False)
        base_output = self.base_model(*args, **kwargs)
        if (
            not self.memory_config.natural_language_memory
            and self.memory_config.direct_logit_scale > 0.0
            and read_memory
            and self.runtime.last_read is not None
            and hasattr(base_output, "logits")
        ):
            read = self.runtime.last_read
            output_embeddings = self.base_model.get_output_embeddings()
            memory_logits = output_embeddings(read.to(output_embeddings.weight.device))
            logits = base_output.logits + self.memory_config.direct_logit_scale * memory_logits.to(
                base_output.logits.device
            )
            base_output.logits = logits
            labels = kwargs.get("labels")
            if labels is not None:
                base_output.loss = F.cross_entropy(
                    logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
                    labels[..., 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
        if (
            not self.memory_config.natural_language_memory
            and self.memory_config.raw_logit_scale > 0.0
            and read_memory
            and self.runtime.raw_memory is not None
            and hasattr(base_output, "logits")
        ):
            output_embeddings = self.base_model.get_output_embeddings()
            raw_logits = output_embeddings(self.runtime.raw_memory.to(output_embeddings.weight.device))
            raw_logits = raw_logits[:, None, :].expand(-1, base_output.logits.shape[1], -1)
            base_output.logits = base_output.logits + self.memory_config.raw_logit_scale * raw_logits.to(
                base_output.logits.device
            )
            labels = kwargs.get("labels")
            if labels is not None:
                base_output.loss = F.cross_entropy(
                    base_output.logits[..., :-1, :].contiguous().view(-1, base_output.logits.shape[-1]),
                    labels[..., 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
        new_state = self.runtime.state
        if new_state is not None and detach_memory:
            new_state = new_state.detach()
        self._persistent_memory = new_state.detach() if new_state is not None else None
        if self.memory_config.persistent_memory and self.runtime.use_persistent_state and new_state is not None:
            self.persistent_memory = new_state.detach()
        if self.memory_config.natural_language_memory and update_memory:
            self.runtime.auto_memory_probability = None
            self.runtime.auto_memory_forget_probability = None
            if (
                self.memory_policy is not None
                and self.memory_config.automatic_memory
                and self._memory_policy_ready
                and isinstance(
                    getattr(
                        self.memory,
                        "last_write_summary",
                        getattr(self.memory, "last_write_representation", None),
                    ),
                    Tensor,
                )
            ):
                policy_input = getattr(
                    self.memory,
                    "last_write_summary",
                    self.memory.last_write_representation,
                )
                policy_logits = self.memory_policy(policy_input)
                self.runtime.auto_memory_probability = torch.sigmoid(policy_logits).detach()
                if hasattr(self.memory_policy, "forget_logits"):
                    forget_logits = self.memory_policy.forget_logits(policy_input)
                    self.runtime.auto_memory_forget_probability = torch.sigmoid(forget_logits).detach()
            self._write_text_memory(
                input_ids,
                kwargs.get("attention_mask"),
                memory_text_input_ids,
                memory_text_attention_mask,
                memory_key_input_ids,
                memory_key_attention_mask,
                memory_storage_input_ids,
                memory_storage_attention_mask,
                force_memory_write,
                memory_text,
            )
        self.runtime.state = new_state
        return QwenDynamicMemoryOutput(base_output, new_state) if return_memory else base_output

    @torch.no_grad()
    def generate(
        self,
        *args: Any,
        memory_state: Optional[Tensor] = None,
        update_memory: bool = True,
        memory_query_input_ids: Optional[Tensor] = None,
        memory_query_attention_mask: Optional[Tensor] = None,
        memory_query_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the native HF generation loop with memory-aware decoder layers."""
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self.runtime.context_compaction = None
        self.runtime.text_read_seconds = 0.0
        reset_mask = self._reset_mask(input_ids)
        if reset_mask is not None and self.memory_os_v2 is not None:
            # Clear before any context compaction so the reset marker can
            # never cause the old bank to be archived again in this call.
            self.clear_hierarchical_memory()
        if input_ids is not None and kwargs.get("attention_mask") is None:
            kwargs["attention_mask"] = torch.ones_like(input_ids)
        if input_ids is not None and reset_mask is None:
            compacted_ids, compacted_mask, compaction_info = self.compact_context_for_kv(
                input_ids,
                kwargs.get("attention_mask"),
            )
            input_ids = compacted_ids
            kwargs["input_ids"] = compacted_ids
            kwargs["attention_mask"] = compacted_mask
            self.runtime.context_compaction = compaction_info
        elif input_ids is not None:
            self.runtime.context_compaction = {
                "compacted": False,
                "reason": "reset_token",
                "archived_records": 0,
            }
        # This is the public prompt returned by generate.  When compaction
        # occurred it intentionally contains the bounded hot window, not a
        # second copy of the potentially enormous source prompt.
        original_input_ids = input_ids
        original_prompt_length = int(input_ids.shape[1]) if input_ids is not None else 0
        if memory_state is not None:
            self.runtime.state = memory_state
        elif (
            self.memory_config.persistent_memory
            and self.runtime.use_persistent_state
            and self.persistent_memory.numel() > 0
        ):
            self.runtime.state = self.persistent_memory
        elif self._persistent_memory is not None and input_ids is not None and self._persistent_memory.shape[0] == input_ids.shape[0]:
            self.runtime.state = self._persistent_memory
        elif input_ids is not None:
            self.runtime.state = self.memory.initial_state(input_ids.shape[0], device=self._find_layer_device())
        if reset_mask is not None and self.runtime.state is not None:
            state = self.runtime.state.clone()
            if state.shape[0] == 1 and reset_mask.shape[0] > 1:
                state = state.expand(reset_mask.shape[0], -1, -1).clone()
            state[reset_mask] = 0
            self.runtime.state = state
        if input_ids is not None:
            if self.memory_config.natural_language_memory:
                self._bind_text_memory(input_ids.shape[0], device=self._find_layer_device())
                if reset_mask is not None:
                    self._clear_text_memory(reset_mask)
            if "attention_mask" not in kwargs or kwargs["attention_mask"] is None:
                kwargs["attention_mask"] = torch.ones_like(input_ids)
            prefix_ids: Optional[Tensor] = None
            prefix_mask: Optional[Tensor] = None
            prefix_length = 0
            retrieval_ids = memory_query_input_ids if memory_query_input_ids is not None else input_ids
            retrieval_mask = (
                memory_query_attention_mask
                if memory_query_attention_mask is not None
                else kwargs["attention_mask"]
            )
            if (
                self.memory_config.natural_language_memory
                and self.runtime.text_slot_valid is not None
                and retrieval_ids is not None
                and retrieval_mask is not None
            ):
                if input_ids.device.type == "cuda":
                    torch.cuda.synchronize(input_ids.device)
                read_started = time.perf_counter()
                prefix_ids, prefix_mask, prefix_length = self._build_text_prefix(
                    retrieval_ids.to(device=input_ids.device),
                    retrieval_mask.to(device=input_ids.device),
                    query_text=memory_query_text or "",
                )
                if input_ids.device.type == "cuda":
                    torch.cuda.synchronize(input_ids.device)
                self.runtime.text_read_seconds = time.perf_counter() - read_started
            if prefix_ids is not None and prefix_mask is not None and prefix_length > 0:
                input_ids = torch.cat((prefix_ids, input_ids), dim=1)
                kwargs["input_ids"] = input_ids
                kwargs["attention_mask"] = torch.cat(
                    (prefix_mask, kwargs["attention_mask"]), dim=1
                )
                # Internal memory tokens are hidden from the public wrapper
                # API, but remain visible to the actual Qwen forward pass.
                self.runtime.input_ids = input_ids
        # In natural-language mode, the retrieved text prefix is the complete
        # read path.  Running the older continuous residual at the same time
        # would distort Qwen's native chat control tokens.  With no prefix the
        # memory path is an exact no-op and the base model remains untouched.
        self.runtime.read_enabled = not self.memory_config.natural_language_memory
        self.runtime.update_enabled = update_memory
        self.runtime.last_read = None
        self.runtime.reset_mask = reset_mask
        self.runtime.input_ids = input_ids
        if self.memory_config.kv_offload or self.memory_config.kv_cache_implementation is not None:
            # Transformers 5.x owns the offload lifecycle.  Supplying an
            # explicit DynamicCache lets us honor the non-sliding-layer
            # setting, which the shorthand ``cache_implementation`` does not
            # expose for dynamic caches.
            kwargs["use_cache"] = True
            cache_implementation = self.memory_config.kv_cache_implementation
            if cache_implementation is None:
                cache_implementation = "offloaded"
            if (
                cache_implementation == "offloaded"
                and kwargs.get("past_key_values") is None
            ):
                try:
                    from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin

                    class _NaturalMemoryOffloadedCache(DynamicCache):
                        """Prefetch the current layer before every cache update.

                        Transformers 5.9 prefetches the next layer as part of
                        ``Cache.update``.  Qwen3.5's hybrid linear/full stack
                        can enter the same layer again on the next decoding
                        step before that next-layer prefetch runs, leaving a
                        CPU tensor to concatenate with CUDA states.  The
                        current-layer prefetch makes the ownership explicit
                        for both attention and recurrent cache updates.
                        """

                        def update(self, key_states, value_states, layer_idx, *args, **extra):
                            if self.offloading:
                                self.prefetch(layer_idx, self.only_non_sliding)
                            return super().update(key_states, value_states, layer_idx, *args, **extra)

                        def update_conv_state(self, conv_states, layer_idx, **extra):
                            if self.offloading:
                                self.prefetch(layer_idx, self.only_non_sliding)
                            value = super().update_conv_state(conv_states, layer_idx, **extra)
                            if self.offloading:
                                self.offload(layer_idx, self.only_non_sliding)
                            return value

                        def update_recurrent_state(self, recurrent_states, layer_idx, **extra):
                            if self.offloading:
                                self.prefetch(layer_idx, self.only_non_sliding)
                            value = super().update_recurrent_state(recurrent_states, layer_idx, **extra)
                            if self.offloading:
                                self.offload(layer_idx, self.only_non_sliding)
                            return value

                        def offload(self, layer_idx, only_non_sliding=True):
                            # Qwen3.5 linear-attention state is tiny and is
                            # read before update.  Keep it on the execution
                            # device; offload only the full-attention KV that
                            # dominates memory usage.
                            if isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
                                return
                            return super().offload(layer_idx, only_non_sliding)

                    decoder_config = (
                        self.base_model.config.get_text_config(decoder=True)
                        if hasattr(self.base_model.config, "get_text_config")
                        else getattr(self.base_model.config, "text_config", self.base_model.config)
                    )
                    kwargs["past_key_values"] = _NaturalMemoryOffloadedCache(
                        config=decoder_config,
                        offloading=True,
                        offload_only_non_sliding=self.memory_config.kv_offload_only_non_sliding,
                    )
                except (ImportError, TypeError, AttributeError):
                    # Older Transformers releases can still understand the
                    # public generation option even if their DynamicCache
                    # constructor has a different signature.
                    kwargs["cache_implementation"] = cache_implementation
            elif kwargs.get("past_key_values") is None:
                kwargs["cache_implementation"] = cache_implementation
        # Natural-language memory is consumed as an internal text prefix.  Do
        # not also add a constant token-level logit bias, because that would
        # repeat the first value token at every generation position.
        if (
            not self.memory_config.natural_language_memory
            and (self.memory_config.direct_logit_scale > 0.0 or self.memory_config.raw_logit_scale > 0.0)
        ):
            from transformers import LogitsProcessor, LogitsProcessorList

            runtime = self.runtime
            output_embeddings = self.base_model.get_output_embeddings()
            prompt_length = int(input_ids.shape[1]) if input_ids is not None else None

            class MemoryLogitsProcessor(LogitsProcessor):
                def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
                    first_generated_token = prompt_length is None or input_ids.shape[1] == prompt_length
                    if first_generated_token and runtime.raw_memory is not None and raw_scale > 0.0:
                        raw = runtime.raw_memory.to(output_embeddings.weight.device)
                        scores = scores + raw_scale * output_embeddings(raw).to(scores.device)
                    if runtime.last_read is None:
                        return scores
                    last_read = runtime.last_read[:, -1].to(output_embeddings.weight.device)
                    memory_logits = output_embeddings(last_read).to(scores.device)
                    return scores + self_scale * memory_logits

            self_scale = self.memory_config.direct_logit_scale
            raw_scale = self.memory_config.raw_logit_scale
            existing = kwargs.get("logits_processor")
            if existing is None:
                kwargs["logits_processor"] = LogitsProcessorList([MemoryLogitsProcessor()])
            else:
                kwargs["logits_processor"] = LogitsProcessorList(
                    list(existing) + [MemoryLogitsProcessor()]
                )
        if input_ids is not None and args:
            args = ()
        replaced_adapters = self._disable_adapters_for_plain_generation()
        try:
            generated = self.base_model.generate(*args, **kwargs)
        finally:
            self._restore_adapters(replaced_adapters)
        if (
            original_input_ids is not None
            and input_ids is not None
            and input_ids.shape[1] != original_prompt_length
        ):
            generated = torch.cat(
                (
                    original_input_ids,
                    generated[:, input_ids.shape[1] :],
                ),
                dim=1,
            )
        if self.runtime.state is not None:
            self._persistent_memory = self.runtime.state.detach()
            if self.memory_config.persistent_memory and self.runtime.use_persistent_state:
                self.persistent_memory = self._persistent_memory
        return generated

    def train(self, mode: bool = True) -> "QwenDynamicMemoryModel":
        self.memory.train(mode)
        if self.memory_router_v2 is not None:
            self.memory_router_v2.train(mode)
        self.base_model.eval()
        return self

    def _disable_adapters_for_plain_generation(self) -> list[tuple[int, nn.Module]]:
        """Temporarily remove inactive layer wrappers from the hot path.

        Natural-language memory is already materialized as a text prefix. If
        the current generation neither reads nor writes the native residual
        memory, keeping Python adapter modules around four decoder layers only
        adds dispatch overhead on every prefill/decode step. The original
        layers are restored by ``_restore_adapters`` even when generation
        raises.
        """

        if self.runtime.read_enabled or self.runtime.update_enabled:
            return []
        layers = resolve_decoder_layers(self.base_model)[1]
        replaced: list[tuple[int, nn.Module]] = []
        for layer_index, adapter in zip(self.layer_indices, self._memory_adapters):
            if layers[layer_index] is adapter:
                replaced.append((layer_index, adapter))
                layers[layer_index] = adapter.inner
        return replaced

    def _restore_adapters(self, replaced: list[tuple[int, nn.Module]]) -> None:
        layers = resolve_decoder_layers(self.base_model)[1]
        for layer_index, adapter in replaced:
            layers[layer_index] = adapter


def load_qwen_base(
    model_path: str | Path,
    *,
    load_in_4bit: bool = True,
    device_map: str | dict[str, Any] = "auto",
    max_memory: Optional[dict[Any, str]] = None,
):
    """Load the unmodified local Qwen checkpoint with the benchmark defaults."""

    import transformers

    load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
    }
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if load_in_4bit:
        load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    loader = getattr(transformers, "AutoModelForImageTextToText", None)
    if loader is not None:
        try:
            return loader.from_pretrained(str(model_path), **load_kwargs)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            # Text-only backbones (Llama/Qwen3/Mistral/OLMoE/Granite/Phi/Cohere...)
            # are not image-text-to-text models; fall back to the plain causal LM
            # so the memory surgery can be applied to them as well.
            logger.debug("AutoModelForImageTextToText failed (%s); trying AutoModelForCausalLM", exc)
    return transformers.AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)


def load_qwen_dynamic(
    model_path: str | Path,
    *,
    memory_config: Optional[QwenMemoryConfig] = None,
    load_in_4bit: bool = True,
    device_map: str | dict[str, Any] = "auto",
    freeze_backbone: bool = True,
    max_memory: Optional[dict[Any, str]] = None,
) -> QwenDynamicMemoryModel:
    """Load Qwen and attach either an explicit adapter or an embedded merge package."""

    model_path = Path(model_path)
    embedded_manifest = model_path / "memory_merge.json"
    if memory_config is None and embedded_manifest.exists():
        embedded_config = model_path / "memory_config.json"
        if not embedded_config.exists():
            raise FileNotFoundError(
                f"embedded memory package is missing {embedded_config}"
            )
        memory_config = load_memory_config(model_path)
    if memory_config is not None and memory_config.memory_storage_mode == "tiered":
        storage_path = Path(memory_config.memory_storage_path) if memory_config.memory_storage_path else None
        if storage_path is None:
            storage_path = model_path / "memory_pages.sqlite"
        elif not storage_path.is_absolute():
            storage_path = model_path / storage_path
        memory_config.memory_storage_path = str(storage_path)

    base_model = load_qwen_base(
        model_path,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        max_memory=max_memory,
    )
    model = QwenDynamicMemoryModel(
        base_model,
        memory_config=memory_config,
        freeze_backbone=freeze_backbone,
    )
    if embedded_manifest.exists():
        model.load_embedded_memory_weights(model_path)
    model.setup_attribute_coverage(model_path)
    return model


def load_tokenizer(model_path: str | Path):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
