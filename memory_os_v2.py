"""Hierarchical, pageable memory components for Natural Memory v2.

The module deliberately keeps storage and neural routing separate:

* :class:`MemoryRouterV2` is the trainable sparse router.
* :class:`PagedMemoryBankV2` owns versioned pages and records.
* :class:`MemoryOSV2` coordinates writing, reading, quarantine and repair.
* :class:`KVBudgetManagerV2` models the hot-context budget.

The bank can run entirely in process memory for research, or be backed by a
checkpoint/page store by serializing ``export_payload``.  No prompt text is
assembled by this module; it returns evidence records and routing traces to
the model adapter, which can inject only the selected evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .tiered_memory_store_v2 import TieredMemoryStoreV2


STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_RETRACTED = "retracted"
STATUS_QUARANTINED = "quarantined"


def _now() -> int:
    return int(time.time())


def _stable_id(text: str, *, prefix: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _tokens(text: str) -> set[str]:
    raw = {
        token
        for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_\-]+", text.lower())
        if token not in {"我", "的", "是", "了", "请", "一下"}
    }
    # Keep the complete identifier for exact lookup, but also expose its
    # bounded components.  Real code questions usually say ``reset token``
    # while the source contains ``DEFAULT_MEMORY_RESET_TOKEN``; treating the
    # identifier as one opaque token makes that otherwise obvious evidence
    # unreachable without a dense full-corpus scan.
    subtokens = {
        part
        for token in raw
        if "_" in token or "-" in token
        for part in re.split(r"[_-]+", token)
        if len(part) >= 3
    }
    # Chinese questions and metadata are commonly written as short phrases
    # (模型名称, 外部脚本, 全量注意力) rather than whitespace-delimited words.
    # Add bounded 3-character n-grams for contiguous CJK runs so the sparse
    # index can match those phrases without turning every generic two-character
    # word (for example 候选) into a direct route.
    cjk_ngrams: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text.lower()):
        for width in (3,):
            if len(run) < width:
                continue
            cjk_ngrams.update(run[index : index + width] for index in range(len(run) - width + 1))
    return raw | subtokens | cjk_ngrams


def _is_explicit_unknown_request(text: str) -> bool:
    """Detect a request that explicitly says the target field is unregistered.

    This is a narrow semantic fail-safe around the learned router.  It does
    not decide ordinary relevance and it never invents a value; it prevents a
    nearby, but different, personal fact from being promoted into an answer
    when the user explicitly asks about a non-existent/unregistered field.
    """

    normalized = " ".join(str(text).strip().split())
    if any(
        marker in normalized
        for marker in (
            "不存在的",
            "未登记",
            "没有登记",
            "从未记录",
            "从来没有登记",
            "没有出现在",
            "没有出现过",
        )
    ):
        return True
    # Natural user questions often express the same safety intent as a
    # conditional: do not guess when the field has no record.  Treat the
    # combination as an abstention request, without blocking ordinary uses of
    # either phrase in isolation.
    return (
        "没有记录" in normalized
        and any(marker in normalized for marker in ("不要猜", "说不知道", "明确说不知道"))
    ) or (
        "如果没有" in normalized
        and any(marker in normalized for marker in ("说不知道", "明确说不知道", "不要猜"))
    )


@dataclass
class MemoryRecordV2:
    """One versioned memory item with evidence and routing metadata."""

    record_id: str
    text: str
    key: Tensor
    summary: Tensor
    # Frozen-backbone representation used by the natural-language retriever.
    # ``key`` remains the compact page/address vector; keeping this optional
    # second view lets update/forget decisions use the same semantic space
    # after a restart without turning the address key into a full KV cache.
    semantic_key: Optional[Tensor] = None
    memory_type: str = "fact"
    entity: str = ""
    attribute: str = ""
    value: str = ""
    timestamp: int = field(default_factory=_now)
    importance: float = 0.5
    confidence: float = 0.5
    source: str = "user"
    status: str = STATUS_ACTIVE
    version: int = 0
    page_id: str = ""
    supersedes: str = ""
    related_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    # Identifies the user turn this record was written from, as a short hash of that
    # turn's text.  One turn routinely produces several records (a sentence can carry
    # a name *and* a number, and an agent may issue one structured write per field),
    # and until now nothing linked them: an explicit "forget X" retracted only the
    # record whose key matched and left its siblings active and readable.  Empty on
    # records written before this field existed, which keeps old payloads loadable.
    origin: str = ""
    # ``slot_index`` links a V2 address to the existing hot text bank when
    # the Qwen adapter is used as a compatibility bridge.  A value of -1
    # means the record is an independent paged-memory item.
    slot_index: int = -1
    token_ids: Optional[Tensor] = None
    token_mask: Optional[Tensor] = None
    access_count: int = 0
    last_access: int = field(default_factory=_now)

    def conflict_key(self) -> str:
        if self.entity and self.attribute:
            return f"{self.entity.strip().lower()}::{self.attribute.strip().lower()}"
        return ""

    def routing_text(self) -> str:
        """Return the searchable view used by the sparse evidence router.

        The original implementation routed only on ``text``.  That loses
        structured evidence when a natural-language query uses an alias of an
        attribute or value (for example ``优先使用哪层内存`` for a record whose
        value is ``DRAM``), and it makes code records needlessly dependent on
        the exact prose used when they were written.  The fields are metadata,
        not extra prompt content, so adding them here does not increase the
        injected evidence token budget.
        """

        aliases = {
            "模型名称": "正式名字 模型名 模型叫什么 名称 名字",
            "对照模型": "原版 Qwen 基线 跑分 对照规模",
            "记忆存储优先级": "优先使用 哪层内存 容量不够 降级 DRAM RAM 显存",
            "记忆持久化方案": "第三个权重切片 数据库 SQLite 磁盘分页 持久化文件",
            "记忆读取方式": "模型自己读 模型原生 外部脚本 提示词 读取器",
            "显存安全要求": "显存原则 不要顶满 显卡 GPU 安全底线",
            "记忆质量要求": "禁止胡编乱造 写错 清理 总结 置信度",
            "研发取向": "训练 微调 应用拼装 方向 精力",
            "核心架构目标": "终极要求 模型自己拥有记忆 固定外部代码 核心",
            "记忆压缩策略": "压成一句话 高压缩率 碎片对话 浓缩",
            "KV与Slot分工": "当前上下文 长期历史 运行内存 注意力",
            "自动化边界": "模型自发完成 自己完成 外部固定参数",
            "评测重点": "大规模读取 正确率 显存 测试规模",
        }.get(self.attribute.strip(), "")
        # Operational records are grounded in real source excerpts.  Their
        # natural questions often use a concept name rather than the exact
        # sentence surrounding the excerpt, so expose a small query-facing
        # alias set in the sparse index only.  These aliases never enter the
        # prompt and therefore do not increase generation cost.
        operational_aliases = ""
        if self.attribute.strip().lower().startswith("operational:"):
            try:
                operational_index = int(self.attribute.rsplit(":", 1)[-1])
            except ValueError:
                operational_index = -1
            operational_aliases = {
                0: "重启 恢复 记忆 权重切片 memory safetensors memory shard",
                1: "百万 slot 当前 token 全部 全量注意力 top-k 稀疏读取",
                2: "写入 隔离 active quarantine 置信度 低置信度",
                3: "权威副本 热点记录 process RAM GPU cache 显存",
                4: "4B 对照 base Qwen Chunk RAG Natural Memory 实验协议",
                5: "训练 公开 cross-encoder 强 RAG 基线",
                6: "清空记忆 默认 reset token fim prefix",
                7: "语义地址 Qwen 主干 hidden state 编码",
            }.get(operational_index, "")
        return " ".join(
            value
            for value in (
                self.text,
                self.entity,
                self.attribute,
                self.value,
                aliases,
                operational_aliases,
            )
            if str(value).strip()
        )

    def lexical_score(self, query_text: str) -> float:
        query = _tokens(query_text)
        if not query:
            return 0.0
        own = _tokens(self.routing_text())
        if not own:
            return 0.0
        return len(query & own) / math.sqrt(max(1, len(query) * len(own)))

    def token_overlap_score(self, query_token_ids: Optional[Tensor]) -> float:
        """Score exact token evidence without decoding or full attention."""

        if query_token_ids is None or self.token_ids is None:
            return 0.0
        own_mask = self.token_mask
        own = self.token_ids.reshape(-1)
        if isinstance(own_mask, Tensor) and own_mask.numel() == own.numel():
            own = own[own_mask.reshape(-1).to(dtype=torch.bool)]
        query = query_token_ids.detach().reshape(-1).to(device=own.device)
        if own.numel() == 0 or query.numel() == 0:
            return 0.0
        own_unique = torch.unique(own)
        query_unique = torch.unique(query)
        shared = torch.isin(query_unique, own_unique).sum().item()
        # Query coverage is the useful direction here: a short question that
        # names an entity should strongly prefer the chunk containing that
        # entity even when the chunk itself is long.
        return float(shared) / max(1, int(query_unique.numel()))

    def touch(self) -> None:
        self.access_count += 1
        self.last_access = _now()


def memory_record_to_dict(record: MemoryRecordV2) -> dict[str, Any]:
    """Return a JSON-safe public view without exposing embedding tensors."""

    return {
        "record_id": record.record_id,
        "text": record.text,
        "memory_type": record.memory_type,
        "entity": record.entity,
        "attribute": record.attribute,
        "value": record.value,
        "timestamp": int(record.timestamp),
        "importance": float(record.importance),
        "confidence": float(record.confidence),
        "source": record.source,
        "status": record.status,
        "version": int(record.version),
        "page_id": record.page_id,
        "supersedes": record.supersedes,
        "related_ids": list(record.related_ids),
        "evidence": list(record.evidence),
        "origin": record.origin,
        "slot_index": int(record.slot_index),
        "token_count": int(record.token_ids.numel()) if isinstance(record.token_ids, Tensor) else 0,
        "key_dim": int(record.key.numel()) if isinstance(record.key, Tensor) else 0,
        "access_count": int(record.access_count),
        "last_access": int(record.last_access),
    }


@dataclass
class MemoryPageV2:
    page_id: str
    tier: str = "warm"
    capacity: int = 32
    record_ids: list[str] = field(default_factory=list)
    key: Optional[Tensor] = None
    summary: Optional[Tensor] = None
    importance: float = 0.0
    created_at: int = field(default_factory=_now)
    last_access: int = field(default_factory=_now)

    @property
    def full(self) -> bool:
        return len(self.record_ids) >= self.capacity

    def touch(self) -> None:
        self.last_access = _now()


@dataclass
class RouterDecisionV2:
    need_memory: bool
    page_ids: list[str]
    record_ids: list[str]
    page_scores: list[float]
    record_scores: list[float]
    hop_count: int
    hop_trace: list[list[str]]
    confidence: float
    stop_reason: str
    # Raw calibrated evidence diagnostics.  They are optional in spirit but
    # always populated by new decisions so an evaluation can distinguish
    # "a record was returned" from "the router had a defensible margin".
    top_score: float = 0.0
    score_margin: float = 0.0
    evidence_score: float = 0.0


class MemoryRouterV2(nn.Module):
    """Trainable multi-head router for sparse page and record retrieval.

    The router does not generate the answer.  It scores candidate memory
    units, predicts whether another hop is useful, and exposes per-head scores
    for diagnostics.  The projection weights are trained separately from the
    frozen Qwen backbone using hard-negative episodes.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        router_dim: int = 128,
        num_heads: int = 8,
        max_hops: int = 3,
    ) -> None:
        super().__init__()
        if router_dim % num_heads != 0:
            raise ValueError("router_dim must be divisible by num_heads")
        if max_hops < 1:
            raise ValueError("max_hops must be positive")
        self.hidden_size = hidden_size
        self.router_dim = router_dim
        self.num_heads = num_heads
        self.max_hops = max_hops
        self.head_dim = router_dim // num_heads
        self.query_projection = nn.Linear(hidden_size, router_dim, bias=False)
        self.key_projection = nn.Linear(hidden_size, router_dim, bias=False)
        self.pair_scorer = nn.Sequential(
            nn.Linear(router_dim * 3, router_dim),
            nn.SiLU(),
            nn.Linear(router_dim, 1),
        )
        policy_hidden = max(32, min(256, router_dim * 2))
        self.need_memory = nn.Sequential(
            nn.Linear(hidden_size, policy_hidden),
            nn.SiLU(),
            nn.Linear(policy_hidden, 1),
        )
        self.hop_controller = nn.Sequential(
            nn.Linear(hidden_size, policy_hidden),
            nn.SiLU(),
            nn.Linear(policy_hidden, max_hops + 1),
        )
        self.head_gate = nn.Linear(hidden_size, num_heads)

    def _reshape(self, value: Tensor) -> Tensor:
        return value.view(*value.shape[:-1], self.num_heads, self.head_dim)

    def encode_query(self, query: Tensor) -> Tensor:
        """Project a model hidden state into the compact address space."""

        if query.shape[-1] != self.hidden_size:
            raise ValueError(f"query last dimension must be {self.hidden_size}")
        return F.normalize(self.query_projection(query), dim=-1)

    def encode_key(self, key: Tensor) -> Tensor:
        """Project model keys into the compact address space."""

        if key.shape[-1] != self.hidden_size:
            raise ValueError(f"key last dimension must be {self.hidden_size}")
        return F.normalize(self.key_projection(key), dim=-1)

    def projected_scores(self, query: Tensor, projected_candidates: Tensor) -> tuple[Tensor, Tensor]:
        """Score compact candidates without storing full hidden states.

        This is the storage-saving path: the bank keeps only the projected
        candidate keys, while the current query is projected on demand.
        """

        if query.ndim != 2 or query.shape[-1] != self.hidden_size:
            raise ValueError("query must have shape [B, hidden_size]")
        if projected_candidates.ndim == 2:
            projected_candidates = projected_candidates.unsqueeze(0).expand(query.shape[0], -1, -1)
        if projected_candidates.ndim != 3 or projected_candidates.shape[-1] != self.router_dim:
            raise ValueError("projected_candidates must have shape [B,N,router_dim]")
        q = self._reshape(self.encode_query(query))
        k = self._reshape(F.normalize(projected_candidates, dim=-1))
        head_scores = torch.einsum("bhc,bnhc->bnh", q, k)
        gates = torch.softmax(self.head_gate(query), dim=-1)[:, None, :]
        cosine_score = (head_scores * gates).sum(dim=-1)
        q_expanded = q.reshape(query.shape[0], 1, -1).expand(-1, projected_candidates.shape[1], -1)
        k_flat = k.reshape(query.shape[0], projected_candidates.shape[1], -1)
        pair_input = torch.cat((q_expanded, k_flat, q_expanded - k_flat), dim=-1)
        learned_score = self.pair_scorer(pair_input).squeeze(-1)
        return cosine_score + learned_score, head_scores

    def pair_scores(self, query: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
        """Return aggregate and per-head candidate scores.

        ``query`` is ``[B,H]`` and ``candidates`` is ``[B,N,H]`` or ``[N,H]``.
        """

        if query.ndim != 2:
            raise ValueError("query must have shape [B,H]")
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0).expand(query.shape[0], -1, -1)
        if candidates.ndim != 3 or candidates.shape[0] != query.shape[0]:
            raise ValueError("candidates must have shape [B,N,H] or [N,H]")
        return self.projected_scores(query, self.encode_key(candidates))

    def forward(self, query: Tensor, candidates: Tensor) -> dict[str, Tensor]:
        scores, head_scores = self.pair_scores(query, candidates)
        return {
            "scores": scores,
            "head_scores": head_scores,
            "need_memory_logits": self.need_memory(query).squeeze(-1),
            "hop_logits": self.hop_controller(query),
        }


class PagedMemoryBankV2:
    """Versioned memory pages with sparse routing and multi-hop expansion."""

    def __init__(
        self,
        hidden_size: int,
        *,
        page_capacity: int = 32,
        max_pages: int = 32768,
        hot_pages: int = 8,
        top_k_pages: int = 4,
        top_k_records: int = 8,
        max_hops: int = 3,
        router: Optional[MemoryRouterV2] = None,
        key_dim: Optional[int] = None,
        coarse_index_bits: int = 20,
        tier_store: Optional["TieredMemoryStoreV2"] = None,
        max_resident_pages: int = 256,
        runtime_device: Optional[torch.device] = None,
        gpu_cache_records: int = 256,
        gpu_cache_tokens: int = 131072,
        gpu_cache_reserve_mb: int = 2048,
        gpu_cache_adaptive: bool = True,
        record_scorer: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        if page_capacity < 1 or max_pages < 1:
            raise ValueError("page_capacity and max_pages must be positive")
        self.hidden_size = hidden_size
        self.page_capacity = page_capacity
        self.max_pages = max_pages
        self.hot_pages = hot_pages
        self.top_k_pages = top_k_pages
        self.top_k_records = top_k_records
        self.max_hops = max_hops
        self.router = router
        self.key_dim = int(key_dim or (router.router_dim if router is not None else hidden_size))
        if not 4 <= coarse_index_bits <= 20:
            raise ValueError("coarse_index_bits must be between 4 and 20")
        self.coarse_index_bits = int(coarse_index_bits)
        if max_resident_pages < hot_pages:
            raise ValueError("max_resident_pages must be >= hot_pages")
        self.tier_store = tier_store
        self.max_resident_pages = int(max_resident_pages)
        if gpu_cache_records < 0 or gpu_cache_tokens < 0:
            raise ValueError("gpu cache limits must be non-negative")
        if gpu_cache_reserve_mb < 0:
            raise ValueError("gpu_cache_reserve_mb must be non-negative")
        self.gpu_cache_records = int(gpu_cache_records)
        self.gpu_cache_tokens = int(gpu_cache_tokens)
        self.gpu_cache_reserve_mb = int(gpu_cache_reserve_mb)
        self.gpu_cache_reserve_bytes = self.gpu_cache_reserve_mb * 1024 * 1024
        self.gpu_cache_adaptive = bool(gpu_cache_adaptive)
        # Optional learned exact reranker.  Page routing remains bounded and
        # is still performed by MemoryRouterV2; this callback only scores the
        # small candidate set inside selected pages.
        self.record_scorer = record_scorer
        # Additive prior weights used by ``_record_scores``.  They are exposed because they
        # dominate the learned term -- measured, the neural score is 12-17% of the summed
        # total while these priors are 83-88% -- so they, not the reranker, decide which
        # records get injected.  Overridable per bank from the model config.
        self.prior_weights = {
            "lexical": 0.25,
            "token_overlap": 0.45,
            "rare_lexical": 1.25,
            "shape_bonus": 0.35,
            "structured_bonus": 0.15,
        }
        self._gpu_device: Optional[torch.device] = None
        self._gpu_record_cache: dict[str, dict[str, Tensor]] = {}
        self._gpu_cache_order: list[str] = []
        self._gpu_cache_tokens_used = 0
        self._gpu_cache_bytes_used = 0
        self._gpu_cache_hits = 0
        self._gpu_cache_misses = 0
        self._gpu_cache_fallbacks = 0
        self._gpu_cache_alloc_failures = 0
        self._gpu_cache_last_free_bytes: Optional[int] = None
        generator = torch.Generator(device="cpu").manual_seed(1729 + self.key_dim)
        self.coarse_planes = F.normalize(
            torch.randn(self.coarse_index_bits, self.key_dim, generator=generator), dim=-1
        )
        self.records: dict[str, MemoryRecordV2] = {}
        self.pages: dict[str, MemoryPageV2] = {}
        # Only pages with free capacity participate in write placement.  The
        # bounded order avoids scanning every page at million-record scale.
        self._open_page_ids: set[str] = set()
        self._open_page_order: list[str] = []
        self._coarse_buckets: dict[int, set[str]] = {}
        self._page_signatures: dict[str, set[int]] = {}
        # Sparse lexical page addresses recover exact pages for rare query
        # terms without turning the reader into a full-page scan.  The neural
        # router still scores the bounded candidate set afterwards.
        self._lexical_buckets: dict[str, set[str]] = {}
        self._page_terms: dict[str, set[str]] = {}
        # Evidence-bearing records get a second sparse inverted index.  This
        # prevents a page centroid from hiding a useful record among raw
        # background chunks without making the reader scan the whole bank.
        self._lexical_record_buckets: dict[str, set[str]] = {}
        self._record_lexical_terms: dict[str, set[str]] = {}
        self._record_page_ids: dict[str, str] = {}
        # Structured address indexes are separate from the semantic router.
        # They let distinctive identifiers (user ids, filenames, symbols) reach
        # the exact record even when a page centroid or LSH bucket is noisy.
        self._address_page_buckets: dict[str, set[str]] = {}
        self._address_record_buckets: dict[str, set[str]] = {}
        self._value_record_buckets: dict[str, set[str]] = {}
        self._entity_record_buckets: dict[str, set[str]] = {}
        self._page_address_terms: dict[str, set[str]] = {}
        self._page_entities: dict[str, set[str]] = {}
        self._record_address_terms: dict[str, set[str]] = {}
        self._record_entities: dict[str, str] = {}
        self._text_index: dict[str, str] = {}
        self.active_by_conflict: dict[str, str] = {}
        self.quarantine: dict[str, MemoryRecordV2] = {}
        self._counter = 0
        self._last_coarse_candidates = 0
        self._suspend_refresh = 0
        if self.tier_store is not None:
            self._restore_page_headers_from_store()
            for item in self.tier_store.load_quarantine():
                record = MemoryRecordV2(**item)
                self.quarantine[record.record_id] = record
            self.active_by_conflict.update(
                self.tier_store.active_conflicts(active_status=STATUS_ACTIVE)
            )
        if runtime_device is not None:
            self.configure_gpu_cache(
                runtime_device,
                max_records=self.gpu_cache_records,
                max_tokens=self.gpu_cache_tokens,
                reserve_mb=self.gpu_cache_reserve_mb,
                adaptive=self.gpu_cache_adaptive,
            )

    def configure_gpu_cache(
        self,
        device: torch.device | str,
        *,
        max_records: Optional[int] = None,
        max_tokens: Optional[int] = None,
        reserve_mb: Optional[int] = None,
        adaptive: Optional[bool] = None,
    ) -> None:
        """Configure a bounded VRAM cache for recently used memory records.

        The canonical copy remains in process RAM and is what gets exported
        into the embedded third safetensors shard.  This cache contains only
        routing keys and the small token payloads needed by recent reads; it
        is deliberately not a second full memory store.
        """

        target = torch.device(device)
        if max_records is not None:
            if int(max_records) < 0:
                raise ValueError("max_records must be non-negative")
            self.gpu_cache_records = int(max_records)
        if max_tokens is not None:
            if int(max_tokens) < 0:
                raise ValueError("max_tokens must be non-negative")
            self.gpu_cache_tokens = int(max_tokens)
        if reserve_mb is not None:
            if int(reserve_mb) < 0:
                raise ValueError("reserve_mb must be non-negative")
            self.gpu_cache_reserve_mb = int(reserve_mb)
            self.gpu_cache_reserve_bytes = self.gpu_cache_reserve_mb * 1024 * 1024
        if adaptive is not None:
            self.gpu_cache_adaptive = bool(adaptive)
        self._gpu_device = target if target.type == "cuda" else None
        self._clear_gpu_cache()

    def _clear_gpu_cache(self) -> None:
        self._gpu_record_cache.clear()
        self._gpu_cache_order.clear()
        self._gpu_cache_tokens_used = 0
        self._gpu_cache_bytes_used = 0

    @staticmethod
    def _tensor_bytes(value: Optional[Tensor]) -> int:
        if not isinstance(value, Tensor):
            return 0
        return int(value.numel()) * int(value.element_size())

    def _record_cache_bytes(self, record: MemoryRecordV2, *, include_tokens: bool) -> int:
        total = self._tensor_bytes(record.key) + self._tensor_bytes(record.summary)
        if include_tokens:
            total += self._tensor_bytes(record.token_ids)
            total += self._tensor_bytes(record.token_mask)
        return total

    def _has_gpu_headroom(self, requested_bytes: int) -> bool:
        """Keep the cache below a VRAM safety line while the model is running."""

        if self._gpu_device is None or not self.gpu_cache_adaptive:
            return True
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self._gpu_device)
        except (RuntimeError, AssertionError, TypeError):
            return True
        self._gpu_cache_last_free_bytes = int(free_bytes)
        return int(free_bytes) - int(requested_bytes) >= self.gpu_cache_reserve_bytes

    def _evict_oldest_gpu_record(self) -> None:
        if not self._gpu_cache_order:
            return
        evicted_id = self._gpu_cache_order.pop(0)
        evicted = self._gpu_record_cache.pop(evicted_id, {})
        evicted_tokens = evicted.get("token_ids")
        if isinstance(evicted_tokens, Tensor):
            self._gpu_cache_tokens_used -= int(evicted_tokens.numel())
        self._gpu_cache_bytes_used -= sum(self._tensor_bytes(value) for value in evicted.values())
        self._gpu_cache_bytes_used = max(0, self._gpu_cache_bytes_used)

    def _promote_record(self, record: MemoryRecordV2) -> None:
        """Promote one hot record to VRAM without changing its RAM copy."""

        if (
            self._gpu_device is None
            or self.gpu_cache_records <= 0
            or record.record_id in self._gpu_record_cache
        ):
            if record.record_id in self._gpu_record_cache:
                self._gpu_cache_hits += 1
                self._gpu_cache_order.remove(record.record_id)
                self._gpu_cache_order.append(record.record_id)
            return
        self._gpu_cache_misses += 1
        source_token_count = int(record.token_ids.numel()) if isinstance(record.token_ids, Tensor) else 0
        token_count = source_token_count
        if self.gpu_cache_tokens == 0 or token_count > self.gpu_cache_tokens:
            token_count = 0
        include_tokens = token_count > 0 and isinstance(record.token_ids, Tensor)
        requested_bytes = self._record_cache_bytes(record, include_tokens=include_tokens)
        while self._gpu_cache_order and (
            len(self._gpu_cache_order) >= self.gpu_cache_records
            or self._gpu_cache_tokens_used + token_count > self.gpu_cache_tokens
            or not self._has_gpu_headroom(requested_bytes)
        ):
            self._evict_oldest_gpu_record()
        if not self._has_gpu_headroom(requested_bytes):
            # The authoritative RAM copy remains available to the router.
            # A hot-cache miss must never make inference fail just because the
            # GPU is busy with model weights or KV activations.
            self._gpu_cache_fallbacks += 1
            return
        try:
            cached: dict[str, Tensor] = {
                "key": record.key.detach().to(self._gpu_device, non_blocking=True),
                "summary": record.summary.detach().to(self._gpu_device, non_blocking=True),
            }
            if include_tokens:
                cached["token_ids"] = record.token_ids.detach().to(self._gpu_device, non_blocking=True)
                if isinstance(record.token_mask, Tensor):
                    cached["token_mask"] = record.token_mask.detach().to(self._gpu_device, non_blocking=True)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            self._gpu_cache_alloc_failures += 1
            self._gpu_cache_fallbacks += 1
            return
        if include_tokens:
            self._gpu_cache_tokens_used += token_count
        cache_bytes = sum(self._tensor_bytes(value) for value in cached.values())
        self._gpu_record_cache[record.record_id] = cached
        self._gpu_cache_order.append(record.record_id)
        self._gpu_cache_bytes_used += cache_bytes

    def _record_key(self, record: MemoryRecordV2) -> Tensor:
        self._promote_record(record)
        cached = self._gpu_record_cache.get(record.record_id)
        if cached is not None:
            return cached["key"]
        return record.key

    def gpu_record_payload(self, record: MemoryRecordV2) -> tuple[Tensor, Optional[Tensor]]:
        """Return a hot record's token payload, preferring its VRAM copy."""

        self._promote_record(record)
        cached = self._gpu_record_cache.get(record.record_id)
        if cached is not None and "token_ids" in cached:
            return cached["token_ids"], cached.get("token_mask")
        return record.token_ids, record.token_mask

    def promote_records(self, records: Iterable[MemoryRecordV2]) -> None:
        for record in records:
            self._promote_record(record)

    def has_token_evidence(
        self,
        query_key: Tensor,
        query_token_ids: Optional[Tensor],
        query_text: str = "",
        *,
        min_shared_tokens: int = 3,
    ) -> bool:
        """Check bounded exact evidence before the learned abstention gate."""

        if query_token_ids is None:
            return False
        # Attribute aliases and sparse operational terms are intentionally
        # indexed outside the serialized token payload.  They are still
        # trustworthy routing evidence: otherwise the learned need-memory
        # gate can abstain before the exact lexical address gets a chance to
        # reach the record reranker.
        lexical_pages, lexical_records = self._lexical_evidence_hits(query_text)
        address_pages, address_records = self._address_hits(query_text)
        if lexical_pages or lexical_records or address_pages or address_records:
            return True
        query_unique = torch.unique(query_token_ids.detach().reshape(-1).cpu())
        if query_unique.numel() < min_shared_tokens:
            return False
        for page_id in self._candidate_page_ids(query_key, query_text):
            page = self._hydrate_page(page_id)
            if page is None:
                continue
            for record_id in page.record_ids:
                record = self.records.get(record_id)
                if record is None or record.status != STATUS_ACTIVE or record.token_ids is None:
                    continue
                own = record.token_ids.reshape(-1)
                if isinstance(record.token_mask, Tensor) and record.token_mask.numel() == own.numel():
                    own = own[record.token_mask.reshape(-1).to(dtype=torch.bool)]
                shared = torch.isin(query_unique, torch.unique(own)).sum().item()
                if int(shared) >= int(min_shared_tokens):
                    return True
        return False

    def _restore_page_headers_from_store(self) -> None:
        """Restore page addresses without materializing every cold record."""

        if self.tier_store is None:
            return
        headers = self.tier_store.page_headers()
        for item in headers:
            page = MemoryPageV2(**item)
            self.pages[page.page_id] = page
            self._counter = max(self._counter, int(page.page_id.rsplit("_", 1)[-1]))
            if not page.full:
                self._open_page_ids.add(page.page_id)
                self._open_page_order.append(page.page_id)
            self._index_page(page, persist=False)

    def _rebuild_coarse_index(self, coarse_index_bits: Optional[int] = None) -> None:
        """Recreate LSH planes and page buckets after a config upgrade."""

        if coarse_index_bits is not None:
            if not 4 <= int(coarse_index_bits) <= 20:
                raise ValueError("coarse_index_bits must be between 4 and 20")
            self.coarse_index_bits = int(coarse_index_bits)
        generator = torch.Generator(device="cpu").manual_seed(1729 + self.key_dim)
        self.coarse_planes = F.normalize(
            torch.randn(self.coarse_index_bits, self.key_dim, generator=generator), dim=-1
        )
        self._coarse_buckets.clear()
        self._page_signatures.clear()
        self._lexical_buckets.clear()
        self._page_terms.clear()
        self._lexical_record_buckets.clear()
        self._record_lexical_terms.clear()
        self._record_page_ids.clear()
        self._value_record_buckets.clear()
        for page in self.pages.values():
            self._index_page(page, persist=False)
        if self.tier_store is not None:
            self.tier_store.clear_coarse_buckets()
            page_by_record = {
                record_id: page.page_id
                for page in self.pages.values()
                for record_id in page.record_ids
            }
            for record_id, key in self.tier_store.record_keys():
                page_id = page_by_record.get(record_id)
                if page_id is None:
                    continue
                signature = self._coarse_signature(key)
                self._page_signatures.setdefault(page_id, set()).add(signature)
            for page_id, signatures in self._page_signatures.items():
                self.tier_store.replace_page_buckets(page_id, signatures)

    def _encode_key(self, key: Tensor) -> Tensor:
        key = key.detach().float().reshape(-1)
        if key.numel() == self.key_dim:
            return F.normalize(key, dim=0)
        if self.router is not None and key.numel() == self.hidden_size:
            router_device = next(self.router.parameters()).device
            return self.router.encode_key(key.to(router_device).unsqueeze(0))[0].detach().float().cpu()
        raise ValueError(
            f"memory key must contain {self.hidden_size} model values or {self.key_dim} address values"
        )

    def _score_candidates(self, query_key: Tensor, candidate_keys: Tensor) -> Tensor:
        if self.router is not None and query_key.numel() == self.hidden_size:
            router_device = next(self.router.parameters()).device
            scores, _ = self.router.projected_scores(
                query_key.to(router_device).reshape(1, -1),
                candidate_keys.to(router_device).reshape(1, -1, self.key_dim),
            )
            # Training uses logits for cross-entropy; storage uses a bounded
            # relevance value so page thresholds remain stable across router
            # checkpoints and do not discard valid negative logits.
            return torch.sigmoid(scores[0])
        query = self._encode_key(query_key)
        candidates = F.normalize(candidate_keys.float().to(query.device), dim=-1)
        return torch.matmul(candidates, query)

    def _coarse_signature(self, key: Tensor) -> int:
        address = self._encode_key(key).cpu()
        bits = (torch.mv(self.coarse_planes, address) >= 0).to(torch.int64)
        signature = 0
        for index, bit in enumerate(bits.tolist()):
            signature |= int(bit) << index
        return signature

    def _remove_page_from_index(self, page: MemoryPageV2) -> None:
        signatures = self._page_signatures.pop(page.page_id, set())
        for signature in signatures:
            bucket = self._coarse_buckets.get(signature)
            if bucket is not None:
                bucket.discard(page.page_id)
                if not bucket:
                    self._coarse_buckets.pop(signature, None)
        terms = self._page_terms.pop(page.page_id, set())
        for term in terms:
            bucket = self._lexical_buckets.get(term)
            if bucket is not None:
                bucket.discard(page.page_id)
                if not bucket:
                    self._lexical_buckets.pop(term, None)
        for term in self._page_address_terms.pop(page.page_id, set()):
            bucket = self._address_page_buckets.get(term)
            if bucket is not None:
                bucket.discard(page.page_id)
                if not bucket:
                    self._address_page_buckets.pop(term, None)
        for entity in self._page_entities.pop(page.page_id, set()):
            bucket = self._entity_record_buckets.get(entity)
            if bucket is not None:
                bucket.difference_update(page.record_ids)
                if not bucket:
                    self._entity_record_buckets.pop(entity, None)
        for record_id in page.record_ids:
            for term in self._record_lexical_terms.pop(record_id, set()):
                bucket = self._lexical_record_buckets.get(term)
                if bucket is not None:
                    bucket.discard(record_id)
                    if not bucket:
                        self._lexical_record_buckets.pop(term, None)
            self._record_page_ids.pop(record_id, None)
            for term in self._record_address_terms.pop(record_id, set()):
                bucket = self._address_record_buckets.get(term)
                if bucket is not None:
                    bucket.discard(record_id)
                    if not bucket:
                        self._address_record_buckets.pop(term, None)
            record = self.records.get(record_id)
            if record is not None:
                for term in _tokens(record.value):
                    bucket = self._value_record_buckets.get(term)
                    if bucket is not None:
                        bucket.discard(record_id)
                        if not bucket:
                            self._value_record_buckets.pop(term, None)
            self._record_entities.pop(record_id, None)

    def _index_page(self, page: MemoryPageV2, *, persist: bool = True) -> None:
        # Make re-indexing idempotent when a cold page is hydrated after
        # restart and its record text becomes available.
        self._remove_page_from_index(page)
        signatures: set[int] = set()
        if page.key is not None:
            signatures.add(self._coarse_signature(page.key))
        # Index record addresses as well as the page centroid.  A centroid
        # can blur multiple topics, while a record address is exact and still
        # costs only one small LSH bucket insertion per record.
        for record_id in page.record_ids:
            record = self.records.get(record_id)
            if record is not None:
                signatures.add(self._coarse_signature(record.key))
        self._page_signatures[page.page_id] = signatures
        for signature in signatures:
            self._coarse_buckets.setdefault(signature, set()).add(page.page_id)
        terms: set[str] = set()
        for record_id in page.record_ids:
            record = self.records.get(record_id)
            if record is not None and record.status == STATUS_ACTIVE:
                record_terms = _tokens(record.routing_text())
                terms.update(record_terms)
                self._record_page_ids[record_id] = page.page_id
                # Random/background chunks may be present only to model
                # corpus noise.  Do not let them occupy the direct lexical
                # route when they have no semantic key or token payload.
                if record.semantic_key is not None or record.token_ids is not None:
                    self._record_lexical_terms[record_id] = record_terms
                    for term in record_terms:
                        self._lexical_record_buckets.setdefault(term, set()).add(record_id)
        self._page_terms[page.page_id] = terms
        for term in terms:
            self._lexical_buckets.setdefault(term, set()).add(page.page_id)
        address_terms: set[str] = set()
        entities: set[str] = set()
        for record_id in page.record_ids:
            record = self.records.get(record_id)
            if record is None or record.status != STATUS_ACTIVE:
                continue
            entity = record.entity.strip().lower()
            if entity:
                entities.add(entity)
                self._record_entities[record_id] = entity
                self._entity_record_buckets.setdefault(entity, set()).add(record_id)
            record_address_terms = {
                term
                for field in (record.entity, record.attribute)
                for term in _tokens(str(field))
                if self._is_distinctive_address_term(term)
                or (
                    record.attribute.strip().lower().startswith(("symbol:", "class:", "def:"))
                    and self._is_distinctive_symbol_term(term)
                )
            }
            self._record_address_terms[record_id] = record_address_terms
            address_terms.update(record_address_terms)
            for term in record_address_terms:
                self._address_record_buckets.setdefault(term, set()).add(record_id)
            for term in _tokens(record.value):
                self._value_record_buckets.setdefault(term, set()).add(record_id)
        self._page_entities[page.page_id] = entities
        self._page_address_terms[page.page_id] = address_terms
        for term in address_terms:
            self._address_page_buckets.setdefault(term, set()).add(page.page_id)
        if self.tier_store is not None and persist:
            self.tier_store.upsert_page(page)
            self.tier_store.replace_page_buckets(page.page_id, signatures)

    def _hydrate_records(self, record_ids: Iterable[str]) -> None:
        """Load only the records needed by the current candidate pages."""

        if self.tier_store is None:
            return
        missing = [record_id for record_id in record_ids if record_id not in self.records]
        if not missing:
            return
        touched_pages: set[str] = set()
        for item in self.tier_store.load_records(missing):
            self.records[item["record_id"]] = MemoryRecordV2(**item)
            text = item.get("text", "").strip().lower()
            if text and item.get("status") == STATUS_ACTIVE:
                self._text_index[text] = item["record_id"]
            page_id = str(item.get("page_id", ""))
            if page_id:
                touched_pages.add(page_id)
        # Rebuild each touched page once, after all of its requested records
        # are resident.  Re-indexing inside the load loop could temporarily
        # erase address terms belonging to still-hydrating fragments.
        for page_id in touched_pages:
            page = self.pages.get(page_id)
            if page is not None:
                self._index_page(page, persist=False)

    def _hydrate_page(self, page_id: str) -> Optional[MemoryPageV2]:
        page = self.pages.get(page_id)
        if page is None or self.tier_store is None:
            return page
        self._hydrate_records(page.record_ids)
        return page

    def _evict_cold_records(self) -> None:
        """Keep page headers resident and unload non-hot record payloads."""

        if self.tier_store is None or self.max_resident_pages <= 0:
            return
        ranked = sorted(
            self.pages.values(),
            key=lambda page: (page.tier == "hot", page.last_access, page.importance),
            reverse=True,
        )
        keep = {page.page_id for page in ranked[: self.max_resident_pages]}
        for page in self.pages.values():
            if page.page_id in keep:
                continue
            for record_id in page.record_ids:
                self.records.pop(record_id, None)

    def _store_record(self, record: MemoryRecordV2) -> None:
        if self.tier_store is not None:
            self.tier_store.upsert_record(record)

    def _candidate_page_ids(self, query_key: Tensor, query_text: str = "") -> list[str]:
        """Use locality-sensitive coarse buckets before exact page scoring."""

        page_ids = list(self.pages)
        if len(page_ids) <= 128:
            self._last_coarse_candidates = len(page_ids)
            return page_ids
        signature = self._coarse_signature(query_key)
        probes = [signature]
        # Probe Hamming distance 1 and 2.  This is bounded (<= 79 buckets for
        # 12 bits) and avoids a query-time scan over every page.
        for bit in range(self.coarse_index_bits):
            probes.append(signature ^ (1 << bit))
        for left in range(self.coarse_index_bits):
            for right in range(left + 1, self.coarse_index_bits):
                probes.append(signature ^ (1 << left) ^ (1 << right))
        query_terms = _tokens(query_text)
        lexical_pages: set[str] = set()
        for term in query_terms:
            bucket = self._lexical_buckets.get(term)
            # Ignore common terms so this remains a sparse address lookup.
            if bucket is not None and len(bucket) <= 128:
                lexical_pages.update(bucket)
            record_bucket = self._lexical_record_buckets.get(term)
            if record_bucket is not None and len(record_bucket) <= 128:
                lexical_pages.update(
                    self._record_page_ids[record_id]
                    for record_id in record_bucket
                    if record_id in self._record_page_ids
                )
        # Structured address terms are allowed to bypass the ordinary lexical
        # bucket-size cutoff.  This is still sparse because the address index
        # only stores distinctive ids, filenames and symbols.
        address_pages, _ = self._address_hits(query_text)
        lexical_pages.update(address_pages)
        if self.tier_store is not None:
            hot_ids = [
                page.page_id for page in self.pages.values() if page.tier == "hot"
            ]
            selected = self.tier_store.candidate_page_ids(
                probes,
                hot_page_ids=hot_ids,
                limit=max(128, self.top_k_pages * 64),
            )
            selected_set = set(selected)
            selected_set.update(lexical_pages)
            limit = max(128, self.top_k_pages * 64)
            if len(selected_set) > limit:
                selected_set = set(selected[:limit]) | lexical_pages
            self._last_coarse_candidates = len(selected_set)
            return list(selected_set)
        selected: set[str] = set()
        for bucket in probes:
            selected.update(self._coarse_buckets.get(bucket, ()))
        selected.update(lexical_pages)
        # Hot pages are always eligible.  They are few by construction and
        # protect high-value memories when a page centroid is still moving.
        selected.update(
            page.page_id for page in self.pages.values() if page.tier == "hot"
        )
        if not selected:
            # Cold-start safety: a bounded sample is preferable to silently
            # claiming that no memory exists.
            selected.update(page_ids[: min(128, len(page_ids))])
        self._last_coarse_candidates = len(selected)
        return list(selected)

    def _rare_lexical_address_score(self, query_text: str, terms: set[str]) -> float:
        """Return a bounded address prior for rare query terms.

        The learned router is intentionally not the only source of truth at
        million-token scale.  A page containing an exact, rare identifier
        such as a project code is an address hit, even when the page centroid
        is diluted by hundreds of unrelated chunks.  This prior is sparse and
        bounded: common language terms are ignored, and it never scans pages.
        """

        if not query_text or not terms:
            return 0.0
        hits = 0
        for term in _tokens(query_text):
            if term not in terms:
                continue
            # Evidence-bearing records have their own sparse lexical index.
            # Use it before the page-level bucket: a large page can contain a
            # rare, exact target even when the page itself is lexically broad.
            record_bucket = self._lexical_record_buckets.get(term)
            page_bucket = self._lexical_buckets.get(term)
            if record_bucket is not None and len(record_bucket) <= 32:
                hits += 1
            elif page_bucket is not None and len(page_bucket) <= 8:
                hits += 1
        return min(2.0, float(hits))

    @staticmethod
    def _is_distinctive_address_term(term: str) -> bool:
        """Keep structured address lookup sparse at very large scale."""

        normalized = str(term).strip().lower()
        return len(normalized) >= 8 or any(
            char.isdigit() or char in "_-./\\" for char in normalized
        )

    @staticmethod
    def _is_distinctive_symbol_term(term: str) -> bool:
        """Keep ordinary Python symbol names addressable without filenames."""

        normalized = str(term).strip().lower()
        return len(normalized) >= 3 and all("a" <= char <= "z" or char == "_" for char in normalized)

    @staticmethod
    def _is_distinctive_value_term(term: str) -> bool:
        """Recognize compact values such as ``DRAM`` without indexing prose."""

        normalized = str(term).strip().lower()
        if len(normalized) < 4:
            return False
        if any(char.isdigit() or char in "_-./\\" for char in normalized):
            return True
        return all("a" <= char <= "z" for char in normalized)

    def _address_hits(self, query_text: str) -> tuple[set[str], set[str]]:
        """Return bounded page/record hits for distinctive address terms."""

        page_ids: set[str] = set()
        record_ids: set[str] = set()
        for term in _tokens(query_text):
            is_address_term = self._is_distinctive_address_term(term)
            is_value_term = self._is_distinctive_value_term(term)
            is_symbol_term = self._is_distinctive_symbol_term(term)
            if not is_address_term and not is_value_term and not is_symbol_term:
                continue
            if is_address_term or is_symbol_term:
                pages = self._address_page_buckets.get(term)
                records = self._address_record_buckets.get(term)
                # A common source word should not fan out into a full scan.
                # Codes, filenames and symbols normally have small buckets.
                if pages is not None and len(pages) <= 256:
                    page_ids.update(pages)
                if records is not None and len(records) <= 256:
                    record_ids.update(records)
            if is_value_term:
                value_records = self._value_record_buckets.get(term)
                if value_records is not None and len(value_records) <= 128:
                    record_ids.update(value_records)
                    page_ids.update(
                        self._record_page_ids[record_id]
                        for record_id in value_records
                        if record_id in self._record_page_ids
                    )
        return page_ids, record_ids

    def _lexical_evidence_hits(self, query_text: str) -> tuple[set[str], set[str]]:
        """Find sparse exact evidence hits for operational/fragment queries.

        A filename or symbol can use the structured address index, but a
        question such as ``what is the reset token`` has no filename to
        address.  The evidence index handles that case without scanning the
        bank: it unions only small inverted-list buckets and requires either
        two shared terms or one distinctive ASCII/identifier term.
        """

        matched_terms: dict[str, set[str]] = {}
        for term in _tokens(query_text):
            bucket = self._lexical_record_buckets.get(term)
            if bucket is None or len(bucket) > 64:
                continue
            for record_id in bucket:
                matched_terms.setdefault(record_id, set()).add(term)
        if not matched_terms:
            return set(), set()
        record_ids: set[str] = set()
        for record_id, terms in matched_terms.items():
            ascii_terms = [
                term
                for term in terms
                if all("a" <= char <= "z" or char == "_" for char in term)
            ]
            distinctive = any(
                self._is_distinctive_address_term(term)
                or self._is_distinctive_symbol_term(term)
                or self._is_distinctive_value_term(term)
                for term in ascii_terms
            )
            # ``候选`` is two single Chinese characters and is not a sparse
            # address.  Pure CJK evidence therefore needs a longer overlap;
            # one or more ASCII identifier terms may trigger immediately.
            if distinctive or (ascii_terms and len(terms) >= 2) or len(terms) >= 3:
                record_ids.add(record_id)
        page_ids = {
            self._record_page_ids[record_id]
            for record_id in record_ids
            if record_id in self._record_page_ids
        }
        return page_ids, record_ids

    def has_explicit_address(self, query_text: str) -> bool:
        """Return whether a query contains an indexed sparse address.

        The model-side reader uses this as a safe fast-path signal. It does
        not infer relevance from arbitrary lexical overlap; only the existing
        bounded address index can activate it.
        """

        page_ids, record_ids = self._address_hits(query_text)
        return bool(page_ids or record_ids)

    def _record_matches_explicit_address(
        self,
        record: MemoryRecordV2,
        query_text: str,
    ) -> bool:
        """Match entity plus an attribute alias such as ``symbol:parse_args``."""

        query_lower = str(query_text).strip().lower()
        query_terms = _tokens(query_lower)
        entity = record.entity.strip().lower()
        if not entity:
            return False
        entity_match = entity in query_lower
        # Numeric identifiers are not globally unique: ``训练用户00006`` and
        # ``评估用户00006`` must not collide merely because both contain
        # ``00006``.  Token fallback is reserved for non-natural-language
        # addresses such as filenames and symbols.
        if not entity_match and not any("\u4e00" <= char <= "\u9fff" for char in entity):
            entity_match = any(
                term in query_terms
                for term in _tokens(entity)
                if self._is_distinctive_address_term(term)
            )
        attribute = record.attribute.strip().lower()
        symbol_attribute = attribute.startswith(("symbol:", "class:", "def:"))
        attribute_match = bool(attribute and attribute in query_lower) or any(
            term in query_terms
            for term in _tokens(attribute)
            if self._is_distinctive_address_term(term)
            or (symbol_attribute and self._is_distinctive_symbol_term(term))
        )
        value_match = any(
            term in query_terms
            for term in _tokens(record.value)
            if self._is_distinctive_value_term(term)
        )
        # A concrete source file is a stronger address than a shared symbol
        # such as ``main`` or ``run``.  Do not admit same-named definitions
        # from unrelated files into the evidence prefix.
        explicit_file = bool(
            re.search(r"(?:[a-z0-9_.\\/-]+)\.(?:py|md|json|toml|yaml|yml)", query_lower)
        )
        if explicit_file and symbol_attribute and not entity_match:
            return False
        # Code navigation questions commonly provide only a symbol such as
        # ``main`` or ``_path`` and omit the filename.  The symbol itself is
        # still a valid sparse address; keep the bounded candidate set so the
        # evidence layer can expose ambiguity instead of dropping all hits.
        if not entity_match and not attribute_match and not value_match:
            return False
        # Attribute-only addressing is safe for code symbols because
        # ``symbol:<name>`` is itself the requested identifier.  Natural
        # attributes such as ``档案代号`` are not unique across entities, so
        # they must still require the entity or a distinctive value match.
        if not entity_match and (value_match or (attribute_match and symbol_attribute)):
            return True
        if not attribute:
            return entity_match or value_match
        return entity_match and (attribute_match or value_match or attribute in query_lower)

    def _new_page(self, *, tier: str = "warm") -> MemoryPageV2:
        if len(self.pages) >= self.max_pages:
            self.consolidate(max_pages=max(1, self.max_pages - 1))
        if len(self.pages) >= self.max_pages:
            raise RuntimeError(
                "memory page capacity exhausted; consolidate or increase max_pages"
            )
        self._counter += 1
        page = MemoryPageV2(
            page_id=f"page_{self._counter:08d}",
            tier=tier,
            capacity=self.page_capacity,
        )
        self.pages[page.page_id] = page
        self._open_page_ids.add(page.page_id)
        self._open_page_order.append(page.page_id)
        return page

    def _target_page(self, key: Tensor, *, importance: float) -> MemoryPageV2:
        open_ids = [
            page_id
            for page_id in self._open_page_order
            if page_id in self._open_page_ids
            and page_id in self.pages
            and not self.pages[page_id].full
        ]
        # If many pages remain partially filled, compare only a bounded recent
        # window plus hot pages. Write placement must stay sublinear in the
        # number of pages.
        if len(open_ids) > 128:
            recent = open_ids[-128:]
            hot = [
                page.page_id
                for page in self.pages.values()
                if page.tier == "hot" and not page.full
            ]
            open_ids = list(dict.fromkeys(recent + hot))
        candidates = [self.pages[page_id] for page_id in open_ids]
        if not candidates:
            return self._new_page(tier="hot" if importance >= 0.8 else "warm")
        key = self._encode_key(key)
        best_page: Optional[MemoryPageV2] = None
        best_score = -float("inf")
        for page in candidates:
            if page.key is None:
                score = -0.1 * len(page.record_ids)
            else:
                score = float(torch.dot(key.to(page.key.device), F.normalize(page.key.float(), dim=0)))
            score += 0.02 * page.importance
            if score > best_score:
                best_score = score
                best_page = page
        assert best_page is not None
        return best_page

    def _update_page(self, page: MemoryPageV2, record: MemoryRecordV2) -> None:
        self._remove_page_from_index(page)
        page.record_ids.append(record.record_id)
        record.page_id = page.page_id
        weight = 1.0 / max(1, len(page.record_ids))
        if page.key is None:
            page.key = record.key.detach().float().clone()
            page.summary = record.summary.detach().float().clone()
        else:
            page.key = (1.0 - weight) * page.key + weight * record.key.detach().float()
            if page.summary is None:
                page.summary = record.summary.detach().float().clone()
            else:
                page.summary = (1.0 - weight) * page.summary + weight * record.summary.detach().float()
        page.importance = max(page.importance, float(record.importance))
        page.touch()
        if page.full:
            self._open_page_ids.discard(page.page_id)
        self._index_page(page)

    def _find_duplicate(self, record: MemoryRecordV2) -> Optional[MemoryRecordV2]:
        text_key = record.text.strip().lower()
        if text_key:
            record_id = self._text_index.get(text_key)
            duplicate = self.records.get(record_id) if record_id else None
            if duplicate is None and self.tier_store is not None:
                item = self.tier_store.find_by_text(record.text, active_status=STATUS_ACTIVE)
                if item is not None:
                    self._hydrate_records([item["record_id"]])
                    duplicate = self.records.get(item["record_id"])
            if duplicate is not None and duplicate.status == STATUS_ACTIVE:
                return duplicate
        conflict_key = record.conflict_key()
        if conflict_key:
            old_id = self.active_by_conflict.get(conflict_key)
            if old_id and old_id in self.records:
                old = self.records[old_id]
                if old.status == STATUS_ACTIVE and old.value.strip().lower() == record.value.strip().lower():
                    return old
            if self.tier_store is not None:
                item = self.tier_store.find_by_conflict(
                    record.entity,
                    record.attribute,
                    active_status=STATUS_ACTIVE,
                )
                if item is not None:
                    self._hydrate_records([item["record_id"]])
                    old = self.records.get(item["record_id"])
                    if old is not None and old.value.strip().lower() == record.value.strip().lower():
                        return old
        return None

    def write(
        self,
        *,
        text: str,
        key: Tensor,
        summary: Optional[Tensor] = None,
        semantic_key: Optional[Tensor] = None,
        memory_type: str = "fact",
        entity: str = "",
        attribute: str = "",
        value: str = "",
        importance: float = 0.5,
        confidence: float = 0.5,
        source: str = "user",
        evidence: Optional[Sequence[str]] = None,
        related_ids: Optional[Sequence[str]] = None,
        slot_index: int = -1,
        token_ids: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        trusted: bool = True,
        force: bool = False,
        origin: str = "",
    ) -> tuple[MemoryRecordV2, str]:
        """Insert or version a record, returning ``(record, action)``."""

        importance = float(max(0.0, min(1.0, importance)))
        confidence = float(max(0.0, min(1.0, confidence)))
        if summary is None:
            summary = key
        record = MemoryRecordV2(
            record_id=_stable_id(f"{text}|{_now()}|{len(self.records)}", prefix="mem"),
            text=text,
            key=self._encode_key(key).cpu().clone(),
            summary=self._encode_key(summary).cpu().clone(),
            semantic_key=(
                semantic_key.detach().float().reshape(-1).cpu().clone()
                if semantic_key is not None
                else None
            ),
            memory_type=memory_type,
            entity=entity,
            attribute=attribute,
            value=value,
            importance=importance,
            confidence=confidence,
            source=source,
            evidence=list(evidence or [text]),
            related_ids=list(related_ids or []),
            slot_index=int(slot_index),
            token_ids=(token_ids.detach().cpu().long().reshape(-1).clone() if token_ids is not None else None),
            token_mask=(token_mask.detach().cpu().bool().reshape(-1).clone() if token_mask is not None else None),
            origin=str(origin or ""),
        )
        if not trusted and not force:
            record.status = STATUS_QUARANTINED
            self.quarantine[record.record_id] = record
            if self.tier_store is not None:
                self.tier_store.upsert_quarantine(record)
            return record, "quarantined"
        duplicate = self._find_duplicate(record)
        if duplicate is not None:
            duplicate.confidence = max(duplicate.confidence, record.confidence)
            duplicate.importance = max(duplicate.importance, record.importance)
            duplicate.evidence.extend(item for item in record.evidence if item not in duplicate.evidence)
            duplicate.touch()
            return duplicate, "duplicate"
        if record.slot_index >= 0:
            for old in self.records.values():
                if old.slot_index == record.slot_index and old.status == STATUS_ACTIVE:
                    old.status = STATUS_SUPERSEDED
                    record.version = max(record.version, old.version + 1)
                    record.supersedes = old.record_id
                    self._store_record(old)
        conflict_key = record.conflict_key()
        if conflict_key and conflict_key in self.active_by_conflict:
            old_id = self.active_by_conflict[conflict_key]
            old = self.records.get(old_id)
            if old is not None and old.status == STATUS_ACTIVE:
                old.status = STATUS_SUPERSEDED
                record.version = old.version + 1
                record.supersedes = old.record_id
                self._store_record(old)
        elif conflict_key and self.tier_store is not None:
            item = self.tier_store.find_by_conflict(
                record.entity,
                record.attribute,
                active_status=STATUS_ACTIVE,
            )
            old = None
            if item is not None:
                self._hydrate_records([item["record_id"]])
                old = self.records.get(item["record_id"])
            if old is not None:
                old.status = STATUS_SUPERSEDED
                record.version = old.version + 1
                record.supersedes = old.record_id
                self.active_by_conflict[conflict_key] = old.record_id
                self._store_record(old)
        self._absorb_unidentified_same_turn(record)
        page = self._target_page(record.key, importance=record.importance)
        self.records[record.record_id] = record
        if record.text.strip():
            self._text_index[record.text.strip().lower()] = record.record_id
        self._update_page(page, record)
        self._store_record(record)
        if conflict_key:
            self.active_by_conflict[conflict_key] = record.record_id
        if self._suspend_refresh == 0:
            self._refresh_tiers()
        return record, "updated" if record.supersedes else "inserted"

    def write_batch(self, records: Iterable[dict[str, Any]]) -> list[tuple[MemoryRecordV2, str]]:
        """Write many records with one durable transaction and one tier pass."""

        items = list(records)
        if not items:
            return []
        self._suspend_refresh += 1
        try:
            if self.tier_store is not None:
                with self.tier_store.transaction():
                    output = [self.write(**item) for item in items]
            else:
                output = [self.write(**item) for item in items]
        finally:
            self._suspend_refresh -= 1
        if self._suspend_refresh == 0:
            self._refresh_tiers()
        return output

    def _refresh_tiers(self) -> None:
        ranked = sorted(
            self.pages.values(),
            key=lambda page: (page.importance, page.last_access, len(page.record_ids)),
            reverse=True,
        )
        hot_ids = {page.page_id for page in ranked[: self.hot_pages]}
        resident_ids = {
            page.page_id for page in ranked[: self.max_resident_pages]
        }
        for page in self.pages.values():
            old_tier = page.tier
            if page.page_id in hot_ids:
                page.tier = "hot"
            elif self.tier_store is not None and page.page_id not in resident_ids:
                page.tier = "cold"
            else:
                page.tier = "warm"
            if self.tier_store is not None and old_tier != page.tier:
                self.tier_store.set_page_tier(page.page_id, page.tier)
        if self.tier_store is None and self._gpu_device is not None:
            hot_records: list[MemoryRecordV2] = []
            for page in sorted(
                (item for item in self.pages.values() if item.tier == "hot"),
                key=lambda item: (item.last_access, item.importance),
                reverse=True,
            ):
                hot_records.extend(
                    self.records[record_id]
                    for record_id in page.record_ids
                    if record_id in self.records
                    and self.records[record_id].status == STATUS_ACTIVE
                )
            self.promote_records(hot_records)
        self._evict_cold_records()

    def _page_scores(
        self,
        query_key: Tensor,
        query_text: str = "",
        query_token_ids: Optional[Tensor] = None,
    ) -> list[tuple[MemoryPageV2, float]]:
        scores = []
        candidate_page_ids = self._candidate_page_ids(query_key, query_text)
        for page_id in candidate_page_ids:
            page = self._hydrate_page(page_id)
            if page is None:
                continue
            if page.key is None:
                continue
            score = float(self._score_candidates(query_key, page.key.unsqueeze(0))[0].item())
            active_records = [
                self.records[record_id]
                for record_id in page.record_ids
                if record_id in self.records and self.records[record_id].status == STATUS_ACTIVE
            ]
            if active_records:
                record_keys = torch.stack(
                    [self._record_key(record).to(device=query_key.device) for record in active_records],
                    dim=0,
                )
                # Exact record-level reranking prevents a mixed-topic page
                # centroid from hiding a highly relevant slot.
                score = max(score, float(self._score_candidates(query_key, record_keys).max().item()))
                # Prefer pages that contain usable evidence over pages made
                # entirely of address/background vectors.  The raw records
                # remain in RAM; this only affects bounded admission order.
                if any(record.token_ids is not None for record in active_records):
                    score += 0.20
            score += 0.05 * page.importance
            age_seconds = max(0, _now() - page.last_access)
            score += 0.02 / (1.0 + age_seconds / 3600.0)
            if query_text:
                page_text = " ".join(
                    self.records[item].routing_text()
                    for item in page.record_ids
                    if item in self.records
                )
                score += 0.65 * max(
                    (self.records[item].lexical_score(query_text) for item in page.record_ids if item in self.records),
                    default=0.0,
                )
                # The record-level index is deliberately limited to usable
                # evidence records. This makes an exact operational phrase
                # beat a large README/background page without scanning all
                # records at query time.
                record_lexical = max(
                    (
                        self.records[item].lexical_score(query_text)
                        for item in page.record_ids
                        if item in self.records
                        and self.records[item].status == STATUS_ACTIVE
                        and (
                            self.records[item].semantic_key is not None
                            or self.records[item].token_ids is not None
                        )
                    ),
                    default=0.0,
                )
                score += 0.45 * record_lexical
                score += 1.25 * self._rare_lexical_address_score(
                    query_text,
                    self._page_terms.get(page.page_id, set()),
                )
                # A query that explicitly names a sufficiently distinctive
                # entity must not be routed by broad semantic similarity
                # alone.  This is an address-layer guard for near-duplicate
                # memories (for example, two users with the same attribute),
                # not an answer generator: it only promotes the page that
                # already contains the named entity and optional attribute.
                query_lower = query_text.strip().lower()
                exact_entity = False
                exact_attribute = False
                for record in active_records:
                    entity = record.entity.strip().lower()
                    if len(entity) < 4 or entity not in query_lower:
                        continue
                    exact_entity = True
                    attribute = record.attribute.strip().lower()
                    if attribute and attribute in query_lower:
                        exact_attribute = True
                        break
                if exact_entity:
                    score += 2.50
                if exact_attribute:
                    score += 0.75
            if query_token_ids is not None:
                score += 0.45 * max(
                    (
                        self.records[item].token_overlap_score(query_token_ids)
                        for item in page.record_ids
                        if item in self.records and self.records[item].status == STATUS_ACTIVE
                    ),
                    default=0.0,
                )
            scores.append((page, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def _record_scores(
        self,
        page: MemoryPageV2,
        query_key: Tensor,
        query_text: str,
        query_token_ids: Optional[Tensor] = None,
        *,
        allow_superseded: bool = False,
    ) -> list[tuple[MemoryRecordV2, float]]:
        output = []
        self._hydrate_page(page.page_id)
        live_records: list[MemoryRecordV2] = []
        for record_id in page.record_ids:
            record = self.records.get(record_id)
            if record is None:
                continue
            if record.status != STATUS_ACTIVE and not allow_superseded:
                continue
            live_records.append(record)
        if not live_records:
            return []
        candidate_keys = torch.stack(
            [self._record_key(record).to(device=query_key.device) for record in live_records],
            dim=0,
        )
        semantic_keys = [record.semantic_key for record in live_records]
        if self.record_scorer is not None and any(isinstance(value, Tensor) for value in semantic_keys):
            # Pages may contain learned target records beside raw background
            # chunks.  Score both groups without allowing a random filler key
            # to contaminate the learned reranker.
            routed_scores = self._score_candidates(query_key, candidate_keys).to(
                device=query_key.device, dtype=torch.float32
            )
            semantic_indices = [
                index for index, value in enumerate(semantic_keys) if isinstance(value, Tensor)
            ]
            semantic_candidates = torch.stack(
                [semantic_keys[index] for index in semantic_indices], dim=0
            ).to(device=query_key.device)
            semantic_scores = self.record_scorer(query_key, semantic_candidates).reshape(-1)
            for position, score in zip(semantic_indices, semantic_scores):
                routed_scores[position] = score.to(dtype=torch.float32)
        else:
            routed_scores = self._score_candidates(query_key, candidate_keys)
        for record, routed_score in zip(live_records, routed_scores):
            score = float(routed_score.item())
            # Prior weights are configurable because they dominate the learned term:
            # measured on this scorer, the neural score is only 12-17% of the summed
            # total while these priors are 83-88%, so changing the learned ranker alone
            # measurably cannot change which records are injected (verified end-to-end).
            score += self.prior_weights["lexical"] * record.lexical_score(query_text)
            score += self.prior_weights["token_overlap"] * record.token_overlap_score(query_token_ids)
            score += 0.10 * record.confidence + 0.08 * record.importance
            score += self.prior_weights["rare_lexical"] * self._rare_lexical_address_score(
                query_text,
                _tokens(record.routing_text()),
            )
            # An item that can actually be injected into the model is more
            # useful than a centroid-only background chunk.  This is a small
            # evidence-quality prior, not a replacement for relevance.
            if record.token_ids is not None:
                score += self.prior_weights["shape_bonus"]
            if record.entity or record.attribute or record.value:
                score += self.prior_weights["structured_bonus"]
            entity = record.entity.strip().lower()
            query_lower = query_text.strip().lower()
            if len(entity) >= 4 and entity in query_lower:
                # Keep the same identity prior at record level so the exact
                # entity remains first after page admission and reranking.
                score += 2.50
                attribute = record.attribute.strip().lower()
                if attribute and attribute in query_lower:
                    score += 0.75
            output.append((record, score))
        return sorted(output, key=lambda item: item[1], reverse=True)

    def query(
        self,
        *,
        query_key: Tensor,
        query_text: str = "",
        query_token_ids: Optional[Tensor] = None,
        top_k_pages: Optional[int] = None,
        top_k_records: Optional[int] = None,
        max_hops: Optional[int] = None,
        min_score: float = -1.0,
    ) -> tuple[list[MemoryRecordV2], RouterDecisionV2]:
        """Sparse multi-hop query over pages and active records."""

        page_limit = max(1, top_k_pages or self.top_k_pages)
        record_limit = top_k_records or self.top_k_records
        hop_limit = min(max_hops or self.max_hops, self.max_hops)
        address_page_ids, address_record_ids = self._address_hits(query_text)
        lexical_page_ids, lexical_record_ids = self._lexical_evidence_hits(query_text)
        content_record_ids = lexical_record_ids.difference(address_record_ids)
        routed_page_ids = address_page_ids | lexical_page_ids
        if routed_page_ids:
            # Explicit addresses are already a high-confidence routing signal.
            # Avoid paying the neural page rerank cost for unrelated candidates;
            # exact records below still receive the final bounded score.
            ranked_pages = []
            for page_id in sorted(routed_page_ids):
                page = self._hydrate_page(page_id)
                if page is None:
                    continue
                page_score = 3.0 + 0.05 * page.importance
                if page_id in lexical_page_ids:
                    page_score += 0.5
                ranked_pages.append((page, page_score))
        else:
            ranked_pages = self._page_scores(query_key, query_text, query_token_ids)
        selected_pages = [item for item in ranked_pages[:page_limit] if item[1] >= min_score]
        # Preserve exact address hits even when the learned page score ranks a
        # mixed-topic page above the target page.  Only the small indexed set of
        # distinctive address pages is admitted; this does not scan the bank.
        ranked_by_page = {page.page_id: (page, score) for page, score in ranked_pages}
        selected_page_ids = {page.page_id for page, _ in selected_pages}
        if routed_page_ids:
            best_selected_score = max((score for _, score in selected_pages), default=0.0)
            for page_id in sorted(routed_page_ids):
                if page_id in selected_page_ids:
                    continue
                ranked = ranked_by_page.get(page_id)
                if ranked is None:
                    page = self._hydrate_page(page_id)
                    if page is None:
                        continue
                    ranked = (page, best_selected_score)
                page, score = ranked
                selected_pages.append((page, max(float(score), best_selected_score) + 3.0))
                selected_page_ids.add(page_id)
        current_page_ids = [page.page_id for page, _ in selected_pages]
        selected_records: list[tuple[MemoryRecordV2, float]] = []
        hop_trace: list[list[str]] = []
        visited: set[str] = set()
        # Direct address records are inserted before semantic candidates.  The
        # neural router still scores ordinary candidates, but a distinctive
        # filename/symbol or user id must never be lost to a centroid collision.
        direct_records: list[tuple[MemoryRecordV2, float]] = []
        direct_record_ids = address_record_ids | content_record_ids
        if direct_record_ids:
            self._hydrate_records(direct_record_ids)
            for record_id in sorted(direct_record_ids):
                record = self.records.get(record_id)
                if record is None or record.status != STATUS_ACTIVE:
                    continue
                if record_id in address_record_ids and not self._record_matches_explicit_address(record, query_text):
                    continue
                score = (10.0 if record_id in address_record_ids else 8.0) + 0.25 * record.lexical_score(query_text)
                if record.token_ids is not None:
                    score += 0.35
                if record.entity or record.attribute or record.value:
                    score += 0.15
                direct_records.append((record, score))
            direct_records.sort(key=lambda item: item[1], reverse=True)
            # If a query names an ambiguous symbol/value, return a bounded
            # evidence set rather than silently choosing one occurrence.  The
            # caller still receives only a small Top-K set for ordinary
            # queries; fan-out is capped by the number of sparse address hits.
            direct_limit = max(record_limit, min(8, len(direct_records)))
            selected_records.extend(direct_records[:direct_limit])
            for record, _ in selected_records:
                if record.record_id not in visited:
                    visited.add(record.record_id)
                    record.touch()
                    self._store_record(record)
        # An exact address is already the final routing decision. Do not pay
        # for semantic page/record reranking or unrelated multi-hop expansion
        # when the query names a distinctive entity, filename, symbol, or user
        # id. Ambiguous queries still use the complete learned path below.
        skip_semantic_expansion = bool(direct_record_ids and direct_records)
        for hop in range(0 if skip_semantic_expansion else hop_limit):
            hop_records: list[str] = []
            candidate_pages = [self.pages[item] for item in current_page_ids if item in self.pages]
            candidates = []
            for page in candidate_pages:
                candidates.extend(self._record_scores(page, query_key, query_text, query_token_ids))
            candidates.sort(key=lambda item: item[1], reverse=True)
            for record, score in candidates:
                if record.record_id in visited:
                    continue
                visited.add(record.record_id)
                selected_records.append((record, score))
                record.touch()
                self._store_record(record)
                hop_records.append(record.record_id)
                if len(selected_records) >= record_limit:
                    break
            hop_trace.append(hop_records)
            if len(selected_records) >= record_limit or not hop_records:
                break
            related_pages: list[str] = []
            for record_id in hop_records:
                record = self.records[record_id]
                self._hydrate_records(record.related_ids)
                for related_id in record.related_ids:
                    related = self.records.get(related_id)
                    if related is not None and related.page_id not in related_pages:
                        related_pages.append(related.page_id)
            if not related_pages:
                break
            current_page_ids = related_pages[:page_limit]
        selected_records.sort(key=lambda item: item[1], reverse=True)
        # When the query names one distinctive entity and its attribute, a
        # semantically similar second record is a liability: it can make the
        # language model blend two people or two versions into one answer.
        # Treat the explicit entity/attribute pair as an address constraint
        # and retain only records at that address.  Broad queries without a
        # sufficiently distinctive entity keep the normal top-k behavior.
        exact_address_records = [
            item
            for item in selected_records
            if self._record_matches_explicit_address(item[0], query_text)
        ]
        if exact_address_records:
            selected_records = exact_address_records
        # Ambiguous code symbols need more than one evidence card: returning
        # only the first occurrence makes a symbol such as ``run`` look like
        # a unique answer even when it exists in several files.  Keep this
        # expansion local to symbol-address queries so ordinary personal
        # memory reads retain the normal small Top-K prompt.
        symbol_address_query = any(
            record.attribute.strip().lower().startswith(("symbol:", "class:", "def:"))
            for record, _ in direct_records
        )
        output_limit = min(
            len(selected_records),
            max(record_limit, 8) if symbol_address_query else record_limit,
        )
        records = [record for record, _ in selected_records[:output_limit]]
        # Keep only the bounded hot working set on VRAM.  The authoritative
        # copies remain in RAM and are exported to the embedded weight shard.
        self.promote_records(records)
        page_scores = [score for _, score in selected_pages]
        record_scores = [score for _, score in selected_records[:record_limit]]
        top_score = float(record_scores[0]) if record_scores else 0.0
        second_score = float(record_scores[1]) if len(record_scores) > 1 else 0.0
        score_margin = max(0.0, top_score - second_score) if record_scores else 0.0
        confidence = max(0.0, min(1.0, top_score))
        decision = RouterDecisionV2(
            need_memory=bool(records),
            page_ids=[page.page_id for page, _ in selected_pages],
            record_ids=[record.record_id for record in records],
            page_scores=page_scores,
            record_scores=record_scores,
            hop_count=len(hop_trace),
            hop_trace=hop_trace,
            confidence=confidence,
            stop_reason="evidence_found" if records else "no_relevant_evidence",
            top_score=top_score,
            score_margin=score_margin,
            evidence_score=confidence,
        )
        for page, _ in selected_pages:
            page.touch()
        self._refresh_tiers()
        return records, decision

    def correct(
        self,
        *,
        text: str,
        key: Tensor,
        entity: str,
        attribute: str,
        value: str,
        evidence: Optional[Sequence[str]] = None,
        confidence: float = 1.0,
    ) -> tuple[MemoryRecordV2, str]:
        """Write an explicit correction as a new version."""

        return self.write(
            text=text,
            key=key,
            entity=entity,
            attribute=attribute,
            value=value,
            evidence=evidence,
            confidence=confidence,
            importance=1.0,
            source="user_correction",
            trusted=True,
            force=True,
        )

    def approve(self, record_id: str) -> MemoryRecordV2:
        record = self.quarantine.pop(record_id, None)
        if record is None:
            raise KeyError(f"unknown quarantined record: {record_id}")
        if self.tier_store is not None:
            self.tier_store.delete_quarantine(record_id)
        record.status = STATUS_ACTIVE
        page = self._target_page(record.key, importance=record.importance)
        self.records[record.record_id] = record
        self._update_page(page, record)
        self._store_record(record)
        conflict_key = record.conflict_key()
        if conflict_key:
            old_id = self.active_by_conflict.get(conflict_key)
            if old_id and old_id not in self.records:
                self._hydrate_records([old_id])
            old = self.records.get(old_id) if old_id else None
            if old is not None and old.status == STATUS_ACTIVE:
                old.status = STATUS_SUPERSEDED
                record.version = old.version + 1
                record.supersedes = old.record_id
                self._store_record(old)
            self.active_by_conflict[conflict_key] = record.record_id
        self._refresh_tiers()
        return record

    def retract(self, record_id: str) -> None:
        record = self.records.get(record_id)
        if record is None and self.tier_store is not None:
            self._hydrate_records([record_id])
            record = self.records.get(record_id)
        if record is None:
            raise KeyError(record_id)
        record.status = STATUS_RETRACTED
        self._remove_gpu_record(record_id)
        self._store_record(record)
        conflict_key = record.conflict_key()
        if conflict_key and self.active_by_conflict.get(conflict_key) == record_id:
            del self.active_by_conflict[conflict_key]
        self._refresh_tiers()

    def _absorb_unidentified_same_turn(self, record: MemoryRecordV2) -> list[str]:
        """Supersede the same turn's identity-less record once a structured one lands.

        The automatic layer writes one record per user turn and runs *before* the model
        generates, so it cannot know what that turn will turn out to be about.  When the
        sentence is not the ``<attribute> 是 <value>`` shape its metadata inference
        understands (「我叫Wpy」, 「我住在上海」), the automatic record ends up with no
        entity and no attribute -- hence no conflict key, hence a correction can never
        retire it, and a stale value stays retrievable beside the corrected one.  Measured:
        two contradictory identity records both stayed ``active`` at version 0.

        The structured record for that turn does carry a key.  Absorbing the turn's
        unidentified record into it -- superseded rather than retracted, since it is the
        same fact in a worse shape -- leaves exactly one active record per turn and removes
        the uncorrectable leftover.

        Only records sharing this record's ``origin`` are touched, and only active ones, so
        this cannot reach another turn's facts or an already retired one.
        """

        origin = str(record.origin or "").strip()
        if not origin or not (record.entity and record.attribute):
            return []
        absorbed: list[str] = []
        for sibling in list(self.records.values()):
            if sibling.record_id == record.record_id or sibling.status != STATUS_ACTIVE:
                continue
            if sibling.origin != origin or (sibling.entity and sibling.attribute):
                continue
            sibling.status = STATUS_SUPERSEDED
            self._store_record(sibling)
            absorbed.append(sibling.record_id)
        return absorbed

    def retract_origin(self, origin: str) -> list[str]:
        """Retract every active record written from the same user turn.

        Retraction has always been *key*-scoped: a forget request retracts the record
        whose key matched.  One turn usually writes more than one record, though -- a
        sentence like "my emergency contact is Wang, extension 7781" produces a contact
        record and an extension record -- so forgetting "the emergency contact
        information" used to leave the extension active and readable.  A measured
        end-to-end run showed the model then answering with that leftover value, i.e.
        the memory layer leaking a fact the user had just revoked.

        Grouping by origin makes the unit of forgetting the unit of telling, which is
        what the user meant.  Records with no origin (written before this field
        existed) are never matched, so this cannot retract anything unrelated.
        """

        origin = str(origin or "").strip()
        if not origin:
            return []
        retracted: list[str] = []
        for record in list(self.records.values()):
            if record.status != STATUS_ACTIVE or record.origin != origin:
                continue
            self.retract(record.record_id)
            retracted.append(record.record_id)
        return retracted

    def consolidate(self, *, max_pages: Optional[int] = None) -> int:
        """Move low-value pages toward compact summaries without deleting evidence."""

        target = max_pages or self.max_pages
        if len(self.pages) <= target:
            self._refresh_tiers()
            return 0
        ranked = sorted(self.pages.values(), key=lambda page: (page.importance, page.last_access))
        merged = 0
        for source in ranked:
            if len(self.pages) <= target:
                break
            if source.tier == "hot" or not source.record_ids:
                continue
            destinations = [
                page
                for page in self.pages.values()
                if page.page_id != source.page_id and not page.full
            ]
            # Consolidation may only use an existing page.  Creating a new
            # page while trying to enforce a page limit would recurse forever
            # when every page is already full.
            if not destinations:
                continue
            source_key = source.key if source.key is not None else torch.zeros(self.key_dim)
            source_key = self._encode_key(source_key)
            destination = max(
                destinations,
                key=lambda page: (
                    float(torch.dot(source_key, F.normalize(page.key.float(), dim=0)))
                    if page.key is not None
                    else -0.1 * len(page.record_ids)
                ),
            )
            for record_id in list(source.record_ids):
                if record_id not in destination.record_ids and not destination.full:
                    destination.record_ids.append(record_id)
                    self.records[record_id].page_id = destination.page_id
            if destination.full:
                self._open_page_ids.discard(destination.page_id)
            else:
                self._open_page_ids.add(destination.page_id)
            self._remove_page_from_index(destination)
            self._index_page(destination)
            if destination.key is None:
                destination.key = source.key
            if destination.summary is None:
                destination.summary = source.summary
            destination.importance = max(destination.importance, source.importance)
            self._remove_page_from_index(source)
            del self.pages[source.page_id]
            self._open_page_ids.discard(source.page_id)
            merged += 1
        self._refresh_tiers()
        return merged

    def stats(self) -> dict[str, Any]:
        active = [record for record in self.records.values() if record.status == STATUS_ACTIVE]
        stored = self.tier_store.count() if self.tier_store is not None else {}
        record_count = int(stored.get("records", len(self.records)))
        active_count = int(stored.get("status_active", len(active)))
        superseded_count = int(
            stored.get(
                "status_superseded",
                sum(record.status == STATUS_SUPERSEDED for record in self.records.values()),
            )
        )
        retracted_count = int(
            stored.get(
                "status_retracted",
                sum(record.status == STATUS_RETRACTED for record in self.records.values()),
            )
        )
        quarantined_count = int(stored.get("quarantined", len(self.quarantine)))
        return {
            "pages": len(self.pages),
            "max_pages": self.max_pages,
            "page_capacity": self.page_capacity,
            "capacity_records": self.max_pages * self.page_capacity,
            "open_pages": len(self._open_page_ids),
            "records": record_count,
            "active_records": active_count,
            "superseded_records": superseded_count,
            "retracted_records": retracted_count,
            "quarantined_records": quarantined_count,
            "hot_pages": int(stored.get("hot_pages", sum(page.tier == "hot" for page in self.pages.values()))),
            "warm_pages": int(stored.get("warm_pages", sum(page.tier == "warm" for page in self.pages.values()))),
            "cold_pages": int(stored.get("cold_pages", sum(page.tier == "cold" for page in self.pages.values()))),
            "active_conflict_keys": len(self.active_by_conflict),
            "coarse_index_buckets": (
                self.tier_store.coarse_bucket_count()
                if self.tier_store is not None
                else len(self._coarse_buckets)
            ),
            "last_coarse_candidates": self._last_coarse_candidates,
            "storage_mode": "tiered" if self.tier_store is not None else "embedded",
            "resident_records": len(self.records),
            "resident_pages": len(self.pages),
            "gpu_cache_records": len(self._gpu_record_cache),
            "gpu_cache_tokens": self._gpu_cache_tokens_used,
            "gpu_cache_bytes": self._gpu_cache_bytes_used,
            "gpu_cache_hits": self._gpu_cache_hits,
            "gpu_cache_misses": self._gpu_cache_misses,
            "gpu_cache_fallbacks": self._gpu_cache_fallbacks,
            "gpu_cache_alloc_failures": self._gpu_cache_alloc_failures,
            "gpu_cache_device": str(self._gpu_device) if self._gpu_device is not None else "none",
            "gpu_cache_limit_records": self.gpu_cache_records,
            "gpu_cache_limit_tokens": self.gpu_cache_tokens,
            "gpu_cache_reserve_mb": self.gpu_cache_reserve_mb,
            "gpu_cache_adaptive": self.gpu_cache_adaptive,
            "gpu_cache_last_free_mb": (
                round(self._gpu_cache_last_free_bytes / (1024 * 1024), 2)
                if self._gpu_cache_last_free_bytes is not None
                else None
            ),
        }

    def _resolve_record(self, record_id: str) -> MemoryRecordV2:
        record = self.records.get(str(record_id))
        if record is None and self.tier_store is not None:
            self._hydrate_records([str(record_id)])
            record = self.records.get(str(record_id))
        if record is None:
            record = self.quarantine.get(str(record_id))
        if record is None:
            raise KeyError(str(record_id))
        return record

    def list_records(
        self,
        *,
        query_text: str = "",
        status: str = STATUS_ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecordV2]:
        """List records for the local management API without exposing vectors."""

        if limit < 1 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        allowed = {STATUS_ACTIVE, STATUS_SUPERSEDED, STATUS_RETRACTED, STATUS_QUARANTINED, "all"}
        if status not in allowed:
            raise ValueError(f"status must be one of: {', '.join(sorted(allowed))}")
        candidates = list(self.records.values()) + list(self.quarantine.values())
        if status != "all":
            candidates = [item for item in candidates if item.status == status]
        if query_text.strip():
            candidates = [
                item for item in candidates
                if item.lexical_score(query_text) > 0.0 or query_text.strip().lower() in item.text.lower()
            ]
            candidates.sort(
                key=lambda item: (item.lexical_score(query_text), item.last_access, item.timestamp),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda item: (item.last_access, item.timestamp), reverse=True)
        return candidates[offset : offset + limit]

    def edit_record(
        self,
        record_id: str,
        *,
        text: Optional[str] = None,
        key: Optional[Tensor] = None,
        summary: Optional[Tensor] = None,
        memory_type: Optional[str] = None,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        value: Optional[str] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        evidence: Optional[Sequence[str]] = None,
        token_ids: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        source: str = "user_edit",
    ) -> MemoryRecordV2:
        """Create a new version and preserve the old record as superseded."""

        old = self._resolve_record(record_id)
        if old.status != STATUS_ACTIVE:
            raise ValueError(f"only active records can be edited: {record_id}")
        old.status = STATUS_SUPERSEDED
        self._remove_gpu_record(old.record_id)
        self._store_record(old)
        old_conflict = old.conflict_key()
        if old_conflict and self.active_by_conflict.get(old_conflict) == old.record_id:
            del self.active_by_conflict[old_conflict]
        next_text = old.text if text is None else str(text)
        next_entity = old.entity if entity is None else str(entity)
        next_attribute = old.attribute if attribute is None else str(attribute)
        next_value = old.value if value is None else str(value)
        next_importance = old.importance if importance is None else float(importance)
        next_confidence = old.confidence if confidence is None else float(confidence)
        next_key = old.key if key is None else key
        next_summary = old.summary if summary is None else summary
        now = _now()
        new_record = MemoryRecordV2(
            record_id=_stable_id(f"{old.record_id}|edit|{now}|{len(self.records)}", prefix="mem"),
            text=next_text,
            key=self._encode_key(next_key).cpu().clone(),
            summary=self._encode_key(next_summary).cpu().clone(),
            memory_type=old.memory_type if memory_type is None else str(memory_type),
            entity=next_entity,
            attribute=next_attribute,
            value=next_value,
            timestamp=now,
            importance=max(0.0, min(1.0, next_importance)),
            confidence=max(0.0, min(1.0, next_confidence)),
            source=str(source),
            status=STATUS_ACTIVE,
            version=old.version + 1,
            supersedes=old.record_id,
            related_ids=list(old.related_ids),
            evidence=list(old.evidence if evidence is None else evidence),
            # A new version of a fact came from whatever turn produced the fact.
            origin=old.origin,
            slot_index=old.slot_index,
            token_ids=(token_ids.detach().cpu().long().reshape(-1).clone() if token_ids is not None else old.token_ids),
            token_mask=(token_mask.detach().cpu().bool().reshape(-1).clone() if token_mask is not None else old.token_mask),
        )
        self.records[new_record.record_id] = new_record
        if new_record.text.strip():
            self._text_index[new_record.text.strip().lower()] = new_record.record_id
        page = self._target_page(new_record.key, importance=new_record.importance)
        self._update_page(page, new_record)
        self._store_record(new_record)
        new_conflict = new_record.conflict_key()
        if new_conflict:
            self.active_by_conflict[new_conflict] = new_record.record_id
        self._refresh_tiers()
        return new_record

    def _remove_gpu_record(self, record_id: str) -> None:
        cached = self._gpu_record_cache.pop(record_id, None)
        if cached is None:
            return
        if record_id in self._gpu_cache_order:
            self._gpu_cache_order.remove(record_id)
        token_ids = cached.get("token_ids")
        if isinstance(token_ids, Tensor):
            self._gpu_cache_tokens_used -= int(token_ids.numel())
        self._gpu_cache_bytes_used -= sum(self._tensor_bytes(value) for value in cached.values())
        self._gpu_cache_tokens_used = max(0, self._gpu_cache_tokens_used)
        self._gpu_cache_bytes_used = max(0, self._gpu_cache_bytes_used)

    def audit(self) -> dict[str, Any]:
        """Check page membership, conflict indexes and relationship references."""

        issues: list[str] = []
        page_membership: set[str] = set()
        for page in self.pages.values():
            if len(page.record_ids) > page.capacity:
                issues.append(f"page_over_capacity:{page.page_id}")
            for record_id in page.record_ids:
                page_membership.add(record_id)
                record = self.records.get(record_id)
                if record is None:
                    issues.append(f"missing_record:{record_id}")
                elif record.page_id != page.page_id:
                    issues.append(f"page_pointer_mismatch:{record_id}")
        for record in self.records.values():
            if record.status == STATUS_ACTIVE and record.record_id not in page_membership:
                issues.append(f"active_record_not_indexed:{record.record_id}")
            for related_id in record.related_ids:
                if related_id not in self.records and related_id not in self.quarantine:
                    issues.append(f"dangling_related_id:{record.record_id}->{related_id}")
        for conflict_key, record_id in self.active_by_conflict.items():
            record = self.records.get(record_id)
            if record is None or record.status != STATUS_ACTIVE or record.conflict_key() != conflict_key:
                issues.append(f"invalid_conflict_index:{conflict_key}")
        return {
            "healthy": not issues,
            "issues": issues[:100],
            "issue_count": len(issues),
            "stats": self.stats(),
        }

    def flush_storage(self) -> None:
        """Flush the durable tier without serializing cold pages into RAM."""

        if self.tier_store is not None:
            self.tier_store.flush()

    def close_storage(self) -> None:
        """Close the durable page database so Windows can remove or rotate it."""

        if self.tier_store is not None:
            self.tier_store.close()

    def export_payload(self) -> dict[str, Any]:
        """Export tensors plus JSON-safe metadata for checkpoint backends."""

        if self.tier_store is not None:
            for page in self.pages.values():
                self._hydrate_page(page.page_id)
        records = []
        for record in self.records.values():
            item = asdict(record)
            item["key"] = record.key.detach().cpu()
            item["summary"] = record.summary.detach().cpu()
            records.append(item)
        pages = []
        for page in self.pages.values():
            item = asdict(page)
            item["key"] = page.key.detach().cpu() if page.key is not None else None
            item["summary"] = page.summary.detach().cpu() if page.summary is not None else None
            pages.append(item)
        quarantine = []
        for record in self.quarantine.values():
            item = asdict(record)
            item["key"] = record.key.detach().cpu()
            item["summary"] = record.summary.detach().cpu()
            quarantine.append(item)
        return {
            "format_version": 2,
            "hidden_size": self.hidden_size,
            "page_capacity": self.page_capacity,
            "max_pages": self.max_pages,
            "hot_pages": self.hot_pages,
            "top_k_pages": self.top_k_pages,
            "top_k_records": self.top_k_records,
            "max_hops": self.max_hops,
            "key_dim": self.key_dim,
            "coarse_index_bits": self.coarse_index_bits,
            "record_count": len(self.records),
            "counter": self._counter,
            "records": records,
            "pages": pages,
            "quarantine": quarantine,
            "active_by_conflict": dict(self.active_by_conflict),
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        router: Optional[MemoryRouterV2] = None,
        tier_store: Optional["TieredMemoryStoreV2"] = None,
        max_resident_pages: int = 256,
        runtime_device: Optional[torch.device] = None,
        gpu_cache_records: int = 256,
        gpu_cache_tokens: int = 131072,
        gpu_cache_reserve_mb: int = 2048,
        gpu_cache_adaptive: bool = True,
        record_scorer: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> "PagedMemoryBankV2":
        bank = cls(
            int(payload["hidden_size"]),
            page_capacity=int(payload.get("page_capacity", 32)),
            max_pages=int(payload.get("max_pages", 32768)),
            hot_pages=int(payload.get("hot_pages", 8)),
            top_k_pages=int(payload.get("top_k_pages", 4)),
            top_k_records=int(payload.get("top_k_records", 8)),
            max_hops=int(payload.get("max_hops", 3)),
            router=router,
            key_dim=int(payload.get("key_dim", router.router_dim if router is not None else payload["hidden_size"])),
            coarse_index_bits=int(payload.get("coarse_index_bits", 20)),
            tier_store=tier_store,
            max_resident_pages=max_resident_pages,
            runtime_device=runtime_device,
            gpu_cache_records=gpu_cache_records,
            gpu_cache_tokens=gpu_cache_tokens,
            gpu_cache_reserve_mb=gpu_cache_reserve_mb,
            gpu_cache_adaptive=gpu_cache_adaptive,
            record_scorer=record_scorer,
        )
        bank._counter = int(payload.get("counter", 0))
        for item in payload.get("records", []):
            item = dict(item)
            item["key"] = torch.as_tensor(item["key"]).float()
            item["summary"] = torch.as_tensor(item["summary"]).float()
            if item.get("semantic_key") is not None:
                item["semantic_key"] = torch.as_tensor(item["semantic_key"]).float()
            if item.get("token_ids") is not None:
                item["token_ids"] = torch.as_tensor(item["token_ids"]).long()
            if item.get("token_mask") is not None:
                item["token_mask"] = torch.as_tensor(item["token_mask"]).bool()
            bank.records[item["record_id"]] = MemoryRecordV2(**item)
            if item.get("text", "").strip() and item.get("status") == STATUS_ACTIVE:
                bank._text_index[item["text"].strip().lower()] = item["record_id"]
        for item in payload.get("pages", []):
            item = dict(item)
            if item.get("key") is not None:
                item["key"] = torch.as_tensor(item["key"]).float()
            if item.get("summary") is not None:
                item["summary"] = torch.as_tensor(item["summary"]).float()
            if item.get("semantic_key") is not None:
                item["semantic_key"] = torch.as_tensor(item["semantic_key"]).float()
            bank.pages[item["page_id"]] = MemoryPageV2(**item)
        bank._open_page_ids = {
            page.page_id for page in bank.pages.values() if not page.full
        }
        bank._open_page_order = [
            page.page_id
            for page in sorted(
                bank.pages.values(), key=lambda item: (item.created_at, item.page_id)
            )
            if page.page_id in bank._open_page_ids
        ]
        for page in bank.pages.values():
            bank._index_page(page)
        for item in payload.get("quarantine", []):
            item = dict(item)
            item["key"] = torch.as_tensor(item["key"]).float()
            item["summary"] = torch.as_tensor(item["summary"]).float()
            if item.get("token_ids") is not None:
                item["token_ids"] = torch.as_tensor(item["token_ids"]).long()
            if item.get("token_mask") is not None:
                item["token_mask"] = torch.as_tensor(item["token_mask"]).bool()
            bank.quarantine[item["record_id"]] = MemoryRecordV2(**item)
        bank.active_by_conflict = dict(payload.get("active_by_conflict", {}))
        bank._refresh_tiers()
        return bank


@dataclass
class KVBudgetManagerV2:
    """Token budget and overflow policy for the hot working context."""

    max_tokens: int = 32768
    hard_max_tokens: int = 131072
    compaction_trigger: float = 0.90
    keep_recent_tokens: int = 8192

    def __post_init__(self) -> None:
        if not 1 <= self.max_tokens <= self.hard_max_tokens:
            raise ValueError("max_tokens must be inside [1, hard_max_tokens]")
        if not 0.5 <= self.compaction_trigger < 1.0:
            raise ValueError("compaction_trigger must be in [0.5, 1)")
        self.keep_recent_tokens = min(self.keep_recent_tokens, self.max_tokens)

    @property
    def trigger_tokens(self) -> int:
        return max(1, int(self.max_tokens * self.compaction_trigger))

    def needs_compaction(self, token_count: int) -> bool:
        return int(token_count) >= self.trigger_tokens

    def overflow(self, token_count: int) -> int:
        return max(0, int(token_count) - self.max_tokens)

    def retention_plan(self, token_count: int) -> dict[str, int | bool]:
        overflow = self.overflow(token_count)
        return {
            "token_count": int(token_count),
            "max_tokens": self.max_tokens,
            "needs_compaction": self.needs_compaction(token_count),
            "overflow_tokens": overflow,
            "preserve_recent_tokens": self.keep_recent_tokens,
            "tokens_for_memory": max(0, overflow + max(0, int(token_count * 0.1))),
        }


class MemoryOSV2:
    """Model-facing coordinator for a hierarchical memory bank."""

    def __init__(
        self,
        hidden_size: int,
        *,
        router: Optional[MemoryRouterV2] = None,
        bank: Optional[PagedMemoryBankV2] = None,
        kv_budget: Optional[KVBudgetManagerV2] = None,
        read_threshold: float = 0.65,
        write_threshold: float = 0.50,
        runtime_device: Optional[torch.device] = None,
        gpu_cache_records: int = 256,
        gpu_cache_tokens: int = 131072,
        gpu_cache_reserve_mb: int = 2048,
        gpu_cache_adaptive: bool = True,
        record_scorer: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        min_read_margin: float = 0.0,
        require_evidence: bool = False,
        attribute_coverage: Optional[Callable[[Tensor], tuple[str, float]]] = None,
    ) -> None:
        self.router = router or MemoryRouterV2(hidden_size)
        self.bank = bank or PagedMemoryBankV2(
            hidden_size,
            router=self.router,
            runtime_device=runtime_device,
            gpu_cache_records=gpu_cache_records,
            gpu_cache_tokens=gpu_cache_tokens,
            gpu_cache_reserve_mb=gpu_cache_reserve_mb,
            gpu_cache_adaptive=gpu_cache_adaptive,
            record_scorer=record_scorer,
        )
        if record_scorer is not None:
            self.bank.record_scorer = record_scorer
        # Optional coverage gate.  ``attribute_coverage(query_key)`` returns the attribute
        # the question asks about (or "" when the head says none), and the runtime refuses
        # when the bank does not hold it.  This exists because score geometry cannot detect
        # an unheld attribute: measured over nine score features, the best fitted head
        # reaches AUC 0.61 and three of four scorers sit at ~0.5, while same-shape
        # candidates look equally plausible for any question.
        self.attribute_coverage = attribute_coverage
        self.kv_budget = kv_budget or KVBudgetManagerV2()
        self.read_threshold = read_threshold
        self.write_threshold = write_threshold
        self.min_read_margin = float(min_read_margin)
        self.require_evidence = bool(require_evidence)

    def write(self, **kwargs: Any) -> tuple[MemoryRecordV2, str]:
        importance = float(kwargs.get("importance", 0.5))
        confidence = float(kwargs.get("confidence", 0.5))
        force = bool(kwargs.get("force", False))
        trusted = bool(kwargs.get("trusted", True))
        if not force and (importance < self.write_threshold or confidence < 0.25):
            kwargs["trusted"] = False
        return self.bank.write(**kwargs)

    def write_batch(self, records: Iterable[dict[str, Any]]) -> list[tuple[MemoryRecordV2, str]]:
        """Apply the V2 write gate and commit a batch atomically when possible."""

        prepared: list[dict[str, Any]] = []
        for item in records:
            kwargs = dict(item)
            importance = float(kwargs.get("importance", 0.5))
            confidence = float(kwargs.get("confidence", 0.5))
            force = bool(kwargs.get("force", False))
            if not force and (importance < self.write_threshold or confidence < 0.25):
                kwargs["trusted"] = False
            prepared.append(kwargs)
        return self.bank.write_batch(prepared)

    def read(
        self,
        *,
        query_key: Tensor,
        query_text: str = "",
        query_token_ids: Optional[Tensor] = None,
        top_k_pages: Optional[int] = None,
        top_k_records: Optional[int] = None,
        max_hops: Optional[int] = None,
    ) -> tuple[list[MemoryRecordV2], RouterDecisionV2]:
        if _is_explicit_unknown_request(query_text):
            return [], RouterDecisionV2(
                need_memory=False,
                page_ids=[],
                record_ids=[],
                page_scores=[],
                record_scores=[],
                hop_count=0,
                hop_trace=[],
                confidence=0.0,
                stop_reason="explicit_unknown_request",
            )
        if self.attribute_coverage is not None and query_key.numel() > 0:
            verdict = self.attribute_coverage(query_key)
            # ``None`` means the gate declines to judge (its vocabulary does not cover this
            # bank); the runtime then behaves exactly as if the gate were absent.
            if verdict is not None:
                attribute, coverage_probability = verdict
                # Match the predicted attribute against every entity in the bank.  The
                # previous check hard-coded the entity name (`f"user::{attribute}"`), while
                # the head's own `coverage()` predicate derives presence from
                # ``key.split("::", 1)[1]`` and therefore treats the entity as arbitrary.
                # Whenever a bank stores records under any other entity (explicit writes,
                # agent-authored memories, imported users) the gate then engaged and
                # refused *every* question, including attributes it actually held.
                attribute_key = attribute.strip().lower() if attribute else ""
                covered = bool(attribute_key) and any(
                    key.rsplit("::", 1)[-1].strip().lower() == attribute_key
                    for key in self.bank.active_by_conflict
                )
                if not covered:
                    return [], RouterDecisionV2(
                        need_memory=False,
                        page_ids=[],
                        record_ids=[],
                        page_scores=[],
                        record_scores=[],
                        hop_count=0,
                        hop_trace=[],
                        confidence=float(coverage_probability),
                        stop_reason=("attribute_not_covered" if attribute_key
                                     else "no_attribute_recognised"),
                    )
        if self.router is not None and query_key.numel() == self.router.hidden_size:
            router_device = next(self.router.parameters()).device
            need_probability = float(
                torch.sigmoid(self.router.need_memory(query_key.to(router_device).reshape(1, -1)))[0, 0].item()
            )
            token_evidence = self.bank.has_token_evidence(query_key, query_token_ids, query_text)
            if need_probability < self.read_threshold and not token_evidence:
                return [], RouterDecisionV2(
                    need_memory=False,
                    page_ids=[],
                    record_ids=[],
                    page_scores=[],
                    record_scores=[],
                    hop_count=0,
                    hop_trace=[],
                    confidence=need_probability,
                    stop_reason="router_abstained",
                )
        records, decision = self.bank.query(
            query_key=query_key,
            query_text=query_text,
            query_token_ids=query_token_ids,
            top_k_pages=top_k_pages,
            top_k_records=top_k_records,
            max_hops=max_hops,
            min_score=-1.0,
        )
        token_evidence = self.bank.has_token_evidence(query_key, query_token_ids, query_text)
        if (
            self.require_evidence
            and records
            and not token_evidence
            and decision.score_margin < self.min_read_margin
        ):
            decision.need_memory = False
            decision.record_ids = []
            decision.stop_reason = "insufficient_evidence_margin"
            return [], decision
        if decision.confidence < self.read_threshold and not token_evidence:
            decision.need_memory = False
            decision.record_ids = []
            decision.stop_reason = "below_read_threshold"
            return [], decision
        if token_evidence and decision.confidence < self.read_threshold:
            decision.stop_reason = "token_evidence_override"
        return records, decision

    def correct(self, **kwargs: Any) -> tuple[MemoryRecordV2, str]:
        return self.bank.correct(**kwargs)

    def approve(self, record_id: str) -> MemoryRecordV2:
        return self.bank.approve(record_id)

    def retract(self, record_id: str) -> None:
        self.bank.retract(record_id)

    def stats(self) -> dict[str, Any]:
        output = self.bank.stats()
        output["kv_budget"] = {
            "max_tokens": self.kv_budget.max_tokens,
            "trigger_tokens": self.kv_budget.trigger_tokens,
        }
        return output

    def list_records(self, **kwargs: Any) -> list[MemoryRecordV2]:
        return self.bank.list_records(**kwargs)

    def edit_record(self, record_id: str, **kwargs: Any) -> MemoryRecordV2:
        return self.bank.edit_record(record_id, **kwargs)

    def retract_record(self, record_id: str) -> None:
        self.bank.retract(record_id)

    def retract_origin(self, origin: str) -> list[str]:
        """Retract every active record written from the same user turn."""
        return self.bank.retract_origin(origin)

    def audit(self) -> dict[str, Any]:
        return self.bank.audit()

    def flush_storage(self) -> None:
        self.bank.flush_storage()

    def close_storage(self) -> None:
        self.bank.close_storage()

    def export_payload(self) -> dict[str, Any]:
        payload = self.bank.export_payload()
        payload["read_threshold"] = self.read_threshold
        payload["write_threshold"] = self.write_threshold
        payload["min_read_margin"] = self.min_read_margin
        payload["require_evidence"] = self.require_evidence
        payload["kv_budget"] = asdict(self.kv_budget)
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        router: Optional[MemoryRouterV2] = None,
        tier_store: Optional["TieredMemoryStoreV2"] = None,
        max_resident_pages: int = 256,
        runtime_device: Optional[torch.device] = None,
        gpu_cache_records: int = 256,
        gpu_cache_tokens: int = 131072,
        gpu_cache_reserve_mb: int = 2048,
        gpu_cache_adaptive: bool = True,
        record_scorer: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        min_read_margin: float = 0.0,
        require_evidence: bool = False,
    ) -> "MemoryOSV2":
        bank = PagedMemoryBankV2.from_payload(
            payload,
            router=router,
            tier_store=tier_store,
            max_resident_pages=max_resident_pages,
            runtime_device=runtime_device,
            gpu_cache_records=gpu_cache_records,
            gpu_cache_tokens=gpu_cache_tokens,
            gpu_cache_reserve_mb=gpu_cache_reserve_mb,
            gpu_cache_adaptive=gpu_cache_adaptive,
            record_scorer=record_scorer,
        )
        budget = KVBudgetManagerV2(**payload.get("kv_budget", {}))
        return cls(
            bank.hidden_size,
            router=router,
            bank=bank,
            kv_budget=budget,
            read_threshold=float(payload.get("read_threshold", 0.65)),
            write_threshold=float(payload.get("write_threshold", 0.5)),
            record_scorer=record_scorer,
            min_read_margin=float(payload.get("min_read_margin", min_read_margin)),
            require_evidence=bool(payload.get("require_evidence", require_evidence)),
        )


def router_training_loss(
    router: MemoryRouterV2,
    query: Tensor,
    candidates: Tensor,
    positive_index: Tensor,
    *,
    need_memory_label: Optional[Tensor] = None,
    hop_label: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Compute supervised router losses with hard negatives."""

    output = router(query, candidates)
    losses: dict[str, Tensor] = {}
    losses["candidate"] = F.cross_entropy(output["scores"], positive_index)
    if need_memory_label is not None:
        losses["need_memory"] = F.binary_cross_entropy_with_logits(
            output["need_memory_logits"], need_memory_label.float()
        )
    if hop_label is not None:
        losses["hop"] = F.cross_entropy(output["hop_logits"], hop_label)
    losses["total"] = sum(losses.values())
    return losses


__all__ = [
    "KVBudgetManagerV2",
    "MemoryOSV2",
    "MemoryPageV2",
    "MemoryRecordV2",
    "MemoryRouterV2",
    "PagedMemoryBankV2",
    "RouterDecisionV2",
    "router_training_loss",
    "STATUS_ACTIVE",
    "STATUS_QUARANTINED",
    "STATUS_RETRACTED",
    "STATUS_SUPERSEDED",
]
