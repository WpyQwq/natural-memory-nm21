"""MemoryRouterXL: a wider, deeper sparse-retrieval router for Natural Memory.

This module defines a **new** router architecture.  It is deliberately not a
patch of :class:`V2_dpskw.memory_os_v2.MemoryRouterV2`: the XL router is a
separate class with its own encoder, interaction features, residual pair trunk
and multi-layer policy heads, so a 512-dim V2 router and an XL router can be
trained, compared and shipped side by side.

What is bigger than the V2 router (4.74M parameters at ``router_dim=512``):

==========================  =========================  ==========================
stage                       V2 router                  XL router
==========================  =========================  ==========================
query/key encoder           one ``Linear``             ``encoder_layers`` MLP
address dimension           ``router_dim``              ``router_dim`` (configurable)
head aggregation            softmax head gate          softmax head gate (kept)
pair features               ``[q, k, q-k]``            ``[q, k, q-k, q*k]`` (+ LayerNorm)
pair scorer                 one hidden layer           residual MLP trunk
outcome heads               single hidden layer        ``policy_layers`` MLP
==========================  =========================  ==========================

The runtime contract is identical to ``MemoryRouterV2`` so that
:class:`V2_dpskw.memory_os_v2.PagedMemoryBankV2` can drive either router
unchanged: ``router_dim``, ``num_heads``, ``head_dim``, ``hidden_size``,
``max_hops``, ``encode_query``, ``encode_key``, ``projected_scores``,
``pair_scores``, ``forward``, plus the ``need_memory`` / ``hop_controller`` /
``head_gate`` submodules.

The only thing an XL checkpoint needs on top of the state dict is the
architecture config, which :meth:`MemoryRouterXL.arch_config` records and
:func:`load_router_xl` consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ARCH_NAME = "router_xl"
ARCH_VERSION = 1

#: Constructor keyword arguments that fully describe an XL router instance.
ARCH_KEYS = (
    "hidden_size",
    "router_dim",
    "num_heads",
    "max_hops",
    "encoder_layers",
    "encoder_hidden",
    "pair_blocks",
    "pair_hidden",
    "pair_expansion",
    "pair_dropout",
    "use_interaction",
    "policy_layers",
    "policy_hidden",
    "policy_dropout",
    "learnable_cosine_scale",
)


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual block: ``x + fc2(silu(fc1(norm(x))))``."""

    def __init__(self, dim: int, *, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, int(dim) * max(1, int(expansion)))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.norm(value)
        hidden = self.fc2(F.silu(self.fc1(hidden)))
        return value + self.dropout(hidden)


def _mlp(
    in_features: int,
    hidden_features: int,
    out_features: int,
    *,
    layers: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Plain MLP with ``layers`` linear stages (``layers >= 1``)."""

    layers = max(1, int(layers))
    if layers == 1:
        return nn.Sequential(nn.Linear(in_features, out_features))
    modules: list[nn.Module] = [nn.Linear(in_features, hidden_features), nn.SiLU()]
    if dropout > 0.0:
        modules.append(nn.Dropout(dropout))
    for _ in range(layers - 2):
        modules.extend([nn.Linear(hidden_features, hidden_features), nn.SiLU()])
        if dropout > 0.0:
            modules.append(nn.Dropout(dropout))
    modules.append(nn.Linear(hidden_features, out_features))
    return nn.Sequential(*modules)


class MemoryRouterXL(nn.Module):
    """High-capacity router: projected addresses, pair interaction, policy heads.

    ``hidden_size`` is the frozen Qwen hidden size (2560 for Qwen3.5-4B) and
    ``router_dim`` is the compact address size stored next to every memory
    record.  Everything else controls capacity inside the router.
    """

    arch_name = ARCH_NAME
    arch_version = ARCH_VERSION

    def __init__(
        self,
        hidden_size: int,
        *,
        router_dim: int = 1024,
        num_heads: int = 16,
        max_hops: int = 3,
        encoder_layers: int = 2,
        encoder_hidden: int = 0,
        pair_blocks: int = 1,
        pair_hidden: int = 0,
        pair_expansion: int = 2,
        pair_dropout: float = 0.05,
        use_interaction: bool = True,
        policy_layers: int = 2,
        policy_hidden: int = 512,
        policy_dropout: float = 0.0,
        learnable_cosine_scale: bool = True,
    ) -> None:
        super().__init__()
        if router_dim % num_heads != 0:
            raise ValueError("router_dim must be divisible by num_heads")
        if max_hops < 1:
            raise ValueError("max_hops must be positive")
        if encoder_layers < 1:
            raise ValueError("encoder_layers must be positive")

        self.hidden_size = int(hidden_size)
        self.router_dim = int(router_dim)
        self.num_heads = int(num_heads)
        self.max_hops = int(max_hops)
        self.head_dim = self.router_dim // self.num_heads
        self.encoder_layers = int(encoder_layers)
        self.pair_blocks = max(0, int(pair_blocks))
        self.pair_expansion = max(1, int(pair_expansion))
        self.pair_dropout = float(pair_dropout)
        self.use_interaction = bool(use_interaction)
        self.policy_layers = max(1, int(policy_layers))
        self.policy_hidden = int(policy_hidden)
        self.policy_dropout = float(policy_dropout)
        self.learnable_cosine_scale = bool(learnable_cosine_scale)

        enc_hidden = int(encoder_hidden) if encoder_hidden > 0 else self.router_dim
        self.encoder_hidden = enc_hidden
        self.query_projection = _mlp(
            self.hidden_size, enc_hidden, self.router_dim, layers=self.encoder_layers
        )
        self.key_projection = _mlp(
            self.hidden_size, enc_hidden, self.router_dim, layers=self.encoder_layers
        )

        pair_in = self.router_dim * (4 if self.use_interaction else 3)
        self.pair_input_dim = pair_in
        pair_hidden = int(pair_hidden) if pair_hidden > 0 else self.router_dim
        self.pair_hidden = pair_hidden
        self.pair_norm = nn.LayerNorm(pair_in)
        self.pair_in = nn.Linear(pair_in, pair_hidden)
        self.pair_activation = nn.SiLU()
        self.pair_blocks_module = nn.ModuleList(
            ResidualMLPBlock(pair_hidden, expansion=self.pair_expansion, dropout=self.pair_dropout)
            for _ in range(self.pair_blocks)
        )
        self.pair_out = nn.Linear(pair_hidden, 1)
        # ``pair_scorer`` keeps the V2 attribute name so existing diagnostics that
        # walk the module tree still find a scorer.
        self.pair_scorer = nn.Sequential(self.pair_in, self.pair_activation, self.pair_out)

        self.need_memory = _mlp(
            self.hidden_size,
            self.policy_hidden,
            1,
            layers=self.policy_layers,
            dropout=self.policy_dropout,
        )
        self.hop_controller = _mlp(
            self.hidden_size,
            self.policy_hidden,
            self.max_hops + 1,
            layers=self.policy_layers,
            dropout=self.policy_dropout,
        )
        self.head_gate = nn.Linear(self.hidden_size, self.num_heads)
        if self.learnable_cosine_scale:
            self.cosine_scale = nn.Parameter(torch.ones(()))

    # ------------------------------------------------------------------ config
    def arch_config(self) -> dict[str, Any]:
        """Everything needed to rebuild this exact router."""

        return {
            "arch": ARCH_NAME,
            "arch_version": ARCH_VERSION,
            "hidden_size": self.hidden_size,
            "router_dim": self.router_dim,
            "num_heads": self.num_heads,
            "max_hops": self.max_hops,
            "encoder_layers": self.encoder_layers,
            "encoder_hidden": self.encoder_hidden,
            "pair_blocks": self.pair_blocks,
            "pair_hidden": self.pair_hidden,
            "pair_expansion": self.pair_expansion,
            "pair_dropout": self.pair_dropout,
            "use_interaction": self.use_interaction,
            "policy_layers": self.policy_layers,
            "policy_hidden": self.policy_hidden,
            "policy_dropout": self.policy_dropout,
            "learnable_cosine_scale": self.learnable_cosine_scale,
        }

    @classmethod
    def from_arch_config(cls, config: dict[str, Any]) -> "MemoryRouterXL":
        kwargs = {key: config[key] for key in ARCH_KEYS if key in config}
        if "hidden_size" not in kwargs:
            raise ValueError("arch config must contain hidden_size")
        return cls(**kwargs)

    def parameter_count(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    # ------------------------------------------------------------------ routing
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

    def _pair_features(self, query_heads: Tensor, key_heads: Tensor) -> Tensor:
        batch, candidates = key_heads.shape[0], key_heads.shape[1]
        query_flat = query_heads.reshape(batch, 1, self.router_dim).expand(-1, candidates, -1)
        key_flat = key_heads.reshape(batch, candidates, self.router_dim)
        parts = [query_flat, key_flat, query_flat - key_flat]
        if self.use_interaction:
            parts.append(query_flat * key_flat)
        return self.pair_norm(torch.cat(parts, dim=-1))

    def _score_pair_features(self, pair_features: Tensor) -> Tensor:
        hidden = self.pair_activation(self.pair_in(pair_features))
        for block in self.pair_blocks_module:
            hidden = block(hidden)
        return self.pair_out(hidden).squeeze(-1)

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
        query_heads = self._reshape(self.encode_query(query))
        key_heads = self._reshape(F.normalize(projected_candidates, dim=-1))
        head_scores = torch.einsum("bhc,bnhc->bnh", query_heads, key_heads)
        gates = torch.softmax(self.head_gate(query), dim=-1)[:, None, :]
        cosine_score = (head_scores * gates).sum(dim=-1)
        if self.learnable_cosine_scale:
            cosine_score = cosine_score * self.cosine_scale
        learned_score = self._score_pair_features(self._pair_features(query_heads, key_heads))
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


def load_router_xl(
    checkpoint_path: str | Path,
    *,
    arch_config: Optional[dict[str, Any]] = None,
    map_location: str | torch.device = "cpu",
) -> MemoryRouterXL:
    """Rebuild an XL router from a checkpoint plus its architecture config.

    ``checkpoint_path`` may be a bare state dict (as written by the trainer's
    ``memory_router_xl.pt``) or a payload containing ``router_state_dict``.
    """

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and "router_state_dict" in payload:
        state = payload["router_state_dict"]
        config = arch_config or payload.get("arch_config")
    else:
        state = payload
        config = arch_config
    if config is None:
        sidecar = checkpoint_path.with_name("router_arch.json")
        if not sidecar.exists():
            raise ValueError(
                f"architecture config not supplied and {sidecar} is missing; "
                "pass arch_config explicitly"
            )
        config = json.loads(sidecar.read_text(encoding="utf-8"))
    router = MemoryRouterXL.from_arch_config(config)
    router.load_state_dict(state, strict=True)
    router.eval()
    return router
