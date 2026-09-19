"""A small causal LM with an explicit writable memory state.

This is intentionally independent from the Qwen checkpoint in the parent
directory. It is a research reference implementation for architecture work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DynamicMemoryConfig:
    vocab_size: int = 128
    max_seq_len: int = 64
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    memory_slots: int = 8
    dropout: float = 0.0


@dataclass
class DynamicMemoryOutput:
    logits: Tensor
    memory: Tensor
    loss: Optional[Tensor] = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DynamicMemoryConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, config: DynamicMemoryConfig) -> None:
        super().__init__()
        hidden = config.d_model * config.mlp_ratio
        self.up = nn.Linear(config.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: DynamicMemoryConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DynamicMemory(nn.Module):
    """A differentiable key-value memory with explicit read/write behavior.

    The memory tensor is returned to the caller and is not a model parameter.
    It can therefore change during inference without changing the backbone.
    """

    def __init__(self, config: DynamicMemoryConfig) -> None:
        super().__init__()
        self.slots = config.memory_slots
        self.dim = config.d_model
        self.read_q = nn.Linear(self.dim, self.dim, bias=False)
        self.read_k = nn.Linear(self.dim, self.dim, bias=False)
        self.read_v = nn.Linear(self.dim, self.dim, bias=False)
        self.read_out = nn.Linear(self.dim, self.dim, bias=False)
        self.read_gate = nn.Linear(self.dim, 1)

        self.slot_keys = nn.Parameter(torch.randn(self.slots, self.dim) / self.dim**0.5)
        self.write_value = nn.Linear(self.dim, self.slots * self.dim, bias=False)
        self.write_gate = nn.Linear(self.dim, self.slots)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.slots, self.dim, device=device, dtype=dtype)

    def read(self, x: Tensor, memory: Tensor) -> Tensor:
        q = self.read_q(x)
        k = self.read_k(memory)
        v = self.read_v(memory)
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.dim**0.5
        retrieved = torch.matmul(scores.softmax(dim=-1), v)
        retrieved = self.read_out(retrieved)
        gate = torch.sigmoid(self.read_gate(x))
        return gate * retrieved

    def update(self, x: Tensor, memory: Tensor) -> Tensor:
        # The final token is used as a compact summary of the newly observed
        # chunk. This makes chunk boundaries explicit and keeps the experiment
        # cheap enough to run repeatedly on a single consumer GPU.
        summary = x[:, -1]
        proposal = self.write_value(summary).view(-1, self.slots, self.dim)
        address = (summary @ self.slot_keys.t()).softmax(dim=-1)
        strength = torch.sigmoid(self.write_gate(summary)) * address
        strength = strength.unsqueeze(-1)
        return memory + strength * (proposal - memory)


class DynamicMemoryLM(nn.Module):
    """Decoder-only Transformer with a persistent, caller-owned memory state."""

    def __init__(self, config: DynamicMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.memory = DynamicMemory(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _ensure_memory(self, memory: Optional[Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if memory is None:
            return self.memory.initial_state(batch_size, device=device, dtype=dtype)
        if memory.ndim != 3 or memory.shape[0] != batch_size:
            raise ValueError("memory must have shape [batch, memory_slots, d_model]")
        return memory

    def forward(
        self,
        input_ids: Tensor,
        *,
        memory: Optional[Tensor] = None,
        update_memory: bool = True,
        labels: Optional[Tensor] = None,
    ) -> DynamicMemoryOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        x = self.token_emb(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.pos_emb(positions)[None, :, :]
        memory = self._ensure_memory(memory, batch_size, input_ids.device, x.dtype)

        # Read uses the state from before this chunk. The write happens only
        # after logits are computed, preventing target-token leakage.
        x = x + self.memory.read(x, memory)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        new_memory = self.memory.update(x, memory) if update_memory else memory
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return DynamicMemoryOutput(logits=logits, memory=new_memory, loss=loss)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
