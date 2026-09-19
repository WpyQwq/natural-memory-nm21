"""Synthetic streaming tasks for testing dynamic learning behavior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AssociativeBatch:
    learn_chunks: list[Tensor]
    query_input: Tensor
    query_labels: Tensor
    expected: Tensor


def _rand_tokens(batch_size: int, low: int, high: int, device: torch.device) -> Tensor:
    return torch.randint(low, high, (batch_size,), device=device)


def sample_associative_batch(
    *,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    overwrite: bool = False,
) -> AssociativeBatch:
    """Generate key-value observations followed by a separated query.

    The query cannot see the learning chunks through attention. It can only
    answer by using the returned dynamic memory state.
    """

    key_low, key_high = 4, vocab_size // 2
    value_low, value_high = vocab_size // 2, vocab_size
    keys = _rand_tokens(batch_size, key_low, key_high, device)
    value = _rand_tokens(batch_size, value_low, value_high, device)
    learn_chunks = [torch.stack((keys, value), dim=1)]

    expected = value
    if overwrite:
        replacement = _rand_tokens(batch_size, value_low, value_high, device)
        learn_chunks.append(torch.stack((keys, replacement), dim=1))
        expected = replacement

    query_input = torch.stack((keys, expected), dim=1)
    query_labels = torch.full_like(query_input, -100)
    query_labels[:, 1] = expected
    return AssociativeBatch(
        learn_chunks=learn_chunks,
        query_input=query_input,
        query_labels=query_labels,
        expected=expected,
    )
