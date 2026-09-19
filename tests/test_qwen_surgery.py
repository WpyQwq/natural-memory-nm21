from __future__ import annotations

import unittest

import torch
from torch import nn

from V2_dpskw.qwen_integration import (
    AutomaticMemoryPolicy,
    MemoryLayerAdapter,
    NaturalLanguageRetriever,
    NativeQwenDynamicMemory,
    QwenMemoryConfig,
    QwenDynamicMemory,
    _MemoryRuntime,
    looks_like_question,
    split_memory_candidates,
)


class _FakeAttention(nn.Module):
    def __init__(self, *, fail_if_called: bool = False) -> None:
        super().__init__()
        self.fail_if_called = fail_if_called
        self.called = False

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        self.called = True
        if self.fail_if_called:
            raise AssertionError("original token mixer was called in replace mode")
        return hidden_states * 2.0, None


class _FakeQwenLayer(nn.Module):
    layer_type = "full_attention"

    def __init__(self, *, fail_if_called: bool = False) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.self_attn = _FakeAttention(fail_if_called=fail_if_called)
        self.mlp = nn.Identity()

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        return hidden_states + self.self_attn(hidden_states)[0]


class QwenSurgeryTest(unittest.TestCase):
    def _runtime(self):
        memory = QwenDynamicMemory(
            hidden_size=8,
            config=QwenMemoryConfig(memory_slots=2, memory_dim=4),
        )
        runtime = _MemoryRuntime(memory)
        runtime.state = memory.initial_state(1, device=torch.device("cpu"))
        return runtime

    def test_blend_exposes_trainable_mixer_weight(self) -> None:
        runtime = self._runtime()
        adapter = MemoryLayerAdapter(
            _FakeQwenLayer(),
            runtime,
            read=True,
            write=False,
            mode="blend",
            blend_init=0.5,
        )
        output = adapter(torch.ones(1, 3, 8))
        output.sum().backward()
        self.assertEqual(tuple(output.shape), (1, 3, 8))
        self.assertIsNotNone(adapter.blend_logit.grad)

    def test_replace_skips_original_token_mixer(self) -> None:
        runtime = self._runtime()
        layer = _FakeQwenLayer(fail_if_called=True)
        adapter = MemoryLayerAdapter(layer, runtime, read=True, write=False, mode="replace")
        output = adapter(torch.ones(1, 3, 8))
        self.assertEqual(tuple(output.shape), (1, 3, 8))
        self.assertFalse(layer.self_attn.called)

    def test_raw_token_write_uses_output_projection_row(self) -> None:
        memory = QwenDynamicMemory(
            hidden_size=8,
            config=QwenMemoryConfig(
                memory_slots=2,
                memory_dim=4,
                write_token_offset=2,
                raw_token_write=True,
                broadcast_write=True,
            ),
        )
        runtime = _MemoryRuntime(memory)
        runtime.state = memory.initial_state(1, device=torch.device("cpu"))
        runtime.read_enabled = False
        runtime.update_enabled = True
        runtime.input_ids = torch.tensor([[5, 6, 7, 8]])
        runtime.attention_mask = torch.ones_like(runtime.input_ids)
        runtime.output_embeddings = nn.Linear(8, 16, bias=False)
        adapter = MemoryLayerAdapter(_FakeQwenLayer(), runtime, read=True, write=True, mode="residual")

        adapter(torch.ones(1, 4, 8))

        expected = runtime.output_embeddings.weight[7]
        self.assertIsNotNone(runtime.raw_memory)
        self.assertTrue(torch.allclose(runtime.raw_memory[0], expected))

    def test_native_controller_exposes_write_forget_and_value_state(self) -> None:
        memory = NativeQwenDynamicMemory(
            hidden_size=8,
            config=QwenMemoryConfig(memory_slots=2, memory_dim=4),
        )
        hidden = torch.randn(1, 3, 8)
        state = memory.initial_state(1, device=torch.device("cpu"))
        updated = memory.update(hidden, state, attention_mask=torch.ones(1, 5, dtype=torch.long))
        self.assertEqual(tuple(updated.shape), (1, 2, 4))
        self.assertEqual(tuple(memory.last_write_probability.shape), (1, 1))
        self.assertEqual(tuple(memory.last_forget_probability.shape), (1, 2))
        self.assertEqual(tuple(memory.last_write_summary.shape), (1, 8))
        self.assertEqual(tuple(memory.last_write_representation.shape), (1, 8))

    def test_natural_language_retriever_scores_single_and_multiple_slots(self) -> None:
        retriever = NaturalLanguageRetriever(hidden_size=8, projection_size=4)
        query = torch.randn(2, 8)
        one_key = torch.randn(2, 8)
        many_keys = torch.randn(2, 3, 8)
        self.assertEqual(tuple(retriever(query, one_key).shape), (2,))
        self.assertEqual(tuple(retriever(query, many_keys).shape), (2, 3))

    def test_automatic_memory_policy_and_candidate_segmentation(self) -> None:
        policy = AutomaticMemoryPolicy(hidden_size=8)
        output = policy(torch.randn(3, 8))
        self.assertEqual(tuple(output.shape), (3,))
        self.assertEqual(
            split_memory_candidates("我叫林浩，我正在开发星火项目。"),
            ["我叫林浩，我正在开发星火项目。"],
        )
        self.assertTrue(looks_like_question("如果我选择 GPU，会发生什么？"))
        self.assertFalse(looks_like_question("我住在上海，正在开发星火项目。"))


if __name__ == "__main__":
    unittest.main()
