from __future__ import annotations

import unittest

import torch

from V2_dpskw.model import DynamicMemoryConfig, DynamicMemoryLM
from V2_dpskw.tasks import sample_associative_batch


class DynamicMemoryModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.config = DynamicMemoryConfig(vocab_size=32, max_seq_len=16, d_model=32, n_layers=1, n_heads=4, memory_slots=2)
        self.model = DynamicMemoryLM(self.config).to(self.device)

    def test_shapes_and_loss(self) -> None:
        batch = sample_associative_batch(batch_size=3, vocab_size=self.config.vocab_size, device=self.device)
        memory = self.model(batch.learn_chunks[0]).memory
        output = self.model(batch.query_input, memory=memory, update_memory=False, labels=batch.query_labels)
        self.assertEqual(tuple(output.logits.shape), (3, 2, self.config.vocab_size))
        self.assertEqual(tuple(output.memory.shape), (3, self.config.memory_slots, self.config.d_model))
        self.assertIsNotNone(output.loss)
        output.loss.backward()

    def test_memory_changes_after_learning_chunk(self) -> None:
        batch = sample_associative_batch(batch_size=2, vocab_size=self.config.vocab_size, device=self.device)
        initial = self.model.memory.initial_state(2, device=self.device, dtype=torch.float32)
        updated = self.model(batch.learn_chunks[0], memory=initial, update_memory=True).memory
        self.assertFalse(torch.allclose(initial, updated))


if __name__ == "__main__":
    unittest.main()
