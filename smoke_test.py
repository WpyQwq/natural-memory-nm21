"""Fast environment and gradient smoke test."""

from __future__ import annotations

import torch

from .model import DynamicMemoryConfig, DynamicMemoryLM, count_parameters
from .tasks import sample_associative_batch


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DynamicMemoryConfig(vocab_size=64, max_seq_len=16, d_model=64, n_layers=2, n_heads=4, memory_slots=4)
    model = DynamicMemoryLM(config).to(device)
    batch = sample_associative_batch(batch_size=8, vocab_size=config.vocab_size, device=device)
    memory = model(batch.learn_chunks[0]).memory
    output = model(batch.query_input, memory=memory, update_memory=False, labels=batch.query_labels)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("non-finite loss")
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("no gradients produced")
    print(f"smoke_ok device={device} parameters={count_parameters(model):,} loss={output.loss.detach().item():.4f}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} memory_allocated_mb={torch.cuda.memory_allocated() / 1024**2:.1f}")


if __name__ == "__main__":
    main()
