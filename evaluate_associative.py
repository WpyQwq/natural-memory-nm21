"""Evaluate a saved dynamic-memory checkpoint."""

from __future__ import annotations

import argparse

import torch

from .model import DynamicMemoryConfig, DynamicMemoryLM
from .tasks import sample_associative_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="V2_dpskw/checkpoints/latest.pt")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = DynamicMemoryConfig(**checkpoint["config"])
    model = DynamicMemoryLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for overwrite in (False, True):
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(args.batches):
                batch = sample_associative_batch(
                    batch_size=256,
                    vocab_size=config.vocab_size,
                    device=device,
                    overwrite=overwrite,
                )
                memory = None
                for chunk in batch.learn_chunks:
                    memory = model(chunk, memory=memory, update_memory=True).memory
                output = model(batch.query_input, memory=memory, update_memory=False)
                prediction = output.logits[:, 0].argmax(dim=-1)
                correct += int((prediction == batch.expected).sum())
                total += batch.expected.numel()
        print(f"overwrite={overwrite} accuracy={correct / total:.3f}")


if __name__ == "__main__":
    main()
