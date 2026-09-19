"""Train the dynamic memory mechanism on a separated key-value task."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from .model import DynamicMemoryConfig, DynamicMemoryLM, count_parameters
from .tasks import sample_associative_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint-dir", default="V2_dpskw/checkpoints")
    parser.add_argument("--resume", default=None, help="path to a checkpoint produced by this script")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def evaluate(model: DynamicMemoryLM, *, device: torch.device, overwrite: bool, batches: int = 20) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for _ in range(batches):
            batch = sample_associative_batch(
                batch_size=256,
                vocab_size=model.config.vocab_size,
                device=device,
                overwrite=overwrite,
            )
            memory = None
            for chunk in batch.learn_chunks:
                memory = model(chunk, memory=memory, update_memory=True).memory
            output = model(batch.query_input, memory=memory, update_memory=False, labels=batch.query_labels)
            total_loss += float(output.loss)
            prediction = output.logits[:, 0].argmax(dim=-1)
            total_correct += int((prediction == batch.expected).sum())
            total_items += batch.expected.numel()
    model.train()
    return total_loss / batches, total_correct / total_items


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = DynamicMemoryConfig()
    model = DynamicMemoryLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    use_amp = device.type == "cuda"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        print(f"resumed_from={args.resume} step={start_step}")

    print(f"device={device} parameters={count_parameters(model):,} overwrite={args.overwrite}")
    for step in range(start_step + 1, start_step + args.steps + 1):
        model.train()
        batch = sample_associative_batch(
            batch_size=args.batch_size,
            vocab_size=config.vocab_size,
            device=device,
            overwrite=args.overwrite,
        )
        memory = None
        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with amp_context:
            for chunk in batch.learn_chunks:
                memory = model(chunk, memory=memory, update_memory=True).memory
            output = model(batch.query_input, memory=memory, update_memory=False, labels=batch.query_labels)
            if output.loss is None:
                raise RuntimeError("training loss was not produced")
            loss = output.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            eval_loss, accuracy = evaluate(model, device=device, overwrite=args.overwrite)
            print(f"step={step:5d} train_loss={loss.detach().item():.4f} eval_loss={eval_loss:.4f} accuracy={accuracy:.3f}")
            state = {
                "config": vars(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "seed": args.seed,
                "overwrite": args.overwrite,
            }
            torch.save(state, checkpoint_dir / "latest.pt")
            (checkpoint_dir / "run.json").write_text(json.dumps({"args": vars(args), "device": str(device)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
