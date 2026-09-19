"""Load the local Qwen checkpoint and run one real memory-aware forward pass."""

from __future__ import annotations

import argparse
import contextlib
import io

import torch

from .qwen_integration import QwenMemoryConfig, load_qwen_dynamic, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--surgery-mode", choices=("residual", "blend", "replace"), default="residual")
    parser.add_argument("--blend-init", type=float, default=0.0)
    parser.add_argument("--layer-indices", type=int, nargs="+", default=None)
    parser.add_argument("--direct-logit-scale", type=float, default=0.0)
    parser.add_argument("--write-token-offset", type=int, default=None)
    parser.add_argument("--broadcast-write", action="store_true")
    parser.add_argument("--raw-token-write", action="store_true")
    parser.add_argument("--raw-logit-scale", type=float, default=0.0)
    parser.add_argument("--native-mode", action="store_true")
    parser.add_argument("--persistent-memory", action="store_true")
    parser.add_argument("--reset-token-id", type=int, default=None)
    parser.add_argument("--no-summary-pooling", action="store_true")
    args = parser.parse_args()

    captured = io.StringIO()
    stdout = contextlib.redirect_stdout(captured)
    stderr = contextlib.redirect_stderr(captured)
    stdout.__enter__()
    stderr.__enter__()
    error = None
    model = None
    tokenizer = None
    try:
        memory_config = QwenMemoryConfig(
            mode=args.surgery_mode,
            blend_init=args.blend_init,
            layer_indices=tuple(args.layer_indices) if args.layer_indices else None,
            direct_logit_scale=args.direct_logit_scale,
            write_token_offset=args.write_token_offset,
            broadcast_write=args.broadcast_write,
            raw_token_write=args.raw_token_write,
            raw_logit_scale=args.raw_logit_scale,
            native_mode=args.native_mode,
            persistent_memory=args.persistent_memory,
            reset_token_id=args.reset_token_id,
            summary_pooling=not args.no_summary_pooling,
        )
        model = load_qwen_dynamic(
            args.model_path,
            memory_config=memory_config,
            load_in_4bit=not args.no_4bit,
        )
        tokenizer = load_tokenizer(args.model_path)
    except Exception as exc:  # pragma: no cover - diagnostic entry point
        error = (type(exc).__name__, str(exc))
    finally:
        stderr.__exit__(None, None, None)
        stdout.__exit__(None, None, None)

    print(f"load_error={error if error else 'none'}")
    if model is None or tokenizer is None:
        return

    print(f"model={type(model.base_model).__name__}")
    print(f"device={model._find_layer_device()}")
    print(f"memory_layers={model.layer_indices}")
    print(
        "layer_types="
        + str(
            tuple(
                getattr(getattr(model.base_model.model.language_model.layers[index], "inner", None), "layer_type", "unknown")
                for index in model.layer_indices
            )
        )
    )
    print(f"surgery_mode={model.memory_config.mode}")
    print(f"blend_init={model.memory_config.blend_init}")
    print(f"trainable_memory_parameters={sum(p.numel() for p in model.trainable_parameters):,}")

    text = "你好，请用一句话介绍你自己。"
    encoded = tokenizer(text, return_tensors="pt")
    device = model._find_layer_device()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded, update_memory=True, return_memory=True)
    print(f"logits_shape={tuple(output.logits.shape)}")
    print(f"memory_shape={tuple(output.memory.shape)}")
    print("forward_ok=true")


if __name__ == "__main__":
    main()
