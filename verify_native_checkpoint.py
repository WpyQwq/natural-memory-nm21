"""Package learned memory into an adapter checkpoint and verify restart/reset."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .evaluate_native_memory import _controller_step, _encode_prompt
from .qwen_integration import (
    DEFAULT_MEMORY_RESET_TOKEN,
    load_memory_config,
    load_qwen_dynamic,
    load_tokenizer,
    resolve_memory_reset_token,
)
from .train_native_memory import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--adapter", default="V2_dpskw/qwen_memory_adapter_native_v3")
    parser.add_argument("--data", default="V2_dpskw/data/native_memory/eval.jsonl")
    parser.add_argument(
        "--output-adapter",
        default="V2_dpskw/qwen_memory_adapter_native_v3_persistent",
    )
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    records = load_records(args.data)
    record = next(record for record in records if record.get("answerable"))
    tokenizer = load_tokenizer(args.model_path)
    config = load_memory_config(args.adapter)
    config.persistent_memory = True
    config.reset_token_id = resolve_memory_reset_token(tokenizer, DEFAULT_MEMORY_RESET_TOKEN)
    model = load_qwen_dynamic(args.model_path, memory_config=config, load_in_4bit=not args.no_4bit)
    model.load_memory_adapter(args.adapter)
    model.reset_memory()
    for chunk in record["memory_chunks"]:
        _controller_step(model, tokenizer, chunk, args.max_length)
    output_adapter = Path(args.output_adapter)
    model.save_persistent_memory_checkpoint(output_adapter)
    saved_norm = float(model.runtime.state.detach().float().norm())
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restart_config = load_memory_config(output_adapter)
    restarted = load_qwen_dynamic(
        args.model_path,
        memory_config=restart_config,
        load_in_4bit=not args.no_4bit,
    )
    restarted.load_memory_adapter(output_adapter)
    answer = str(record["answer"])
    prompt = _encode_prompt(tokenizer, record["query"][:-1])
    device = restarted._find_layer_device()
    generated = restarted.generate(
        input_ids=prompt.to(device),
        attention_mask=torch.ones_like(prompt, device=device),
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=False,
        update_memory=False,
    )
    response = tokenizer.decode(
        generated[:, prompt.shape[1] :][0].detach().cpu().tolist(),
        skip_special_tokens=True,
    ).strip()
    reset_prompt = _encode_prompt(tokenizer, [{"role": "user", "content": DEFAULT_MEMORY_RESET_TOKEN}])
    restarted.generate(
        input_ids=reset_prompt.to(device),
        attention_mask=torch.ones_like(reset_prompt, device=device),
        max_new_tokens=1,
        do_sample=False,
        use_cache=False,
        update_memory=False,
    )
    reset_norm = float(restarted.runtime.state.detach().float().norm())
    report = {
        "source_adapter": str(args.adapter),
        "output_adapter": str(output_adapter),
        "record_id": record.get("id"),
        "answer": answer,
        "generated_after_restart": response,
        "restart_contains_answer": answer in response,
        "saved_memory_norm": saved_norm,
        "reset_token": DEFAULT_MEMORY_RESET_TOKEN,
        "reset_token_id": restart_config.reset_token_id,
        "memory_norm_after_reset_token": reset_norm,
        "reset_zeroed_memory": reset_norm < 1e-5,
    }
    (output_adapter / "native_checkpoint_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
