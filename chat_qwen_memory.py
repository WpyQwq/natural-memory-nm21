"""Minimal interactive chat using the persistent Qwen dynamic memory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .qwen_integration import (
    DEFAULT_MEMORY_RESET_TOKEN,
    QwenMemoryConfig,
    load_memory_config,
    load_qwen_dynamic,
    load_tokenizer,
    resolve_memory_reset_token,
    split_memory_candidates,
)


def _memory_system_prefix(tokenizer, content: str):
    """Encode a valid system-message prefix without adding a fake query."""

    full = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": content},
            {"role": "user", "content": "__memory_query_boundary__"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    input_ids = full["input_ids"]
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    positions = (input_ids[0] == int(im_start)).nonzero(as_tuple=False).flatten()
    if positions.numel() < 2:
        raise RuntimeError("could not locate the system/user memory boundary")
    end = int(positions[1].item())
    return {
        "input_ids": input_ids[:, :end],
        "attention_mask": torch.ones((1, end), dtype=torch.long),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--adapter", default=None)
    parser.add_argument(
        "--natural-language-memory",
        action="store_true",
        help="enable the model-owned exact text memory bank and internal retrieval prefix",
    )
    parser.add_argument(
        "--memory-state",
        default=None,
        help="user runtime memory file; it is loaded at startup and saved after each turn",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--persistent-memory",
        action="store_true",
        help="keep native memory inside the model instance across turns",
    )
    parser.add_argument("--reset-token", default=None)
    parser.add_argument("--reset-token-id", type=int, default=None)
    parser.add_argument(
        "--persist-in-adapter",
        action="store_true",
        help="also checkpoint current user memory into the adapter package",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    memory_config = load_memory_config(args.adapter) if args.adapter else None
    if args.natural_language_memory and memory_config is None:
        memory_config = QwenMemoryConfig(natural_language_memory=True)
    if memory_config is not None and args.natural_language_memory:
        memory_config.natural_language_memory = True
    if memory_config is not None and memory_config.native_mode and args.persistent_memory:
        memory_config.persistent_memory = True
    if memory_config is not None:
        if args.reset_token_id is not None:
            memory_config.reset_token_id = args.reset_token_id
        elif args.reset_token is not None:
            memory_config.reset_token_id = resolve_memory_reset_token(tokenizer, args.reset_token)
        elif memory_config.native_mode and memory_config.reset_token_id is None:
            memory_config.reset_token_id = resolve_memory_reset_token(tokenizer)
    model = load_qwen_dynamic(
        args.model_path,
        memory_config=memory_config,
        load_in_4bit=not args.no_4bit,
    )
    if args.adapter:
        model.load_memory_adapter(args.adapter)
    model.eval()
    device = model._find_layer_device()
    state_path = Path(args.memory_state) if args.memory_state else None
    if state_path is not None and state_path.exists():
        model.load_runtime_memory(state_path, device=device)
        print(f"已加载用户 memory_state：{state_path}")
    print("普通消息会自动判断并保存重要信息；/remember <事实> 强制写入，/reset 清空，/save 保存，/quit 退出。")
    if memory_config is not None and memory_config.reset_token_id is not None:
        print(f"也可在用户消息中发送重置 token：{args.reset_token or DEFAULT_MEMORY_RESET_TOKEN}")

    def save_state() -> None:
        if state_path is not None and model.runtime.state is not None:
            model.save_runtime_memory(state_path)
            print(f"已保存：{state_path}")
        if args.persist_in_adapter and args.adapter and model.runtime.state is not None:
            model.save_persistent_memory_checkpoint(args.adapter)
            print(f"已将当前用户记忆写入模型适配器：{args.adapter}")

    try:
        while True:
            user_text = input("你> ").strip()
            if user_text == "/quit":
                break
            if user_text == "/reset":
                model.reset_memory(batch_size=1, device=device)
                save_state()
                print("已清空动态记忆。")
                continue
            if user_text == "/save":
                save_state()
                continue
            if user_text.startswith("/remember "):
                fact = user_text[len("/remember ") :].strip()
                if not fact:
                    continue
                messages = [
                    {"role": "user", "content": fact},
                    {"role": "assistant", "content": "好的，我会记住这件事。"},
                ]
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=False,
                )
                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                    if isinstance(value, torch.Tensor)
                }
                memory_text = _memory_system_prefix(
                    tokenizer,
                    "以下是与当前用户相关的已保存长期记忆。仅在问题相关时使用，不要编造：\n" + fact,
                )
                memory_text_input_ids = memory_text["input_ids"].to(device)
                memory_text_attention_mask = memory_text.get("attention_mask")
                if memory_text_attention_mask is None:
                    memory_text_attention_mask = torch.ones_like(memory_text_input_ids)
                memory_text_attention_mask = memory_text_attention_mask.to(device)
                memory_key = tokenizer(fact, add_special_tokens=False, return_tensors="pt")
                memory_key_input_ids = memory_key["input_ids"].to(device)
                memory_key_attention_mask = memory_key.get("attention_mask")
                if memory_key_attention_mask is None:
                    memory_key_attention_mask = torch.ones_like(memory_key_input_ids)
                memory_key_attention_mask = memory_key_attention_mask.to(device)
                memory_storage = tokenizer(
                    fact,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    model(
                        **encoded,
                        read_memory=False,
                        update_memory=True,
                        return_memory=True,
                        use_cache=False,
                        memory_text_input_ids=memory_text_input_ids,
                        memory_text_attention_mask=memory_text_attention_mask,
                        memory_key_input_ids=memory_key_input_ids,
                        memory_key_attention_mask=memory_key_attention_mask,
                        memory_storage_input_ids=memory_storage["input_ids"].to(device),
                        memory_storage_attention_mask=memory_storage.get(
                            "attention_mask",
                            torch.ones_like(memory_storage["input_ids"]),
                        ).to(device),
                        force_memory_write=True,
                    )
                save_state()
                print("已写入动态记忆。")
                continue
            if not user_text:
                continue

            messages = [{"role": "user", "content": user_text}]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            memory_query = tokenizer(
                user_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            memory_query_input_ids = memory_query["input_ids"].to(device)
            memory_query_attention_mask = memory_query.get("attention_mask")
            if memory_query_attention_mask is None:
                memory_query_attention_mask = torch.ones_like(memory_query_input_ids)
            memory_query_attention_mask = memory_query_attention_mask.to(device)
            with torch.no_grad():
                # Native mode first gives the prompt to the learned controller
                # so it can decide whether each fact-sized candidate is worth
                # storing. The generation itself is read-only, preventing the
                # model from accidentally memorizing its own answer text.
                if memory_config is not None and memory_config.native_mode:
                    for candidate in split_memory_candidates(user_text):
                        candidate_encoded = tokenizer.apply_chat_template(
                            [{"role": "user", "content": candidate}],
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True,
                            enable_thinking=False,
                        )
                        candidate_encoded = {
                            key: value.to(device)
                            for key, value in candidate_encoded.items()
                            if isinstance(value, torch.Tensor)
                        }
                        memory_text = _memory_system_prefix(
                            tokenizer,
                            "以下是与当前用户相关的已保存长期记忆。仅在问题相关时使用，不要编造：\n"
                            + candidate,
                        )
                        memory_key = tokenizer(
                            candidate,
                            add_special_tokens=False,
                            return_tensors="pt",
                        )
                        memory_key_input_ids = memory_key["input_ids"].to(device)
                        memory_key_attention_mask = memory_key.get("attention_mask")
                        if memory_key_attention_mask is None:
                            memory_key_attention_mask = torch.ones_like(memory_key_input_ids)
                        memory_storage = tokenizer(
                            candidate,
                            add_special_tokens=False,
                            return_tensors="pt",
                        )
                        model(
                            **candidate_encoded,
                            read_memory=False,
                            update_memory=True,
                            return_memory=True,
                            use_cache=False,
                            memory_text_input_ids=memory_text["input_ids"].to(device),
                            memory_text_attention_mask=torch.ones_like(
                                memory_text["input_ids"], device=device
                            ),
                            memory_key_input_ids=memory_key_input_ids,
                            memory_key_attention_mask=memory_key_attention_mask.to(device),
                            memory_storage_input_ids=memory_storage["input_ids"].to(device),
                            memory_storage_attention_mask=memory_storage.get(
                                "attention_mask",
                                torch.ones_like(memory_storage["input_ids"]),
                            ).to(device),
                        )
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    update_memory=False,
                    memory_query_input_ids=memory_query_input_ids,
                    memory_query_attention_mask=memory_query_attention_mask,
                )
            response_ids = output_ids[0, encoded["input_ids"].shape[1] :]
            print(f"AI> {tokenizer.decode(response_ids, skip_special_tokens=True)}")
            save_state()
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        save_state()


if __name__ == "__main__":
    main()
