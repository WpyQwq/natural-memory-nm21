"""Streaming chat for natural-language memory with restart-safe autosave.

Every turn is encoded independently.  The model's internal reader decides
whether a saved memory prefix is relevant; this script never reconstructs
conversation history.  A user memory state is atomically saved before the
streaming generation starts, so restarting the process is safe at any time.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
from pathlib import Path

import torch

from .qwen_integration import (
    DEFAULT_MEMORY_RESET_TOKEN,
    QwenMemoryConfig,
    format_memory_evidence,
    infer_memory_metadata,
    load_memory_config,
    load_qwen_dynamic,
    load_tokenizer,
    resolve_memory_reset_token,
    split_memory_candidates,
)


def _memory_system_prefix(tokenizer, content: str) -> dict[str, torch.Tensor]:
    """Encode a valid system prefix without adding a fake user question."""

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


def _chat_tensor(tokenizer, user_text: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    return {
        key: value
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def _atomic_save(model, path: Path) -> None:
    """Save one user's state without exposing a partially written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        model.save_runtime_memory(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _persist_memory(
    model,
    *,
    embedded_dir: Path | None,
    state_path: Path | None,
) -> None:
    """Persist either into the merged shard or into a normal runtime file."""

    if embedded_dir is not None:
        if getattr(model.memory_config, "memory_storage_mode", "embedded") == "tiered":
            model.flush_memory_storage()
            return
        model.save_embedded_memory_weights(embedded_dir)
        return
    if state_path is None:
        raise ValueError("no persistence target is configured")
    _atomic_save(model, state_path)


def _slot_count(model) -> int:
    valid = model.runtime.text_slot_valid
    return int(valid.sum().item()) if isinstance(valid, torch.Tensor) else 0


@torch.inference_mode()
def _write_turn(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    force_write: bool = False,
) -> bool:
    """Run the learned write controller for one user turn."""
    changed = False
    for candidate in split_memory_candidates(text):
        encoded = _chat_tensor(tokenizer, candidate)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        metadata = infer_memory_metadata(candidate)
        evidence_text = format_memory_evidence(
            candidate,
            entity=str(metadata.get("entity", "")),
            attribute=str(metadata.get("attribute", "")),
            value=str(metadata.get("value", "")),
        )
        memory_prefix = _memory_system_prefix(
            tokenizer,
            "以下是与当前用户相关的已保存长期记忆。仅在问题相关时使用，"
            "只能依据明确证据；先核对实体、属性和已确认值；冲突优先最新可靠来源，"
            "不要拼接不确定候选，证据不足就明确说不知道；涉及名称、路径、token、"
            "参数或结论时，原样复述证据中的关键短语：\n"
            + evidence_text,
        )
        memory_text_ids = memory_prefix["input_ids"].to(device)
        memory_text_mask = memory_prefix["attention_mask"].to(device)
        memory_key = tokenizer(candidate, add_special_tokens=False, return_tensors="pt")
        memory_key_ids = memory_key["input_ids"].to(device)
        memory_key_mask = memory_key.get("attention_mask")
        if memory_key_mask is None:
            memory_key_mask = torch.ones_like(memory_key_ids)
        memory_key_mask = memory_key_mask.to(device)
        memory_storage = tokenizer(
            candidate,
            add_special_tokens=False,
            return_tensors="pt",
        )
        memory_storage_ids = memory_storage["input_ids"].to(device)
        memory_storage_mask = memory_storage.get("attention_mask")
        if memory_storage_mask is None:
            memory_storage_mask = torch.ones_like(memory_storage_ids)
        memory_storage_mask = memory_storage_mask.to(device)
        model(
            **encoded,
            # The write controller must see only the current user turn.  If
            # it reads the already-retrieved prefix first, the policy can
            # mistake recalled facts for a new fact and write questions back.
            read_memory=False,
            update_memory=True,
            return_memory=True,
            use_cache=False,
            memory_text_input_ids=memory_text_ids,
            memory_text_attention_mask=memory_text_mask,
            memory_key_input_ids=memory_key_ids,
            memory_key_attention_mask=memory_key_mask,
            memory_storage_input_ids=memory_storage_ids,
            memory_storage_attention_mask=memory_storage_mask,
            force_memory_write=force_write,
            memory_text=candidate,
        )
        last_written = model.runtime.text_last_written_slot
        if isinstance(last_written, torch.Tensor):
            changed = changed or bool((last_written >= 0).any())
    return changed


def _stream_answer(
    model,
    tokenizer,
    encoded: dict[str, torch.Tensor],
    max_new_tokens: int,
    memory_query_input_ids: torch.Tensor,
    memory_query_attention_mask: torch.Tensor,
    memory_query_text: str,
) -> None:
    """Stream one answer through TextIteratorStreamer in a worker thread."""

    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    errors: list[BaseException] = []
    stop_event = threading.Event()

    class StopOnEvent(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full(
                (input_ids.shape[0],),
                stop_event.is_set(),
                dtype=torch.bool,
                device=input_ids.device,
            )

    def worker() -> None:
        try:
            model.generate(
                **encoded,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                update_memory=False,
                memory_query_input_ids=memory_query_input_ids,
                memory_query_attention_mask=memory_query_attention_mask,
                memory_query_text=memory_query_text,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnEvent()]),
            )
        except BaseException as error:  # propagate through the main thread
            errors.append(error)
            streamer.on_finalized_text("", stream_end=True)

    thread = threading.Thread(target=worker, name="qwen-stream-generation", daemon=True)
    thread.start()
    interrupted = False
    try:
        for chunk in streamer:
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        print("\n[已中断生成；最近一次 memory state 已保存，可直接重启]", flush=True)
    finally:
        thread.join(timeout=10.0)
    if thread.is_alive():
        raise RuntimeError("generation did not stop after interrupt; refusing concurrent state save")
    if interrupted:
        raise KeyboardInterrupt
    if errors:
        raise RuntimeError("streaming generation failed") from errors[0]
    print()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".")
    parser.add_argument(
        "--adapter",
        default=None,
        help="external adapter; omitted means use an embedded merge package or v13",
    )
    parser.add_argument(
        "--memory-state",
        default=None,
        help="optional external state file; omitted for merged models means write back to the memory shard",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--natural-language-memory", action="store_true")
    parser.add_argument("--reset-token", default=None)
    parser.add_argument("--reset-token-id", type=int, default=None)
    parser.add_argument(
        "--kv-offload",
        action="store_true",
        help="keep generation KV on CPU when supported by the installed Transformers",
    )
    parser.add_argument(
        "--kv-cache-implementation",
        default=None,
        help="optional Transformers cache implementation, for example offloaded",
    )
    parser.add_argument(
        "--no-auto-compact",
        action="store_true",
        help="disable model-owned old-context archiving when the hot KV budget is exceeded",
    )
    parser.add_argument(
        "--tiered-memory-path",
        default=None,
        help="optional SQLite page store for warm/cold memory; relative paths use the model package",
    )
    parser.add_argument(
        "--memory-resident-pages",
        type=int,
        default=None,
        help="maximum number of tiered pages kept resident",
    )
    args = parser.parse_args()

    if args.adapter is None and not (Path(args.model_path) / "memory_merge.json").exists():
        args.adapter = "V2_dpskw/qwen_memory_adapter_natural_auto_v13"

    embedded_dir = (
        Path(args.model_path)
        if (
            (Path(args.model_path) / "memory_merge.json").exists()
            and args.memory_state is None
            and args.adapter is None
        )
        else None
    )

    tokenizer = load_tokenizer(args.model_path)
    memory_config = load_memory_config(args.adapter) if args.adapter else None
    if args.tiered_memory_path is not None and memory_config is None:
        memory_config = load_memory_config(args.model_path)
    if args.tiered_memory_path is not None:
        memory_config.memory_storage_mode = "tiered"
        memory_config.memory_storage_path = args.tiered_memory_path
    if args.memory_resident_pages is not None:
        if memory_config is None:
            memory_config = QwenMemoryConfig()
        memory_config.memory_resident_pages = args.memory_resident_pages
    if args.natural_language_memory and memory_config is None:
        memory_config = QwenMemoryConfig(natural_language_memory=True)
    if memory_config is not None and args.natural_language_memory:
        memory_config.natural_language_memory = True
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
    if memory_config is None:
        # An embedded merge package supplies its architecture metadata during
        # load_qwen_dynamic(); keep the CLI state machine in sync with it.
        memory_config = model.memory_config
        if memory_config.native_mode and memory_config.reset_token_id is None:
            memory_config.reset_token_id = resolve_memory_reset_token(tokenizer)
    model.configure_memory_grounding_guard(tokenizer)
    if args.kv_offload:
        model.memory_config.kv_offload = True
    if args.kv_cache_implementation is not None:
        model.memory_config.kv_cache_implementation = args.kv_cache_implementation
    if args.no_auto_compact:
        model.memory_config.auto_compact_context = False
    if args.adapter:
        model.load_memory_adapter(args.adapter)
    model.eval()
    device = model._find_layer_device()
    state_path = (
        Path(args.memory_state)
        if args.memory_state
        else (None if embedded_dir is not None else Path("V2_dpskw/data/stream_user_memory.pt"))
    )
    if state_path is not None and state_path.exists():
        model.load_runtime_memory(state_path, device=device)
        print(f"已恢复 memory state：{state_path}，有效记忆槽 {_slot_count(model)} 个")
    elif model.memory_config.persistent_memory and model.runtime.state is not None:
        print(f"已使用合并权重内置的 memory state，有效记忆槽 {_slot_count(model)} 个")
    else:
        model.reset_memory(batch_size=1, device=device)
        if state_path is None:
            print("新建 memory state：合并权重回写模式")
        else:
            print(f"新建 memory state：{state_path}")

    reset_token = args.reset_token or DEFAULT_MEMORY_RESET_TOKEN
    print("流式聊天已启动。不会发送历史聊天记录。")
    print("命令：/remember <事实>、/reset、/save、/quit；也可发送 reset token。")
    print(f"reset token：{reset_token}")

    try:
        while True:
            try:
                user_text = input("你> ").strip()
            except EOFError:
                break
            if user_text == "/quit":
                break
            if user_text == "/save":
                _persist_memory(model, embedded_dir=embedded_dir, state_path=state_path)
                target = "主权重切片" if embedded_dir is not None else str(state_path)
                print(f"已保存到 {target}，当前有效记忆槽 {_slot_count(model)} 个")
                continue
            if user_text == "/reset":
                model.reset_memory(batch_size=1, device=device)
                _persist_memory(model, embedded_dir=embedded_dir, state_path=state_path)
                print("已清空并保存。")
                continue
            if user_text.startswith("/remember "):
                fact = user_text[len("/remember ") :].strip()
                if fact:
                    _write_turn(model, tokenizer, fact, device, force_write=True)
                    _persist_memory(model, embedded_dir=embedded_dir, state_path=state_path)
                    target = "主权重切片" if embedded_dir is not None else str(state_path)
                    print(f"已写入并保存到 {target}，当前有效记忆槽 {_slot_count(model)} 个")
                continue
            if not user_text:
                continue

            encoded = _chat_tensor(tokenizer, user_text)
            encoded = {key: value.to(device) for key, value in encoded.items()}
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
            if (
                memory_config is not None
                and memory_config.reset_token_id is not None
                and bool((encoded["input_ids"] == memory_config.reset_token_id).any())
            ):
                # Persist the clear operation before generation, so an
                # interrupted response cannot resurrect the old memory.
                model.reset_memory(batch_size=1, device=device)
                _persist_memory(model, embedded_dir=embedded_dir, state_path=state_path)
            elif memory_config is not None and memory_config.native_mode:
                # The controller sees the current turn only.  Save before
                # generation; generation itself is read-only.
                changed = _write_turn(model, tokenizer, user_text, device)
                if changed:
                    _persist_memory(model, embedded_dir=embedded_dir, state_path=state_path)

            print("AI> ", end="", flush=True)
            _stream_answer(
                model,
                tokenizer,
                encoded,
                args.max_new_tokens,
                memory_query_input_ids,
                memory_query_attention_mask,
                user_text,
            )
    except KeyboardInterrupt:
        print("\n[已退出；最近一次记忆已保存，可直接重启]", flush=True)
    finally:
        model.close_memory_storage()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
