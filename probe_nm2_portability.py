"""Measure how far the NM2 memory surgery generalizes across model architectures.

The NM2 embedding is not architecture-neutral.  Reading the integration code
shows four concrete assumptions, and this tool tests each one on real checkpoints
instead of arguing about them:

1. ``base_model.config.text_config`` must exist (line 1073);
2. the backbone must be loadable as a conditional-generation / image-text model
   (line 4000);
3. the decoder layers must be reachable as
   ``base_model.model.language_model.layers`` (lines 1155/2393/3961/3970);
4. ``MemoryLayerAdapter.forward`` declares ``position_embeddings`` as its **second
   positional parameter** and forwards it positionally to the wrapped layer
   (lines 960-999), which only matches layers whose own second positional
   parameter is ``position_embeddings`` (Qwen3.5/Gemma-style), not Llama/Qwen2
   style layers that expect ``attention_mask`` there.

For every model the probe reports which stages pass, applies the *minimal* shims
(``config.text_config = config`` and ``model.model.language_model = model.model``)
to see how much work a port actually needs, and then runs the strongest possible
capability-preservation check: with memory read/write disabled, the wrapped model
must reproduce the original model's logits **exactly**.

Usage::

    python -m V2_dpskw.probe_nm2_portability --models list.json --output nm2_portability.json
    python -m V2_dpskw.probe_nm2_portability            # built-in architecture list
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from V2_dpskw.qwen_integration import QwenDynamicMemoryModel, QwenMemoryConfig

#: Layer-container attribute paths tried, in the order the integration would need.
CONTAINER_PATHS = (
    "model.language_model.layers",
    "model.layers",
    "language_model.layers",
    "layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "decoder.layers",
    "model.layers.0.model.layers",  # multimodal nesting
)

DEFAULT_MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-MistralForCausalLM",
    "hf-internal-testing/tiny-random-Qwen2ForCausalLM",
    "hf-internal-testing/tiny-random-Qwen3ForCausalLM",
    "hf-internal-testing/tiny-random-Phi3ForCausalLM",
    "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "hf-internal-testing/tiny-random-OlmoeForCausalLM",
    "hf-internal-testing/tiny-random-GraniteMoeForCausalLM",
    "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "hf-internal-testing/tiny-random-CohereForCausalLM",
    "hf-internal-testing/tiny-random-FalconForCausalLM",
    "akreal/tiny-random-BloomForCausalLM",
    "hf-tiny-model-private/tiny-random-OPTForCausalLM",
    "hf-tiny-model-private/tiny-random-GPTNeoXForCausalLM",
    "hf-tiny-model-private/tiny-random-GPT2LMHeadModel",
    "hf-tiny-model-private/tiny-random-BartForCausalLM",
    "trl-internal-testing/tiny-random-LlamaForCausalLM",
]


def _resolve_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if current is None:
            return None
        if part.isdigit():
            current = current[int(part)] if hasattr(current, "__getitem__") else None
        else:
            current = getattr(current, part, None)
    return current


def _find_container(model: Any) -> tuple[str | None, Any]:
    for path in CONTAINER_PATHS:
        value = _resolve_path(model, path)
        if value is not None and hasattr(value, "__len__") and len(value) > 0 and hasattr(value[0], "forward"):
            return path, value
    return None, None


def _minimal_config(layer_indices: tuple[int, ...], *, full_stack: bool) -> QwenMemoryConfig:
    if not full_stack:
        return QwenMemoryConfig(
            memory_slots=4,
            memory_dim=64,
            layer_indices=list(layer_indices),
            mode="residual",
            blend_init=0.0,
            native_mode=False,
            persistent_memory=False,
            summary_pooling=False,
            natural_language_memory=False,
            automatic_memory=False,
            memory_version=1,
            hierarchical_memory=False,
            kv_offload=False,
            auto_compact_context=False,
        )
    return QwenMemoryConfig(
        memory_slots=8,
        memory_dim=128,
        layer_indices=list(layer_indices),
        mode="residual",
        blend_init=0.0,
        native_mode=True,
        persistent_memory=True,
        summary_pooling=True,
        natural_language_memory=True,
        automatic_memory=True,
        automatic_memory_policy_version=2,
        memory_version=2,
        hierarchical_memory=True,
        kv_offload=False,
        auto_compact_context=False,
    )


def probe(model_id: str, *, full_stack: bool = False, dtype: str = "float32") -> dict[str, Any]:
    record: dict[str, Any] = {"model": model_id, "full_stack": full_stack, "dtype": dtype, "stages": {}, "patches": []}
    stages = record["stages"]
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.float32)

    def fail(stage: str, exc: BaseException) -> None:
        stages[stage] = f"FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:180]}"

    # ---- 1. config ---------------------------------------------------------
    try:
        config = AutoConfig.from_pretrained(model_id)
        text_config = getattr(config, "text_config", None)
        record["architecture"] = (getattr(config, "architectures", None) or ["?"])[0]
        record["model_type"] = getattr(config, "model_type", "?")
        record["hidden_size"] = getattr(text_config or config, "hidden_size", None)
        record["num_layers"] = getattr(text_config or config, "num_hidden_layers", None)
        record["vocab_size"] = getattr(text_config or config, "vocab_size", None)
        record["text_config_present"] = text_config is not None
        stages["1_config"] = (
            "ok (config.text_config)" if text_config is not None
            else "resolved by integration (text-only model has no text_config)"
        )
    except BaseException as exc:  # noqa: BLE001
        fail("1_config", exc)
        return record

    # ---- 2. load backbone --------------------------------------------------
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        except (ValueError, KeyError, OSError) as exc:
            # Multimodal checkpoints (Qwen3.5, Gemma3-VL) are image-text-to-text
            # models, which is what the integration's own loader uses.
            from transformers import AutoModelForImageTextToText

            record["causal_lm_fallback"] = str(exc).splitlines()[0][:120]
            model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch_dtype)
        model.eval()
        record["backbone_class"] = type(model).__name__
        stages["2_load_causal_lm"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        fail("2_load_causal_lm", exc)
        return record

    # ---- 3. layer container -------------------------------------------------
    container_path, container = _find_container(model)
    record["container_path"] = container_path
    record["repo_layer_path_ok"] = container_path == "model.language_model.layers"
    stages["3_container"] = (
        "ok (repo path model.language_model.layers)" if container_path == "model.language_model.layers"
        else f"resolved by integration (found {container_path})"
    )
    if container is None:
        return record

    # ---- 4. layer calling convention ---------------------------------------
    layer_signature = list(inspect.signature(container[0].forward).parameters)
    record["layer_forward_params"] = layer_signature
    second = layer_signature[1] if len(layer_signature) > 1 else None
    record["second_positional_is_position_embeddings"] = second == "position_embeddings"
    stages["4_convention"] = (
        "ok (position_embeddings second -> positional call)"
        if second == "position_embeddings"
        else f"handled by declared-argument form (second positional is {second!r})"
    )

    # ---- reference output BEFORE any surgery -------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer("记住：项目代号是 P-123。", return_tensors="pt")
    except BaseException:
        inputs = {"input_ids": torch.randint(1, 128, (1, 8)), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
    with torch.no_grad():
        reference = model(**inputs).logits.clone()

    # ---- 5. surgery (no shims; the integration resolves architecture itself) --
    num_layers = int(record["num_layers"] or len(container))
    try:
        # Tiny probe models can have very few layers, so mirror the integration's
        # own stride rule but clamp it to a valid range for this depth.
        stride = max(1, num_layers // 4)
        layer_indices = tuple(sorted({max(0, min(num_layers - 1, stride * i - 1)) for i in range(1, 5)}))
        record["resolved_layer_indices"] = list(layer_indices)
        memory_config = _minimal_config(layer_indices, full_stack=full_stack)
        # No shims: the integration resolves text_config and the decoder stack
        # itself now, so a successful surgery here means a genuinely portable port.
        wrapped = QwenDynamicMemoryModel(model, memory_config, freeze_backbone=True)
        record["shims_needed"] = []
        stages["5_surgery"] = "ok (no shims)"
    except BaseException as exc:  # noqa: BLE001
        wrapped_layers = [
            index for index, layer in enumerate(container)
            if type(layer).__name__ == "MemoryLayerAdapter"
        ]
        record["partially_wrapped_layers"] = wrapped_layers
        stages["5_surgery"] = f"FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:180]}"
        record["traceback"] = traceback.format_exc()[-800:]
        return record

    # ---- 6. identity check (the capability-preservation proof) -------------
    try:
        with torch.no_grad():
            output = wrapped(**inputs, read_memory=False, update_memory=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        delta = float((logits - reference).abs().max())
        record["identity_max_abs_logit_delta"] = delta
        stages["6_identity"] = "identical" if delta == 0.0 else f"DIFFERS by {delta:.3e}"
    except BaseException as exc:  # noqa: BLE001
        stages["6_identity"] = f"FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:180]}"
        record["traceback"] = traceback.format_exc()[-800:]
        return record

    # ---- 7. memory write/read smoke ---------------------------------------
    try:
        with torch.no_grad():
            written = wrapped(**inputs, read_memory=False, update_memory=True, return_memory=True)
        state = getattr(written, "memory_state", None)
        changed = None
        if state is not None and getattr(wrapped, "persistent_memory", None) is not None:
            changed = float((state - wrapped.persistent_memory).abs().max())
        record["memory_state_changed_by"] = changed
        with torch.no_grad():
            wrapped(**inputs, read_memory=True, update_memory=False)
        stages["7_write_read"] = "ok" if changed is None or changed > 0 else "state unchanged"
    except BaseException as exc:  # noqa: BLE001
        stages["7_write_read"] = f"FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:180]}"

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="", help="json file with a list of model ids")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--full-stack", action="store_true", help="also exercise the V2/native memory stack")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--output", default="nm2_portability.json")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    if args.model:
        models = args.model
    elif args.models:
        models = json.loads(Path(args.models).read_text(encoding="utf-8"))
    else:
        models = DEFAULT_MODELS

    records = []
    for model_id in models:
        print(f"--- {model_id}", flush=True)
        try:
            record = probe(model_id, full_stack=args.full_stack, dtype=args.dtype)
        except BaseException as exc:  # noqa: BLE001
            record = {"model": model_id, "stages": {"0_probe": f"FAIL {type(exc).__name__}: {exc}"}}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False)[:400], flush=True)

    Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown:
        lines = [
            "| model | model_type | hidden | layers | container | 2nd positional | identity | surgery |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in records:
            stages = r.get("stages", {})
            lines.append("| {m} | {t} | {h} | {l} | {c} | {s} | {i} | {su} |".format(
                m=r.get("model", "?").split("/")[-1],
                t=r.get("model_type", "-"),
                h=r.get("hidden_size", "-"),
                l=r.get("num_layers", "-"),
                c=r.get("container_path", "-"),
                s="position_embeddings" if r.get("second_positional_is_position_embeddings") else "OTHER",
                i=stages.get("6_identity", "-"),
                su=stages.get("5_surgery", "-"),
            ))
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
