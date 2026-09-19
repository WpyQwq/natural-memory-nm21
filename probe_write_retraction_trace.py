"""READ-ONLY probe: which condition retires 8 of 20 freshly written facts?

Symptom (already established, not re-derived here): writing the 20 facts of
``eval_router_critical_e2e.build_cases(...)["bank"]`` into a fresh V2 bank leaves
``records_in_bank == 20`` but only ``active_records == 12`` -- 8 records are
already non-active before any query runs, deterministically.

This probe answers *which code path and which condition* does it, by observing
the real production code instead of re-implementing it:

1. ``PagedMemoryBankV2.retract`` (the single sink behind
   ``MemoryOSV2.retract_record``) is wrapped: every call records the full
   ``traceback`` stack, the record's text/attribute/status/slot and whether it
   happened inside a write turn.
2. ``PagedMemoryBankV2.write`` is wrapped: every call records the record text,
   the ``slot_index`` actually passed (i.e. ``v2_slot_index``), the conflict key,
   the action returned ("inserted"/"updated"/"duplicate"/"quarantined") and the
   resulting record id/status.
3. ``QwenDynamicMemoryModel._write_text_memory`` is wrapped: per write turn it
   snapshots every record status before/after, so *any* status transition
   (``retracted`` AND ``superseded`` AND ``quarantined``) is attributed to a
   concrete write index, whichever mechanism caused it.
4. ``sys.settrace`` is installed only around the write loop and returns a local
   trace function *only* for the ``_write_text_memory`` frame, so the exact
   locals at the guard lines are captured without editing any source file and
   without paying line-trace cost inside the model forward pass.  Captured
   lines: 2137 (``confirmed_update`` seed), 2147-2155 (guard result + chosen
   slot), 2183-2195 (``v2_slot_index`` and the retract loop).
5. The packaged ``text_retriever`` is replaced in memory by a logging proxy that
   delegates to the original module and records, per call, the max/min sigmoid
   score, how many candidates scored >= 0.95 and the top scores -- tagged with
   the write turn it happened in.

Nothing is edited on disk and no weight is modified; the retriever is wrapped
(a submodule swap on the loaded model only).

Usage::

    $env:PYTHONPATH='H:\\Memory'; $env:PYTHONIOENCODING='utf-8'
    & $py -m V2_dpskw.probe_write_retraction_trace --repeat 2
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import V2_dpskw.memory_os_v2 as mos
import V2_dpskw.qwen_integration as qi
from V2_dpskw.eval_end_to_end_memory import write_fact
from V2_dpskw.eval_router_critical_e2e import build_cases
from V2_dpskw.qwen_integration import (
    format_memory_evidence,
    infer_memory_metadata,
    load_memory_config,
    load_qwen_dynamic,
    load_tokenizer,
)

TARGET_FILE_SUFFIX = "qwen_integration.py"
TARGET_FUNC = "_write_text_memory"

#: line -> local names worth capturing at that line event
WATCH_LINES: dict[int, tuple[str, ...]] = {
    2137: (
        "exact_slots", "learned_best", "learned_slot", "lexical_similarity",
        "valid_slots", "best_slot", "best_similarity", "key_length",
        "valid_key_ids", "old_key_ids", "old_key_mask", "similarities",
    ),
    2147: ("confirmed_update", "learned_best", "lexical_similarity", "exact_slots"),
    2148: ("confirmed_update", "best_slot", "slot"),
    2155: ("slot", "confirmed_update", "valid_slots"),
    2178: ("slot", "confirmed_update", "valid_slots"),
    2183: ("v2_slot_index", "slot", "confirmed_update"),
    2187: ("v2_slot_index", "slot", "record", "score", "shared"),
    2191: ("record", "score", "shared", "slot"),
    2195: ("record", "score", "shared", "slot"),
}

CURRENT: dict[str, object] = {"write_index": 0, "phase": "setup", "repetition": 0}
WRITE_CALLS: list[dict] = []
RETRACTIONS: list[dict] = []
WTM_CALLS: list[dict] = []
RETRIEVER_CALLS: list[dict] = []
TRACE: list[dict] = []
TRACE_ERRORS: list[str] = []
TOKENIZER = None


# ---------------------------------------------------------------------------
# summarisation helpers (never keep live tensors/graphs in the report)
# ---------------------------------------------------------------------------


def _summarize(value):
    if isinstance(value, torch.Tensor):
        det = value.detach()
        out: dict = {"shape": list(det.shape), "dtype": str(det.dtype)}
        if det.dtype == torch.bool:
            out["true"] = int(det.sum().item())
            return out
        if det.dtype in (torch.int32, torch.int64):
            flat = det.reshape(-1)
            if flat.numel() <= 160:
                out["values"] = [int(x) for x in flat]
            else:
                out["numel"] = int(flat.numel())
            return out
        flat = det.reshape(-1).float()
        if flat.numel() == 0:
            out["numel"] = 0
            return out
        out["min"] = round(float(flat.min().item()), 6)
        out["max"] = round(float(flat.max().item()), 6)
        out["n_ge_0.95"] = int((flat >= 0.95).sum().item())
        out["n_ge_0.65"] = int((flat >= 0.65).sum().item())
        if flat.numel() <= 20:
            out["values"] = [round(float(x), 6) for x in flat]
        return out
    if isinstance(value, mos.MemoryRecordV2):
        return {
            "record_id": value.record_id,
            "status": value.status,
            "slot_index": value.slot_index,
            "conflict_key": value.conflict_key(),
            "attribute": value.attribute,
            "text": value.text[:90].replace("\n", " "),
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:160]
    if isinstance(value, (list, tuple)):
        return [_summarize(item) for item in list(value)[:8]]
    return {"repr": repr(value)[:160]}


def _manual_exact_match(snapshot_locals: dict) -> dict:
    """Independently evaluate the ``exact_match`` arithmetic of line 2093-2099."""

    valid_key_ids = snapshot_locals.get("valid_key_ids")
    old_key_ids = snapshot_locals.get("old_key_ids")
    old_key_mask = snapshot_locals.get("old_key_mask")
    valid_slots = snapshot_locals.get("valid_slots")
    if not all(isinstance(item, torch.Tensor) for item in (valid_key_ids, old_key_ids, old_key_mask, valid_slots)):
        return {}
    key_length = int(valid_key_ids.numel())
    slots: list[int] = []
    same_length_slots: list[int] = []
    for slot in range(int(old_key_mask.shape[0])):
        if not bool(valid_slots[slot].item()):
            continue
        old_length = int(old_key_mask[slot].sum().item())
        if old_length == key_length:
            same_length_slots.append(slot)
            if key_length > 0 and bool((old_key_ids[slot, :key_length] == valid_key_ids).all().item()):
                slots.append(slot)
    return {
        "key_length": key_length,
        "valid_slot_count": int(valid_slots.sum().item()),
        "slots_with_equal_old_key_length": same_length_slots,
        "slots_with_identical_tokens": slots,
        "exact_slots_numel": len(slots),
    }


def _snapshot(frame, lineno: int, names: tuple[str, ...]) -> dict:
    row: dict = {
        "write": CURRENT["write_index"],
        "repetition": CURRENT["repetition"],
        "line": lineno,
    }
    for name in names:
        if name in frame.f_locals:
            row[name] = _summarize(frame.f_locals[name])
    if lineno == 2137:
        row["manual_exact_match"] = _manual_exact_match(frame.f_locals)
    return row


# ---------------------------------------------------------------------------
# trace hook: armed only for the _write_text_memory frame
# ---------------------------------------------------------------------------


def _local_trace(frame, event, arg):
    if event == "line":
        names = WATCH_LINES.get(frame.f_lineno)
        if names is not None:
            try:
                TRACE.append(_snapshot(frame, frame.f_lineno, names))
            except Exception as exc:  # never break the traced code
                TRACE_ERRORS.append(repr(exc))
        return _local_trace
    if event == "return":
        return None
    return _local_trace


def _global_trace(frame, event, arg):
    if event == "call":
        if frame.f_code.co_name == TARGET_FUNC and frame.f_code.co_filename.endswith(TARGET_FILE_SUFFIX):
            return _local_trace
        return None
    return None


# ---------------------------------------------------------------------------
# production-code proxies
# ---------------------------------------------------------------------------

_ORIG_BANK_RETRACT = mos.PagedMemoryBankV2.retract


def _logging_bank_retract(self, record_id):
    record = self.records.get(record_id)
    stack = traceback.extract_stack()
    frames = [
        f"{Path(item.filename).name}:{item.lineno}:{item.name}"
        for item in stack
        if "V2_dpskw" in item.filename or "dynamic_memory_lab" in item.filename
    ]
    RETRACTIONS.append({
        "repetition": CURRENT["repetition"],
        "write_index": CURRENT["write_index"],
        "phase": CURRENT["phase"],
        "record_id": record_id,
        "record_text": (record.text[:120].replace("\n", " ") if record is not None else None),
        "record_attribute": (record.attribute if record is not None else None),
        "record_slot_index": (record.slot_index if record is not None else None),
        "record_status_before": (record.status if record is not None else None),
        "stack_callsite": [line for line in frames if "retract" in line][-3:],
        "stack_tail": frames[-8:],
        "stack_full": frames,
    })
    return _ORIG_BANK_RETRACT(self, record_id)


_ORIG_BANK_WRITE = mos.PagedMemoryBankV2.write


def _logging_bank_write(self, **kwargs):
    record, action = _ORIG_BANK_WRITE(self, **kwargs)
    WRITE_CALLS.append({
        "repetition": CURRENT["repetition"],
        "write_index": CURRENT["write_index"],
        "phase": CURRENT["phase"],
        "text": str(kwargs.get("text", ""))[:60].replace("\n", " "),
        "passed_slot_index": int(kwargs.get("slot_index", -1)),
        "entity": kwargs.get("entity"),
        "attribute": kwargs.get("attribute"),
        "value": kwargs.get("value"),
        "source": kwargs.get("source"),
        "trusted": kwargs.get("trusted"),
        "action": action,
        "record_id": record.record_id,
        "record_slot_index": record.slot_index,
        "record_status": record.status,
        "record_conflict_key": record.conflict_key(),
        "record_supersedes": record.supersedes,
        "record_version": record.version,
    })
    return record, action


_ORIG_WRITE_TEXT_MEMORY = qi.QwenDynamicMemoryModel._write_text_memory
_WTM_SIGNATURE = inspect.signature(_ORIG_WRITE_TEXT_MEMORY)


def _decode(ids, mask=None) -> str:
    if TOKENIZER is None or ids is None:
        return ""
    flat = ids.detach().reshape(-1).cpu()
    if mask is not None:
        keep = mask.detach().reshape(-1).cpu().bool()
        flat = flat[keep]
    try:
        return TOKENIZER.decode(flat.tolist())
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"<decode failed: {exc!r}>"


def _slot_table(model) -> list[dict]:
    runtime = model.runtime
    if runtime.text_slot_valid is None:
        return []
    valid = runtime.text_slot_valid[0]
    key_ids = runtime.text_key_token_ids[0]
    key_mask = runtime.text_key_token_mask[0]
    rows = []
    for slot in range(int(valid.shape[0])):
        length = int(key_mask[slot].sum().item())
        rows.append({
            "slot": slot,
            "valid": bool(valid[slot].item()),
            "stored_key_token_count": length,
            "age": int(runtime.text_slot_age[0][slot].item()),
            "stored_key_text": _decode(key_ids[slot], key_mask[slot]),
        })
    return rows


def _logging_write_text_memory(self, *args, **kwargs):
    index = int(CURRENT["write_index"]) + 1
    CURRENT["write_index"] = index
    CURRENT["phase"] = "write"
    try:
        bound = _WTM_SIGNATURE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
    except Exception:
        arguments = {}

    bank = self.memory_os_v2.bank if self.memory_os_v2 is not None else None
    pre = {rid: record.status for rid, record in bank.records.items()} if bank is not None else {}
    key_ids = arguments.get("key_input_ids")
    storage_ids = arguments.get("storage_input_ids")
    memory_text = arguments.get("memory_text")
    runtime = self.runtime
    row: dict = {
        "repetition": CURRENT["repetition"],
        "write_index": index,
        "memory_text": memory_text,
        "force_write": arguments.get("force_write"),
        "text_retriever_ready": bool(getattr(self, "_text_retriever_ready", False)),
        "slot_valid_before": (int(runtime.text_slot_valid[0].sum().item())
                              if runtime.text_slot_valid is not None else None),
        "slot_ages_before": ([int(x) for x in runtime.text_slot_age[0]]
                             if runtime.text_slot_age is not None else None),
        "write_counter_before": (int(runtime.text_write_counter[0].item())
                                 if runtime.text_write_counter is not None else None),
        "key_input_ids_shape": (list(key_ids.shape) if isinstance(key_ids, torch.Tensor) else None),
        "storage_input_ids_shape": (list(storage_ids.shape) if isinstance(storage_ids, torch.Tensor) else None),
        "key_ids_decoded": _decode(key_ids),
        "storage_ids_decoded": _decode(storage_ids),
        "metadata": safe_metadata(memory_text),
        "records_before": len(pre),
        "trace": [],
        "trace_cursor": len(TRACE),
    }
    if isinstance(memory_text, str):
        meta = infer_memory_metadata(memory_text)
        row["evidence_card_that_would_be_stored"] = format_memory_evidence(
            memory_text, entity=str(meta.get("entity", "")),
            attribute=str(meta.get("attribute", "")), value=str(meta.get("value", "")),
        )[:120]

    _ORIG_WRITE_TEXT_MEMORY(self, *args, **kwargs)

    post = {rid: record.status for rid, record in bank.records.items()} if bank is not None else {}
    transitions = [
        {
            "record_id": rid,
            "from": pre[rid],
            "to": post[rid],
            "text": bank.records[rid].text[:70].replace("\n", " ")
            if rid in bank.records else "",
            "slot_index": bank.records[rid].slot_index if rid in bank.records else None,
        }
        for rid in post
        if rid in pre and pre[rid] != post[rid]
    ]
    added = [rid for rid in post if rid not in pre]
    row.update({
        "records_after": len(post),
        "status_transitions_caused_by_this_write": transitions,
        "records_added_by_this_write": [
            {
                "record_id": rid,
                "slot_index": bank.records[rid].slot_index,
                "status": bank.records[rid].status,
                "conflict_key": bank.records[rid].conflict_key(),
                "attribute": bank.records[rid].attribute,
                "value": bank.records[rid].value,
                "supersedes": bank.records[rid].supersedes,
                "text": bank.records[rid].text[:70].replace("\n", " "),
            }
            for rid in added
        ],
        "last_written_slot": (int(runtime.text_last_written_slot[0].item())
                              if runtime.text_last_written_slot is not None else None),
        "slot_table_after": _slot_table(self),
        "trace": TRACE[row.pop("trace_cursor"):],
    })
    CURRENT["phase"] = "idle"
    WTM_CALLS.append(row)
    return None


def safe_metadata(memory_text):
    if not isinstance(memory_text, str):
        return None
    try:
        return infer_memory_metadata(memory_text)
    except Exception as exc:
        return {"error": repr(exc)}


class _LoggingRetriever(torch.nn.Module):
    """Delegate to the packaged pair scorer and record every score it returns."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        out = self.inner(query, keys)
        try:
            with torch.no_grad():
                probs = torch.sigmoid(out.detach().float()).reshape(-1)
                top = probs.topk(min(6, probs.numel())).values if probs.numel() else probs
                RETRIEVER_CALLS.append({
                    "repetition": CURRENT["repetition"],
                    "write_index": CURRENT["write_index"],
                    "phase": CURRENT["phase"],
                    "query_shape": list(query.shape),
                    "keys_shape": list(keys.shape),
                    "candidates": int(probs.numel()),
                    "max": round(float(probs.max().item()), 6) if probs.numel() else None,
                    "min": round(float(probs.min().item()), 6) if probs.numel() else None,
                    "argmax": int(probs.argmax().item()) if probs.numel() else None,
                    "n_ge_0.95": int((probs >= 0.95).sum().item()),
                    "n_ge_0.90": int((probs >= 0.90).sum().item()),
                    "top": [round(float(x), 6) for x in top],
                })
        except Exception as exc:  # pragma: no cover
            RETRIEVER_CALLS.append({"error": repr(exc)})
        return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def pairwise_scores(model, tokenizer, facts: list[str], device, label: str) -> dict:
    """Score every unordered pair of ``facts`` with the packaged text_retriever.

    This is the same call shape the write path uses at line 2105-2110
    (``retriever(key_i[None], keys[None])``) and the same shape the earlier
    ``measure_write_path_updates.py`` measurement used, so the two fact sets are
    comparable.
    """

    encoded = tokenizer(facts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        keys = model._encode_model_key(encoded["input_ids"], encoded["attention_mask"]).reshape(len(facts), -1)
        matrix = torch.zeros(len(facts), len(facts))
        for index in range(len(facts)):
            row = torch.sigmoid(model.text_retriever(
                keys[index].reshape(1, -1), keys.unsqueeze(0)
            )).reshape(-1)
            matrix[index] = row.detach().float().cpu()
    pairs = []
    for left in range(len(facts)):
        for right in range(left + 1, len(facts)):
            pairs.append({
                "left": facts[left], "right": facts[right],
                "score": round(float(matrix[left, right].item()), 6),
            })
    scores = sorted(item["score"] for item in pairs)
    at_or_above = [item for item in pairs if item["score"] >= 0.95]
    return {
        "label": label,
        "facts": len(facts),
        "pairs": len(pairs),
        "max": scores[-1] if scores else None,
        "min": scores[0] if scores else None,
        "p50": scores[len(scores) // 2] if scores else None,
        "p90": scores[int(0.9 * len(scores))] if scores else None,
        "pairs_at_or_above_0.95": len(at_or_above),
        "worst_pairs": sorted(pairs, key=lambda item: item["score"], reverse=True)[:10],
        "pairs_ge_0.95": at_or_above,
    }


def _inventory(bank) -> list[dict]:
    return [
        {
            "record_id": record.record_id,
            "status": record.status,
            "slot_index": record.slot_index,
            "conflict_key": record.conflict_key(),
            "attribute": record.attribute,
            "value": record.value,
            "supersedes": record.supersedes,
            "version": record.version,
            "text": record.text[:60].replace("\n", " "),
        }
        for record in sorted(bank.records.values(), key=lambda item: item.timestamp)
    ]


def main() -> int:
    global TOKENIZER

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=2,
                        help="how many fresh banks to write the 20 facts into")
    parser.add_argument("--output", default="write_retraction_trace.json")
    args = parser.parse_args()

    cases = build_cases(args.cases, args.seed)
    facts = list(cases[0]["bank"])
    model_path = Path(args.package)
    print(json.dumps({"stage": "loading", "package": str(model_path.resolve()),
                      "facts": len(facts), "bank": facts}, ensure_ascii=False), flush=True)

    memory_config = load_memory_config(model_path)
    model = load_qwen_dynamic(model_path, memory_config=memory_config, load_in_4bit=True,
                              max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    TOKENIZER = load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ready = {
        "device": str(device),
        "memory_slots": memory_config.memory_slots,
        "text_memory_key_tokens": memory_config.text_memory_key_tokens,
        "text_memory_tokens": memory_config.text_memory_tokens,
        "text_memory_semantic_update_threshold": memory_config.text_memory_semantic_update_threshold,
        "text_memory_update_overlap_threshold": memory_config.text_memory_update_overlap_threshold,
        "text_memory_overlap_threshold": memory_config.text_memory_overlap_threshold,
        "text_retriever_ready": bool(model._text_retriever_ready),
        "memory_router_v2_ready": bool(model._memory_router_v2_ready),
        "text_retriever_params": sum(p.numel() for p in model.text_retriever.parameters()),
        "retriever_class": type(model.text_retriever).__name__,
    }
    print(json.dumps({"stage": "environment", **ready}, ensure_ascii=False, indent=2), flush=True)

    # --- install the read-only proxies -----------------------------------
    mos.PagedMemoryBankV2.retract = _logging_bank_retract
    mos.PagedMemoryBankV2.write = _logging_bank_write
    qi.QwenDynamicMemoryModel._write_text_memory = _logging_write_text_memory
    model.text_retriever = _LoggingRetriever(model.text_retriever).to(device).eval()

    # --- direct pairwise retriever measurement on both fact sets ----------
    # The earlier refutation of "the retriever is over-confident" measured the
    # 190 pairs of a *different* fact set ("我的<attr>是 VAL-A%07d。").  Repeat the
    # same measurement on the fact set the end-to-end protocol actually writes.
    from V2_dpskw.make_zero_overlap_paraphrase_data import ATTRIBUTE_PARAPHRASES

    comparison_facts = [
        "我的%s是 VAL-%s。" % (name, "A%07d" % index)
        for index, name in enumerate([name for name, _ in ATTRIBUTE_PARAPHRASES][: len(facts)])
    ]
    pairwise = [
        pairwise_scores(model, TOKENIZER, facts, device, "e2e_bank_20_facts"),
        pairwise_scores(model, TOKENIZER, comparison_facts, device, "previous_measurement_20_facts"),
    ]
    print(json.dumps({"stage": "pairwise", "pairwise": [
        {key: value for key, value in item.items() if key != "pairs_ge_0.95"}
        for item in pairwise]}, ensure_ascii=False, indent=2), flush=True)

    repetitions = []
    for repetition in range(1, args.repeat + 1):
        CURRENT.update({"repetition": repetition, "write_index": 0, "phase": "idle"})
        model.reset_memory(batch_size=1, device=device)
        sys.settrace(_global_trace)
        try:
            for fact in facts:
                write_fact(model, tokenizer=TOKENIZER, text=fact, device=device)
        finally:
            sys.settrace(None)
        inventory = _inventory(model.memory_os_v2.bank)
        non_active = [item for item in inventory if item["status"] != "active"]
        by_status: dict[str, int] = {}
        for item in inventory:
            by_status[item["status"]] = by_status.get(item["status"], 0) + 1
        fact_index = {fact: position for position, fact in enumerate(facts)}
        retired_facts = []
        for item in non_active:
            matched = next((fact for fact in facts if fact in item["text"]), None)
            retired_facts.append({
                "record_id": item["record_id"],
                "status": item["status"],
                "slot_index": item["slot_index"],
                "attribute": item["attribute"],
                "fact": matched,
                "fact_write_index": (fact_index[matched] + 1) if matched in fact_index else None,
            })
        repetitions.append({
            "repetition": repetition,
            "records_in_bank": len(inventory),
            "active_records": by_status.get("active", 0),
            "status_counts": by_status,
            "non_active_count": len(non_active),
            "retired_records": retired_facts,
            "inventory": inventory,
            "write_calls": len(WTM_CALLS),
            "retractions_in_this_repetition": sum(
                1 for item in RETRACTIONS if item["repetition"] == repetition),
        })
        print(json.dumps({
            "stage": "repetition_done", "repetition": repetition,
            "records_in_bank": len(inventory), "status_counts": by_status,
            "non_active": len(non_active),
            "retired": [(item["status"], item["fact_write_index"], item["attribute"])
                        for item in retired_facts],
        }, ensure_ascii=False), flush=True)

    # --- per-write summary ------------------------------------------------
    write_guard = []
    for row in WTM_CALLS:
        trace = row.get("trace") or []
        guard_line = next((item for item in trace if item.get("line") == 2147), {})
        slot_line = next((item for item in trace if item.get("line") == 2183), {})
        seed_line = next((item for item in trace if item.get("line") == 2137), {})
        write_guard.append({
            "repetition": row["repetition"],
            "write_index": row["write_index"],
            "memory_text": row["memory_text"],
            "metadata_attribute": (row.get("metadata") or {}).get("attribute"),
            "span": f"{row['records_before']}->{row['records_after']}",
            "slot_valid_before": row["slot_valid_before"],
            "last_written_slot": row["last_written_slot"],
            "exact_slots_numel": _summarize(seed_line.get("exact_slots")) if "exact_slots" in seed_line else None,
            "manual_exact_match": seed_line.get("manual_exact_match"),
            "learned_best": seed_line.get("learned_best"),
            "lexical_similarity": seed_line.get("lexical_similarity"),
            "best_similarity": seed_line.get("best_similarity"),
            "confirmed_update": guard_line.get("confirmed_update"),
            "slot_chosen": slot_line.get("slot"),
            "v2_slot_index": slot_line.get("v2_slot_index"),
            "bank_write_passed_slot_index": [
                item["passed_slot_index"] for item in WRITE_CALLS
                if item["repetition"] == row["repetition"] and item["write_index"] == row["write_index"]
            ],
            "bank_write_actions": [
                item["action"] for item in WRITE_CALLS
                if item["repetition"] == row["repetition"] and item["write_index"] == row["write_index"]
            ],
            "retract_calls": sum(
                1 for item in RETRACTIONS
                if item["repetition"] == row["repetition"] and item["write_index"] == row["write_index"]
                and item["phase"] == "write"),
            "status_transitions": row["status_transitions_caused_by_this_write"],
            "retriever_calls_during_write": [
                {key: value for key, value in item.items()
                 if key in {"candidates", "max", "min", "argmax", "n_ge_0.95", "n_ge_0.90", "top", "phase"}}
                for item in RETRIEVER_CALLS
                if item.get("repetition") == row["repetition"]
                and item.get("write_index") == row["write_index"]
                and item.get("phase") == "write"
            ],
        })

    write_phase_retriever = [
        item for item in RETRIEVER_CALLS
        if item.get("phase") == "write" and "max" in item
    ]
    retriever_summary = {
        "calls_during_writes": len(write_phase_retriever),
        "max_score_observed_during_writes": (
            max(item["max"] for item in write_phase_retriever) if write_phase_retriever else None),
        "calls_with_any_score_ge_0.95": sum(
            1 for item in write_phase_retriever if item["n_ge_0.95"] > 0),
        "calls_with_any_score_ge_0.90": sum(
            1 for item in write_phase_retriever if item["n_ge_0.90"] > 0),
        "calls_with_any_score_ge_0.65": sum(
            1 for item in write_phase_retriever if (item.get("top") or [0])[0] >= 0.65),
        "all_calls": len(RETRIEVER_CALLS),
    }

    transitions = [
        {"repetition": item["repetition"], "write_index": item["write_index"],
         "transition": transition}
        for item in WTM_CALLS for transition in item["status_transitions_caused_by_this_write"]
    ]
    transition_kinds: dict[str, int] = {}
    for item in transitions:
        key = f"{item['transition']['from']}->{item['transition']['to']}"
        transition_kinds[key] = transition_kinds.get(key, 0) + 1

    report = {
        "environment": ready,
        "facts": facts,
        "pairwise": pairwise,
        "repetitions": repetitions,
        "write_guard_timeline": write_guard,
        "write_calls_full": WTM_CALLS,
        "retriever_summary": retriever_summary,
        "retraction_calls": RETRACTIONS,
        "bank_write_calls": WRITE_CALLS,
        "status_transition_summary": transition_kinds,
        "status_transitions": transitions,
        "trace_errors": TRACE_ERRORS,
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "stage": "verdict",
        "status_transition_summary": transition_kinds,
        "retract_calls_by_write": [
            {"repetition": item["repetition"], "write_index": item["write_index"],
             "phase": item["phase"], "record_attribute": item["record_attribute"],
             "callsite": item["stack_callsite"][-1:]}
            for item in RETRACTIONS
        ],
        "bank_write_slot_index_by_write": [
            {"repetition": item["repetition"], "write_index": item["write_index"],
             "passed_slot_index": item["passed_slot_index"], "action": item["action"],
             "attribute": item["attribute"]}
            for item in WRITE_CALLS
        ],
        "retriever_summary": retriever_summary,
        "per_write_confirmed_update": [
            {"repetition": row["repetition"], "write_index": row["write_index"],
             "confirmed_update": row["confirmed_update"],
             "learned_best": row["learned_best"],
             "v2_slot_index": row["v2_slot_index"],
             "slot_chosen": row["slot_chosen"],
             "exact_slots": row["exact_slots_numel"],
             "retract_calls": row["retract_calls"]}
            for row in write_guard
        ],
        "output": str(output),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
