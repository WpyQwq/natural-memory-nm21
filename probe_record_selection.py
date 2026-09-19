"""READ-ONLY probe: which additive term actually decides record selection.

Why this exists
---------------
`eval_router_critical_e2e.py` shows that swapping the trained V2 router
checkpoint (or randomising it) does not change end-to-end answer accuracy on the
16 router-critical cases.  This probe measures, per case:

* the real `PagedMemoryBankV2._record_scores` call (captured by an in-memory-only
  logging proxy that delegates to the original implementation, so the real
  `(page, query_key, query_text, query_token_ids)` and the real returned scores
  are observed - no scoring code is replaced or re-implemented);
* every additive component named in the task, reconstructed from the public
  record API, plus the residual `implied_routed = real_score - prior_sum` and a
  comparison of that residual against the trained text retriever and the V2
  router sigmoid (which shows *which* scorer produced `routed_score`);
* the V2 read gate (`need_memory` sigmoid vs `read_threshold`, token evidence);
* the legacy 16-slot text path, which is what injects evidence when the V2 gate
  abstains.

Counterfactual orderings (priors only, learned only, V2-router only, random
routed_score draws, a re-initialised *copy* of the router) are computed from the
same reconstruction.  No weight is modified on the loaded model and nothing is
written to disk.

Usage::

    $env:PYTHONPATH='H:\\Memory'; $env:PYTHONIOENCODING='utf-8'
    & $py -m V2_dpskw.probe_record_selection --cases 4
    & $py -m V2_dpskw.probe_record_selection --gate-scan
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import V2_dpskw.memory_os_v2 as mos
from V2_dpskw.eval_end_to_end_memory import answer, write_fact
from V2_dpskw.eval_router_critical_e2e import CASE_PAIRS, build_cases
from V2_dpskw.memory_os_v2 import _tokens
from V2_dpskw.qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer

DEPLOYED_ROUTER = Path(
    r"H:\Memory\dynamic_memory_lab\checkpoints\natural_memory_v2_qwen_router_entities\memory_router_v2.pt"
)
SIDECAR_RETRIEVER = DEPLOYED_ROUTER.parent / "text_retriever.pt"

TERM_KEYS = ("lexical_0.25", "overlap_0.45", "rare_1.25",
             "token_ids_0.35", "struct_0.15", "entity_2.50", "attribute_0.75")

# ---------------------------------------------------------------------------
# in-memory-only logging proxy for the production scoring call
# ---------------------------------------------------------------------------

CALLS: list[dict] = []
_ORIGINAL_RECORD_SCORES = mos.PagedMemoryBankV2._record_scores


def _logging_record_scores(self, page, query_key, query_text, query_token_ids=None, *,
                           allow_superseded: bool = False):
    out = _ORIGINAL_RECORD_SCORES(
        self, page, query_key, query_text, query_token_ids, allow_superseded=allow_superseded
    )
    CALLS.append({
        "page_id": page.page_id,
        "query_key": query_key.detach().clone(),
        "query_text": query_text,
        "query_token_ids": None if query_token_ids is None else query_token_ids.detach().clone(),
        "scored": [(record, float(score)) for record, score in out],
        "record_scorer_is_none": self.record_scorer is None,
    })
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def router_scores(router, query_key: torch.Tensor, candidate_keys: torch.Tensor) -> torch.Tensor:
    """Mirror of `PagedMemoryBankV2._score_candidates` for an arbitrary router."""

    device = next(router.parameters()).device
    scores, _ = router.projected_scores(
        query_key.to(device).reshape(1, -1),
        candidate_keys.to(device).reshape(1, -1, candidate_keys.shape[-1]),
    )
    return torch.sigmoid(scores[0]).detach().float().cpu()


def need_probability(router, query_key: torch.Tensor) -> float:
    device = next(router.parameters()).device
    logits = router.need_memory(query_key.to(device).reshape(1, -1))
    return float(torch.sigmoid(logits)[0, 0].item())


def components(bank, record, query_text, query_token_ids, routed_score: float) -> dict:
    """Reconstruct every additive term named in the task from the record API."""

    lexical = record.lexical_score(query_text)
    overlap = record.token_overlap_score(query_token_ids)
    conf_imp = 0.10 * record.confidence + 0.08 * record.importance
    rare = bank._rare_lexical_address_score(query_text, _tokens(record.routing_text()))
    token_bonus = 0.35 if record.token_ids is not None else 0.0
    struct_bonus = 0.15 if (record.entity or record.attribute or record.value) else 0.0
    entity = record.entity.strip().lower()
    attribute = record.attribute.strip().lower()
    query_lower = query_text.strip().lower()
    entity_bonus = 0.0
    attribute_bonus = 0.0
    if len(entity) >= 4 and entity in query_lower:
        entity_bonus = 2.50
        if attribute and attribute in query_lower:
            attribute_bonus = 0.75
    terms = {
        "routed": float(routed_score),
        "lexical_0.25": 0.25 * lexical,
        "overlap_0.45": 0.45 * overlap,
        "conf_imp_0.10_0.08": conf_imp,
        "rare_1.25": 1.25 * rare,
        "token_ids_0.35": token_bonus,
        "struct_0.15": struct_bonus,
        "entity_2.50": entity_bonus,
        "attribute_0.75": attribute_bonus,
    }
    terms["total"] = sum(terms.values())
    terms["prior_total"] = terms["total"] - terms["routed"]
    terms["lexical_raw"] = float(lexical)
    terms["overlap_raw"] = float(overlap)
    terms["rare_raw"] = float(rare)
    return terms


def order(records: list) -> list:
    """Stable ordering by score then record id (mirrors the production sort)."""

    return [record.record_id for record, _ in sorted(
        records, key=lambda item: (item[1], item[0].record_id), reverse=True)]


def rank_of(ordering: list, record_id) -> int:
    return ordering.index(record_id) + 1 if record_id in ordering else -1


def first_text(ordering: list, by_id: dict) -> str:
    return by_id[ordering[0]].text[:30] if ordering else "-"


def merged_tensor_counts(model_path: Path):
    """Read the merged shard index only (no weights) to confirm the package
    itself carries the trained record reranker."""

    try:
        manifest = json.loads((model_path / "memory_merge.json").read_text(encoding="utf-8"))
        index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        prefix = str(manifest.get("tensor_prefix", "dynamic_memory."))
        keys = list(index["weight_map"])
        return (sum(key.startswith(prefix + "text_retriever.") for key in keys),
                sum(key.startswith(prefix + "memory_router_v2.") for key in keys))
    except Exception as error:  # pragma: no cover - diagnostics only
        return f"unavailable: {error}", "unavailable"


def random_router_copy(router, device, seed: int):
    """A re-initialised COPY of the router; the loaded model is never touched."""

    clone = copy.deepcopy(router).to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in clone.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator).to(parameter.device) * 0.05)
    return clone


def legacy_slot_report(model, input_ids, attention_mask, tokenizer):
    """Reproduce the legacy 16-slot read selection (the path used when the V2
    gate abstains) and report the slot-level scores and injected texts."""

    runtime = model.runtime
    if runtime.text_slot_valid is None:
        return {"available": False}
    with torch.inference_mode():
        address, relevance = model._probe_text_retrieval(input_ids, attention_mask)
    if address is None:
        return {"available": False}
    valid = runtime.text_slot_valid
    scores = address.masked_fill(~valid, torch.finfo(address.dtype).min)
    top_k = min(model.memory_config.text_memory_top_k, scores.shape[-1])
    top_scores, top_slots = scores.topk(top_k, dim=-1)
    selected = top_scores >= model.memory_config.text_memory_threshold
    # `relevance` is the per-batch max score (see _probe_text_retrieval), used as
    # a batch-level gate; `text_read_overlap` is the per-slot token overlap.
    batch_relevance = float(relevance[0].item()) if relevance is not None else None
    if relevance is not None:
        selected &= relevance[:, None] >= model.memory_config.text_memory_threshold
    overlap = runtime.text_read_overlap
    bank_ids = runtime.text_token_ids
    bank_mask = runtime.text_token_mask
    rows = []
    for rank in range(top_k):
        slot = int(top_slots[0, rank].item())
        ids = bank_ids[0, slot][bank_mask[0, slot]] if bank_ids is not None else None
        rows.append({
            "slot": slot,
            "score": float(top_scores[0, rank].item()),
            "slot_overlap": float(overlap[0, slot].item()) if overlap is not None else None,
            "selected": bool(selected[0, rank].item()),
            "text": tokenizer.decode(ids, skip_special_tokens=True)[:60] if ids is not None else "",
        })
    return {
        "available": True,
        "top_k": top_k,
        "threshold": model.memory_config.text_memory_threshold,
        "batch_relevance": batch_relevance,
        "used": bool(runtime.text_prefix_used),
        "prefix_tokens": int(runtime.text_prefix_tokens or 0),
        "slots": rows,
        "retriever_driven": bool(model.text_retriever is not None and model._text_retriever_ready),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--router", default=str(DEPLOYED_ROUTER))
    parser.add_argument("--random-draws", type=int, default=50)
    parser.add_argument("--gate-scan", action="store_true",
                        help="run all 16 cases and report only the read-gate behaviour "
                             "(deployed router vs 3 re-initialised router copies)")
    args = parser.parse_args()

    case_count = len(CASE_PAIRS) if args.gate_scan else args.cases
    cases = build_cases(case_count, args.seed)
    model_path = Path(args.package)
    print(json.dumps({"stage": "loading", "package": str(model_path.resolve()),
                      "cases": len(cases), "mode": "gate-scan" if args.gate_scan else "full"},
                     ensure_ascii=False), flush=True)

    memory_config = load_memory_config(model_path)
    model = load_qwen_dynamic(model_path, memory_config=memory_config, load_in_4bit=True,
                             max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    tokenizer = load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- identical swap to eval_router_critical_e2e.py --------------------
    if args.router.strip():
        state = torch.load(args.router.strip(), map_location="cpu", weights_only=True)
        model.memory_router_v2.load_state_dict(state.get("router_state_dict", state), strict=True)
        model.memory_router_v2.to(device).eval()

    bank = model.memory_os_v2.bank
    retriever_tensors, router_tensors = merged_tensor_counts(model_path)
    env = {
        "device": str(device),
        "router_swapped_from": str(args.router),
        "model__memory_router_v2_ready": bool(model._memory_router_v2_ready),
        "model__text_retriever_ready": bool(model._text_retriever_ready),
        "text_retriever_is_not_none": model.text_retriever is not None,
        "text_retriever_param_count": (sum(p.numel() for p in model.text_retriever.parameters())
                                       if model.text_retriever is not None else 0),
        "memory_router_v2_param_count": sum(p.numel() for p in model.memory_router_v2.parameters()),
        "merged_package_text_retriever_tensor_count": retriever_tensors,
        "merged_package_memory_router_v2_tensor_count": router_tensors,
        "sidecar_text_retriever_pt_exists": SIDECAR_RETRIEVER.exists(),
        "bank_record_scorer_is_bound_method": getattr(bank.record_scorer, "__self__", None) is model,
        "bank_top_k_pages": bank.top_k_pages,
        "bank_top_k_records": bank.top_k_records,
        "bank_hidden_size": bank.hidden_size,
        "bank_key_dim": bank.key_dim,
        "read_threshold": model.memory_os_v2.read_threshold,
        "text_memory_top_k": model.memory_config.text_memory_top_k,
        "text_memory_threshold": model.memory_config.text_memory_threshold,
    }
    print(json.dumps({"stage": "environment", **env}, ensure_ascii=False, indent=2), flush=True)

    random_routers = [random_router_copy(model.memory_router_v2, device, seed)
                      for seed in (1234, 5678, 9012)]
    mos.PagedMemoryBankV2._record_scores = _logging_record_scores

    summary_rows = []
    gate_rows = []
    for index, case in enumerate(cases, 1):
        print(json.dumps({"stage": "case_start", "case": index, "query": case["query"]},
                         ensure_ascii=False), flush=True)
        CALLS.clear()
        model.reset_memory(batch_size=1, device=device)
        fact_probe = case["fact"].strip().rstrip("。")
        written = 0
        for fact in case["bank"]:
            written += int(write_fact(model, tokenizer, fact, device))
        # snapshot taken BEFORE the query: proves write-path retraction is not a
        # side effect of reading
        pre_status = {record.record_id: record.status
                      for record in model.memory_os_v2.bank.records.values()}
        pre_target_states = [status for record_id, status in pre_status.items()
                             if fact_probe in model.memory_os_v2.bank.records[record_id].text]
        started = time.perf_counter()
        reply = answer(model, tokenizer, case["query"], device, args.max_new_tokens)
        elapsed = time.perf_counter() - started
        production_calls = list(CALLS)
        decision = dict(model.runtime.v2_last_decisions[-1]) if model.runtime.v2_last_decisions else {}
        runtime = model.runtime
        query_key = runtime.v2_query_key
        if query_key is not None and query_key.ndim == 2:
            query_key = query_key[0]
        correct = case["code"].lower() in reply.lower()

        tokens = tokenizer(case["query"], add_special_tokens=False, return_tensors="pt")
        query_token_ids = tokens["input_ids"].to(device)
        bank = model.memory_os_v2.bank
        active = [record for record in bank.records.values() if record.status == "active"]
        inventory = [{
            "record_id": record.record_id,
            "status": record.status,
            "slot_index": record.slot_index,
            "entity": record.entity,
            "attribute": record.attribute,
            "value": record.value,
            "conflict_key": record.conflict_key(),
            "text": record.text[:44].replace("\n", " "),
        } for record in sorted(bank.records.values(), key=lambda item: item.timestamp)]
        target_records = [item for item in inventory if fact_probe in item["text"]]
        target_status = target_records[0]["status"] if target_records else "ABSENT_FROM_BANK"
        address_pages, address_records = bank._address_hits(case["query"])
        lexical_pages, lexical_records = bank._lexical_evidence_hits(case["query"])
        with torch.inference_mode():
            token_evidence = bool(bank.has_token_evidence(query_key, query_token_ids, case["query"]))
            need = need_probability(model.memory_router_v2, query_key)
            need_random = [need_probability(router, query_key) for router in random_routers]
            gate_open = (need >= model.memory_os_v2.read_threshold) or token_evidence
            gate_open_random = [bool((value >= model.memory_os_v2.read_threshold) or token_evidence)
                                for value in need_random]

        # direct (address / lexical evidence) path: hardcoded 10.0 / 8.0 scores
        direct_inventory = []
        for record_id in sorted(set(address_records) | set(lexical_records)):
            record = bank.records.get(record_id)
            if record is None or record.status != "active":
                continue
            if record_id in address_records and not bank._record_matches_explicit_address(record, case["query"]):
                continue
            score = (10.0 if record_id in address_records else 8.0) + 0.25 * record.lexical_score(case["query"])
            score += 0.35 if record.token_ids is not None else 0.0
            score += 0.15 if (record.entity or record.attribute or record.value) else 0.0
            direct_inventory.append({
                "record_id": record_id, "is_target": fact_probe in record.text,
                "path": "address" if record_id in address_records else "lexical", "score": score,
            })
        direct_inventory.sort(key=lambda item: item["score"], reverse=True)
        direct_scores = sorted({round(item["score"], 6) for item in direct_inventory}, reverse=True)
        direct_top_ties = sum(1 for item in direct_inventory
                              if direct_scores and abs(item["score"] - direct_scores[0]) < 1e-9)
        direct_target_rank = next((position for position, item in enumerate(direct_inventory, 1)
                                   if item["is_target"]), -1)

        with torch.inference_mode():
            legacy = legacy_slot_report(model, query_token_ids, torch.ones_like(query_token_ids), tokenizer)

        gate_rows.append({
            "case": index, "query": case["query"], "code": case["code"], "correct": bool(correct),
            "need_probability_deployed": need,
            "need_probability_random_router_copies": need_random,
            "read_threshold": model.memory_os_v2.read_threshold,
            "token_evidence": token_evidence,
            "gate_open_deployed": bool(gate_open),
            "gate_open_random": gate_open_random,
            "stop_reason": decision.get("stop_reason"),
            "records_selected": len(decision.get("record_ids") or []),
            "legacy_prefix_used": bool(legacy.get("used")),
            "legacy_injected_texts": [row["text"][:24] for row in legacy.get("slots", []) if row["selected"]],
            "target_status": target_status,
            "target_status_pre_query": pre_target_states,
            "pre_query_non_active": sum(1 for status in pre_status.values() if status != "active"),
            "active_records": len(active), "records_in_bank": len(inventory),
            "address_hits": len(address_records), "lexical_hits": len(lexical_records),
            "direct_records": len(direct_inventory),
            "direct_target_rank": direct_target_rank,
            "direct_top_ties": direct_top_ties,
        })

        if args.gate_scan:
            print(f"CASE {index:2d} need={need:.4f} need_random={[round(value, 3) for value in need_random]} "
                  f"token_ev={token_evidence} gate_open={bool(gate_open)} "
                  f"stop={decision.get('stop_reason')} selected={len(decision.get('record_ids') or [])} "
                  f"lex={len(lexical_records)} addr={len(address_records)} direct={len(direct_inventory)} "
                  f"target={target_status} correct={bool(correct)} "
                  f"injected={[row['text'][:18] for row in legacy.get('slots', []) if row['selected']]}",
                  flush=True)
            continue

        prefix_path = (
            f"V2 record prefix ({len(decision.get('record_ids') or [])} records, "
            f"{legacy.get('prefix_tokens')} tokens)"
            if (decision.get("record_ids") and legacy.get("used"))
            else (f"legacy 16-slot prefix ({legacy.get('prefix_tokens')} tokens)"
                  if legacy.get("used") else "no memory prefix injected")
        )

        # ---- capture the real production call, else force an analysis call --
        if production_calls:
            calls = production_calls
            analysis = "production (V2 read() gate passed)"
        else:
            CALLS.clear()
            with torch.inference_mode():
                bank.query(query_key=query_key, query_text=case["query"],
                           query_token_ids=query_token_ids[0],
                           top_k_pages=bank.top_k_pages, top_k_records=bank.top_k_records,
                           max_hops=bank.max_hops, min_score=-1.0)
            calls = list(CALLS)
            analysis = "forced probe query (production V2 read() gate ABSTAINED)"
        query_text = calls[0]["query_text"] if calls else case["query"]
        record_scorer_none = calls[0]["record_scorer_is_none"] if calls else None

        # ---- reconstruct --------------------------------------------------
        by_id: dict = {}
        rows: list[dict] = []
        residual_error = 0.0
        with torch.inference_mode():
            for call in calls:
                candidate_keys = torch.stack([record.key for record, _ in call["scored"]], dim=0)
                semantic_keys = [record.semantic_key for record, _ in call["scored"]
                                 if isinstance(record.semantic_key, torch.Tensor)]
                retriever_scores = None
                if semantic_keys:
                    retriever_scores = model._score_semantic_memory_records(
                        query_key, torch.stack(semantic_keys, dim=0)
                    ).detach().float().cpu()
                router_sig = router_scores(model.memory_router_v2, query_key, candidate_keys)
                random_sig = router_scores(random_routers[0], query_key, candidate_keys)
                semantic_index = 0
                for position, (record, real_score) in enumerate(call["scored"]):
                    used_retriever = None
                    if isinstance(record.semantic_key, torch.Tensor) and retriever_scores is not None:
                        used_retriever = float(retriever_scores[semantic_index].item())
                        semantic_index += 1
                    zero = components(bank, record, query_text, query_token_ids[0], 0.0)
                    implied_routed = real_score - zero["total"]
                    terms = components(bank, record, query_text, query_token_ids[0], implied_routed)
                    residual_error = max(residual_error,
                                         abs(zero["total"] + float(router_sig[position].item()) - real_score))
                    by_id[record.record_id] = record
                    rows.append({
                        "record_id": record.record_id,
                        "text": record.text[:40],
                        "is_target": fact_probe in record.text,
                        "page_id": call["page_id"],
                        "route": ("address" if record.record_id in address_records else
                                  ("lexical" if record.record_id in lexical_records else "-")),
                        "semantic_key_tensor": isinstance(record.semantic_key, torch.Tensor),
                        "semantic_key_dim": (tuple(record.semantic_key.shape)
                                             if isinstance(record.semantic_key, torch.Tensor) else None),
                        "record_key_dim": tuple(record.key.shape),
                        "real_final": real_score,
                        "implied_routed": implied_routed,
                        "retriever_sigmoid": used_retriever,
                        "router_v2_sigmoid": float(router_sig[position].item()),
                        "random_router_sigmoid": float(random_sig[position].item()),
                        "matches_retriever": (used_retriever is not None
                                              and abs(implied_routed - used_retriever) < 1e-6),
                        "matches_router_v2": abs(implied_routed - float(router_sig[position].item())) < 1e-6,
                        **terms,
                    })

        direct_ids = set(address_records) | set(lexical_records)
        direct_rows = []
        for record_id in sorted(direct_ids):
            record = bank.records.get(record_id)
            if record is None or record.status != "active":
                continue
            if record_id in address_records and not bank._record_matches_explicit_address(record, case["query"]):
                continue
            direct_score = (10.0 if record_id in address_records else 8.0) + 0.25 * record.lexical_score(query_text)
            if record.token_ids is not None:
                direct_score += 0.35
            if record.entity or record.attribute or record.value:
                direct_score += 0.15
            by_id[record_id] = record
            direct_rows.append({"record_id": record_id, "text": record.text[:40],
                                "is_target": fact_probe in record.text,
                                "path": "address" if record_id in address_records else "lexical",
                                "direct_score": direct_score})

        # ---- orderings -----------------------------------------------------
        actual = order([(by_id[row["record_id"]], row["real_final"]) for row in rows]) if rows else []
        priors_only = order([(by_id[row["record_id"]], row["prior_total"]) for row in rows]) if rows else []
        learned_only = order([(by_id[row["record_id"]], row["implied_routed"]) for row in rows]) if rows else []
        router_only = order([(by_id[row["record_id"]], row["router_v2_sigmoid"]) for row in rows]) if rows else []
        random_router_final = order(
            [(by_id[row["record_id"]], row["prior_total"] + row["random_router_sigmoid"]) for row in rows]
        ) if rows else []
        target_id = next((row["record_id"] for row in rows if row["is_target"]), None)
        if target_id is None:
            target_id = next((row["record_id"] for row in direct_rows if row["is_target"]), None)

        draw_top1: dict = {}
        target_rank_draws: list = []
        generator_cpu = torch.Generator().manual_seed(99)
        for _ in range(args.random_draws):
            draws = torch.rand(len(rows), generator=generator_cpu).tolist() if rows else []
            ordering = order([(by_id[row["record_id"]], row["prior_total"] + draws[position])
                              for position, row in enumerate(rows)]) if rows else []
            if ordering:
                draw_top1[ordering[0]] = draw_top1.get(ordering[0], 0) + 1
                if target_id:
                    target_rank_draws.append(rank_of(ordering, target_id))
        stable_under_random = (len(draw_top1) == 1) if draw_top1 else None

        pair_total = 0
        pair_router_proof = 0
        prior_diffs: list = []
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                pair_total += 1
                diff = abs(rows[left]["prior_total"] - rows[right]["prior_total"])
                prior_diffs.append(diff)
                if diff > 1.0:
                    pair_router_proof += 1

        decision_ids = list(decision.get("record_ids") or [])
        top_row = max(rows, key=lambda row: row["real_final"]) if rows else None
        target_row = next((row for row in rows if row["is_target"]), None)
        gap = None
        if top_row is not None and target_row is not None and top_row is not target_row:
            gap = top_row["real_final"] - target_row["real_final"]
        flip = "target already ranked #1 by the final score"
        if target_id is None and direct_inventory:
            flip = (f"direct address/lexical path only (no record-level learned scoring): target rank "
                    f"{direct_target_rank}/{len(direct_inventory)} at score "
                    f"{next((item['score'] for item in direct_inventory if item['is_target']), float('nan')):.3f}, "
                    f"top score {direct_scores[0]:.3f} shared by {direct_top_ties} records "
                    f"(all ties broken by input order)")
        elif target_id is None:
            flip = (f"the target record is NOT in the candidate set at all (bank status: {target_status}); "
                    f"no scoring weight can fix this")
        elif gap is not None and gap > 0:
            delta_routed = target_row["implied_routed"] - top_row["implied_routed"]
            improvable = {key: target_row[key] - top_row[key]
                          for key in TERM_KEYS if target_row[key] - top_row[key] > 1e-9}
            if delta_routed > 1e-9:
                flip = (f"routed_score weight would need >= {gap / delta_routed:.2f} "
                        f"(gap {gap:.3f} / learned delta {delta_routed:.3f})")
            else:
                best = max(improvable.items(), key=lambda item: item[1], default=None)
                if best is None:
                    flip = (f"no single additive term can flip it: gap {gap:.3f}, "
                            f"learned routed delta {delta_routed:.3f} <= 0, no prior favours the target")
                else:
                    flip = (f"no positive routed_score weight works (learned delta {delta_routed:.3f} <= 0); "
                            f"the only single-term fix is a further +{gap - best[1]:.3f} on {best[0]} "
                            f"(it already favours the target by {best[1]:.3f})")

        # ---------------- report -------------------------------------------
        print("=" * 120)
        print(f"CASE {index}: query={case['query']}  target={case['code']}  fact={case['fact']}")
        print(f"  reply={reply[:70]!r}  correct={correct}  written={written}/{len(case['bank'])}  "
              f"gen_s={elapsed:.1f}  active_records={len(active)}/{len(inventory)}")
        if target_records:
            print(f"  TARGET RECORD: status={target_status} (pre-query status={pre_target_states}) "
                  f"slot_index={target_records[0]['slot_index']} "
                  f"id={target_records[0]['record_id'][:14]} conflict_key={target_records[0]['conflict_key']!r} "
                  f"| {target_records[0]['text']!r}")
        else:
            print(f"  TARGET RECORD: ABSENT_FROM_BANK (no record contains {fact_probe!r})")
        if len(inventory) != written:
            print(f"  WRITE-PATH ATTRITION: wrote {written} facts -> {len(inventory)} records "
                  f"({len(active)} active, "
                  f"{sum(1 for item in inventory if item['status'] != 'active')} not active)")
            for item in inventory:
                if item["status"] != "active":
                    print(f"    {item['status']:10s} slot={item['slot_index']:2d} "
                          f"{item['record_id'][:14]} | {item['text'][:52]!r}")
        print(f"  V2 GATE: stop_reason={decision.get('stop_reason')} "
              f"need_prob={need:.4f} (threshold {model.memory_os_v2.read_threshold}) "
              f"token_evidence={token_evidence} gate_open={bool(gate_open)} "
              f"need_prob(random router copies)={[round(value, 3) for value in need_random]} "
              f"records_selected={len(decision_ids)}")
        print(f"  PREFIX ACTUALLY USED: {prefix_path}")
        print(f"  LEGACY 16-slot path (recomputed, only reached when the V2 path yields no tokens): "
              f"used={legacy.get('used')} prefix_tokens={legacy.get('prefix_tokens')} "
              f"top_k={legacy.get('top_k')} threshold={legacy.get('threshold')} "
              f"batch_relevance={legacy.get('batch_relevance')} "
              f"retriever_driven={legacy.get('retriever_driven')}")
        for row in legacy.get("slots", []):
            print(f"    slot {row['slot']:2d} score={row['score']:7.4f} overlap={row['slot_overlap']:6.4f} "
                  f"selected={row['selected']} | {row['text']!r}")
        print(f"  ROUTE: has_explicit_address={bank.has_explicit_address(case['query'])} "
              f"address_hits={len(address_records)} lexical_hits={len(lexical_records)} "
              f"direct_records={len(direct_rows)}")
        print(f"  ANALYSIS: {analysis}; _record_scores_calls={len(calls)} "
              f"pages={len({call['page_id'] for call in calls})} records_scored={len(rows)} "
              f"record_scorer_is_None={record_scorer_none} reconstruction_residual={residual_error:.2e}")
        print(f"  flags: router_v2_ready={env['model__memory_router_v2_ready']} "
              f"text_retriever_ready={env['model__text_retriever_ready']} "
              f"semantic_key_is_Tensor={sum(1 for row in rows if row['semantic_key_tensor'])}/{len(rows)} "
              f"semantic_key_None={sum(1 for row in rows if not row['semantic_key_tensor'])}/{len(rows)} "
              f"semantic_key_dim={rows[0]['semantic_key_dim'] if rows else '-'} "
              f"record_key_dim={rows[0]['record_key_dim'] if rows else '-'}")
        print(f"  decision order: {[rid[:12] for rid in decision_ids[:8]]}")
        if rows:
            print(f"  {'record':14s} {'rt':2s} {'routed':>7s} {'retr':>7s} {'rtr2':>7s} {'lex':>6s} "
                  f"{'ovlp':>6s} {'cf+im':>6s} {'rare':>6s} {'+0.35':>6s} {'+0.15':>6s} {'ent':>5s} "
                  f"{'attr':>5s} {'FINAL':>7s} {'src':>9s} {'T':>2s}  text")
            for row in sorted(rows, key=lambda item: item["real_final"], reverse=True):
                source = ("retriever" if row["matches_retriever"] else
                          ("router_v2" if row["matches_router_v2"] else "?"))
                print(f"  {row['record_id'][:14]:14s} {row['route'][:2]:2s} "
                      f"{row['routed']:7.3f} "
                      f"{(row['retriever_sigmoid'] if row['retriever_sigmoid'] is not None else float('nan')):7.4f} "
                      f"{row['router_v2_sigmoid']:7.4f} "
                      f"{row['lexical_0.25']:6.3f} {row['overlap_0.45']:6.3f} "
                      f"{row['conf_imp_0.10_0.08']:6.3f} {row['rare_1.25']:6.3f} "
                      f"{row['token_ids_0.35']:6.2f} {row['struct_0.15']:6.2f} "
                      f"{row['entity_2.50']:5.2f} {row['attribute_0.75']:5.2f} "
                      f"{row['real_final']:7.3f} {source:>9s} {'T' if row['is_target'] else ' ':2s}  "
                      f"{row['text'][:26]!r}")
        for row in direct_rows:
            print(f"  DIRECT {row['record_id'][:14]:14s} {row['path']:8s} score={row['direct_score']:7.3f} "
                  f"{'TARGET' if row['is_target'] else '      '} | {row['text'][:40]!r}")
        if direct_inventory:
            print(f"  DIRECT-PATH ORDER (hardcoded 10.0/8.0 + 0.25*lexical + 0.35/0.15): "
                  f"target_rank={direct_target_rank}/{len(direct_inventory)} "
                  f"top_ties={direct_top_ties} distinct_top_scores={direct_scores[:4]}")
        print(f"  ORDERINGS (target={(target_id or '-')[:14]}):")
        print(f"    actual final        #1={first_text(actual, by_id):32s} target_rank={rank_of(actual, target_id)}")
        print(f"    priors only         #1={first_text(priors_only, by_id):32s} target_rank={rank_of(priors_only, target_id)}")
        print(f"    learned only        #1={first_text(learned_only, by_id):32s} target_rank={rank_of(learned_only, target_id)}")
        print(f"    router_v2 only      #1={first_text(router_only, by_id):32s} target_rank={rank_of(router_only, target_id)}")
        print(f"    priors+rand router  #1={first_text(random_router_final, by_id):32s} "
              f"changed_vs_actual={random_router_final != actual}")
        print(f"    priors+U(0,1) random routed: distinct_winners={len(draw_top1)} stable={stable_under_random} "
              f"target_rank_min/max={min(target_rank_draws) if target_rank_draws else '-'}/"
              f"{max(target_rank_draws) if target_rank_draws else '-'}")
        print(f"  pair dominance: {pair_router_proof}/{pair_total} candidate pairs have |prior diff| > 1.0 "
              f"(router-proof); mean |prior diff|={sum(prior_diffs)/max(1,len(prior_diffs)):.3f} "
              f"max={max(prior_diffs) if prior_diffs else 0:.3f}")
        routed_range = ((max(row["implied_routed"] for row in rows)
                         - min(row["implied_routed"] for row in rows)) if rows else None)
        prior_range = ((max(row["prior_total"] for row in rows)
                        - min(row["prior_total"] for row in rows)) if rows else None)
        print(f"  HYPOTHESIS TEST: learned routed_score range={routed_range if routed_range is None else round(routed_range, 3)} vs "
              f"additive-prior range={prior_range if prior_range is None else round(prior_range, 3)}; "
              f"winner changes if routed_score is randomised={not bool(stable_under_random) if stable_under_random is not None else 'n/a'}")
        print(f"  single-term fix for the target: {flip}")

        summary_rows.append({
            "case": index, "query": case["query"], "code": case["code"], "correct": bool(correct),
            "stop_reason": decision.get("stop_reason"), "analysis": analysis,
            "prefix_path": prefix_path,
            "need_probability": need, "gate_open": bool(gate_open), "token_evidence": token_evidence,
            "legacy_used": bool(legacy.get("used")),
            "legacy_injected": [row["text"][:24] for row in legacy.get("slots", []) if row["selected"]],
            "records_scored": len(rows), "records_selected": len(decision_ids),
            "active_records": len(active), "records_in_bank": len(inventory),
            "non_active_records": sum(1 for item in inventory if item["status"] != "active"),
            "written": written, "target_status": target_status,
            "target_status_pre_query": pre_target_states,
            "pre_query_non_active": sum(1 for status in pre_status.values() if status != "active"),
            "address_hits": len(address_records), "lexical_hits": len(lexical_records),
            "direct_records": len(direct_inventory), "direct_target_rank": direct_target_rank,
            "direct_top_ties": direct_top_ties,
            "routed_range": routed_range, "prior_range": prior_range,
            "target_id": target_id,
            "target_rank_actual": rank_of(actual, target_id),
            "target_rank_priors_only": rank_of(priors_only, target_id),
            "target_rank_learned_only": rank_of(learned_only, target_id),
            "target_rank_router_only": rank_of(router_only, target_id),
            "top1_actual_is_target": bool(top_row and top_row["is_target"]),
            "priors_only_equals_actual_order": priors_only == actual,
            "random_routed_stable_top1": stable_under_random,
            "random_routed_top1_distinct": len(draw_top1),
            "random_routed_target_rank_min": min(target_rank_draws) if target_rank_draws else None,
            "random_routed_target_rank_max": max(target_rank_draws) if target_rank_draws else None,
            "router_v2_random_top1_changed": random_router_final != actual,
            "implied_routed_matches_retriever": sum(1 for row in rows if row["matches_retriever"]),
            "implied_routed_matches_router_v2": sum(1 for row in rows if row["matches_router_v2"]),
            "router_proof_pairs": pair_router_proof, "pairs": pair_total,
            "prior_diff_mean": sum(prior_diffs) / max(1, len(prior_diffs)),
            "prior_diff_max": max(prior_diffs) if prior_diffs else 0.0,
            "target_score": target_row["real_final"] if target_row else None,
            "top_score_real": top_row["real_final"] if top_row else None,
            "gap_to_top": gap, "single_term_fix": flip,
            "routed_share_of_final_pct": (100 * sum(row["routed"] for row in rows)
                                          / max(1e-9, sum(row["real_final"] for row in rows))) if rows else None,
            "prior_share_of_final_pct": (100 * sum(row["prior_total"] for row in rows)
                                         / max(1e-9, sum(row["real_final"] for row in rows))) if rows else None,
        })

    n = len(gate_rows)
    print("=" * 120)
    print("GATE SCAN" if args.gate_scan else "AGGREGATE")
    print(json.dumps({
        "cases": n,
        "answer_accuracy_pct": 100 * sum(row["correct"] for row in gate_rows) / max(1, n),
        "v2_gate_open_pct_deployed_router":
            100 * sum(row["gate_open_deployed"] for row in gate_rows) / max(1, n),
        "v2_gate_open_pct_random_router_copies": [
            100 * sum(row["gate_open_random"][position] for row in gate_rows) / max(1, n)
            for position in range(len(gate_rows[0]["gate_open_random"]))
        ] if gate_rows else [],
        "token_evidence_pct": 100 * sum(row["token_evidence"] for row in gate_rows) / max(1, n),
        "need_prob_deployed_min": min(row["need_probability_deployed"] for row in gate_rows),
        "need_prob_deployed_max": max(row["need_probability_deployed"] for row in gate_rows),
        "stop_reasons": {reason: sum(1 for row in gate_rows if row["stop_reason"] == reason)
                         for reason in {row["stop_reason"] for row in gate_rows}},
        "legacy_prefix_used_pct": 100 * sum(row["legacy_prefix_used"] for row in gate_rows) / max(1, n),
        "target_record_retracted_or_absent": sum(1 for row in gate_rows
                                                 if row["target_status"] != "active"),
        "target_record_active": sum(1 for row in gate_rows if row["target_status"] == "active"),
        "cases_with_direct_records": sum(1 for row in gate_rows if row["direct_records"]),
        "cases_with_address_hits": sum(1 for row in gate_rows if row["address_hits"]),
        "cases_with_lexical_hits": sum(1 for row in gate_rows if row["lexical_hits"]),
        "cases_where_target_is_direct_record": sum(1 for row in gate_rows if row["direct_target_rank"] > 0),
        "cases_where_target_is_direct_top1": sum(1 for row in gate_rows if row["direct_target_rank"] == 1),
        "active_records_min": min(row["active_records"] for row in gate_rows),
        "active_records_max": max(row["active_records"] for row in gate_rows),
    }, ensure_ascii=False, indent=2))
    if not args.gate_scan:
        print(json.dumps({
            "target_ranked_1_actual_pct": 100 * sum(row["target_rank_actual"] == 1 for row in summary_rows) / max(1, n),
            "target_ranked_1_by_priors_only_pct":
                100 * sum(row["target_rank_priors_only"] == 1 for row in summary_rows) / max(1, n),
            "target_ranked_1_by_learned_only_pct":
                100 * sum(row["target_rank_learned_only"] == 1 for row in summary_rows) / max(1, n),
            "priors_only_reproduces_actual_order_pct":
                100 * sum(row["priors_only_equals_actual_order"] for row in summary_rows) / max(1, n),
            "random_routed_top1_stable_pct":
                100 * sum(bool(row["random_routed_stable_top1"]) for row in summary_rows) / max(1, n),
            "random_router_v2_changes_final_top1_pct":
                100 * sum(row["router_v2_random_top1_changed"] for row in summary_rows) / max(1, n),
            "implied_routed_matches_retriever_records":
                sum(row["implied_routed_matches_retriever"] for row in summary_rows),
            "implied_routed_matches_router_v2_records":
                sum(row["implied_routed_matches_router_v2"] for row in summary_rows),
            "records_scored_total": sum(row["records_scored"] for row in summary_rows),
            "router_proof_pairs_total": sum(row["router_proof_pairs"] for row in summary_rows),
            "pairs_total": sum(row["pairs"] for row in summary_rows),
            "mean_routed_share_of_final_pct":
                sum(row["routed_share_of_final_pct"] or 0 for row in summary_rows) / max(1, n),
            "mean_prior_share_of_final_pct":
                sum(row["prior_share_of_final_pct"] or 0 for row in summary_rows) / max(1, n),
        }, ensure_ascii=False, indent=2))
        for row in summary_rows:
            print(json.dumps(row, ensure_ascii=False))
    else:
        for row in gate_rows:
            print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
