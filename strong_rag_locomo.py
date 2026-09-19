"""Strong-RAG baselines on the same corpus, model and budget as the memory system.

The point of this script is a *fair* head-to-head, so every axis that could
flatter either side is held equal:

* **same model** -- the identical 4-bit NM2.1 checkpoint generates both answers;
  the RAG path simply never touches the memory module (no ``memory_query_*``).
* **same candidate pool** -- the same per-question turn pool the memory run writes
  into its bank.
* **same injection budget** -- ``--top-k`` defaults to the package's own
  ``memory_top_k_records`` (8), so both sides put the same number of records in
  front of the model.
* **same scorer** -- ``eval_scoring.score_case``, whitespace-insensitive with
  pattern-based refusal detection.

Methods:

* ``dense``   -- cosine retrieval in the frozen Qwen key space.  This uses the
  very same ``_encode_model_key`` the memory system's router ranks with, so the
  representation is not a handicap invented for the baseline.
* ``bm25``    -- classic lexical retrieval, the standard non-neural baseline.
* ``full``    -- the entire pool stuffed into the prompt.  This is the *upper
  bound for both sides*: the memory run also has all 16 records in its bank, so
  a memory system that retrieved perfectly could not beat it.

Result rows are written in the same shape as ``eval_end_to_end_memory`` so both
can be re-scored and compared offline by one script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.eval_end_to_end_memory import _chat_tensor, build_cases
from V2_dpskw.eval_scoring import score_case
from V2_dpskw.qwen_integration import load_memory_config, load_qwen_dynamic, load_tokenizer

SYSTEM_PROMPT = (
    "You are a memory assistant. Answer the user's question using ONLY the "
    "conversation memories provided. If the memories do not contain the answer, "
    "say that you don't know -- do not guess."
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Minimal BM25 over one question's candidate pool."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in documents]
        self.k1, self.b = k1, b
        self.lengths = [len(d) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        self.df: Counter[str] = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

    def rank(self, query: str) -> list[int]:
        n = len(self.docs)
        q = tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                score += idf * tf[term] * (self.k1 + 1) / denom
            scores.append(score)
        return sorted(range(n), key=lambda i: scores[i], reverse=True)


@torch.inference_mode()
def encode_keys(model, tokenizer, texts: list[str], device, batch: int = 32) -> torch.Tensor:
    out = []
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        encoded = tokenizer(chunk, padding=True, truncation=True, max_length=256,
                            return_tensors="pt")
        keys = model._encode_model_key(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        out.append(keys.float().cpu())
    return torch.cat(out, dim=0)


@torch.inference_mode()
def generate(model, tokenizer, system: str, user: str, device, max_new_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, enable_thinking=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    output = model.generate(
        **encoded, max_new_tokens=max_new_tokens, do_sample=False,
        update_memory=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
    )
    generated = output[0][encoded["input_ids"].shape[1]:] if isinstance(output, torch.Tensor) else output
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_prompt(question: str, memories: list[str]) -> str:
    listing = "\n".join(f"- {m}" for m in memories)
    return f"Conversation memories:\n{listing}\n\nQuestion: {question}\nAnswer:"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="qwen3_5_4b_natural_memory_v2_1")
    parser.add_argument("--eval-file", default="data/net_locomo/eval.jsonl")
    parser.add_argument("--per-category", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=0, help="0 = use the package's memory_top_k_records")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--methods", default="dense,bm25,full")
    parser.add_argument("--output", default="strong_rag_locomo.json")
    parser.add_argument("--limit", type=int, default=0, help="smoke-test cap")
    args = parser.parse_args()

    cases = build_cases(Path(args.eval_file), args.per_category)
    if args.limit:
        cases = cases[:args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(json.dumps({"cases": len(cases), "methods": methods,
                      "categories": sorted({c["category"] for c in cases})}, ensure_ascii=False), flush=True)

    model_path = Path(args.package)
    memory_config = load_memory_config(model_path)
    top_k = args.top_k or int(memory_config.memory_top_k_records)
    print(json.dumps({"top_k": top_k, "source": "cli" if args.top_k else "package memory_top_k_records"}),
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_qwen_dynamic(model_path, memory_config=memory_config, load_in_4bit=True,
                              max_memory={0: "10.5GiB", "cpu": "48GiB"})
    model.eval()
    tokenizer = load_tokenizer(model_path)
    print(json.dumps({"model_loaded": True}), flush=True)

    # Encode every unique candidate text once; pools overlap heavily across questions.
    unique_texts: list[str] = []
    seen: dict[str, int] = {}
    for case in cases:
        for text in case["facts"]:
            if text not in seen:
                seen[text] = len(unique_texts)
                unique_texts.append(text)
    print(json.dumps({"unique_candidate_texts": len(unique_texts)}), flush=True)
    started = time.perf_counter()
    candidate_keys = encode_keys(model, tokenizer, unique_texts, device)
    print(json.dumps({"encoded_s": round(time.perf_counter() - started, 1)}), flush=True)

    query_keys = encode_keys(model, tokenizer, [c["query"] for c in cases], device)

    results: dict[str, dict] = {}
    for method in methods:
        rows = []
        started = time.perf_counter()
        for index, case in enumerate(cases, 1):
            pool = case["facts"]
            if method == "full":
                chosen = list(range(len(pool)))
            elif method == "dense":
                pool_keys = candidate_keys[[seen[t] for t in pool]]
                sims = pool_keys @ query_keys[index - 1]
                chosen = torch.argsort(sims, descending=True)[:top_k].tolist()
            elif method == "bm25":
                chosen = BM25(pool).rank(case["query"])[:top_k]
            else:
                raise SystemExit(f"unknown method {method}")
            memories = [pool[i] for i in chosen]
            reply = generate(model, tokenizer, SYSTEM_PROMPT,
                             build_prompt(case["query"], memories), device, args.max_new_tokens)
            scored = score_case(case, reply)
            rows.append({
                "category": case["category"], "query": case["query"], "reply": reply[:200],
                "written": len(memories), "retrieved": chosen, **scored,
            })
            if index % 25 == 0:
                print(json.dumps({"method": method, "case": index, "total": len(cases),
                                  "elapsed_s": round(time.perf_counter() - started, 1),
                                  "running_accuracy_pct": round(
                                      100 * sum(r["correct"] for r in rows) / len(rows), 2)}), flush=True)

        total = len(rows)
        answerable = [r for r, c in zip(rows, cases) if c["answerable"]]
        unknown = [r for r, c in zip(rows, cases) if not c["answerable"]]
        per_category = defaultdict(lambda: {"n": 0, "ok": 0, "ans": 0, "ans_ok": 0, "unk": 0, "unk_ok": 0})
        for row, case in zip(rows, cases):
            block = per_category[case["category"]]
            block["n"] += 1
            block["ok"] += int(row["correct"])
            if case["answerable"]:
                block["ans"] += 1
                block["ans_ok"] += int(row["correct"])
            else:
                block["unk"] += 1
                block["unk_ok"] += int(row["correct"])
        results[method] = {
            "summary": {
                "router": f"strong_rag_{method}",
                "cases": total,
                "top_k": top_k if method != "full" else len(cases[0]["facts"]),
                "accuracy_pct": 100 * sum(r["correct"] for r in rows) / max(1, total),
                "answerable_cases": len(answerable),
                "answerable_accuracy_pct": 100 * sum(r["correct"] for r in answerable) / max(1, len(answerable)),
                "unknown_cases": len(unknown),
                "unknown_refusal_pct": 100 * sum(r["correct"] for r in unknown) / max(1, len(unknown)),
                "wrong_abstention_pct": 100 * sum(r["wrongly_abstained"] for r in answerable) / max(1, len(answerable)),
                "seconds": round(time.perf_counter() - started, 1),
                "per_category": {
                    name: {
                        "cases": b["n"],
                        "accuracy_pct": 100 * b["ok"] / max(1, b["n"]),
                        "answerable": b["ans"],
                        "answerable_accuracy_pct": 100 * b["ans_ok"] / max(1, b["ans"]),
                        "unknown": b["unk"],
                        "unknown_refusal_pct": 100 * b["unk_ok"] / max(1, b["unk"]),
                    }
                    for name, b in sorted(per_category.items())
                },
            },
            "rows": rows,
        }
        print(json.dumps(results[method]["summary"], ensure_ascii=False), flush=True)

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
