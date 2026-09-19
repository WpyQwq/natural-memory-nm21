"""Exercise the RAG baseline's model-free logic (BM25, prompt, corpus loading).

Runs without touching the GPU so it can be checked while the memory run owns the
card.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from V2_dpskw.strong_rag_locomo import BM25, build_prompt, tokenize
from V2_dpskw.eval_end_to_end_memory import build_cases

# 1. BM25 must rank the turn that actually states the answer first.
pool = [
    "Melanie: I finally finished that painting of a sunrise.",
    "Caroline: I went to the LGBTQ support group on 7 May 2023.",
    "Caroline: Work has been really busy this month.",
    "Melanie: That sounds rough, hope you are okay.",
]
ranking = BM25(pool).rank("When did Caroline go to the LGBTQ support group?")
print("BM25 ranking:", ranking)
print("top doc:", pool[ranking[0]])
assert ranking[0] == 1, "BM25 failed to rank the answering turn first"

# 2. A question with no lexical overlap should not crash and should still rank.
ranking2 = BM25(pool).rank("What colour was the painting?")
print("BM25 (no-overlap) top:", pool[ranking2[0]])

# 3. Prompt shape.
prompt = build_prompt("When did Caroline go?", pool[:2])
assert "Question: When did Caroline go?" in prompt and prompt.rstrip().endswith("Answer:")
print("prompt ok, length", len(prompt))

# 4. Tokenizer sanity.
print("tokens:", tokenize("I don't know — 7 May 2023!"))
assert tokenize("7 May 2023") == ["7", "may", "2023"]

# 5. Corpus loads with the harness' own loader and the answerable split is right.
corpus_path = Path(sys.argv[1] if len(sys.argv) > 1 else r"H:\Memory\V2_dpskw\data\net_locomo\eval.jsonl")
cases = build_cases(corpus_path, 40)
ans = sum(1 for c in cases if c["answerable"])
print(f"corpus: {len(cases)} cases, answerable {ans}, adversarial {len(cases) - ans}")
assert ans == 155 and len(cases) == 195, "unexpected answerable split"

# 6. Every pool has at least a few candidates and no duplicate text.
for case in cases:
    assert len(case["facts"]) >= 4, case["query"]
    assert len(set(case["facts"])) == len(case["facts"]), "duplicate candidate text"
print("all pools >= 4 candidates, no duplicates")
print("\nOK")
