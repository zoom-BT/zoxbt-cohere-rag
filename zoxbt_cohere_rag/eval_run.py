from __future__ import annotations

import json
from pathlib import Path

from zoxbt_cohere_rag.retrieve import retrieve

EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"

def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(relevant & top) / len(relevant)

def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0
    

def load_eval() -> list[dict]:
    rows: list[dict] = []
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def run_eval(k: int = 5) -> dict:
    questions = load_eval()
    methods = ("bm25", "embed", "embed_rerank")
    summary = {m: {"recall": [], "mrr": []} for m in methods}

    for q in questions:
        relevant = set(q.get("relevant_chunk_ids", []))
        query = q["question"]
        for method in methods:
            hits = retrieve(query, method=method, top_k=15, top_n=k)
            ids = [h["id"] for h in hits]
            summary[method]["recall"].append(recall_at_k(relevant, ids, k))
            summary[method]["mrr"].append(mrr(relevant, ids))

    report = {}
    for method in methods:
        r = summary[method]["recall"]
        m = summary[method]["mrr"]
        report[method] = {
            f"recall@{k}": round(sum(r) / len(r), 4) if r else 0.0,
            "mrr": round(sum(m) / len(m), 4) if m else 0.0,
            "n": len(questions),
        }
    return report

def main() -> None:
    report = run_eval()
    print(json.dumps(report, indent=2))
    