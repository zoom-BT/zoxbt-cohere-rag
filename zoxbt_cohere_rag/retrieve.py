from __future__ import annotations

import json

import cohere
import numpy as np
from rank_bm25 import BM25Okapi

from zoxbt_cohere_rag.config import (
    COHERE_API_KEY,
    EMBED_MODEL,
    INDEX_META_PATH,
    EMBEDDINGS_PATH,
    RERANK_MODEL,
)
from zoxbt_cohere_rag.ingest import load_chunks_jsonl

def _client() -> cohere.Client:
    if not COHERE_API_KEY:
        raise RuntimeError("Set COHERE_API_KEY in .env ")
    return cohere.Client(api_key=COHERE_API_KEY)

def _load_index():
    chunks = load_chunks_jsonl()
    emb = np.load(EMBEDDINGS_PATH)
    meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
    if len(chunks) != emb.shape[0]:
        raise RuntimeError("chunks.jsonl and embeddings.npy are out of sync - run build-index ")
    return chunks, emb, meta

def _tokenizer(s: str) -> list[str]:
    return s.lower().split()

def bm25_search(query: str, chunks: list[dict], top_k: int = 10) -> list[tuple[str, float]]:
    corpus = [_tokenizer(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenizer(query))
    ranked = sorted(enumerate(scores), key=lambda   x: x[1], reverse=True)[:top_k]
    return [(chunks[i]["id"], float(s)) for i,s in ranked]

def embd_query(client: cohere.Client, query: str) -> np.ndarray:
    resp = client.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    return np.array(resp.embeddings.float[0], dtype=np.float32)

def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]

def embed_search(query: str, top_k: int = 10) -> list[dict]:
    chunks, emb, _ = _load_index()
    client = _client()
    qv = embd_query(client, query)
    hits = cosine_topk(qv, emb, top_k)
    id_to_chunk = {c["id"]: c for c in chunks}
    return [{**id_to_chunk[chunks[i]["id"]], "score": score} for i, score in hits]

def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    if not candidates:
        return []
    client = _client()
    docs = [c["text"] for c in candidates]
    resp = client.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=min(top_n, len(docs)))
    out: list[dict] = []
    for r in resp.results:
        c = dict(candidates[r.index])
        c["rerank_score"] = r.relevance_score
        out.append(c)
    return out

def retrieve(query: str, method: str = "embed_rerank", top_k: int = 15, top_n: int = 5) -> list[dict]:
    chunks, _, _ = _load_index()
    if method == "bm25":
        ids = bm25_search(query, chunks, top_k=top_k)
        id_map = {c["id"]: c for c in chunks}
        return [{**id_map[cid], "score": sc} for cid, sc in ids[:top_n]]

    if method == "embed":
        return embed_search(query, top_k=top_n)

    cands = embed_search(query, top_k=top_k)
    return rerank(query, cands, top_n=top_n)
    
