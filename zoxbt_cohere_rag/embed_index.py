from __future__ import annotations

import json

import cohere
import numpy as np

from zoxbt_cohere_rag.config import (
COHERE_API_KEY,
EMBED_MODEL,
EMBEDDINGS_PATH,
INDEX_DIR,
INDEX_META_PATH,

)
from zoxbt_cohere_rag.ingest import load_chunks_jsonl, run_ingest

def _client() -> cohere.Client:
    if not COHERE_API_KEY:
        raise RuntimeError("Set COHERE_API_KEY in .env (see .env.example)")
    return cohere.Client(api_key=COHERE_API_KEY)

def embed_texts(client: cohere.Client, texts: list[str], batch_size: int = 90) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embed(
            texts=batch,
            model=EMBED_MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        vectors.extend(resp.embeddings.float_)
    return np.array(vectors, dtype=np.float32)

def build_index(reingest: bool = False) -> None:
    from zoxbt_cohere_rag.config import CHUNKS_PATH

    if reingest or not CHUNKS_PATH.exists():
        run_ingest()
    chunks = load_chunks_jsonl()
    if not chunks:
        raise RuntimeError("No chunks to embed.")
    client = _client()
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    emb = embed_texts(client, texts)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, emb)
    meta = {
        "embed_model": EMBED_MODEL,
        "num_chunks": len(chunks),
        "chunk_ids": [c["id"] for c in chunks],
    }
    INDEX_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved index -> {INDEX_DIR}")
