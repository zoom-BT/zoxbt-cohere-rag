from __future__ import annotations

from zoxbt_cohere_rag.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text:str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks