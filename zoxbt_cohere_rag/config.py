from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BLOGFOLIO_PATH = Path(os.getenv("BLOGFOLIO_PATH", ROOT / "_blogfolio_src")).resolve()
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_DIR = DATA_DIR / "index"
INDEX_META_PATH = INDEX_DIR / "meta.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
SITE_BASE = "https://zoxbt.is-a.dev"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
