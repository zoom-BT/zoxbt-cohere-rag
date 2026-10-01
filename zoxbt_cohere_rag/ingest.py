from __future__ import annotations

import json
import re
from pathlib import Path

from zoxbt_cohere_rag.chunking import chunk_text
from zoxbt_cohere_rag.mdx_utils import parse_frontmatter, strip_mdx_body
from zoxbt_cohere_rag.config import BLOGFOLIO_PATH, CHUNKS_PATH, ROOT, SITE_BASE

CONTENT_KINDS = ("posts", "projects", "research", "certifications")
LOCALES = ("fr", "en")

def _url_for(kind: str, locale: str, slug: str) -> str:
    if kind == "posts":
        return f"{SITE_BASE}/{locale}/blog/{slug}"
    if kind == "projects":
        return f"{SITE_BASE}/{locale}/projects/{slug}"
    if kind == "research":
        return f"{SITE_BASE}/{locale}/research/{slug}"
    if kind == "certifications":
        return f"{SITE_BASE}/{locale}/certifications/{slug}"
    return f"{SITE_BASE}/{locale}"

def _safe_rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def _slug_from_filename(name: str, kind: str) -> str:
    base = re.sub(r"\.mdx|md", "", name)
    if kind == "posts":
        return re.sub(r"^(\d{4}-\d{2}-\d{2})-", "", base)
    return base

def _iter_mdx_files() -> list[tuple[str, str, str]]:
    content_root = BLOGFOLIO_PATH /"src" /"content"
    out: list[tuple[Path, str, str]] = []
    for kind in CONTENT_KINDS:
        for locale in LOCALES:
            d = content_root / kind / locale
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.mdx")) + sorted(d.glob("*.md")):
                out.append((f, kind, locale))
    static = ROOT /"corpus_static"
    if static.is_dir():
        for f in sorted(static.glob("*.md")):
            meta, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
            locale = meta.get("locale", "fr")
            kind = meta.get("kind", "about")
            out.append((f, kind, locale))
    return out

def ingest_documents() -> list[dict]:
    if not BLOGFOLIO_PATH.is_dir():
        raise FileNotFoundError(
            f"BLOGFOLIO_PATH not found: {BLOGFOLIO_PATH}. " 
            "Clone https://github.com/zoxbt/blogfolio into _blogfolio_src or set BLOGFOLIO_PATH."

        )   
    docs: list[dict] = []
    for path, kind, locale in _iter_mdx_files():
        raw  = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        plain = strip_mdx_body(body)
        if len(plain) < 40:
            continue
        slug = meta.get("slug") or _slug_from_filename(path.name, kind if kind in CONTENT_KINDS else "posts")
        title = meta.get("title") or slug
        url = meta.get("url") or _url_for(kind if kind in CONTENT_KINDS else "posts", locale, slug)
        doc_id = f"{kind}:{locale}:{slug}"
        for i, piece in enumerate(chunk_text(plain)):
            chunk_id = f"{doc_id}#{i}"
            docs.append(
                {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "text": piece,
                    "title": title,
                    "kind": kind,
                    "locale": locale,
                    "slug": slug,
                    "url": url,
                    "source_path": _safe_rel_path(path, ROOT),
                }
            )
    return docs


def write_chunks_jsonl(chunks: list[dict], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_chunks_jsonl(path: Path = CHUNKS_PATH) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def run_ingest() -> int:
    chunks = ingest_documents()
    write_chunks_jsonl(chunks)
    print(f"ingested {len(chunks)} chunks -> {CHUNKS_PATH}")
    return len(chunks)
