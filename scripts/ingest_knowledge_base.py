"""Chunk knowledge_base/*.md files and (re-)upsert them into Qdrant.

Run whenever knowledge_base/ content changes, against whichever Qdrant QDRANT_URL
points at (local docker compose, or the production Qdrant Cloud cluster):

    python -m scripts.ingest_knowledge_base
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from app.core.config import get_settings
from app.services.vector_store import get_vector_store

KB_ROOT = Path(__file__).resolve().parent.parent / "knowledge_base"

# ~400 tokens per chunk, comfortably under the embedding model's 512-token limit,
# with a small overlap so a chunk boundary doesn't sever one idea in two.
MAX_CHUNK_WORDS = 300
OVERLAP_WORDS = 40

_HEADING_SPLIT = re.compile(r"\n(?=#{1,6} )")


def split_into_chunks(text: str) -> list[str]:
    """Split on markdown headings first, then cap each section's word count."""
    chunks: list[str] = []
    for section in _HEADING_SPLIT.split(text.strip()):
        words = section.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + MAX_CHUNK_WORDS
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - OVERLAP_WORDS
    return chunks


def iter_source_files() -> list[tuple[str, Path]]:
    return [
        (domain_dir.name, path)
        for domain_dir in sorted(p for p in KB_ROOT.iterdir() if p.is_dir())
        for path in sorted(domain_dir.glob("*.md"))
    ]


def main() -> None:
    settings = get_settings()
    # Same client the search tool uses, so ingestion embeds the same way (Qdrant Cloud
    # Inference for a cloud URL, local fastembed otherwise).
    store = get_vector_store()

    all_chunks = []
    current_sources: set[str] = set()
    for domain, path in iter_source_files():
        source = str(path.relative_to(KB_ROOT))
        current_sources.add(source)
        text = path.read_text(encoding="utf-8")
        for chunk_index, chunk_text in enumerate(split_into_chunks(text)):
            all_chunks.append({"text": chunk_text, "source": source, "domain": domain, "chunk_index": chunk_index})

    if not all_chunks:
        print(f"No knowledge base content found under {KB_ROOT}", file=sys.stderr)
        return

    store.upsert_chunks(all_chunks)
    store.delete_stale_sources(current_sources)
    print(
        f"Ingested {len(all_chunks)} chunks from {len(current_sources)} source files "
        f"into '{settings.qdrant_collection}' at {settings.qdrant_url} "
        f"({'Qdrant Cloud Inference' if settings.qdrant_cloud_inference else 'local fastembed'} embeddings)."
    )


if __name__ == "__main__":
    main()
