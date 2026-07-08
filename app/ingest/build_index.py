"""Build the dense (Qdrant) + lexical (bm25s) indices for the corpus.

Usage:
    uv run python -m app.ingest.build_index
"""
from __future__ import annotations

import time

from app.core.config import settings
from app.index.embedder import Embedder
from app.index.lexical import LexicalIndex
from app.index.vector_store import VectorStore
from app.ingest.corpus import document_passage, load_documents


def main() -> None:
    t0 = time.time()
    print(f"Loading corpus: {settings.corpus_dataset}")
    docs = load_documents()
    passages = [document_passage(d) for d in docs]
    print(f"  {len(docs)} documents")

    print(f"Embedding with {settings.embedding_model} ...")
    embedder = Embedder()
    vectors = embedder.encode_documents(passages)

    print(f"Indexing dense vectors into Qdrant ({settings.qdrant_location}) ...")
    store = VectorStore()
    store.recreate(dim=embedder.dim)
    store.upsert(docs, vectors)

    print(f"Building BM25 index ({settings.bm25_path}) ...")
    lexical = LexicalIndex()
    lexical.build(docs, passages)
    lexical.save()

    print(f"Done in {time.time() - t0:.1f}s: {len(docs)} docs indexed (dense + lexical).")


if __name__ == "__main__":
    main()
