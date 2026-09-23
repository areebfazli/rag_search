"""Build the dense (Qdrant) + lexical (bm25s) indices for the corpus.

Usage:
    uv run python -m app.ingest.build_index
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.config import settings
from app.index.embedder import Embedder
from app.index.lexical import LexicalIndex
from app.index.vector_store import VectorStore
from app.ingest.corpus import document_passage, load_documents

# Describes what the on-disk indices were built from. The retrieval eval hashes it into
# its cache signature, so a reindex with a different corpus/model/BM25 setup cannot
# serve cached scores computed against the old index.
MANIFEST_PATH = Path("data/index_manifest.json")


def main() -> None:
    t0 = time.time()
    # A manifest must never outlive the index it describes: drop it before touching
    # the indices, so a build that dies halfway can't leave a stale one vouching for
    # a half-rebuilt index.
    MANIFEST_PATH.unlink(missing_ok=True)
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

    manifest = {
        "corpus_dataset": settings.corpus_dataset,
        "doc_count": len(docs),
        "embedding_model": settings.embedding_model,
        "embedding_dim": embedder.dim,
        "bm25": lexical.describe(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys + fixed indent: the eval hashes these bytes, so identical inputs must
    # serialize identically.
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote index manifest ({MANIFEST_PATH})")

    print(f"Done in {time.time() - t0:.1f}s: {len(docs)} docs indexed (dense + lexical).")


if __name__ == "__main__":
    main()
