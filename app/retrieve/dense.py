"""Dense retriever — adapts the embedder + vector store to the Retriever interface."""
from __future__ import annotations

from app.core.interfaces import SearchHit
from app.index.embedder import Embedder
from app.index.vector_store import VectorStore


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        vec = self.embedder.encode_query(query)
        return self.store.search(vec, top_k)
