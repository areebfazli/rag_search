"""SearchService — the single retrieval entry point shared by the API and the
eval harness, so both exercise identical logic.

Modes: bm25 | dense | hybrid | hybrid_rerank. `candidate_k` controls the first-stage
depth that gets fused/reranked; `top_k` is how many results come back.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.index.embedder import Embedder
from app.index.lexical import LexicalIndex
from app.index.vector_store import VectorStore
from app.retrieve.dense import DenseRetriever
from app.retrieve.fusion import fuse_hits

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


class SearchService:
    def __init__(self, dense=None, lexical=None, reranker=None):
        # Components may be injected (used by tests); otherwise they are built from
        # the on-disk index — which must exist.
        if dense is None or lexical is None:
            missing = [p for p in (settings.qdrant_location, settings.bm25_path) if not Path(p).exists()]
            if missing:
                raise RuntimeError(
                    f"Search index not found ({', '.join(missing)}). Build it first with "
                    "`make index` (uv run python -m app.ingest.build_index)."
                )
            self.embedder = Embedder()
            self.store = VectorStore()
            dense = DenseRetriever(self.embedder, self.store)
            lexical = LexicalIndex().load()
        self.dense = dense
        self.lexical = lexical
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None:  # lazy — only load the cross-encoder when needed
            from app.rerank.cross_encoder import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
        return self._reranker

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[SearchHit]:
        top_k = top_k if (top_k and top_k > 0) else settings.rerank_top_k
        if mode == "bm25":
            return self.lexical.search(query, top_k)
        if mode == "dense":
            return self.dense.search(query, top_k)

        candidate_k = candidate_k or max(settings.dense_top_k, top_k)
        fused = fuse_hits(
            [self.dense.search(query, candidate_k), self.lexical.search(query, candidate_k)],
            k=settings.rrf_k,
            top_k=candidate_k,
        )
        if mode == "hybrid":
            return fused[:top_k]
        if mode == "hybrid_rerank":
            return self.reranker.rerank(query, fused, top_k)
        raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")
