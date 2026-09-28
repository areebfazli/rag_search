"""SearchService — the single retrieval entry point shared by the API and the
eval harness, so both exercise identical logic.

Modes: bm25 | dense | hybrid | hybrid_rerank, plus the optional networked web | hybrid_web
(Semantic Scholar; see app/retrieve/semantic_scholar.py). `candidate_k` controls the
first-stage depth that gets fused/reranked; `top_k` is how many results come back.
"""
from __future__ import annotations

import threading
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.index.embedder import Embedder
from app.index.lexical import LexicalIndex
from app.index.vector_store import VectorStore
from app.retrieve.dense import DenseRetriever
from app.retrieve.fusion import fuse_hits
from app.retrieve.semantic_scholar import S2_MAX_LIMIT, S2Error

LOCAL_MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
# Optional, networked (Semantic Scholar) — never the default. `web` is S2 alone;
# `hybrid_web` RRF-fuses the local hybrid list with the S2 list.
WEB_MODES = ("web", "hybrid_web")
MODES = LOCAL_MODES + WEB_MODES


def _demote(tail: list[SearchHit], head: list[SearchHit]) -> list[SearchHit]:
    """Restamp `tail` scores to sit strictly below `head`, preserving its order.

    The reranked head carries cross-encoder logits (roughly -11..+11) while the
    un-reranked tail carries RRF scores (~0.016 and down). Concatenating them leaves
    the ORDER right but the `score` field non-monotonic, so a client that sorts by
    score — or a UI that simply displays it, as ours does — contradicts the ranking.
    Rank order is the contract; the score must not argue with it.
    """
    if not tail or not head:
        return tail
    floor = min(h.score for h in head)
    return [
        SearchHit(h.doc_id, floor - (i + 1), h.text, h.metadata) for i, h in enumerate(tail)
    ]


class SearchService:
    def __init__(self, dense=None, lexical=None, reranker=None, web=None):
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
        self._web = web
        self._web_lock = threading.Lock()

    @property
    def web(self):
        """The Semantic Scholar retriever — built lazily, only when a web mode is used."""
        if self._web is None:
            with self._web_lock:
                if self._web is None:
                    from app.retrieve.semantic_scholar import SemanticScholarRetriever

                    self._web = SemanticScholarRetriever.for_api()
        return self._web

    @property
    def reranker(self):
        if self._reranker is None:  # lazy — only load the cross-encoder when needed
            from app.rerank.cross_encoder import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
        return self._reranker

    @reranker.setter
    def reranker(self, reranker) -> None:
        # Lets the eval harness swap reranker models without rebuilding the service
        # (and re-acquiring the single-process embedded-Qdrant lock).
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        candidate_k: int | None = None,
        warnings: list[str] | None = None,
        local_lock: AbstractContextManager | None = None,
    ) -> list[SearchHit]:
        """Ranked hits for `query`.

        Web modes only: `warnings` (if given) collects human-readable notes, e.g. that
        Semantic Scholar failed and hybrid_web fell back to local results; `local_lock`
        (if given) is held around the LOCAL retrieval only, so the S2 network call never
        runs under the API's retrieval lock. `web` raises S2Error when S2 fails — there
        is no local fallback to degrade to — while hybrid_web never raises for it.
        """
        top_k = top_k if (top_k and top_k > 0) else settings.rerank_top_k
        if mode in WEB_MODES:
            return self._retrieve_web(query, mode, top_k, candidate_k, warnings, local_lock)
        if mode == "bm25":
            return self.lexical.search(query, top_k)
        if mode == "dense":
            return self.dense.search(query, top_k)

        candidate_k = candidate_k or max(settings.dense_top_k, top_k)
        fused = self._hybrid(query, candidate_k)
        if mode == "hybrid":
            return fused[:top_k]
        if mode == "hybrid_rerank":
            # Rerank only the top slice (bounded cost — see settings.rerank_candidates)
            # and keep the tail in fused order, so recall past the slice is preserved.
            depth = min(len(fused), settings.rerank_candidates)
            reranked = self.reranker.rerank(query, fused[:depth], depth)
            return (reranked + _demote(fused[depth:], reranked))[:top_k]
        raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")

    def _hybrid(self, query: str, candidate_k: int) -> list[SearchHit]:
        return fuse_hits(
            [self.dense.search(query, candidate_k), self.lexical.search(query, candidate_k)],
            k=settings.rrf_k,
            top_k=candidate_k,
        )

    def _retrieve_web(
        self,
        query: str,
        mode: str,
        top_k: int,
        candidate_k: int | None,
        warnings: list[str] | None,
        local_lock: AbstractContextManager | None,
    ) -> list[SearchHit]:
        if mode == "web":
            return self.web.search(query, min(top_k, S2_MAX_LIMIT))
        # hybrid_web: the local hybrid list and the S2 list, fused with the same RRF.
        # fuse_hits dedupes by doc_id (SciFact ids are S2 corpus ids), and the local list
        # goes first, so a paper in both keeps the local text/title and gains S2's url/year.
        candidate_k = candidate_k or max(settings.dense_top_k, top_k)
        with local_lock or nullcontext():
            local = self._hybrid(query, candidate_k)
        local = [
            SearchHit(h.doc_id, h.score, h.text, {**h.metadata, "source": "local"}) for h in local
        ]
        try:
            web = self.web.search(query, min(candidate_k, S2_MAX_LIMIT))
        except S2Error as e:
            if warnings is not None:
                warnings.append(
                    f"Semantic Scholar unavailable ({e}); showing local results only."
                )
            return local[:top_k]
        return fuse_hits([local, web], k=settings.rrf_k, top_k=candidate_k)[:top_k]
