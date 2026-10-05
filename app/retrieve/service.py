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
from app.retrieve.web_search import WebResult, WebSearch

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


def _kinds(value: str) -> tuple[str, ...]:
    return tuple(k.strip() for k in value.split(",") if k.strip()) or ("rewrite",)


def api_dense_cap() -> int | None:
    """The API's embedding cap: web_dense_cap (the measured value, which the eval uses),
    bounded by web_api_dense_cap — the rerank runs under the API's global retrieval lock,
    so fewer embeddings per request means less time holding it. 0 disables either."""
    caps = [c for c in (settings.web_dense_cap, settings.web_api_dense_cap) if c]
    return min(caps) if caps else None


def web_extras(s2) -> dict:
    """WebSearch kwargs for the extra candidate sources enabled in settings (API path:
    short waits and at most one retry, like SemanticScholarRetriever.for_api; caches
    only with SSR_S2_API_CACHE; a per-request deadline; the API embedding cap)."""
    kw: dict = {
        "multi_query": settings.s2_multi_query,
        "snippets": settings.web_snippets,
        "citation_seeds": settings.web_citation_seeds,
        "citation_cap": settings.web_citation_cap,
        "dense_cap": api_dense_cap(),
        "pubmed_queries": _kinds(settings.web_pubmed_queries),
        "snippet_queries": _kinds(settings.web_snippet_queries),
        "deadline_s": settings.web_deadline_s or None,
    }
    if settings.web_snippets and settings.web_snippet_dataset_filter:
        from app.retrieve.s2_extra import DatasetSnippetFilter

        kw["snippet_filter"] = DatasetSnippetFilter.from_scifact()
    if settings.web_pubmed:
        from app.retrieve.pubmed import PubMedSource
        from app.retrieve.semantic_scholar import ResponseCache

        kw["pubmed"] = PubMedSource(
            cache=ResponseCache(Path(settings.s2_cache_dir).parent / "web_cache") if settings.s2_api_cache else None,
            rate=settings.pubmed_rate_per_s,
            email=settings.ncbi_email,
            api_key=settings.ncbi_api_key,
            max_retries=1,
            max_wait_s=settings.s2_max_wait_s,
            max_backoff_s=settings.s2_max_wait_s,
        )
    if settings.web_pubmed or settings.web_snippets or settings.web_citation_seeds:
        from app.retrieve.s2_extra import S2IdResolver

        kw["resolver"] = S2IdResolver(s2, Path(settings.s2_cache_dir) / "ids" if settings.s2_api_cache else None)
    return kw


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
        """The web pipeline (Semantic Scholar + optional rewrite/rerank, per settings) —
        built lazily, only when a web mode is used."""
        if self._web is None:
            with self._web_lock:
                if self._web is None:
                    from app.retrieve.semantic_scholar import SemanticScholarRetriever
                    from app.retrieve.web_search import EmbeddingCache

                    s2 = SemanticScholarRetriever.for_api()
                    self._web = WebSearch(
                        s2,
                        rewrite=settings.s2_query_rewrite,
                        rerank=settings.s2_rerank,
                        embedder=self._web_embedder,
                        rarity=self._term_rarity,
                        # Same opt-in as the response cache: public queries must not
                        # grow data/ without bound.
                        emb_cache=(
                            EmbeddingCache(Path(settings.s2_cache_dir) / "emb")
                            if settings.s2_api_cache
                            else None
                        ),
                        **web_extras(s2),
                    )
        return self._web

    def _web_embedder(self):
        """The local dense embedder when this service built one (shared, so the model
        loads once), else a fresh one."""
        embedder = getattr(self, "embedder", None)
        if embedder is None:
            embedder = Embedder()
        return embedder

    def _term_rarity(self):
        """Local-corpus term rarity for the rewrite's term cap, from the loaded BM25
        index's documents; None (entity/length priority) if the index carries none."""
        docs = getattr(self.lexical, "docs", None)
        if not docs:
            return None
        from app.ingest.corpus import document_passage
        from app.retrieve.query_rewrite import TermRarity

        return TermRarity.from_texts(document_passage(d) for d in docs)

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

        Web modes only: `warnings` (if given) collects human-readable notes — that
        Semantic Scholar failed and hybrid_web fell back to local results, or that an
        extra web source (PubMed, snippet search, id resolution) failed or ran out of
        time and was skipped (see retrieve_web); `local_lock`
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

    def retrieve_web(
        self,
        query: str,
        mode: str = "web",
        top_k: int | None = None,
        candidate_k: int | None = None,
        local_lock: AbstractContextManager | None = None,
    ) -> WebResult:
        """A web-mode retrieval with its per-request side information, as a WebResult:

            hits      the ranked hits (what retrieve() returns)
            warnings  human-readable notes for the response — hybrid_web's local-only
                      fallback when S2 fails, and every extra source (PubMed, S2 snippet
                      search, id resolution, ...) that failed or ran out of time and was
                      skipped. Status/reason text only, never keys or headers.
            errors    the raw "<source>: <reason>" strings behind those warnings
            entries   the S2 paper/search cache entries behind the search (eval dates)

        retrieve() calls this and appends `warnings` to its `warnings` argument, so the
        API gets them through the list it already passes. Raises S2Error for `web` when
        the base S2 search fails (hybrid_web never does).
        """
        if mode not in WEB_MODES:
            raise ValueError(f"not a web mode: {mode!r} (expected one of {WEB_MODES})")
        top_k = top_k if (top_k and top_k > 0) else settings.rerank_top_k
        if mode == "web":
            res = self._web_search(query, min(top_k, S2_MAX_LIMIT), local_lock)
            return WebResult(res.hits[:top_k], res.errors, res.entries, _source_warnings(res.errors))
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
            res = self._web_search(query, min(candidate_k, S2_MAX_LIMIT), local_lock)
        except S2Error as e:
            return WebResult(
                local[:top_k], [], [], [f"Semantic Scholar unavailable ({e}); showing local results only."]
            )
        fused = fuse_hits([local, res.hits], k=settings.rrf_k, top_k=candidate_k)[:top_k]
        return WebResult(fused, res.errors, res.entries, _source_warnings(res.errors))

    def _retrieve_web(
        self,
        query: str,
        mode: str,
        top_k: int,
        candidate_k: int | None,
        warnings: list[str] | None,
        local_lock: AbstractContextManager | None,
    ) -> list[SearchHit]:
        res = self.retrieve_web(query, mode, top_k, candidate_k, local_lock)
        if warnings is not None:
            warnings.extend(res.warnings)
        return res.hits

    def _web_search(
        self, query: str, k: int, local_lock: AbstractContextManager | None
    ) -> WebResult:
        """S2 results for `query`. The pipeline's local rerank (embedder work) runs under
        `local_lock`, like every other use of the shared models; the S2 calls never do.
        An injected plain retriever (tests) is called as-is (no errors, no entries)."""
        if isinstance(self.web, WebSearch):
            return self.web.run(query, k, lock=local_lock)
        return WebResult(self.web.search(query, k))


def _source_warnings(errors: list[str]) -> list[str]:
    """One response warning per failed extra web source (deduped, in order)."""
    return [
        f"Web source skipped ({e}); results may be incomplete."
        for e in dict.fromkeys(errors)
    ]
