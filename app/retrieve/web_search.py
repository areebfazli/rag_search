"""The `web` retrieval pipeline: Semantic Scholar candidates, optionally fetched with a
keyword rewrite of the claim and re-ranked locally with the hand-rolled stack.

    claim ──rewrite?──► S2 paper/search (6-term primary + 3-term fallback, pooled)
          ──► candidate pool (≤100 per query, deduped by corpus id)
          ──rerank?──► RRF( bge-small dense similarity , BM25 over the pool )

Both steps are deterministic and free (no LLM), and both are off unless enabled
(settings.s2_query_rewrite / settings.s2_rerank; web_eval measures each). The claim
itself — never the rewrite — is what the local re-ranker scores against, with the
bge-small query prefix on the CLAIM ONLY; S2 papers are embedded as documents (title +
abstract, exactly like the local index's passages), without it.

Candidate embeddings are cached on disk (EmbeddingCache, keyed by model + passage text),
so an eval re-run re-embeds nothing. The rerank stays within S2's own page: it reorders
what S2 returned, so it can raise Recall@5/10 and nDCG but not Recall@100 of that pool.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.index.lexical import LexicalIndex
from app.retrieve.fusion import fuse_hits
from app.retrieve.query_rewrite import TermRarity, rewrite_queries
from app.retrieve.semantic_scholar import S2_MAX_LIMIT, S2Error, parse_search


def candidate_passage(hit: SearchHit) -> str:
    """The text a candidate is embedded/BM25-scored as: title + abstract, the same shape
    as a local passage (ingest.corpus.document_passage). A no-abstract paper (web_any
    view only) is its title alone."""
    from app.ingest.corpus import document_passage  # lazy: keeps ir_datasets off import

    return document_passage({"title": hit.metadata.get("title", ""), "text": hit.text})


class EmbeddingCache:
    """Document embeddings of S2 candidates, on disk (one SQLite file per model under
    `directory`), keyed by sha256(model, passage). SQLite gives atomic writes and safe
    concurrent readers without a file per paper."""

    def __init__(self, directory: str | Path, model_name: str | None = None):
        self.model = model_name or settings.embedding_model
        slug = "".join(c if c.isalnum() else "-" for c in self.model).strip("-")
        self.path = Path(directory) / f"{slug}.sqlite"
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def key(self, passage: str) -> str:
        return hashlib.sha256(f"{self.model}\n{passage}".encode()).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
        return con

    def get_many(self, passages: list[str]) -> dict[str, np.ndarray]:
        keys = {self.key(p): p for p in passages}
        found: dict[str, np.ndarray] = {}
        if not keys:
            return found
        with self._lock:
            con = self._connect()
            try:
                ks = list(keys)
                for i in range(0, len(ks), 500):
                    chunk = ks[i : i + 500]
                    rows = con.execute(
                        f"SELECT key, dim, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})",
                        chunk,
                    ).fetchall()
                    for k, dim, blob in rows:
                        vec = np.frombuffer(blob, dtype=np.float32)
                        if vec.shape == (dim,):
                            found[keys[k]] = vec
            finally:
                con.close()
        self.hits += len(found)
        self.misses += len(set(passages)) - len(found)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        rows = []
        for passage, vec in items.items():
            v = np.asarray(vec, dtype=np.float32).ravel()
            rows.append((self.key(passage), int(v.shape[0]), v.tobytes()))
        with self._lock:
            con = self._connect()
            try:
                with con:
                    con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows)
            finally:
                con.close()


def embed_passages(embedder, passages: list[str], cache: EmbeddingCache | None) -> np.ndarray:
    """Document embeddings (NO query prefix) for `passages`, through the cache."""
    unique = list(dict.fromkeys(passages))
    known = cache.get_many(unique) if cache is not None else {}
    todo = [p for p in unique if p not in known]
    if todo:
        vecs = np.asarray(embedder.encode_documents(todo, show_progress=False), dtype=np.float32)
        fresh = dict(zip(todo, vecs))
        if cache is not None:
            cache.put_many(fresh)
        known.update(fresh)
    return np.stack([known[p] for p in passages])


def dense_order(claim: str, hits: list[SearchHit], embedder, cache: EmbeddingCache | None) -> list[SearchHit]:
    """`hits` by cosine similarity to the claim, best first (ties keep S2 order). The
    claim goes through encode_query (bge prefix); papers through encode_documents."""
    if not hits:
        return []
    q = np.asarray(embedder.encode_query(claim), dtype=np.float32).ravel()
    docs = embed_passages(embedder, [candidate_passage(h) for h in hits], cache)
    sims = docs @ q  # embeddings are L2-normalised: dot product == cosine
    order = sorted(range(len(hits)), key=lambda i: (-float(sims[i]), i))
    return [SearchHit(hits[i].doc_id, float(sims[i]), hits[i].text, hits[i].metadata) for i in order]


def bm25_order(claim: str, hits: list[SearchHit]) -> list[SearchHit]:
    """`hits` by BM25 of the claim over an index of just these candidates — the same
    LexicalIndex (bm25s + English stemming + stopwords) as the local corpus."""
    if not hits:
        return []
    index = LexicalIndex()
    index.build(
        [{"doc_id": h.doc_id, "text": "", "title": ""} for h in hits],
        [candidate_passage(h) for h in hits],
        show_progress=False,
    )
    by_id = {h.doc_id: h for h in hits}
    ranked = index.search(claim, len(hits))
    # bm25s returns every doc at k=len; zero-score ties come back in index order.
    return [SearchHit(r.doc_id, r.score, by_id[r.doc_id].text, by_id[r.doc_id].metadata) for r in ranked]


def rerank(
    claim: str,
    hits: list[SearchHit],
    embedder,
    cache: EmbeddingCache | None = None,
    rrf_k: int | None = None,
) -> list[SearchHit]:
    """RRF(dense, BM25) over the S2 candidate pool — the same fusion as local hybrid."""
    if not hits:
        return []
    lists = [dense_order(claim, hits, embedder, cache), bm25_order(claim, hits)]
    return fuse_hits(lists, k=rrf_k or settings.rrf_k)


def _restamp(hits: list[SearchHit]) -> list[SearchHit]:
    return [SearchHit(h.doc_id, 1.0 / (i + 1), h.text, h.metadata) for i, h in enumerate(hits)]


class WebSearch:
    """Retriever over S2 with optional claim->keyword rewrite and local re-ranking.

    `s2` is a SemanticScholarRetriever (or anything with `fetch(query, limit)` returning
    its cache-entry shape). `embedder` is an Embedder or a zero-arg factory for one,
    resolved only when a rerank runs. `rarity` (a TermRarity or factory) ranks rewrite
    terms by local-corpus rarity; None falls back to entity/length priority.
    `keep_no_abstract` keeps title-only papers (the eval's web_any view). With `strict`
    a failed fallback request raises (the eval must not silently score a degraded
    pool); otherwise the primary results are served alone.
    """

    def __init__(
        self,
        s2,
        *,
        rewrite: bool = False,
        rerank: bool = False,
        embedder=None,
        rarity: TermRarity | Callable[[], TermRarity | None] | None = None,
        emb_cache: EmbeddingCache | None = None,
        keep_no_abstract: bool = False,
        strict: bool = False,
        fetch_k: int = S2_MAX_LIMIT,
    ):
        self.s2 = s2
        self.rewrite = rewrite
        self.do_rerank = rerank
        self._embedder = embedder
        self._rarity = rarity
        self.emb_cache = emb_cache
        self.keep_no_abstract = keep_no_abstract
        self.strict = strict
        self.fetch_k = max(1, min(int(fetch_k), S2_MAX_LIMIT))
        self._init_lock = threading.Lock()
        self.last_entries: list[dict] = []  # cache entries behind the last search (eval dates)

    @property
    def embedder(self):
        with self._init_lock:
            if self._embedder is None:
                from app.index.embedder import Embedder

                self._embedder = Embedder()
            elif not hasattr(self._embedder, "encode_query"):  # a factory
                self._embedder = self._embedder()
            return self._embedder

    @property
    def rarity(self) -> TermRarity | None:
        with self._init_lock:
            if self._rarity is not None and not isinstance(self._rarity, TermRarity):
                self._rarity = self._rarity()  # a factory (may return None)
            return self._rarity

    def queries(self, claim: str) -> list[str]:
        """What is sent to S2: the claim, or [primary, fallback] keyword queries."""
        if self.rewrite:
            qs = rewrite_queries(claim, self.rarity)
            if qs:
                return qs
        return [claim]

    def candidates(self, claim: str, limit: int) -> list[SearchHit]:
        """The S2 candidate pool for `claim` in fetch order (primary results first,
        deduped). The fallback query is sent unless the primary alone filled the page
        (`limit` papers WITH an abstract) — decided on that count in both views, so web
        and web_any always come from the same requests."""
        self.last_entries = []
        pool: list[SearchHit] = []
        served: list[SearchHit] = []
        seen: set[str] = set()
        for i, q in enumerate(self.queries(claim)):
            if i > 0 and len(served) >= limit:
                break
            try:
                entry = self.s2.fetch(q, limit)
            except S2Error:
                if i == 0 or self.strict:
                    raise
                break  # API: a failed fallback serves the primary results alone
            self.last_entries.append(entry)
            response = entry.get("response")
            if i == 0:
                served = parse_search(response, limit)
            for h in parse_search(response, limit, keep_no_abstract=self.keep_no_abstract):
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    pool.append(h)
        return pool

    def search(
        self, query: str, top_k: int, lock: AbstractContextManager | None = None
    ) -> list[SearchHit]:
        """Top `top_k` for `query`. Without rerank, S2 is asked for `top_k` papers per
        query (as before); with it, for `fetch_k` (the full page) to rerank. `lock`, if
        given, is held around the local rerank only — the S2 calls never run under it."""
        top_k = max(1, min(int(top_k), S2_MAX_LIMIT))
        limit = self.fetch_k if self.do_rerank else top_k
        pool = self.candidates(query, limit)
        if not self.do_rerank:
            return _restamp(pool)[:top_k]
        with lock or nullcontext():
            return rerank(query, pool, self.embedder, self.emb_cache)[:top_k]
