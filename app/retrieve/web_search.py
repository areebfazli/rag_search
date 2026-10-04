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

With extra sources enabled (settings.web_pubmed / web_snippets / s2_multi_query /
web_citation_seeds), search_pooled widens the pool beyond S2's own page — PubMed Best
Match and S2 snippet search, ids resolved to S2 corpus ids — and rerank_pool re-ranks the
merged pool, embedding only its pre-ranked head (settings.web_dense_cap).

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


def merge_pools(lists: list[list[SearchHit]], keep_no_abstract: bool = False) -> list[SearchHit]:
    """Union of several sources' candidate lists, deduped by doc_id, in RRF order over
    the sources' own rankings (a paper several sources found comes first).

    Text and metadata are merged across sources: the first non-empty abstract wins (so a
    paper S2 serves without an abstract can borrow PubMed's or OpenAlex's), title/url/
    year likewise, and metadata["sources"] lists every source that returned it. Papers
    with no text anywhere are dropped unless `keep_no_abstract`."""
    by_id: dict[str, SearchHit] = {}
    srcs: dict[str, list[str]] = {}
    for hits in lists:
        for h in hits:
            cur = by_id.get(h.doc_id)
            if cur is None:
                by_id[h.doc_id] = SearchHit(h.doc_id, 0.0, h.text, dict(h.metadata))
                srcs[h.doc_id] = []
            else:
                if not cur.text and h.text:
                    cur.text = h.text
                for k, v in h.metadata.items():
                    if v and not cur.metadata.get(k):
                        cur.metadata[k] = v
            src = h.metadata.get("source")
            if src and src not in srcs[h.doc_id]:
                srcs[h.doc_id].append(src)
    order = fuse_hits([[SearchHit(h.doc_id, 0.0) for h in hits] for hits in lists], k=settings.rrf_k)
    out = []
    for f in order:
        h = by_id[f.doc_id]
        if not h.text and not keep_no_abstract:
            continue
        h.metadata["sources"] = srcs[h.doc_id]
        out.append(SearchHit(h.doc_id, f.score, h.text, h.metadata))
    return out


def prerank(claim: str, pool: list[SearchHit], rrf_k: int | None = None) -> list[SearchHit]:
    """Cheap (no embedding) ranking of a pool: RRF(BM25 of the claim over the pool, the
    pool's own source order). Picks rerank_pool's dense head and citation seeds."""
    if not pool:
        return []
    return fuse_hits([bm25_order(claim, pool), pool], k=rrf_k or settings.rrf_k)


def rerank_pool(
    claim: str,
    pool: list[SearchHit],
    embedder,
    cache: EmbeddingCache | None = None,
    dense_cap: int | None = None,
    source_vote: bool = False,
    rrf_k: int | None = None,
) -> list[SearchHit]:
    """rerank() for a multi-source pool (`pool` in merge_pools order), bounded.

    Embedding is the expensive step (~60 ms per candidate on the reference CPU), so with
    `dense_cap` only the top `dense_cap` candidates of a cheap pre-ranking — RRF(BM25 of
    the claim over the pool, the pool's source order) — are embedded and re-ranked by
    RRF(dense, BM25); the rest follow in pre-rank order. `source_vote` adds the pool's
    source order as a third RRF list in the final fusion."""
    if not pool:
        return []
    k = rrf_k or settings.rrf_k
    head, tail = pool, []
    if dense_cap is not None and len(pool) > dense_cap:
        pre = prerank(claim, pool, k)
        head, tail = pre[:dense_cap], pre[dense_cap:]
        src_order = {h.doc_id: i for i, h in enumerate(pool)}
        head_src = sorted(head, key=lambda h: src_order[h.doc_id])
    else:
        head_src = pool
    lists = [dense_order(claim, head, embedder, cache), bm25_order(claim, head)]
    if source_vote:
        lists.append(head_src)
    ranked = fuse_hits(lists, k=k)
    if not tail:
        return ranked
    floor = min(h.score for h in ranked)
    return ranked + [SearchHit(h.doc_id, floor * (1 - (i + 1) / (len(tail) + 2)), h.text, h.metadata)
                     for i, h in enumerate(tail)]


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
        multi_query: int = 0,
        pubmed=None,
        snippets: bool = False,
        citation_seeds: int = 0,
        citation_cap: int = 30,
        resolver=None,
        dense_cap: int | None = None,
        pubmed_queries: tuple[str, ...] = ("rewrite",),
        snippet_queries: tuple[str, ...] = ("rewrite",),
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
        # Extra candidate sources (all off by default; see search_pooled). `pubmed` is a
        # PubMedSource, `resolver` an S2IdResolver (needed by pubmed/snippets/citations).
        self.multi_query = max(0, int(multi_query))
        self.pubmed = pubmed
        self.snippets = snippets
        self.citation_seeds = max(0, int(citation_seeds))
        self.citation_cap = citation_cap
        self.resolver = resolver
        self.dense_cap = dense_cap
        # Which query each extra source gets: "rewrite" (primary keyword query) / "claim".
        for kinds in (pubmed_queries, snippet_queries):
            if not kinds or set(kinds) - {"rewrite", "claim"}:
                raise ValueError("query kinds must be a non-empty subset of ('rewrite', 'claim')")
        self.pubmed_queries = tuple(pubmed_queries)
        self.snippet_queries = tuple(snippet_queries)
        self.last_errors: list[str] = []  # extra sources that failed in the last search
        if (pubmed is not None or snippets or self.citation_seeds) and resolver is None:
            raise ValueError("pubmed / snippets / citation expansion need an S2IdResolver")

    @property
    def pooled(self) -> bool:
        return bool(self.multi_query or self.pubmed is not None or self.snippets or self.citation_seeds)

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
        if self.pooled:
            return self.search_pooled(query, top_k, lock)
        limit = self.fetch_k if self.do_rerank else top_k
        pool = self.candidates(query, limit)
        if not self.do_rerank:
            return _restamp(pool)[:top_k]
        with lock or nullcontext():
            return rerank(query, pool, self.embedder, self.emb_cache)[:top_k]

    def _extra(self, name: str, fn, default):
        """Run one extra source; a failure degrades to `default` (recorded in
        last_errors) unless strict, where it raises — the eval must not score a thinner
        pool silently."""
        from app.retrieve.web_sources import SourceError

        try:
            return fn()
        except (S2Error, SourceError) as e:
            if self.strict:
                raise
            self.last_errors.append(f"{name}: {e}")
            return default

    def search_pooled(
        self, claim: str, top_k: int, lock: AbstractContextManager | None = None
    ) -> list[SearchHit]:
        """Multi-source candidate generation, then the local rerank of the whole pool:

            S2 rewrite queries (+ `multi_query` extra keyword queries)   ─┐
            PubMed Best Match (own host, parallel thread)                 ├─► resolve
            S2 snippet search                                             ─┘   ids (1 S2
            ──► citation expansion of the top `citation_seeds` (pre-ranked)     batch)
            ──► merge_pools ──► rerank_pool(dense_cap)

        The base S2 queries are required (their failure raises, as before); every extra
        source degrades to nothing on failure. S2 calls share the process-wide limiter;
        PubMed runs in a worker thread against its own host limiter."""
        from concurrent.futures import ThreadPoolExecutor

        from app.retrieve.query_rewrite import multi_queries
        from app.retrieve.s2_extra import expand_citations, snippet_search

        self.last_errors = []
        cache = getattr(self.s2, "cache", None)
        rws = self.queries(claim)
        primary = rws[0]
        by_kind = {"rewrite": primary, "claim": claim}
        pm_qs = list(dict.fromkeys(by_kind[k] for k in self.pubmed_queries))
        snip_qs = list(dict.fromkeys(by_kind[k] for k in self.snippet_queries))
        lists: list[list[SearchHit]] = []
        with ThreadPoolExecutor(max_workers=1) as pool_exec:
            pm_future = (
                pool_exec.submit(lambda: [self._extra("pubmed", lambda q=q: self.pubmed.search(q, 100), []) for q in pm_qs])
                if self.pubmed is not None
                else None
            )
            keep, self.keep_no_abstract = self.keep_no_abstract, True
            try:
                base = self.candidates(claim, self.fetch_k)  # raises if S2 fails
            finally:
                self.keep_no_abstract = keep
            lists.append(_tag(base, "s2"))
            if self.multi_query:
                extra_q = multi_queries(claim, self.rarity, self.multi_query) if self.rewrite else []
                lists.append(_tag(self._extra("s2_multi", lambda: self._fetch_all(extra_q), []), "s2_multi"))
            groups: list[list] = []
            if self.snippets:
                for q in snip_qs:
                    groups.append(self._extra("snippets", lambda q=q: snippet_search(self.s2, q, 100, cache), []))
            if pm_future is not None:
                groups += pm_future.result()
        groups = [g for g in groups if g]
        if groups:
            # One S2 batch resolves every source's ids; each source stays its own ranked
            # list for the pool's RRF order.
            # A failed resolve batch drops these sources for this query (recorded in
            # last_errors): unresolved ids could not be deduped against S2's.
            resolved = self._extra("resolve", lambda: (self.resolver.prefetch([p for g in groups for p in g]), True)[1], False)
            if resolved:
                for g in groups:
                    lists.append(self._extra("resolve", lambda g=g: self.resolver.to_hits(g, keep_no_abstract=True), []))
        if self.citation_seeds:
            # Seeds come from the abstract-bearing S2 base pool (BM25 needs text to rank).
            seeds = [h.doc_id for h in prerank(claim, merge_pools(lists[:1]))[: self.citation_seeds]
                     if h.doc_id.isdigit()]
            if seeds:
                cited = self._extra(
                    "citations",
                    lambda: self.resolver.to_hits(
                        expand_citations(self.s2, seeds, max_refs_per_seed=self.citation_cap,
                                         max_cites_per_seed=self.citation_cap, cache=cache),
                        keep_no_abstract=True,
                    ),
                    [],
                )
                lists.append(cited)
        merged = merge_pools(lists, keep_no_abstract=self.keep_no_abstract)
        if not self.do_rerank:
            return _restamp(merged)[:top_k]
        with lock or nullcontext():
            return rerank_pool(claim, merged, self.embedder, self.emb_cache, self.dense_cap)[:top_k]

    def _fetch_all(self, queries: list[str]) -> list[SearchHit]:
        out: list[SearchHit] = []
        seen: set[str] = set()
        for q in queries:
            entry = self.s2.fetch(q, self.fetch_k)
            self.last_entries.append(entry)
            for h in parse_search(entry.get("response"), self.fetch_k, keep_no_abstract=True):
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    out.append(h)
        return out


def _tag(hits: list[SearchHit], source: str) -> list[SearchHit]:
    return [SearchHit(h.doc_id, h.score, h.text, {**h.metadata, "source": source}) for h in hits]
