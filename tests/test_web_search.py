"""Web pipeline (app/retrieve/web_search.py): rewrite -> S2 pool -> local rerank. Fakes only:
no network, no model download, no index."""
import threading

import httpx
import numpy as np
import pytest

from app.core.interfaces import SearchHit
from app.index.embedder import Embedder
from app.retrieve.semantic_scholar import S2Error, SemanticScholarRetriever
from app.retrieve.service import SearchService
from app.retrieve.web_search import (
    EmbeddingCache,
    WebSearch,
    bm25_order,
    candidate_passage,
    dense_order,
    rerank,
)

VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]


def bag(text: str) -> np.ndarray:
    v = np.array([text.lower().count(w) for w in VOCAB], dtype=np.float32) + 1e-3
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Bag-of-words vectors; records what was encoded as a query vs as documents."""

    def __init__(self):
        self.queries: list[str] = []
        self.documents: list[str] = []

    def encode_query(self, text):
        self.queries.append(text)
        return bag(text)

    def encode_documents(self, texts, batch_size=32, show_progress=True):
        self.documents += list(texts)
        return np.stack([bag(t) for t in texts])


def paper(cid, title, abstract="An abstract."):
    return {"corpusId": cid, "title": title, "abstract": abstract, "url": None, "year": 2020}


class FakeS2:
    """`fetch(query, limit)` with scripted payloads per query; records every call."""

    def __init__(self, by_query=None, error_on=()):
        self.by_query = by_query or {}
        self.error_on = set(error_on)
        self.calls: list[tuple[str, int]] = []

    def fetch(self, query, limit):
        self.calls.append((query, limit))
        if query in self.error_on:
            raise S2Error("HTTP 429")
        return {"response": {"data": self.by_query.get(query, [])[:limit]}, "fetched_at": "2026-09-30"}


def hits(*specs):
    return [SearchHit(str(i), 1.0, abstract, {"title": title}) for i, (title, abstract) in enumerate(specs)]


def test_rerank_orders_candidates_by_the_claim_not_by_s2():
    # S2 put the on-topic paper last; dense + BM25 agree it belongs first.
    pool = hits(("Gamma study", "gamma gamma"), ("Delta note", "delta"), ("Alpha beta", "alpha beta alpha"))
    out = rerank("alpha beta", pool, FakeEmbedder())
    assert [h.doc_id for h in out][0] == "2"
    assert sorted(h.doc_id for h in out) == ["0", "1", "2"]  # reorders, never drops
    assert [h.score for h in out] == sorted((h.score for h in out), reverse=True)
    emb = FakeEmbedder()
    assert rerank("alpha", [], emb) == [] and dense_order("alpha", [], emb, None) == []
    assert bm25_order("alpha", []) == [] and emb.queries == emb.documents == []


def test_prefix_goes_on_the_claim_only():
    class FakeModel:
        def __init__(self):
            self.seen: list = []

        def encode(self, texts, **kw):
            self.seen.append(texts)
            n = 1 if isinstance(texts, str) else len(texts)
            out = np.ones((n, len(VOCAB)), dtype=np.float32) / np.sqrt(len(VOCAB))
            return out[0] if isinstance(texts, str) else out

    emb = Embedder.__new__(Embedder)  # real Embedder logic, fake model: no download
    emb.model, emb.query_prefix = FakeModel(), "PREFIX: "
    pool = hits(("T1", "abstract one"), ("T2", "abstract two"))
    dense_order("the claim", pool, emb, None)
    query_calls = [t for t in emb.model.seen if isinstance(t, str)]
    doc_calls = [t for t in emb.model.seen if not isinstance(t, str)]
    assert query_calls == ["PREFIX: the claim"]
    assert doc_calls == [[candidate_passage(h) for h in pool]]
    assert not any(p.startswith("PREFIX") for p in doc_calls[0])


def test_candidate_passage_matches_the_local_passage_shape():
    assert candidate_passage(SearchHit("1", 1.0, "Body.", {"title": "Title"})) == "Title\n\nBody."
    assert candidate_passage(SearchHit("1", 1.0, "", {"title": "Title only"})) == "Title only"


def test_embedding_cache_makes_a_rerun_embed_nothing(tmp_path):
    pool = hits(("Alpha", "alpha"), ("Beta", "beta"))
    first = FakeEmbedder()
    cache = EmbeddingCache(tmp_path / "emb", "fake-model")
    a = rerank("alpha", pool, first, cache)
    assert len(first.documents) == 2 and cache.misses == 2 and cache.hits == 0

    second = FakeEmbedder()
    fresh = EmbeddingCache(tmp_path / "emb", "fake-model")  # new instance: read from disk
    b = rerank("alpha", pool, second, fresh)
    assert second.documents == [] and fresh.hits == 2 and fresh.misses == 0
    assert [h.doc_id for h in a] == [h.doc_id for h in b]
    # Keyed by model: another model never reuses these vectors.
    other = EmbeddingCache(tmp_path / "emb", "other-model")
    assert other.get_many([candidate_passage(h) for h in pool]) == {}


def test_plain_pipeline_is_the_old_web_mode():
    papers = [paper(1, "A"), paper(2, "B", abstract=None), paper(3, "C")]

    def handler(request):
        return httpx.Response(200, json={"data": papers})

    s2 = SemanticScholarRetriever(
        api_key="", client=httpx.Client(transport=httpx.MockTransport(handler)),
        limiter=type("L", (), {"acquire": lambda self, timeout=None: True})(), max_retries=0,
    )
    old = s2.search("some claim", 10)
    new = WebSearch(s2).search("some claim", 10)
    assert [(h.doc_id, h.score, h.text, h.metadata) for h in new] == [
        (h.doc_id, h.score, h.text, h.metadata) for h in old
    ]
    assert [h.doc_id for h in WebSearch(s2, keep_no_abstract=True).search("x", 10)] == ["1", "2", "3"]


CLAIM = "Citrullinated proteins externalized in neutrophil extracellular traps disrupt inflammatory cycles in mice."


def test_rewrite_sends_keywords_and_pools_the_fallback_unless_the_primary_fills_the_page():
    probe = WebSearch(FakeS2(), rewrite=True)
    primary, fallback = probe.queries(CLAIM)
    assert len(primary.split()) > len(fallback.split())
    full = [paper(i, f"P{i}") for i in range(1, 11)]
    s2 = FakeS2({primary: full, fallback: [paper(99, "F")]})
    out = WebSearch(s2, rewrite=True).search(CLAIM, 10)
    assert s2.calls == [(primary, 10)] and len(out) == 10  # page full: no fallback

    s2 = FakeS2({primary: [paper(1, "P1"), paper(2, "P2", abstract=None)], fallback: [paper(2, "P2"), paper(3, "P3")]})
    out = WebSearch(s2, rewrite=True).search(CLAIM, 100)
    assert [q for q, _ in s2.calls] == [primary, fallback]
    assert [h.doc_id for h in out] == ["1", "2", "3"]  # primary first, deduped
    # web_any decides on the same count (papers WITH an abstract), so it sends the same requests.
    s2_any = FakeS2(dict(s2.by_query))
    WebSearch(s2_any, rewrite=True, keep_no_abstract=True).search(CLAIM, 100)
    assert s2_any.calls == s2.calls


def test_failed_fallback_degrades_in_the_api_and_raises_in_strict_mode():
    primary, fallback = WebSearch(FakeS2(), rewrite=True).queries(CLAIM)
    by_query = {primary: [paper(1, "P1")]}
    out = WebSearch(FakeS2(by_query, error_on={fallback}), rewrite=True).search(CLAIM, 10)
    assert [h.doc_id for h in out] == ["1"]
    with pytest.raises(S2Error):
        WebSearch(FakeS2(by_query, error_on={fallback}), rewrite=True, strict=True).search(CLAIM, 10)
    with pytest.raises(S2Error):  # a failed primary always raises (web -> 503)
        WebSearch(FakeS2(error_on={primary}), rewrite=True).search(CLAIM, 10)


def test_rerank_fetches_a_full_page_scores_the_raw_claim_and_cuts_to_top_k():
    s2 = FakeS2({CLAIM: [paper(1, "Gamma", "gamma"), paper(2, "Alpha", "alpha alpha")]})
    emb = FakeEmbedder()
    out = WebSearch(s2, rerank=True, embedder=lambda: emb).search(CLAIM, 1)
    assert s2.calls == [(CLAIM, 100)]  # the whole page, not top_k
    assert emb.queries == [CLAIM] and len(out) == 1
    # With rewrite too, S2 sees the keywords but the reranker still scores the CLAIM.
    primary = WebSearch(FakeS2(), rewrite=True).queries(CLAIM)[0]
    s2 = FakeS2({primary: [paper(i, f"P{i}") for i in range(1, 20)]})
    emb = FakeEmbedder()
    out = WebSearch(s2, rewrite=True, rerank=True, embedder=emb).search(CLAIM, 5)
    assert [c for c in s2.calls] == [(q, 100) for q in WebSearch(FakeS2(), rewrite=True).queries(CLAIM)]
    assert CLAIM not in [q for q, _ in s2.calls]
    assert emb.queries == [CLAIM] and len(out) == 5


def test_service_runs_the_rerank_under_the_local_lock_and_s2_outside_it():
    events: list[str] = []

    class Lock:
        def __enter__(self):
            events.append("lock")

        def __exit__(self, *exc):
            events.append("unlock")

    class RecordingS2(FakeS2):
        def fetch(self, query, limit):
            events.append("s2")
            return super().fetch(query, limit)

    class RecordingEmbedder(FakeEmbedder):
        def encode_query(self, text):
            events.append("embed")
            return super().encode_query(text)

    s2 = RecordingS2({"alpha claim": [paper(1, "Alpha", "alpha")]})
    web = WebSearch(s2, rerank=True, embedder=RecordingEmbedder())

    class NoLocal:
        def search(self, q, k):
            raise AssertionError("web mode touched the local index")

    svc = SearchService(dense=NoLocal(), lexical=NoLocal(), web=web)
    svc.retrieve("alpha claim", mode="web", top_k=3, local_lock=Lock())
    assert events == ["s2", "lock", "embed", "unlock"]


def test_service_builds_the_pipeline_from_settings(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "s2_query_rewrite", True)
    monkeypatch.setattr(settings, "s2_rerank", True)
    monkeypatch.setattr(settings, "s2_api_cache", False)

    class Lexical:
        docs = [{"doc_id": "1", "title": "Rare words", "text": "zymogen"}]

        def search(self, q, k):
            return []

    svc = SearchService(dense=Lexical(), lexical=Lexical())
    web = svc.web
    assert isinstance(web, WebSearch) and web.rewrite and web.do_rerank and web.emb_cache is None
    assert web.rarity.n_docs == 1 and web.rarity.df["zymogen"] == 1  # from the BM25 docs


def test_embedder_is_resolved_once_under_concurrency():
    made: list = []

    def factory():
        made.append(1)
        return FakeEmbedder()

    web = WebSearch(FakeS2(), rerank=True, embedder=factory)
    threads = [threading.Thread(target=lambda: web.embedder) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1
