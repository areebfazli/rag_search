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


# --- multi-source pooling ---------------------------------------------------------
from app.retrieve.web_search import merge_pools, rerank_pool  # noqa: E402


def _h(doc_id, text="", source="s2", **meta):
    return SearchHit(doc_id, 1.0, text, {"source": source, **meta})


def test_merge_pools_dedupes_borrows_text_and_records_sources():
    s2 = [_h("1", "", title="One"), _h("2", "two abstract")]
    pm = [_h("3", "three", "pubmed"), _h("1", "one from pubmed", "pubmed", url="https://x.org/1")]
    out = merge_pools([s2, pm])
    by = {h.doc_id: h for h in out}
    assert sorted(by) == ["1", "2", "3"]
    assert by["1"].text == "one from pubmed" and by["1"].metadata["title"] == "One"
    assert by["1"].metadata["url"] == "https://x.org/1"
    assert by["1"].metadata["sources"] == ["s2", "pubmed"]
    assert out[0].doc_id == "1"  # found by both sources -> first in RRF order
    # Text-less papers are dropped unless asked for.
    assert [h.doc_id for h in merge_pools([[_h("9")], [_h("8", "x")]])] == ["8"]
    assert {h.doc_id for h in merge_pools([[_h("9")]], keep_no_abstract=True)} == {"9"}
    assert merge_pools([]) == [] and merge_pools([[], []]) == []


def test_rerank_pool_without_cap_matches_rerank():
    pool = hits(("Gamma study", "gamma gamma"), ("Delta note", "delta"), ("Alpha beta", "alpha beta alpha"))
    a = rerank("alpha beta", pool, FakeEmbedder())
    b = rerank_pool("alpha beta", pool, FakeEmbedder())
    assert [h.doc_id for h in a] == [h.doc_id for h in b]
    assert rerank_pool("x", [], FakeEmbedder()) == []


def test_rerank_pool_dense_cap_embeds_only_the_head_and_keeps_everything():
    pool = hits(*[(f"T{i}", "gamma" if i else "alpha beta") for i in range(10)])
    emb = FakeEmbedder()
    out = rerank_pool("alpha beta", pool, emb, dense_cap=3)
    assert len(emb.documents) == 3  # only the pre-ranked head is embedded
    assert sorted(h.doc_id for h in out) == sorted(h.doc_id for h in pool)
    assert out[0].doc_id == "0"
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True) and min(scores[:3]) > max(scores[3:])
    voted = rerank_pool("alpha beta", pool, FakeEmbedder(), dense_cap=3, source_vote=True)
    assert sorted(h.doc_id for h in voted) == sorted(h.doc_id for h in pool)


# --- WebSearch.search_pooled (multi-source) ---------------------------------------
from app.retrieve import s2_extra  # noqa: E402
from app.retrieve.query_rewrite import multi_queries  # noqa: E402
from app.retrieve.web_sources import ExternalPaper, SourceError  # noqa: E402


class FakePubMed:
    """`search(query, limit)` -> scripted ExternalPapers; records calls; can raise."""

    def __init__(self, papers=(), error=None):
        self.papers = list(papers)
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def search(self, query, limit):
        self.calls.append((query, limit))
        if self.error is not None:
            raise self.error
        return self.papers[:limit]


class FakeResolver:
    """prefetch/to_hits with the S2IdResolver contract, minus S2: doc_id = corpus id, else
    "pmid:<n>"; text = the source abstract."""

    def __init__(self, error=None):
        self.error = error
        self.prefetched: list[ExternalPaper] = []

    def prefetch(self, papers):
        if self.error is not None:
            raise self.error
        self.prefetched += list(papers)
        return 0

    def to_hits(self, papers, keep_no_abstract=False):
        out = []
        for p in papers:
            if not p.abstract and not keep_no_abstract:
                continue
            doc_id = p.corpus_id or f"pmid:{p.pmid}"
            out.append(SearchHit(doc_id, 1.0 / (len(out) + 1), p.abstract,
                                 {"title": p.title, "source": p.source}))
        return out


def _pm(rank, title, abstract, *, pmid=None, corpus_id=None, source="pubmed"):
    return ExternalPaper(source=source, rank=rank, title=title, abstract=abstract, pmid=pmid, corpus_id=corpus_id)


def test_pooled_is_off_by_default_and_on_with_any_extra_source():
    assert WebSearch(FakeS2()).pooled is False
    assert WebSearch(FakeS2(), rewrite=True, rerank=True, embedder=FakeEmbedder()).pooled is False
    assert WebSearch(FakeS2(), multi_query=2).pooled is True
    assert WebSearch(FakeS2(), pubmed=FakePubMed(), resolver=FakeResolver()).pooled is True
    assert WebSearch(FakeS2(), snippets=True, resolver=FakeResolver()).pooled is True
    assert WebSearch(FakeS2(), citation_seeds=2, resolver=FakeResolver()).pooled is True
    # multi_query alone needs no resolver; unpooled search() never enters search_pooled.
    s2 = FakeS2({"alpha": [paper(1, "A", "alpha")]})
    web = WebSearch(s2)
    web.search_pooled = lambda *a, **k: pytest.fail("pooled path used")
    assert [h.doc_id for h in web.search("alpha", 5)] == ["1"]


@pytest.mark.parametrize("kw", [{"pubmed": FakePubMed()}, {"snippets": True}, {"citation_seeds": 3}])
def test_extra_sources_without_a_resolver_are_refused(kw):
    with pytest.raises(ValueError):
        WebSearch(FakeS2(), **kw)
    WebSearch(FakeS2(), resolver=FakeResolver(), **kw)  # fine with one


def test_pubmed_papers_join_the_pool_and_lend_missing_abstracts():
    s2 = FakeS2({"alpha": [
        paper(1, "S2 no abstract", abstract=None),   # PubMed has this one's text
        paper(2, "S2 with abstract", "alpha two"),
        paper(3, "S2 text-less, unknown to PubMed", abstract=None),
    ]})
    pm = FakePubMed([
        _pm(0, "PM title for one", "pubmed text for one", pmid="11", corpus_id="1"),
        _pm(1, "PubMed only", "pubmed only text", pmid="99"),
    ])
    web = WebSearch(s2, pubmed=pm, resolver=FakeResolver())
    out = {h.doc_id: h for h in web.search("alpha", 10)}
    assert pm.calls == [("alpha", 100)]  # primary query (no rewrite -> the claim), 100 papers
    assert set(out) == {"1", "2", "pmid:99"}  # 3 has no text anywhere: dropped
    assert out["1"].text == "pubmed text for one"  # S2 paper without abstract borrowed it
    assert out["1"].metadata["title"] == "S2 no abstract"  # S2's own title wins
    assert out["1"].metadata["sources"] == ["s2", "pubmed"]
    assert out["pmid:99"].metadata["sources"] == ["pubmed"]
    assert out["2"].text == "alpha two"
    # keep_no_abstract=True keeps the text-less paper too.
    keep = WebSearch(FakeS2(dict(s2.by_query)), pubmed=FakePubMed(pm.papers), resolver=FakeResolver(),
                     keep_no_abstract=True)
    assert {h.doc_id for h in keep.search("alpha", 10)} == {"1", "2", "3", "pmid:99"}
    assert web.run("alpha", 10).errors == []


def test_pooled_search_ranks_papers_found_by_both_sources_first_and_cuts_to_top_k():
    s2 = FakeS2({"alpha": [paper(1, "A1", "one"), paper(2, "A2", "two")]})
    pm = FakePubMed([_pm(0, "PM", "pm text", pmid="5"), _pm(1, "A2 again", "two", pmid="6", corpus_id="2")])
    out = WebSearch(s2, pubmed=pm, resolver=FakeResolver()).search("alpha", 2)
    assert len(out) == 2 and out[0].doc_id == "2"  # in both lists -> RRF head
    assert [h.score for h in out] == sorted((h.score for h in out), reverse=True)


def test_multi_query_sends_extra_keyword_queries_to_s2():
    probe = WebSearch(FakeS2(), rewrite=True)
    rewrites = probe.queries(CLAIM)
    extras = multi_queries(CLAIM, None, 2)
    assert extras and not set(extras) & set(rewrites)
    s2 = FakeS2({extras[0]: [paper(7, "Extra hit", "extra abstract")], rewrites[0]: [paper(1, "Base", "base")]})
    web = WebSearch(s2, rewrite=True, multi_query=2)
    res = web.run(CLAIM, 10)
    out = res.hits
    sent = [q for q, _ in s2.calls]
    assert sent[: len(rewrites)] == rewrites and sent[len(rewrites):] == extras
    assert all(limit == 100 for _, limit in s2.calls)
    assert {h.doc_id for h in out} == {"1", "7"}
    assert {h.doc_id: h.metadata["sources"] for h in out} == {"1": ["s2"], "7": ["s2_multi"]}
    assert len(res.entries) == len(sent)
    # Without rewrite there are no extra keyword queries: only the claim goes out.
    s2 = FakeS2({CLAIM: [paper(1, "Base", "base")]})
    WebSearch(s2, multi_query=2).search(CLAIM, 10)
    assert [q for q, _ in s2.calls] == [CLAIM]


@pytest.mark.parametrize("source", ["multi", "pubmed", "snippets", "resolve", "citations"])
def test_a_failing_extra_source_is_skipped_unless_strict(source, monkeypatch):
    base = {CLAIM: [paper(1, "Base", "alpha base")]}
    extras = multi_queries(CLAIM, None, 1)

    def build(strict):
        kw = {"strict": strict, "resolver": FakeResolver()}
        s2 = FakeS2({**base, **{q: [paper(2, "X", "x")] for q in extras}})
        if source == "multi":
            kw.update(multi_query=1, rewrite=True)
            s2 = FakeS2(dict(s2.by_query), error_on=set(extras))
            # rewrite=True sends keyword queries, not the claim: serve the base under them.
            primary = WebSearch(FakeS2(), rewrite=True).queries(CLAIM)[0]
            s2.by_query[primary] = base[CLAIM]
        elif source == "pubmed":
            kw["pubmed"] = FakePubMed(error=SourceError("HTTP 500"))
        elif source == "snippets":
            kw["snippets"] = True
            monkeypatch.setattr(s2_extra, "snippet_search", lambda *a, **k: (_ for _ in ()).throw(S2Error("HTTP 429")))
        elif source == "resolve":
            kw["pubmed"] = FakePubMed([_pm(0, "PM", "t", pmid="3")])
            kw["resolver"] = FakeResolver(error=S2Error("HTTP 504"))
        else:
            kw["citation_seeds"] = 1
            monkeypatch.setattr(s2_extra, "expand_citations", lambda *a, **k: (_ for _ in ()).throw(S2Error("HTTP 429")))
        return WebSearch(s2, **kw)

    web = build(strict=False)
    res = web.run(CLAIM, 10)
    assert [h.doc_id for h in res.hits] == ["1"]  # the S2 base results still come back
    assert len(res.errors) >= 1 and res.errors[0].split(":")[0] in {
        "s2_multi", "pubmed", "snippets", "resolve", "citations"}
    with pytest.raises((S2Error, SourceError)):
        build(strict=True).search(CLAIM, 10)
    # Errors belong to the request that hit them: a later clean search reports none.
    if source == "pubmed":
        web.pubmed.error = None
        assert web.run(CLAIM, 10).errors == []


def test_a_failing_base_s2_query_always_raises_even_with_extra_sources():
    pm = FakePubMed([_pm(0, "PM", "t", pmid="3")])
    web = WebSearch(FakeS2(error_on={CLAIM}), pubmed=pm, resolver=FakeResolver())  # strict=False
    with pytest.raises(S2Error):
        web.search(CLAIM, 10)
    # The PubMed worker is never joined (test_web_concurrency): one not yet started is
    # cancelled, one already running finishes its single call. Never more than that.
    assert len(pm.calls) <= 1
    with pytest.raises(S2Error):
        WebSearch(FakeS2(error_on={CLAIM}), multi_query=1, strict=True).search(CLAIM, 10)


def test_snippet_papers_are_resolved_into_the_pool(monkeypatch):
    seen = {}

    def fake_snippets(s2, query, limit, cache, **kw):
        seen["args"] = (s2, query, limit, cache)
        seen["kw"] = kw
        return [_pm(0, "Snip", "snippet abstract", corpus_id="42", source="s2_snippet")]

    monkeypatch.setattr(s2_extra, "snippet_search", fake_snippets)
    s2 = FakeS2({"alpha": [paper(1, "Base", "alpha")]})
    resolver = FakeResolver()
    out = {h.doc_id: h for h in WebSearch(s2, snippets=True, resolver=resolver).search("alpha", 10)}
    assert seen["args"] == (s2, "alpha", 100, None)
    assert set(out) == {"1", "42"} and out["42"].metadata["sources"] == ["s2_snippet"]
    assert [p.corpus_id for p in resolver.prefetched] == ["42"]


def test_citation_expansion_is_seeded_from_the_s2_base_pool(monkeypatch):
    calls = []

    def fake_expand(s2, seeds, *, max_refs_per_seed, max_cites_per_seed, cache):
        calls.append((s2, list(seeds), max_refs_per_seed, max_cites_per_seed, cache))
        return [_pm(0, "Cited", "cited alpha", corpus_id="500", source="s2_refs"),
                _pm(1, "Citing", "citing text", corpus_id="1", source="s2_cites")]

    monkeypatch.setattr(s2_extra, "expand_citations", fake_expand)
    s2 = FakeS2({"alpha beta": [paper(1, "Gamma", "gamma gamma"), paper(2, "Delta", "delta"),
                                paper(3, "Alpha", "alpha beta alpha")]})
    resolver = FakeResolver()
    web = WebSearch(s2, citation_seeds=2, citation_cap=7, resolver=resolver)
    out = {h.doc_id: h for h in web.search("alpha beta", 10)}
    assert len(calls) == 1
    got_s2, seeds, refs, cites, cache = calls[0]
    assert got_s2 is s2 and (refs, cites) == (7, 7) and cache is None
    assert len(seeds) == 2 and seeds[0] == "3"  # best pre-ranked base paper first
    assert set(seeds) <= {"1", "2", "3"} and all(s.isdigit() for s in seeds)
    assert {"1", "2", "3", "500"} == set(out)  # cited papers join the pool, deduped with base
    assert out["500"].metadata["sources"] == ["s2_refs"]
    assert out["1"].metadata["sources"] == ["s2", "s2_cites"]

    # No base papers -> no seeds -> expand_citations is never called.
    calls.clear()
    assert WebSearch(FakeS2(), citation_seeds=2, resolver=FakeResolver()).search("alpha", 5) == []
    assert calls == []


def test_dense_cap_bounds_the_embedding_work_but_not_the_candidates():
    s2 = FakeS2({"alpha": [paper(i, f"S2 paper {i}", "gamma" if i else "alpha alpha") for i in range(5)]})
    pm = FakePubMed([_pm(i, f"PM paper {i}", f"delta {i}", pmid=str(100 + i)) for i in range(4)])
    emb = FakeEmbedder()
    web = WebSearch(s2, rerank=True, embedder=emb, pubmed=pm, resolver=FakeResolver(), dense_cap=3)
    out = web.search("alpha", 100)
    assert len(emb.documents) == 3  # exactly dense_cap passages embedded
    assert len(out) == 9 and len({h.doc_id for h in out}) == 9  # the whole pool is returned
    assert out[0].doc_id == "0"
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True)
    emb2 = FakeEmbedder()
    out2 = WebSearch(FakeS2(dict(s2.by_query)), rerank=True, embedder=emb2, pubmed=FakePubMed(pm.papers),
                     resolver=FakeResolver(), dense_cap=3).search("alpha", 5)
    assert len(out2) == 5 and len(emb2.documents) == 3
    # No cap: every candidate is embedded.
    emb3 = FakeEmbedder()
    WebSearch(FakeS2(dict(s2.by_query)), rerank=True, embedder=emb3, pubmed=FakePubMed(pm.papers),
              resolver=FakeResolver()).search("alpha", 100)
    assert len(emb3.documents) == 9


def _offline_s2():
    def handler(request):
        pytest.fail("web_extras must not touch the network")

    return SemanticScholarRetriever(
        api_key="", client=httpx.Client(transport=httpx.MockTransport(handler)),
        limiter=type("L", (), {"acquire": lambda self, timeout=None: True})(), max_retries=0,
    )


def test_web_extras_defaults_add_nothing(monkeypatch):
    from app.core.config import settings
    from app.retrieve.service import web_extras

    for name, value in [("s2_multi_query", 0), ("web_pubmed", False), ("web_snippets", False),
                        ("web_citation_seeds", 0), ("web_citation_cap", 30), ("web_dense_cap", 0),
                        ("web_api_dense_cap", 0)]:
        monkeypatch.setattr(settings, name, value)
    kw = web_extras(_offline_s2())
    assert kw["multi_query"] == 0 and kw["snippets"] is False and kw["citation_seeds"] == 0
    assert kw["citation_cap"] == 30 and kw["dense_cap"] is None
    assert "pubmed" not in kw and "resolver" not in kw
    assert WebSearch(_offline_s2(), **kw).pooled is False


def test_web_extras_builds_pubmed_and_resolver_from_settings(monkeypatch):
    from app.core.config import settings
    from app.retrieve.pubmed import PubMedSource
    from app.retrieve.s2_extra import S2IdResolver
    from app.retrieve.service import web_extras

    for name, value in [("s2_multi_query", 2), ("web_pubmed", True), ("web_snippets", True),
                        ("web_citation_seeds", 3), ("web_citation_cap", 12), ("web_dense_cap", 40),
                        ("web_api_dense_cap", 0), ("s2_api_cache", False),
                        ("web_snippet_dataset_filter", False)]:
        monkeypatch.setattr(settings, name, value)
    s2 = _offline_s2()
    kw = web_extras(s2)
    assert isinstance(kw["pubmed"], PubMedSource) and isinstance(kw["resolver"], S2IdResolver)
    assert kw["resolver"].s2 is s2
    assert (kw["multi_query"], kw["snippets"], kw["citation_seeds"], kw["citation_cap"], kw["dense_cap"]) == (
        2, True, 3, 12, 40)
    web = WebSearch(s2, **kw)  # accepted by the constructor (resolver present)
    assert web.pooled and web.dense_cap == 40
    # Resolver alone (no PubMed) when only snippets/citations are on.
    monkeypatch.setattr(settings, "web_pubmed", False)
    kw = web_extras(s2)
    assert "pubmed" not in kw and isinstance(kw["resolver"], S2IdResolver)
