"""Routing tests for SearchService using injected fake components — no models or
on-disk index required, so they're fast and deterministic."""
import pytest

from app.core.config import settings
from app.core.interfaces import SearchHit
from app.retrieve.service import SearchService


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


class FakeReranker:
    """Reverses the candidate order so a test can prove the reranker actually ran."""

    def __init__(self):
        self.seen: int | None = None  # how many candidates the last call was handed

    def rerank(self, query, hits, top_k):
        self.seen = len(hits)
        return list(reversed(hits))[:top_k]


def make_service():
    dense = FakeRetriever([SearchHit("d1", 0.9, "t1"), SearchHit("d2", 0.8, "t2"), SearchHit("shared", 0.7, "ts")])
    lexical = FakeRetriever([SearchHit("l1", 5.0, "t3"), SearchHit("shared", 4.0, "ts"), SearchHit("l2", 3.0, "t4")])
    return SearchService(dense=dense, lexical=lexical, reranker=FakeReranker())


def test_bm25_mode_uses_lexical():
    assert [h.doc_id for h in make_service().retrieve("q", mode="bm25", top_k=2)] == ["l1", "shared"]


def test_dense_mode_uses_dense():
    assert [h.doc_id for h in make_service().retrieve("q", mode="dense", top_k=2)] == ["d1", "d2"]


def test_hybrid_fuses_and_rewards_shared_doc():
    ids = [h.doc_id for h in make_service().retrieve("q", mode="hybrid", top_k=5)]
    assert ids[0] == "shared"  # appears in both rankings → RRF ranks it first
    assert set(ids) == {"d1", "d2", "shared", "l1", "l2"}


def test_hybrid_rerank_invokes_reranker():
    svc = make_service()
    plain = [h.doc_id for h in svc.retrieve("q", mode="hybrid", top_k=5)]
    reranked = [h.doc_id for h in svc.retrieve("q", mode="hybrid_rerank", top_k=5)]
    assert reranked == list(reversed(plain))  # fake reranker reverses order


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        make_service().retrieve("q", mode="bogus")


def test_nonpositive_top_k_falls_back_to_default():
    # top_k=0 must not silently return nothing; it falls back to the configured default
    assert len(make_service().retrieve("q", mode="dense", top_k=0)) == 3


def test_rerank_candidate_slice_is_bounded_and_keeps_the_tail(monkeypatch):
    # Cross-encoder cost is linear in candidates, so an unbounded slice is both slow
    # and a DoS vector: the whole call is held under the API's retrieval lock.
    monkeypatch.setattr(settings, "rerank_candidates", 4)
    dense = FakeRetriever([SearchHit(f"d{i}", 1.0 - i / 100, f"t{i}") for i in range(20)])
    lexical = FakeRetriever([SearchHit(f"l{i}", 20.0 - i, f"s{i}") for i in range(20)])
    reranker = FakeReranker()
    svc = SearchService(dense=dense, lexical=lexical, reranker=reranker)

    plain = [h.doc_id for h in svc.retrieve("q", mode="hybrid", top_k=40, candidate_k=40)]
    reranked = [h.doc_id for h in svc.retrieve("q", mode="hybrid_rerank", top_k=40, candidate_k=40)]

    assert reranker.seen == 4  # bounded regardless of how deep the fused pool is
    assert reranked[:4] == plain[:4][::-1]  # the slice really was reordered
    assert reranked[4:] == plain[4:]  # ...and the tail kept its fused order
    assert len(reranked) == len(plain)  # so recall past the slice is unchanged


def test_rerank_scores_stay_monotonic_across_the_slice_boundary(monkeypatch):
    # The reranked head carries cross-encoder logits and the tail carries RRF scores —
    # two incompatible scales. Concatenating them raw leaves the order right but the
    # score field non-monotonic, so a client sorting by score (or a UI displaying it)
    # would contradict the ranking.
    monkeypatch.setattr(settings, "rerank_candidates", 3)

    class LogitReranker:
        def rerank(self, query, hits, top_k):
            # Cross-encoder-like scores, including negatives, on a different scale
            # from RRF's ~0.016.
            return [
                SearchHit(h.doc_id, -5.0 - i, h.text, h.metadata)
                for i, h in enumerate(reversed(hits))
            ][:top_k]

    dense = FakeRetriever([SearchHit(f"d{i}", 1.0 - i / 100, f"t{i}") for i in range(10)])
    lexical = FakeRetriever([SearchHit(f"l{i}", 10.0 - i, f"s{i}") for i in range(10)])
    svc = SearchService(dense=dense, lexical=lexical, reranker=LogitReranker())

    hits = svc.retrieve("q", mode="hybrid_rerank", top_k=20, candidate_k=20)
    scores = [h.score for h in hits]
    assert len(scores) > 3, "need results past the reranked slice to exercise the boundary"
    assert scores == sorted(scores, reverse=True), f"non-monotonic scores: {scores}"


# --- optional web modes (Semantic Scholar), with a fake web retriever -----------------

from app.retrieve.semantic_scholar import S2Error  # noqa: E402


class FakeWeb:
    def __init__(self, hits=None, error=None):
        self._hits = hits or []
        self._error = error
        self.calls: list[int] = []

    def search(self, query, top_k):
        self.calls.append(top_k)
        if self._error:
            raise self._error
        return self._hits[:top_k]


def web_hit(doc_id, title="web title"):
    return SearchHit(
        doc_id,
        1.0,
        f"web abstract {doc_id}",
        {"title": title, "url": f"https://www.semanticscholar.org/paper/{doc_id}", "year": 2001,
         "source": "semantic_scholar"},
    )


def make_web_service(web):
    svc = make_service()
    svc._web = web
    return svc


def test_web_mode_uses_only_semantic_scholar():
    web = FakeWeb([web_hit("w1"), web_hit("w2")])
    svc = make_web_service(web)
    assert [h.doc_id for h in svc.retrieve("q", mode="web", top_k=1)] == ["w1"]
    assert web.calls == [1]


def test_web_mode_caps_depth_at_the_s2_page_size():
    web = FakeWeb([])
    make_web_service(web).retrieve("q", mode="web", top_k=500)
    assert web.calls == [100]


def test_web_mode_propagates_s2_failure():
    # No local fallback exists for `web`, so the caller (API -> 503) must see it.
    with pytest.raises(S2Error):
        make_web_service(FakeWeb(error=S2Error("HTTP 500"))).retrieve("q", mode="web")


def test_hybrid_web_fuses_and_dedupes_papers_in_both_sources():
    # "shared" is in the local corpus AND returned by S2 (same corpus id): it must
    # appear once, keep the local text/title, gain S2's url/year, and — being in both
    # the local hybrid list and the web list — rank first under RRF.
    web = FakeWeb([web_hit("w1"), web_hit("shared", title="S2 title")])
    svc = make_web_service(web)
    hits = svc.retrieve("q", mode="hybrid_web", top_k=10)
    ids = [h.doc_id for h in hits]
    assert ids.count("shared") == 1 and ids[0] == "shared"
    assert set(ids) == {"d1", "d2", "shared", "l1", "l2", "w1"}
    shared = hits[0]
    assert shared.text == "ts" and shared.metadata["source"] == "local"
    assert shared.metadata["url"].endswith("/shared") and shared.metadata["year"] == 2001
    w1 = next(h for h in hits if h.doc_id == "w1")
    assert w1.metadata["source"] == "semantic_scholar"
    assert all(h.metadata.get("source") == "local" for h in hits if h.doc_id[0] in "dl")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_web_matches_rrf_of_local_hybrid_and_web_lists():
    from app.retrieve.fusion import reciprocal_rank_fusion

    web = FakeWeb([web_hit("w1"), web_hit("l2"), web_hit("w2")])
    svc = make_web_service(web)
    local = [h.doc_id for h in svc.retrieve("q", mode="hybrid", top_k=100)]
    expected = [d for d, _ in reciprocal_rank_fusion([local, ["w1", "l2", "w2"]], k=settings.rrf_k)]
    assert [h.doc_id for h in svc.retrieve("q", mode="hybrid_web", top_k=100)] == expected


def test_hybrid_web_degrades_to_local_on_s2_failure():
    svc = make_web_service(FakeWeb(error=S2Error("HTTP 429")))
    warnings: list[str] = []
    hits = svc.retrieve("q", mode="hybrid_web", top_k=5, warnings=warnings)
    assert [h.doc_id for h in hits] == [h.doc_id for h in svc.retrieve("q", mode="hybrid", top_k=5)]
    assert len(warnings) == 1 and "Semantic Scholar unavailable" in warnings[0]


def test_hybrid_web_holds_local_lock_only_around_local_retrieval():
    events: list[str] = []

    class Lock:
        def __enter__(self):
            events.append("lock")

        def __exit__(self, *exc):
            events.append("unlock")

    class RecordingWeb(FakeWeb):
        def search(self, query, top_k):
            events.append("web")
            return super().search(query, top_k)

    svc = make_web_service(RecordingWeb([web_hit("w1")]))
    svc.retrieve("q", mode="hybrid_web", top_k=3, local_lock=Lock())
    assert events == ["lock", "unlock", "web"]  # the S2 call never runs under the lock


def test_local_modes_never_touch_the_web_retriever():
    web = FakeWeb([web_hit("w1")])
    svc = make_web_service(web)
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        svc.retrieve("q", mode=mode, top_k=3)
    assert web.calls == []


def test_hybrid_is_still_the_default_mode():
    import inspect

    assert inspect.signature(SearchService.retrieve).parameters["mode"].default == "hybrid"


def test_web_title_only_hits_are_marked_never_empty():
    from app.retrieve.service import NO_ABSTRACT_TEXT

    bare = SearchHit("w2", 0.9, "", {"title": "Only a title", "source": "s2"})
    svc = make_web_service(FakeWeb([web_hit("w1"), bare]))
    hits = svc.retrieve("q", mode="web", top_k=2)
    assert hits[0].text == "web abstract w1" and "no_abstract" not in hits[0].metadata
    assert hits[1].text == NO_ABSTRACT_TEXT and hits[1].metadata["no_abstract"] is True
    fused = {h.doc_id: h for h in svc.retrieve("q", mode="hybrid_web", top_k=10)}
    assert fused["w2"].text == NO_ABSTRACT_TEXT


def test_web_keep_no_abstract_is_off_by_default_and_reaches_the_pipeline(monkeypatch):
    import app.retrieve.service as service_mod

    assert make_service().web_keep_no_abstract is False
    seen = {}

    class SpyWebSearch:
        def __init__(self, s2, **kw):
            seen.update(kw)

    monkeypatch.setattr(service_mod, "WebSearch", SpyWebSearch)
    monkeypatch.setattr(service_mod, "web_extras", lambda s2: {})
    monkeypatch.setattr("app.retrieve.semantic_scholar.SemanticScholarRetriever.for_api", classmethod(lambda cls: None))
    svc = SearchService(dense=FakeRetriever([]), lexical=FakeRetriever([]), web_keep_no_abstract=True)
    svc.web  # noqa: B018 - builds the pipeline
    assert seen["keep_no_abstract"] is True
