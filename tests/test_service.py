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
