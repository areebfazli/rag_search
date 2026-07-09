"""Routing tests for SearchService using injected fake components — no models or
on-disk index required, so they're fast and deterministic."""
import pytest

from app.core.interfaces import SearchHit
from app.retrieve.service import SearchService


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


class FakeReranker:
    """Reverses the candidate order so a test can prove the reranker actually ran."""

    def rerank(self, query, hits, top_k):
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
